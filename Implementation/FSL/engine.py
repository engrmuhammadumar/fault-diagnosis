import os
from typing import Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from config import Config
from utils import ensure_dir
from datasets import ClassBalancedSplit, EpisodicSampler, episode_collate
from model import EmbeddingNet
from proto import AdaptiveProtoBank
from metrics import mahalanobis_logits

class FSLEngine:
    def __init__(self, cfg: Config, device: torch.device):
        self.cfg = cfg
        self.device = device
        ensure_dir(cfg.out_dir)

        self.train_tf = transforms.Compose([
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.eval_tf = transforms.Compose([
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Full dataset object (we'll restrict sampling via the episodic sampler)
        self.dataset = datasets.ImageFolder(cfg.dataset_path, transform=self.eval_tf)
        self.targets = [y for _, y in self.dataset.samples]

        # Build class-balanced splits (indices are GLOBAL wrt self.dataset)
        self.splitter = ClassBalancedSplit(
            targets=self.targets,
            per_train=cfg.per_class_train,
            per_val=cfg.per_class_val,
            per_test=cfg.per_class_test,
            seed=cfg.seed,
        )

        # Separate datasets pointing to the same folder but different transforms
        self.train_set = datasets.ImageFolder(cfg.dataset_path, transform=self.train_tf)
        self.val_set = datasets.ImageFolder(cfg.dataset_path, transform=self.eval_tf)
        self.test_set = datasets.ImageFolder(cfg.dataset_path, transform=self.eval_tf)

        self.model = EmbeddingNet(embed_dim=cfg.embed_dim).to(device)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

        # Prototypes across all known classes
        self.bank = AdaptiveProtoBank(
            n_classes=len(self.dataset.classes),
            dim=cfg.embed_dim,
            beta=cfg.ema_beta,
            device=device
        )

    def _episode_loader(self, split: str):
        # Allowed GLOBAL dataset indices for this split
        allowed_indices = self.splitter.indices(split)

        # Episodic sampler chooses ONLY from allowed_indices and yields GLOBAL indices
        sampler = EpisodicSampler(
            targets=self.targets,
            indices=allowed_indices,
            n_way=self.cfg.n_way,
            k_shot=self.cfg.k_shot,
            q_query=self.cfg.q_query,
            episodes=self.cfg.episodes_per_epoch if split == "train" else max(50, self.cfg.episodes_per_epoch // 4),
            seed=self.cfg.seed if split != "train" else np.random.randint(0, 1_000_000),
        )

        # Use the full dataset for that split; sampler will index into it with GLOBAL indices
        ds = self.train_set if split == "train" else (self.val_set if split == "val" else self.test_set)
        return DataLoader(
            ds,
            batch_sampler=sampler,              # episodic sampler as batch_sampler
            num_workers=self.cfg.num_workers,
            persistent_workers=False,           # friendlier on Windows
            collate_fn=episode_collate
        )

    def _split_support_query(self, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        k, q, n = self.cfg.k_shot, self.cfg.q_query, self.cfg.n_way
        support_mask = torch.zeros_like(labels, dtype=torch.bool)
        query_mask = torch.zeros_like(labels, dtype=torch.bool)
        for i in range(n):
            start = i * (k + q)
            support_mask[start:start + k] = True
            query_mask[start + k:start + k + q] = True
        return support_mask, query_mask

    def train(self):
        cfg = self.cfg
        best_val = -1e9
        for epoch in range(1, cfg.max_epochs + 1):
            self.model.train()
            ep_loader = self._episode_loader("train")
            running_loss = 0.0
            num_batches = 0
            for imgs, labels in ep_loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                support_mask, query_mask = self._split_support_query(labels)

                embs = self.model(imgs)
                support_embs = embs[support_mask]
                query_embs = embs[query_mask]
                support_labels = labels[support_mask]
                query_labels = labels[query_mask]

                # Update bank with supports (detached inside bank)
                self.bank.update(support_labels, support_embs)
                self.bank.set_covariance(support_embs, shrink=cfg.cov_shrink)

                classes_in_episode = support_labels.unique()
                protos = self.bank.protos[classes_in_episode].detach()
                cov = self.bank.cov.detach()

                logits = mahalanobis_logits(query_embs, protos, cov)
                label_map = {c.item(): i for i, c in enumerate(classes_in_episode)}
                y = torch.tensor([label_map[int(l.item())] for l in query_labels], device=self.device)

                loss = torch.nn.functional.cross_entropy(logits, y)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                running_loss += loss.item()
                num_batches += 1

            val_acc = self.evaluate(split="val", episodes=100, silent=True)
            avg_loss = running_loss / max(1, num_batches)
            print(f"Epoch {epoch:02d} | loss {avg_loss:.4f} | val_acc {val_acc:.2f}%")

            # Save best
            if val_acc > best_val:
                best_val = val_acc
                ensure_dir(cfg.out_dir)
                torch.save({
                    "model": self.model.state_dict(),
                    "bank_protos": self.bank.protos,
                    "cfg": cfg.__dict__,
                }, os.path.join(cfg.out_dir, "best.pt"))

            # Periodic snapshot
            if epoch % cfg.save_every == 0:
                ensure_dir(cfg.out_dir)
                torch.save({
                    "model": self.model.state_dict(),
                    "bank_protos": self.bank.protos,
                    "cfg": cfg.__dict__,
                }, os.path.join(cfg.out_dir, f"epoch_{epoch}.pt"))

    @torch.no_grad()
    def evaluate(self, split="test", episodes: int = 200, silent: bool = False) -> float:
        self.model.eval()
        loader = self._episode_loader(split)
        correct = 0
        total = 0
        eps_done = 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            support_mask, query_mask = self._split_support_query(labels)
            embs = self.model(imgs)
            support_embs = embs[support_mask]
            query_embs = embs[query_mask]
            support_labels = labels[support_mask]
            query_labels = labels[query_mask]

            # Build episode-local protos and pooled cov from supports
            class_ids = support_labels.unique()
            protos = torch.stack([support_embs[support_labels == c].mean(0) for c in class_ids], dim=0)
            x = support_embs - support_embs.mean(dim=0, keepdim=True)
            cov = (x.t() @ x) / max(1, x.shape[0] - 1)
            eye = torch.eye(cov.shape[0], device=cov.device)
            cov = (1 - self.cfg.cov_shrink) * cov + self.cfg.cov_shrink * eye

            logits = mahalanobis_logits(query_embs, protos, cov)
            y_map = {c.item(): i for i, c in enumerate(class_ids)}
            y_true = torch.tensor([y_map[int(l.item())] for l in query_labels], device=self.device)
            pred = logits.argmax(dim=1)

            correct += (pred == y_true).sum().item()
            total += y_true.numel()

            eps_done += 1
            if eps_done >= episodes:
                break

        acc = 100.0 * correct / max(1, total)
        if not silent:
            print(f"{split} episodic accuracy over {eps_done} episodes: {acc:.2f}%")
        return acc

    @torch.no_grad()
    def export_confusion_and_tsne(self, split="test", outfile_prefix="fsl"):
        self.model.eval()
        loader = self._episode_loader(split)
        all_preds = []
        all_true = []
        tsne_embs = []
        tsne_labels = []

        episodes = 100
        done = 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            support_mask, query_mask = self._split_support_query(labels)
            embs = self.model(imgs)
            support_embs = embs[support_mask]
            query_embs = embs[query_mask]
            support_labels = labels[support_mask]
            query_labels = labels[query_mask]

            class_ids = support_labels.unique()
            protos = torch.stack([support_embs[support_labels == c].mean(0) for c in class_ids], dim=0)
            x = support_embs - support_embs.mean(dim=0, keepdim=True)
            cov = (x.t() @ x) / max(1, x.shape[0] - 1)
            eye = torch.eye(cov.shape[0], device=cov.device)
            cov = (1 - self.cfg.cov_shrink) * cov + self.cfg.cov_shrink * eye

            logits = mahalanobis_logits(query_embs, protos, cov)
            y_map = {c.item(): i for i, c in enumerate(class_ids)}
            pred_local = logits.argmax(dim=1)
            inv_map = {i: c.item() for i, c in enumerate(class_ids)}
            global_preds = [inv_map[int(p.item())] for p in pred_local]
            all_preds.extend(global_preds)
            all_true.extend([int(l.item()) for l in query_labels])

            tsne_embs.append(query_embs.cpu().numpy())
            tsne_labels.extend([int(l.item()) for l in query_labels])

            done += 1
            if done >= episodes:
                break

        ensure_dir(self.cfg.out_dir)
        cm = confusion_matrix(all_true, all_preds, labels=list(range(len(self.dataset.classes))))
        report = classification_report(all_true, all_preds, target_names=self.dataset.classes, digits=4)
        np.save(os.path.join(self.cfg.out_dir, f"{outfile_prefix}_confusion.npy"), cm)
        with open(os.path.join(self.cfg.out_dir, f"{outfile_prefix}_report.txt"), "w") as f:
            f.write(report)

        X = np.concatenate(tsne_embs, axis=0)
        tsne = TSNE(n_components=2, perplexity=30, learning_rate="auto", init="pca", random_state=self.cfg.seed)
        X2 = tsne.fit_transform(X)
        np.save(os.path.join(self.cfg.out_dir, f"{outfile_prefix}_tsne.npy"), X2)
        plt.figure()
        plt.scatter(X2[:, 0], X2[:, 1], c=np.array(tsne_labels), s=10)
        plt.title("t-SNE of query embeddings")
        plt.savefig(os.path.join(self.cfg.out_dir, f"{outfile_prefix}_tsne.png"), dpi=200)
        plt.close()
        print("Saved confusion matrix (.npy), classification report (.txt), and t-SNE (.npy/.png).")
