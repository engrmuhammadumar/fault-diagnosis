import os, json, argparse, random
import numpy as np
from datasets import ImageFolderIndexed, EpisodicBatcher, build_transforms
from model import Backbone, pairwise_distance
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def set_seed(seed=1337):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def compute_prototypes(emb_support, y_support, n_way):
    # mean embedding per class
    protos = []
    for c in range(n_way):
        protos.append(emb_support[y_support == c].mean(dim=0))
    return torch.stack(protos, dim=0)

def shrinkage_cov(features, y, n_way, alpha=0.1, per_class=False, eps=1e-4):
    # Ledoit-Wolf style shrinkage toward I for stability
    D = features.shape[1]
    I = torch.eye(D, device=features.device)
    if per_class:
        covs = []
        for c in range(n_way):
            X = features[y == c]
            Xc = X - X.mean(dim=0, keepdim=True)
            S = (Xc.t() @ Xc) / max(1, X.shape[0]-1) + eps * I
            S = (1 - alpha) * S + alpha * I
            covs.append(torch.inverse(S))
        return torch.stack(covs, dim=0)  # N x D x D (inverse)
    else:
        Xc = features - features.mean(dim=0, keepdim=True)
        S = (Xc.t() @ Xc) / max(1, features.shape[0]-1) + eps * I
        S = (1 - alpha) * S + alpha * I
        return torch.inverse(S)  # D x D

def episode_logits(emb_q, prototypes, metric, inv_cov):
    dists = pairwise_distance(emb_q, prototypes, metric=metric, inv_cov=inv_cov)  # B x N
    return -dists  # higher is better

def run_epoch(model, ds, optimizer, device, n_way, k_shot, q_query, episodes, distance, cov_mode, maha_alpha):
    model.train()
    losses, accs = [], []
    epi = EpisodicBatcher(ds, n_way=n_way, k_shot=k_shot, q_query=q_query)

    for _ in range(episodes):
        s_x, s_y, q_x, q_y = epi.sample_episode()
        s_x, s_y, q_x, q_y = s_x.to(device), s_y.to(device), q_x.to(device), q_y.to(device)

        emb_s = model(s_x); emb_q = model(q_x)
        protos = compute_prototypes(emb_s, s_y, n_way)

        inv_cov = None
        if distance == "mahalanobis":
            inv_cov = shrinkage_cov(emb_s, s_y, n_way, alpha=maha_alpha, per_class=(cov_mode == "per_class"))

        logits = episode_logits(emb_q, protos, distance, inv_cov)
        loss = F.cross_entropy(logits, q_y)

        optimizer.zero_grad(); loss.backward(); optimizer.step()

        acc = (logits.argmax(1) == q_y).float().mean().item()
        losses.append(loss.item()); accs.append(acc)

    return float(np.mean(losses)), float(np.mean(accs))

@torch.no_grad()
def evaluate(model, ds, device, n_way, k_shot, q_query, episodes, distance, cov_mode, maha_alpha,
             save_confusion=None, save_tsne=None):
    model.eval()
    epi = EpisodicBatcher(ds, n_way=n_way, k_shot=k_shot, q_query=q_query)
    accs = []; all_targets = []; all_preds = []
    tsne_emb = None; tsne_labels = None

    for e in range(episodes):
        s_x, s_y, q_x, q_y = epi.sample_episode()
        s_x, s_y, q_x, q_y = s_x.to(device), s_y.to(device), q_x.to(device), q_y.to(device)

        emb_s = model(s_x); emb_q = model(q_x)
        protos = compute_prototypes(emb_s, s_y, n_way)

        inv_cov = None
        if distance == "mahalanobis":
            inv_cov = shrinkage_cov(emb_s, s_y, n_way, alpha=maha_alpha, per_class=(cov_mode == "per_class"))

        logits = episode_logits(emb_q, protos, distance, inv_cov)
        preds = logits.argmax(1)
        accs.append((preds == q_y).float().mean().item())
        all_targets += q_y.cpu().tolist()
        all_preds += preds.cpu().tolist()

        if e == 0:
            tsne_emb = torch.cat([emb_s, emb_q], dim=0).cpu().numpy()
            tsne_labels = torch.cat([s_y, q_y], dim=0).cpu().numpy()

    mean_acc = float(np.mean(accs))

    if save_confusion is not None:
        cm = confusion_matrix(all_targets, all_preds, labels=list(range(n_way)))
        plt.figure(figsize=(5,4))
        plt.imshow(cm, interpolation="nearest")
        plt.title("Confusion Matrix"); plt.xlabel("Predicted"); plt.ylabel("True")
        plt.colorbar(); plt.tight_layout()
        os.makedirs(os.path.dirname(save_confusion), exist_ok=True)
        plt.savefig(save_confusion, dpi=200); plt.close()

    if save_tsne is not None and tsne_emb is not None:
        tsne = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto", random_state=42)
        Y = tsne.fit_transform(tsne_emb)
        plt.figure(figsize=(5,4))
        for c in range(n_way):
            idx = (tsne_labels == c)
            plt.scatter(Y[idx,0], Y[idx,1], s=12, label=f"class {c}")
        plt.legend(loc="best", fontsize=8)
        plt.title("t-SNE of Support+Query Embeddings")
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_tsne), exist_ok=True)
        plt.savefig(save_tsne, dpi=200); plt.close()

    return mean_acc

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="./runs/exp")
    p.add_argument("--backbone", type=str, default="resnet50")
    p.add_argument("--pretrained", type=lambda x: str(x).lower()=="true", default=True)
    p.add_argument("--embedding_dim", type=int, default=512)
    p.add_argument("--use_attention", type=lambda x: str(x).lower()=="true", default=True)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--imagenet_norm", type=lambda x: str(x).lower()=="true", default=True)
    p.add_argument("--batch_aug", type=str, default="light", choices=["none","light","strong"])

    p.add_argument("--n_way", type=int, default=5)
    p.add_argument("--k_shot", type=int, default=5)
    p.add_argument("--q_query", type=int, default=15)

    p.add_argument("--distance", type=str, default="mahalanobis", choices=["euclidean","cosine","mahalanobis"])
    p.add_argument("--cov_mode", type=str, default="shared", choices=["shared","per_class"])
    p.add_argument("--maha_alpha", type=float, default=0.1)

    p.add_argument("--episodes_per_epoch", type=int, default=400)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=1337)

    p.add_argument("--eval_only", type=lambda x: str(x).lower()=="true", default=False)
    p.add_argument("--episodes", type=int, default=600)
    p.add_argument("--load_checkpoint", type=str, default=None)
    p.add_argument("--save_confusion", type=str, default=None)
    p.add_argument("--save_tsne", type=str, default=None)

    args = p.parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    transform = build_transforms(img_size=args.img_size, imagenet_norm=args.imagenet_norm,
                                 batch_aug=args.batch_aug if not args.eval_only else "none")
    dataset = ImageFolderIndexed(args.data_dir, transform=transform)

    model = Backbone(name=args.backbone, pretrained=args.pretrained,
                     embedding_dim=args.embedding_dim, use_attention=args.use_attention).to(device)

    if args.load_checkpoint:
        ckpt = torch.load(args.load_checkpoint, map_location=device)
        model.load_state_dict(ckpt["model"])

    if args.eval_only:
        acc = evaluate(model, dataset, device, args.n_way, args.k_shot, args.q_query, args.episodes,
                       args.distance, args.cov_mode, args.maha_alpha,
                       args.save_confusion, args.save_tsne)
        print(f"[EVAL] {args.n_way}-way {args.k_shot}-shot acc over {args.episodes} episodes: {acc*100:.2f}%")
        return

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best = 0.0
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, dataset, optimizer, device, args.n_way, args.k_shot, args.q_query,
                                    args.episodes_per_epoch, args.distance, args.cov_mode, args.maha_alpha)
        val_acc = evaluate(model, dataset, device, args.n_way, args.k_shot, args.q_query, 100,
                           args.distance, args.cov_mode, args.maha_alpha)
        print(f"Epoch {epoch:03d} | train {tr_acc*100:.2f}% | val {val_acc*100:.2f}% | loss {tr_loss:.4f}")
        if val_acc > best:
            best = val_acc
            torch.save({"model": model.state_dict()}, os.path.join(args.output_dir, "best.pt"))
    print(f"Best val acc: {best*100:.2f}%")

if __name__ == "__main__":
    main()
