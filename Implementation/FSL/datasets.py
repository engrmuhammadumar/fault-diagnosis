from typing import List, Dict
import numpy as np
import torch
from torch.utils.data import Sampler
from torchvision import datasets, transforms

class ClassBalancedSplit:
    """Splits each class's indices into train/val/test for ImageFolder dataset."""
    def __init__(self, targets: List[int], per_train: int, per_val: int, per_test: int, seed: int = 42):
        self.targets = np.array(targets)
        self.per_train = per_train
        self.per_val = per_val
        self.per_test = per_test
        self.seed = seed
        self.splits = self._compute_splits()

    def _compute_splits(self) -> Dict[int, Dict[str, np.ndarray]]:
        rng = np.random.default_rng(self.seed)
        splits = {}
        classes = np.unique(self.targets)
        for c in classes:
            idxs = np.where(self.targets == c)[0]
            idxs = idxs.copy()
            rng.shuffle(idxs)
            n = len(idxs)
            want = self.per_train + self.per_val + self.per_test
            if n < want:
                raise ValueError(f"Class {c} has {n} samples, need {want}. Adjust per_class_* in Config.")
            train_idx = idxs[:self.per_train]
            val_idx = idxs[self.per_train:self.per_train+self.per_val]
            test_idx = idxs[self.per_train+self.per_val:self.per_train+self.per_val+self.per_test]
            splits[c] = {"train": train_idx, "val": val_idx, "test": test_idx}
        return splits

    def indices(self, split: str) -> List[int]:
        out = []
        for _, sub in self.splits.items():
            out.extend(sub[split].tolist())
        return out

class EpisodicSampler(Sampler):
    """Yields indices for episodes: choose n_way classes, sample k_shot + q_query per class."""
    def __init__(self, targets: List[int], indices: List[int], n_way: int, k_shot: int, q_query: int, episodes: int, seed: int = 42):
        self.targets = np.array(targets)
        self.indices = np.array(indices)
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.episodes = episodes
        self.rng = np.random.default_rng(seed)
        self.classes = np.unique(self.targets[self.indices])

    def __len__(self):
        return self.episodes

    def __iter__(self):
        for _ in range(self.episodes):
            chosen = self.rng.choice(self.classes, size=self.n_way, replace=False)
            episode_idxs = []
            for c in chosen:
                cls_idxs = self.indices[self.targets[self.indices] == c]
                self.rng.shuffle(cls_idxs)
                need = self.k_shot + self.q_query
                if len(cls_idxs) < need:
                    raise ValueError(f"Class {c} needs at least {need} samples for an episode.")
                take = cls_idxs[:need].tolist()
                episode_idxs.extend(take)
            yield episode_idxs

def episode_collate(batch):
    imgs, labels = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)
    return imgs, labels
