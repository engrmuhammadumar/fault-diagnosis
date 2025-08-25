import os, random
from typing import List, Dict
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

def build_transforms(img_size=224, imagenet_norm=True, batch_aug="light"):
    # Base resize + toTensor
    t_list = [transforms.Resize((img_size, img_size))]

    # Augmentations (applied to both support/query during training)
    if batch_aug == "light":
        t_list += [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.1,0.1,0.1,0.05)], p=0.3),
            transforms.RandomRotation(degrees=5),
        ]
    elif batch_aug == "strong":
        t_list += [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.2,0.2,0.2,0.1)], p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.RandomAffine(degrees=0, translate=(0.05,0.05), scale=(0.95,1.05)),
        ]

    t_list.append(transforms.ToTensor())

    if imagenet_norm:
        t_list.append(transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]))
    else:
        # If your images are spectrogram-like already in [0,1], this is often better.
        t_list.append(transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]))

    return transforms.Compose(t_list)

class ImageFolderIndexed(Dataset):
    # Simple ImageFolder that also tracks file paths per class (for episodic sampling).
    # Layout: root/class_x/*.png|jpg|jpeg|bmp|tif
    def __init__(self, root: str, transform=None):
        super().__init__()
        self.root = root
        self.transform = transform
        self.class_to_idx = {}
        self.samples = []  # (path, class_idx)
        self.paths_by_class: Dict[int, List[str]] = {}

        classes = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
        for ci, cname in enumerate(classes):
            self.class_to_idx[cname] = ci
            cdir = os.path.join(root, cname)
            imgs = [os.path.join(cdir, f) for f in os.listdir(cdir)
                    if f.lower().endswith((".png",".jpg",".jpeg",".bmp",".tif",".tiff"))]
            imgs.sort()
            if len(imgs) == 0:
                continue
            self.paths_by_class[ci] = imgs
            for p in imgs:
                self.samples.append((p, ci))

        self.classes = list(self.class_to_idx.keys())

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, ci = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, ci

class EpisodicBatcher:
    # Yields support/query tensors for N-way K-shot episodes.
    def __init__(self, dataset: ImageFolderIndexed, n_way=5, k_shot=5, q_query=10):
        self.ds = dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.valid_classes = [c for c in dataset.paths_by_class.keys()
                              if len(dataset.paths_by_class[c]) >= (k_shot + q_query)]
        if len(self.valid_classes) < n_way:
            raise ValueError("Not enough classes with sufficient images for the given N-way/K-shot/Q.")

    def sample_episode(self):
        classes = random.sample(self.valid_classes, self.n_way)
        support_imgs, support_labels = [], []
        query_imgs, query_labels = [], []
        for i, c in enumerate(classes):
            paths = random.sample(self.ds.paths_by_class[c], self.k_shot + self.q_query)
            s_paths = paths[:self.k_shot]
            q_paths = paths[self.k_shot:]
            for p in s_paths:
                img = Image.open(p).convert("RGB")
                img = self.ds.transform(img) if self.ds.transform else img
                support_imgs.append(img); support_labels.append(i)
            for p in q_paths:
                img = Image.open(p).convert("RGB")
                img = self.ds.transform(img) if self.ds.transform else img
                query_imgs.append(img); query_labels.append(i)

        support = torch.stack(support_imgs, dim=0)
        query = torch.stack(query_imgs, dim=0)
        return support, torch.tensor(support_labels), query, torch.tensor(query_labels)
