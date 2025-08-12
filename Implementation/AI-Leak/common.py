# common.py
from pathlib import Path
import json
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def load_classes_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

class CSVDataset(Dataset):
    """
    CSV columns expected: filepath,label,target
    If target missing, it will be derived from classes.json mapping using label
    """
    def __init__(self, csv_path: Path, classes_json: Path, img_size: int = 224, is_train: bool = True):
        self.df = pd.read_csv(csv_path)
        self.classes = load_classes_json(classes_json)  # dict: class_name -> index
        self.is_train = is_train

        # define transforms
        aug_train = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.RandomAffine(degrees=5, translate=(0.02,0.02))], p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        aug_eval = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        self.tf = aug_train if is_train else aug_eval

        # precompute targets
        if "target" in self.df.columns:
            self.targets = self.df["target"].tolist()
        else:
            self.targets = [self.classes[str(lbl)] for lbl in self.df["label"].tolist()]

        self.filepaths = self.df["filepath"].tolist()

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        path = self.filepaths[idx]
        img = Image.open(path).convert("RGB")
        x = self.tf(img)
        y = torch.tensor(self.targets[idx], dtype=torch.long)
        return x, y, path

def make_loader(csv_path: Path, classes_json: Path, batch_size=32, shuffle=True, img_size=224, is_train=True, num_workers=0):
    ds = CSVDataset(csv_path, classes_json, img_size=img_size, is_train=is_train)
    return DataLoader(
    ds,
    batch_size=batch_size,
    shuffle=shuffle,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),  # was True
)