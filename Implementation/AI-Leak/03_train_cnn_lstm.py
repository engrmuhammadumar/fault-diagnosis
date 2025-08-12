# 03_train_cnn_lstm.py
from pathlib import Path
import time
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import pandas as pd
from tqdm import tqdm
import json
import math
import warnings

# -------------------
# CONFIG
# -------------------
DATA_ROOT = Path(r"E:\Upwork Project\AI_Leak_Detection_Project\images\cwt_log")
OUT_DIR = DATA_ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EPOCHS = 12
BATCH_SIZE = 16            # a bit smaller; each sample is a sequence
IMG_SIZE = 224             # each segment is resized to this
NUM_SEGMENTS = 8           # how many time-steps per image (try 6–12)
LR = 3e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0            # safest on Windows
LABEL_SMOOTH = 0.05

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# -------------------
# UTILITIES
# -------------------
def load_classes_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

class SeqStripeDataset(Dataset):
    """
    Reads CSV with columns: filepath, label, target.
    For each image, splits along width into NUM_SEGMENTS vertical stripes,
    resizes each stripe to (IMG_SIZE, IMG_SIZE), and returns a tensor of
    shape (S, 3, IMG_SIZE, IMG_SIZE) + target.
    """
    def __init__(self, csv_path: Path, classes_json: Path, num_segments=8, img_size=224, is_train=True):
        self.df = pd.read_csv(csv_path)
        self.classes = load_classes_json(classes_json)
        self.num_segments = num_segments
        self.img_size = img_size
        self.is_train = is_train

        # build targets
        if "target" in self.df.columns:
            self.targets = self.df["target"].tolist()
        else:
            self.targets = [self.classes[str(lbl)] for lbl in self.df["label"].tolist()]
        self.paths = self.df["filepath"].tolist()

        # transforms for each segment
        # (light jitter; you can add RandomResizedCrop for more)
        self.tf_train = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.RandomAffine(degrees=5, translate=(0.02, 0.02))], p=0.3),
            transforms.ColorJitter(brightness=0.05, contrast=0.05),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        self.tf_eval = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        self.tf = self.tf_train if is_train else self.tf_eval

    def _stripe_bounds(self, w: int) -> List[Tuple[int, int]]:
        """Return [(x0,x1), ...] for num_segments approximately equal-width splits."""
        seg_w = w / self.num_segments
        bounds = []
        for s in range(self.num_segments):
            x0 = int(round(s * seg_w))
            x1 = int(round((s + 1) * seg_w))
            # ensure at least 1 pixel wide
            if x1 <= x0:
                x1 = min(w, x0 + 1)
            bounds.append((x0, min(x1, w)))
        # fix last x1
        bounds[-1] = (bounds[-1][0], w)
        return bounds

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        y = torch.tensor(self.targets[idx], dtype=torch.long)
        img = Image.open(p).convert("RGB")
        W, H = img.size

        segments = []
        for (x0, x1) in self._stripe_bounds(W):
            stripe = img.crop((x0, 0, x1, H))  # (left, top, right, bottom)
            segments.append(self.tf(stripe))   # (3, IMG_SIZE, IMG_SIZE)

        # (S, 3, H, W)
        x = torch.stack(segments, dim=0)
        return x, y, p

def make_loader_seq(csv_path: Path, classes_json: Path, batch_size=16, shuffle=True, img_size=224, num_segments=8, is_train=True, num_workers=0):
    ds = SeqStripeDataset(csv_path, classes_json, num_segments=num_segments, img_size=img_size, is_train=is_train)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available())

# -------------------
# MODEL
# -------------------
class CNNEncoder(nn.Module):
    """
    ResNet-18 encoder returning a 512-d feature vector per image (no final FC).
    """
    def __init__(self):
        super().__init__()
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # take all but the final FC
        self.feature_extractor = nn.Sequential(
            m.conv1, m.bn1, m.relu, m.maxpool,
            m.layer1, m.layer2, m.layer3, m.layer4,
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.out_dim = 512

    def forward(self, x):
        # x: (B, 3, H, W) -> (B, 512)
        f = self.feature_extractor(x)           # (B, 512, 1, 1)
        f = f.view(f.size(0), -1)               # (B, 512)
        return f

class CNNLSTM(nn.Module):
    """
    Time-distributed CNN encoder over S segments, then LSTM over sequence.
    """
    def __init__(self, num_classes: int, cnn_out_dim=512, hidden=256, num_layers=1, bidirectional=True, dropout=0.1):
        super().__init__()
        self.cnn = CNNEncoder()
        self.lstm = nn.LSTM(
            input_size=cnn_out_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0
        )
        d = hidden * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, num_classes)
        )

    def forward(self, x_seq):
        # x_seq: (B, S, 3, H, W)
        B, S, C, H, W = x_seq.shape
        x = x_seq.view(B * S, C, H, W)      # merge time into batch
        feats = self.cnn(x)                 # (B*S, 512)
        feats = feats.view(B, S, -1)        # (B, S, 512)

        out, (h_n, c_n) = self.lstm(feats)  # out: (B, S, d)
        # take last timestep
        last = out[:, -1, :]                # (B, d)
        logits = self.head(last)            # (B, num_classes)
        return logits

# -------------------
# TRAIN / EVAL
# -------------------
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    crit = nn.CrossEntropyLoss()
    total = 0
    correct = 0
    loss_sum = 0.0
    for x_seq, y, _ in loader:
        x_seq = x_seq.to(device)   # (B, S, 3, H, W)
        y = y.to(device)
        logits = model(x_seq)
        loss = crit(logits, y)
        preds = logits.argmax(dim=1)
        total += y.size(0)
        correct += (preds == y).sum().item()
        loss_sum += loss.item() * y.size(0)
    return loss_sum / total, correct / total

def get_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def main():
    device = get_device()
    print("Device:", device)

    classes_json = DATA_ROOT / "classes.json"
    classes = load_classes_json(classes_json)
    num_classes = len(classes)
    print("Classes:", classes)

    train_csv = DATA_ROOT / "manifest_train.csv"
    val_csv   = DATA_ROOT / "manifest_val.csv"

    train_loader = make_loader_seq(train_csv, classes_json, batch_size=BATCH_SIZE, shuffle=True,
                                   img_size=IMG_SIZE, num_segments=NUM_SEGMENTS, is_train=True, num_workers=NUM_WORKERS)
    val_loader   = make_loader_seq(val_csv, classes_json, batch_size=BATCH_SIZE, shuffle=False,
                                   img_size=IMG_SIZE, num_segments=NUM_SEGMENTS, is_train=False, num_workers=NUM_WORKERS)

    model = CNNLSTM(num_classes=num_classes).to(device)

    crit = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
    optimiz = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimiz, T_max=EPOCHS)

    best_val_acc = 0.0
    stamp = time.strftime("%Y%m%d_%H%M%S")
    ckpt_path = OUT_DIR / f"cnn_lstm_{stamp}.pt"

    for epoch in range(1, EPOCHS + 1):
        model.train()
        run_loss = 0.0
        run_correct = 0
        run_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", ncols=80)
        for x_seq, y, _ in pbar:
            x_seq = x_seq.to(device)  # (B, S, 3, H, W)
            y = y.to(device)

            optimiz.zero_grad()
            logits = model(x_seq)
            loss = crit(logits, y)
            loss.backward()
            optimiz.step()

            preds = logits.argmax(dim=1)
            bs = y.size(0)
            run_loss += loss.item() * bs
            run_correct += (preds == y).sum().item()
            run_total += bs

            pbar.set_postfix(loss=f"{run_loss/run_total:.4f}",
                             acc=f"{run_correct/run_total:.3f}")

        val_loss, val_acc = evaluate(model, val_loader, device)
        print(f"\nVal: loss={val_loss:.4f}, acc={val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"model_state": model.state_dict(),
                        "classes": classes,
                        "img_size": IMG_SIZE,
                        "num_segments": NUM_SEGMENTS},
                       ckpt_path)
            print(f"✅ Saved best checkpoint to: {ckpt_path}")

        scheduler.step()

    print("\nTraining done.")
    print("Best val acc:", round(best_val_acc, 4))

if __name__ == "__main__":
    main()
