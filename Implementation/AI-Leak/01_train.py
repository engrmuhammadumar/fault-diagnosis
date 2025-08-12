# 01_train.py
from pathlib import Path
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm import tqdm
import json

from common import make_loader, load_classes_json

# >>> EDIT THESE IF NEEDED
DATA_ROOT = Path(r"E:\Upwork Project\AI_Leak_Detection_Project\images\cwt_log")
OUT_DIR = DATA_ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EPOCHS = 20
BATCH_SIZE = 32
IMG_SIZE = 224
LR = 3e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0  # 0 is safest on Windows; increase later if you like

def get_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def build_model(num_classes: int):
    # ResNet-18 pretrained on ImageNet
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # replace final layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

def evaluate(model, loader, device):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    crit = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = crit(logits, y)
            loss_sum += loss.item() * y.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return loss_sum / total, correct / total

def main():
    device = get_device()
    print("Device:", device)

    classes_json = DATA_ROOT / "classes.json"
    classes = load_classes_json(classes_json)
    num_classes = len(classes)
    print("Classes:", classes)

    train_csv = DATA_ROOT / "manifest_train.csv"
    val_csv   = DATA_ROOT / "manifest_val.csv"

    train_loader = make_loader(train_csv, classes_json, batch_size=BATCH_SIZE, shuffle=True,
                               img_size=IMG_SIZE, is_train=True, num_workers=NUM_WORKERS)
    val_loader   = make_loader(val_csv, classes_json, batch_size=BATCH_SIZE, shuffle=False,
                               img_size=IMG_SIZE, is_train=False, num_workers=NUM_WORKERS)

    model = build_model(num_classes).to(device)
    crit = nn.CrossEntropyLoss()
    optimiz = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimiz, T_max=EPOCHS)


    best_val_acc = 0.0
    stamp = time.strftime("%Y%m%d_%H%M%S")
    ckpt_path = OUT_DIR / f"resnet18_{stamp}.pt"

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss, running_correct, running_total = 0.0, 0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", ncols=80)
        for x, y, _ in pbar:
            x, y = x.to(device), y.to(device)
            optimiz.zero_grad()
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            optimiz.step()

            preds = logits.argmax(dim=1)
            running_loss += loss.item() * y.size(0)
            running_correct += (preds == y).sum().item()
            running_total += y.size(0)

            pbar.set_postfix(loss=f"{running_loss/running_total:.4f}",
                             acc=f"{running_correct/running_total:.3f}")

        val_loss, val_acc = evaluate(model, val_loader, device)
        print(f"\nVal: loss={val_loss:.4f}, acc={val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state": model.state_dict(),
                "classes": classes,
                "img_size": IMG_SIZE,
            }, ckpt_path)
            print(f"✅ Saved best checkpoint to: {ckpt_path}")

    print("\nTraining done.")
    print("Best val acc:", round(best_val_acc, 4))
    print("If no checkpoint saved, please share the logs.")

if __name__ == "__main__":
    main()
