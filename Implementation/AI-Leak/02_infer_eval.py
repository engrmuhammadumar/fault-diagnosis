# 02_infer_eval.py (enhanced)
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import models
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from common import make_loader, load_classes_json
import matplotlib.pyplot as plt

DATA_ROOT = Path(r"E:\Upwork Project\AI_Leak_Detection_Project\images\cwt_log")
OUT_DIR = DATA_ROOT / "outputs"

def find_latest_ckpt(out_dir: Path):
    cand = sorted(list(out_dir.glob("*.pt")))
    if not cand:
        raise SystemExit("No checkpoints found in outputs/. Train first.")
    return cand[-1]

def build_model_from_ckpt(ckpt, num_classes: int):
    # decide architecture by filename prefix
    name = ckpt_path.name.lower()
    if name.startswith("vit_b16_"):
        model = models.vit_b_16(weights=None)
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)
    else:
        model = models.resnet18(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    return model

def compute_prf(cm):
    import numpy as np
    cm = np.array(cm)
    tp = cm.diagonal()
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    precision = tp / (tp + fp + 1e-9)
    recall    = tp / (tp + fn + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)
    return precision, recall, f1

if __name__ == "__main__":
    ckpt_path = find_latest_ckpt(OUT_DIR)
    print("Loading checkpoint:", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    classes = ckpt["classes"]
    img_size = ckpt.get("img_size", 224)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    num_classes = len(classes)
    model = build_model_from_ckpt(ckpt, num_classes)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    test_csv = DATA_ROOT / "manifest_test.csv"
    classes_json = DATA_ROOT / "classes.json"
    loader = make_loader(test_csv, classes_json, batch_size=32, shuffle=False, img_size=img_size, is_train=False)

    total, correct = 0, 0
    cm = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    all_true, all_pred = [], []

    with torch.no_grad():
        for x, y, paths in tqdm(loader, ncols=70):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            for t, p in zip(y.tolist(), preds.tolist()):
                cm[t][p] += 1
                all_true.append(t)
                all_pred.append(p)

    acc = correct / total if total else 0.0
    print(f"Test accuracy: {acc:.4f} ({correct}/{total})")

    # confusion matrix figure
    idx_to_class = {v: k for k, v in classes.items()}
    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(cm)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(num_classes))
    ax.set_xticklabels([idx_to_class[i] for i in range(num_classes)], rotation=45, ha="right")
    ax.set_yticks(range(num_classes))
    ax.set_yticklabels([idx_to_class[i] for i in range(num_classes)])
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, str(cm[i][j]), ha="center", va="center")
    fig.tight_layout()
    fig_path = OUT_DIR / "confusion_matrix.png"
    fig.savefig(fig_path, dpi=150)
    print("Saved confusion matrix:", fig_path)

    # per-class precision / recall / f1
    import numpy as np
    prec, rec, f1 = compute_prf(cm)
    rows = []
    for i in range(num_classes):
        rows.append({
            "class": idx_to_class[i],
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "f1": float(f1[i]),
            "support": int(sum(cm[i])),
        })
    df = pd.DataFrame(rows).sort_values("class")
    print("\nPer-class metrics:")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    out_csv = OUT_DIR / "per_class_metrics.csv"
    df.to_csv(out_csv, index=False)
    print("Saved per-class metrics CSV:", out_csv)
