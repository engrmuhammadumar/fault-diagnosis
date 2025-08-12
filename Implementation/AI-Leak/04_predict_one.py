from pathlib import Path
import sys
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json

DATA_ROOT = Path(r"E:\Upwork Project\AI_Leak_Detection_Project\images\cwt_log")
OUT_DIR = DATA_ROOT / "outputs"

def load_latest_ckpt():
    ckpts = sorted(OUT_DIR.glob("*.pt"))
    if not ckpts:
        raise SystemExit("No checkpoints in outputs/")
    return ckpts[-1]

def build_model(arch_name, num_classes):
    if arch_name.startswith("vit_b16_"):
        m = models.vit_b_16(weights=None)
        in_features = m.heads.head.in_features
        m.heads.head = nn.Linear(in_features, num_classes)
    else:
        m = models.resnet18(weights=None)
        in_features = m.fc.in_features
        m.fc = nn.Linear(in_features, num_classes)
    return m

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python 04_predict_one.py <image_path>")
        sys.exit(1)
    img_path = Path(sys.argv[1])

    ckpt_path = load_latest_ckpt()
    print("Using checkpoint:", ckpt_path.name)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    classes = ckpt["classes"]
    idx_to_class = {v:k for k,v in classes.items()}
    img_size = ckpt.get("img_size", 224)

    model = build_model(ckpt_path.name.lower(), len(classes))
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    img = Image.open(img_path).convert("RGB")
    x = tf(img).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        prob = torch.softmax(logits, dim=1)[0]
        pred = int(prob.argmax().item())
        print(f"Prediction: {idx_to_class[pred]} | confidence: {prob[pred].item():.3f}")
        # show top-3
        topk = torch.topk(prob, k=min(3, len(classes)))
        print("Top-k:")
        for i in range(topk.indices.numel()):
            cls = idx_to_class[int(topk.indices[i])]
            print(f"  {cls}: {topk.values[i].item():.3f}")
