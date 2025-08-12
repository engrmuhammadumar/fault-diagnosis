# 00_prepare_data.py
from pathlib import Path
from collections import Counter
import pandas as pd
from sklearn.model_selection import train_test_split
import json

# >>> EDIT THIS ONLY IF YOUR PATH IS DIFFERENT
DATA_ROOT = Path(r"E:\Upwork Project\AI_Leak_Detection_Project\images\cwt_log")

EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}

def scan_images(data_root: Path):
    files = [p for p in data_root.rglob("*") if p.suffix.lower() in EXTS]
    if not files:
        raise SystemExit(f"No images found under: {data_root}")
    labels = [p.parent.name for p in files]
    df = pd.DataFrame({"filepath": [str(p) for p in files], "label": labels})
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    return df

def save_class_map(df: pd.DataFrame, out_path: Path):
    classes = sorted(df["label"].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(class_to_idx, f, ensure_ascii=False, indent=2)
    return class_to_idx

def main():
    print("Scanning images ...")
    df = scan_images(DATA_ROOT)
    print(f"Total images: {len(df)}")
    counts = Counter(df["label"].tolist())
    print("Class counts:")
    for k, v in counts.items():
        print(f"  {k}: {v}")

    manifest_all = DATA_ROOT / "manifest_all.csv"
    df.to_csv(manifest_all, index=False)
    print("Saved:", manifest_all)

    # create classes.json
    classes_json = DATA_ROOT / "classes.json"
    class_to_idx = save_class_map(df, classes_json)
    print("Classes mapping saved:", classes_json)
    print(class_to_idx)

    # stratified split 70/15/15
    train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df['label'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df['label'], random_state=42)

    # add numeric target
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    train_df["target"] = train_df["label"].map(class_to_idx)
    val_df["target"]   = val_df["label"].map(class_to_idx)
    test_df["target"]  = test_df["label"].map(class_to_idx)

    for name, d in [("train", train_df), ("val", val_df), ("test", test_df)]:
        out = DATA_ROOT / f"manifest_{name}.csv"
        d.to_csv(out, index=False)
        print(f"{name}: {len(d)} -> {out}")

    print("\n✅ Done. Next run: 01_train.py")

if __name__ == "__main__":
    main()
