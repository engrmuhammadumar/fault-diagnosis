import argparse, torch
from config import Config
from utils import set_seed
from engine import FSLEngine

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--outfile_prefix", type=str, default="fsl")
    args = parser.parse_args()

    cfg = Config()
    if args.dataset_path: cfg.dataset_path = args.dataset_path

    set_seed(cfg.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {dev}\nConfig: {cfg}")

    engine = FSLEngine(cfg, dev)
    engine.export_confusion_and_tsne(split="test", outfile_prefix=args.outfile_prefix)

if __name__ == "__main__":
    main()
