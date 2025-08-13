import argparse, torch
from config import Config
from utils import set_seed
from engine import FSLEngine

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=200)
    args = parser.parse_args()

    cfg = Config()
    if args.dataset_path: cfg.dataset_path = args.dataset_path

    set_seed(cfg.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {dev}\nConfig: {cfg}")

    engine = FSLEngine(cfg, dev)
    acc = engine.evaluate(split="test", episodes=args.episodes)
    print(f"Test episodic accuracy: {acc:.2f}%")

if __name__ == "__main__":
    main()
