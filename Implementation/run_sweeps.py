import os, argparse, subprocess, sys

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--ways", type=int, choices=[5,7], required=True)
    p.add_argument("--shots", type=int, nargs="+", required=True, help="e.g. 1 3 5 10")
    p.add_argument("--base_out", type=str, default="./runs")
    args = p.parse_args()

    for k in args.shots:
        exp = f"{args.ways}way{k}shot_maha"
        out_dir = os.path.join(args.base_out, exp)

        # Train
        cmd = [
            sys.executable, "train_eval_protonet.py",
            "--data_dir", args.data_dir,
            "--output_dir", out_dir,
            "--n_way", str(args.ways),
            "--k_shot", str(k),
            "--q_query", "10",
            "--distance", "mahalanobis",
            "--cov_mode", "shared",
            "--episodes_per_epoch", "400",
            "--epochs", "50",
            "--lr", "1e-4",
            "--batch_aug", "strong"
        ]
        print(">>> Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)

        # Eval + Figures
        ev_cmd = [
            sys.executable, "train_eval_protonet.py",
            "--data_dir", args.data_dir,
            "--load_checkpoint", os.path.join(out_dir, "best.pt"),
            "--eval_only", "True",
            "--n_way", str(args.ways),
            "--k_shot", str(k),
            "--q_query", "10",
            "--episodes", "1000",
            "--save_confusion", os.path.join(out_dir, f"confusion_{args.ways}w{k}s.png"),
            "--save_tsne", os.path.join(out_dir, f"tsne_{args.ways}w{k}s.png")
        ]
        print(">>> Evaluating:", " ".join(ev_cmd))
        subprocess.run(ev_cmd, check=True)

if __name__ == "__main__":
    main()
