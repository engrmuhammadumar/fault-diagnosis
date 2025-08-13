from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Config:
    dataset_path: str = r"E:\\1 Paper MCT\\Cutting Tool Paper\\Dataset\\cutting tool data\\test_data_40_images"
    class_names: Optional[List[str]] = None
    image_size: int = 224
    num_workers: int = 4
    seed: int = 42
    n_way: int = 5
    k_shot: int = 5
    q_query: int = 5
    episodes_per_epoch: int = 200
    batch_size: int = 1
    lr: float = 1e-4
    weight_decay: float = 1e-4
    max_epochs: int = 30
    embed_dim: int = 256
    ema_beta: float = 0.9
    cov_shrink: float = 0.05
    per_class_train: int = 24
    per_class_val: int = 8
    per_class_test: int = 8
    out_dir: str = "./runs/fsl_run"
    save_every: int = 5
