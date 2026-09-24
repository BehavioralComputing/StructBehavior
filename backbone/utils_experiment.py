"""
Shared utility functions for SEAD fusion experiments.

- set_model_seed: seed for model initialization, dropout, CUDA randomness
- summarize_split: print label distribution of train/val/test vs global
"""

import os
import random
import numpy as np
import torch
from collections import Counter


def set_model_seed(seed: int):
    """Set all random seeds for reproducible model initialization and training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def summarize_split(labels, train_idx, val_idx, test_idx):
    """Print label distribution of train/val/test splits vs global distribution.

    Args:
        labels: np.array or torch.Tensor of all labels
        train_idx, val_idx, test_idx: index arrays
    """
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()
    else:
        labels = np.asarray(labels)
    summary = {}
    for name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        y = labels[idx]
        cnt = Counter(y.tolist())
        total = len(y)
        ratio = {k: v / total for k, v in sorted(cnt.items())}
        summary[name] = {
            "total": total,
            "count": dict(sorted(cnt.items())),
            "ratio": ratio,
        }

    global_cnt = Counter(labels.tolist())
    global_total = len(labels)
    global_ratio = {k: v / global_total for k, v in sorted(global_cnt.items())}

    print("=== Split Distribution ===")
    print(f"  Global  (n={global_total}): {global_ratio}")
    for name in ["train", "val", "test"]:
        s = summary[name]
        print(f'  {name:6s} (n={s["total"]:<6d}): {s["ratio"]}')
    print("==========================")
    return summary
