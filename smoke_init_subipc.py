"""CPU smoke test for the sub-1-IPC poster geometries (IPC 0.5 and 0.25).

Verifies, without a GPU or dataset, that the init path of main.py is sound for the
new poster sizes: fade-combined poster init (including the fully-overlapping class
rows at 320x128), the (2,5) label grid from PoDDL.init_label_array, patch index
uniqueness, and label cropping normalization. Run inside the podd conda env.
"""
import sys

import numpy as np
import torch

from src.PoDD_utils import combine_images_with_fade, _get_patch_index_lst
from src.PoDDL import PoDDL

FAILS = []


def check(name, cond, detail=""):
    status = "OK" if cond else "FAIL"
    print(f"[{status}] {name} {detail}")
    if not cond:
        FAILS.append(name)


for name, w, h, px, py in [("ipc05", 450, 180, 5, 4), ("ipc025", 320, 128, 20, 1)]:
    torch.manual_seed(0)
    class_areas = torch.randn(10, 3, 128, 128)
    poster = combine_images_with_fade(class_areas, w, h, 5, 2).unsqueeze(0)
    check(f"{name} poster shape", tuple(poster.shape) == (1, 3, h, w), str(tuple(poster.shape)))
    check(f"{name} poster finite", torch.isfinite(poster).all().item())

    class_order = np.arange(10).reshape(2, 5)
    y = PoDDL.init_label_array(np.array(poster.shape), class_order, 1)
    check(f"{name} label grid (2,5)", tuple(y.shape) == (1, 10, 2, 5), str(tuple(y.shape)))
    per_class = y.sum(dim=(0, 2, 3))
    check(f"{name} every class has label mass", bool((per_class > 0).all()), str(per_class.tolist()))

    idx = _get_patch_index_lst(128, 128, w, h, px, py)
    uniq = len(set(idx))
    check(f"{name} patch count", len(idx) == px * py, f"{len(idx)}")
    check(f"{name} patches unique", uniq == len(idx), f"{uniq}/{len(idx)}")
    in_bounds = all(0 <= x1 and x2 <= w and 0 <= y1 and y2 <= h for ((x1, x2), (y1, y2)) in idx)
    check(f"{name} patches in bounds", in_bounds)

    sf_x = 1 / (poster.shape[3] / y.shape[3])
    sf_y = 1 / (poster.shape[2] / y.shape[2])
    labels = PoDDL.get_labels_from_array(y, sf_x, sf_y, 128, 128, px, py)
    check(f"{name} cropped labels shape", tuple(labels.shape) == (px * py, 10), str(tuple(labels.shape)))
    rowsum_ok = torch.allclose(labels.sum(1), torch.ones(px * py))
    check(f"{name} labels normalized, no NaN", rowsum_ok and torch.isfinite(labels).all().item())

print("RESULT:", "FAIL " + ",".join(FAILS) if FAILS else "ALL_OK")
sys.exit(1 if FAILS else 0)
