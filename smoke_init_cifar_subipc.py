"""CPU smoke test for the CIFAR-10 sub-1-IPC poster geometries (IPC 0.5/0.4/0.3/0.2/0.1).

CIFAR-10's protocol does not use --train_y, so labels are the deterministic
PoDDL.get_poster_labels(...) function of the geometry. This verifies, without a GPU or
dataset, that each geometry is sound: fade-combined poster init, patch index bounds
(catches posters smaller than the 32x32 crop), crop shapes, label normalization, and —
informatively — how many classes are actually distinguishable by the fixed soft labels.

Expected structural degeneracies (reported, not failed):
- IPC 0.2 (64x32): height is exactly one class area, so the 5x2 layout would fully
  overlap its two rows (label-identical class pairs -> ~50% ceiling); we use 10x1.
- IPC 0.1 (32x32): the poster IS the single 32x32 crop; all class regions coincide and
  the soft label is uniform — PoDD cannot express IPC 0.1 non-degenerately.

Run inside the podd conda env.
"""
import sys

import numpy as np
import torch

from src.PoDD_utils import combine_images_with_fade, _get_patch_index_lst, get_crops_from_poster
from src.PoDDL import PoDDL

FAILS = []


def check(name, cond, detail=""):
    status = "OK" if cond else "FAIL"
    print(f"[{status}] {name} {detail}")
    if not cond:
        FAILS.append(name)


# name, poster_w, poster_h, class_num_x, class_num_y, patch_num_x, patch_num_y
GEOMS = [
    ("ipc05", 113, 45, 5, 2, 16, 6),
    ("ipc04", 101, 40, 5, 2, 16, 6),
    ("ipc03", 87, 35, 5, 2, 16, 6),
    ("ipc02", 64, 32, 10, 1, 96, 1),
    ("ipc01", 32, 32, 10, 1, 96, 1),
]

AREA = 32  # class_area == image size == crop size for CIFAR

for name, w, h, cx, cy, px, py in GEOMS:
    torch.manual_seed(0)
    budget = w * h / 10240.0
    print(f"--- {name}: poster {w}x{h} ({w * h} px, ratio {budget:.4f}), "
          f"classes {cx}x{cy}, patches {px}x{py}")
    check(f"{name} budget <= ratio", w * h <= round(10240 * float(name[3:]) / 10),
          f"{w * h} px")

    class_areas = torch.randn(10, 3, AREA, AREA)
    poster = combine_images_with_fade(class_areas, w, h, cx, cy).unsqueeze(0)
    check(f"{name} poster shape", tuple(poster.shape) == (1, 3, h, w), str(tuple(poster.shape)))
    check(f"{name} poster finite", torch.isfinite(poster).all().item())

    idx = _get_patch_index_lst(AREA, AREA, w, h, px, py)
    check(f"{name} patch count", len(idx) == px * py, f"{len(idx)}")
    in_bounds = all(0 <= x1 and x2 <= w and 0 <= y1 and y2 <= h for ((x1, x2), (y1, y2)) in idx)
    check(f"{name} patches in bounds (poster >= crop)", in_bounds)
    uniq = len(set(idx))
    print(f"[INFO] {name} distinct patch positions: {uniq}/{len(idx)}")

    crops = get_crops_from_poster(poster, AREA, AREA, px, py)
    check(f"{name} crop shapes", tuple(crops.shape) == (px * py, 3, AREA, AREA),
          str(tuple(crops.shape)))

    class_order = np.arange(10).reshape(cy, cx)
    labels = PoDDL.get_poster_labels(class_order, AREA, AREA, AREA, AREA,
                                     w, h, cx, cy, px, py)
    check(f"{name} labels shape", tuple(labels.shape) == (px * py, 10), str(tuple(labels.shape)))
    check(f"{name} labels normalized, finite",
          torch.allclose(labels.sum(1), torch.ones(px * py)) and torch.isfinite(labels).all().item())
    per_class_mass = labels.sum(0)
    check(f"{name} every class has label mass", bool((per_class_mass > 0).all()),
          str([round(v, 2) for v in per_class_mass.tolist()]))

    # Distinguishability diagnostics (informative): a class is "covered" if it is the
    # argmax of at least one patch label; pairs tied everywhere cannot be separated.
    argmaxes = labels.argmax(1)
    covered = len(set(argmaxes.tolist()))
    max_prob = labels.max(1).values
    print(f"[INFO] {name} classes that are argmax of some patch: {covered}/10, "
          f"max-prob mean {max_prob.mean():.3f} (uniform would be 0.100)")
    if covered < 10:
        print(f"[WARN] {name} only {covered}/10 classes distinguishable by fixed labels "
              f"— structural degeneracy of this budget")

print("RESULT:", "FAIL " + ",".join(FAILS) if FAILS else "ALL_OK")
sys.exit(1 if FAILS else 0)
