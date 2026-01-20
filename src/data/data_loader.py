# src/data/data_loader.py

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple
import sys
import numpy as np
import scipy.io

# Dynamically add the project root to PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[2] # 2 levels up from src/data
sys.path.insert(0, str(PROJECT_ROOT))
print(f"Project root set to: {PROJECT_ROOT}")

__all__: list[str] = [
    "DATASET_PATHS",
    "DATASET_NAME_LIST",
    "load_dataset",
]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = Path(os.getenv("HSI_DATA_ROOT", PROJECT_ROOT / "src"/ "Dataset")).resolve()

print(DEFAULT_DATA_ROOT)
# Centralised relative path map -------------------------------------------------
DATASET_PATHS: Dict[str, Dict[str, Path]] = {
    "Botswana": {
        "image": DEFAULT_DATA_ROOT / "Botswana" / "Botswana.mat",
        "ground_truth": DEFAULT_DATA_ROOT / "Botswana" / "Botswana_gt.mat",
    },
    "Houston13": {
        "image": DEFAULT_DATA_ROOT / "Houston13" / "Houston13.mat",
        "ground_truth": DEFAULT_DATA_ROOT / "Houston13" / "Houston13_gt.mat",
    },
    "Indian_Pines": {
        "image": DEFAULT_DATA_ROOT / "Indian_Pines" / "Indian_Pines_corrected.mat",
        "ground_truth": DEFAULT_DATA_ROOT / "Indian_Pines" / "Indian_Pines_gt.mat",
    },
    "Kennedy_Space_Center": {
        "image": DEFAULT_DATA_ROOT / "Kennedy_Space_Center" / "Kennedy_Space_Center.mat",
        "ground_truth": DEFAULT_DATA_ROOT / "Kennedy_Space_Center" / "Kennedy_Space_Center_gt.mat",
    },
    "Pavia_Centre": {
        "image": DEFAULT_DATA_ROOT / "Pavia_Centre" / "Pavia_Centre.mat",
        "ground_truth": DEFAULT_DATA_ROOT / "Pavia_Centre" / "Pavia_Centre_gt.mat",
    },
    "Pavia_University": {
        "image": DEFAULT_DATA_ROOT / "Pavia_University" / "Pavia_University.mat",
        "ground_truth": DEFAULT_DATA_ROOT / "Pavia_University" / "Pavia_University_gt.mat",
    },
    "Salinas": {
        "image": DEFAULT_DATA_ROOT / "Salinas" / "Salinas_corrected.mat",
        "ground_truth": DEFAULT_DATA_ROOT / "Salinas" / "Salinas_gt.mat",
    },
    "SalinasA": {
        "image": DEFAULT_DATA_ROOT / "SalinasA" / "SalinasA_corrected.mat",
        "ground_truth": DEFAULT_DATA_ROOT / "SalinasA" / "SalinasA_gt.mat",
    },
}

DATASET_NAME_LIST: list[str] = list(DATASET_PATHS.keys())

def _first_valid_var(mat_dict: dict) -> np.ndarray:
    """Return the first entry that is not a MATLAB header field."""
    for k, v in mat_dict.items():
        if not k.startswith("__"):
            return v
    raise ValueError("No valid variable found in .mat file")


def _load_mat(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    return _first_valid_var(scipy.io.loadmat(path))


def _align_cube_to_gt(cube: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Ensure *cube* becomes (H, W, B) to match ground‑truth spatial dims."""
    h_gt, w_gt = gt.shape
    if cube.shape[:2] == (h_gt, w_gt):
        return cube  # already channel‑last
    if cube.shape[1:] == (h_gt, w_gt):       # (B, H, W)
        return cube.transpose(1, 2, 0)
    if cube.shape[0] == h_gt and cube.shape[2] == w_gt:  # (H, B, W)
        return cube.transpose(0, 2, 1)
    raise ValueError(
        "Could not align cube with ground‑truth – shapes are incompatible: "
        f"cube {cube.shape}, gt {gt.shape}"
    )

def load_dataset(dataset_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load *cube* and *ground‑truth* with guaranteed channel‑last orientation."""
    if dataset_name not in DATASET_PATHS:
        raise KeyError(f"Unknown dataset '{dataset_name}'.")

    cfg = DATASET_PATHS[dataset_name]
    cube_path, gt_path = cfg["image"], cfg["ground_truth"]
    cube = _load_mat(cube_path)
    gt   = _load_mat(gt_path)

    cube = _align_cube_to_gt(cube, gt)
    
    if cube.shape[:2] != gt.shape:
        raise ValueError(f"Image {cube.shape} and GT {gt.shape} spatial dimensions do not match -> {cube.shape[:2]} ≠ {gt.shape}")
    return cube, gt

if __name__ == "__main__":
    # Importing the dataset name list and labels function
    from src.data.data_loader import DATASET_NAME_LIST
    from src.data.dataset_info import get_dataset_labels
    
    print(DATASET_NAME_LIST)
    for name in DATASET_NAME_LIST:
        cube, gt = load_dataset(name)
        print(f"✔ {name:18s} → cube {cube.shape}, labels {gt.shape}")
    
    # print all class values in ground truth matrix for each dataset
    print("\nGround truth class values:")

    dataset_class_dicts = {}

    for name in DATASET_NAME_LIST:
        _, gt = load_dataset(name)
        unique_classes = np.unique(gt)
        labels = get_dataset_labels(name)

        class_dict = {int(c): labels[int(c)] for c in unique_classes if int(c) < len(labels)}
        dataset_class_dicts[name] = class_dict

    for name, class_dict in dataset_class_dicts.items():
        print(f"{name:18s} → {class_dict}")