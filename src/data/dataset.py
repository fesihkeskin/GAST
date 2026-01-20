# src/data/dataset.py

"""
PyTorch Dataset for hyperspectral‑image classification (patch‑based).
This module provides a `HyperspectralDataset` class that can be used to load
hyperspectral image cubes and their corresponding ground truth labels.
It supports patch extraction, data augmentation, and normalization.
It is designed to work with datasets like Indian Pines, Botswana, etc.
It is compatible with PyTorch's `Dataset` and `DataLoader` classes.
This code is part of the GAST project, which implements GAST for hyperspectral image classification.
"""

import numpy as np
import scipy.io
import torch
from torch.utils.data import Dataset
from pathlib import Path
import os

class HyperspectralDataset(Dataset):
    def __init__(
        self,
        cube_path: str,
        gt_path: str,
        patch_size: int = 11,
        mode: str = "train",
        indices: np.ndarray | None = None,
        train_ratio: float = 0.05,
        val_ratio: float = 0.05,
        random_seed: int = 242,
        mean: np.ndarray | None = None,
        std:  np.ndarray | None = None,
        augment: bool = False,
        stride: int = 1,
    ):
        if not os.path.exists(cube_path):
            raise FileNotFoundError(f"Cube file not found: {cube_path}")
        if not os.path.exists(gt_path):
            raise FileNotFoundError(f"Ground truth file not found: {gt_path}")

        self.mode = mode
        self.patch_size = patch_size
        self.augment = augment and mode == "train"

        # ── load cube & GT ────────────────────────────────────────────────────
        cube = self._read_mat(cube_path)
        gt   = self._read_mat(gt_path)

        # make cube (H, W, B)
        if cube.ndim == 3:
            pass
        elif cube.ndim == 2:
            raise ValueError("2‑D cube is not supported.")
        else:                     # (B, H, W)
            cube = np.transpose(cube, (1, 2, 0))
        self.cube = cube
        self.gt   = gt
        self.H, self.W, self.B = cube.shape

        # ── decide which pixels we use ───────────────────────────────────────
        # --- PATCH: Select all pixels as center ---
        all_idx = np.array([[r, c] for r in range(gt.shape[0]) for c in range(gt.shape[1])])
        if stride > 1:
            mask = ((all_idx[:, 0] % stride == 0) & (all_idx[:, 1] % stride == 0))
            all_idx = all_idx[mask]
        if indices is None:                     # first call → create split
            rng = np.random.default_rng(random_seed)
            perm = rng.permutation(len(all_idx))

            n_train = int(len(all_idx) * train_ratio)
            n_val   = int(len(all_idx) * val_ratio)

            split = {
                "train": perm[:n_train],
                "val"  : perm[n_train : n_train + n_val],
                "test" : perm[n_train + n_val :],
            }[mode]
            indices = all_idx[split]
        else:                                   # indices provided
            indices = np.asarray(indices)
        self.indices = indices
        self.labels  = gt[indices[:, 0], indices[:, 1]] - 1  # make class labels zero‑indexed (background = -1)

        # ── stats ────────────────────────────────────────────────────────────
        if mean is not None and std is not None:
            self.mean, self.std = mean, std
        elif mode == "train":
            self.mean, self.std = self._compute_mean_std()
        else:
            raise ValueError("mean/std must be given for val & test sets")

        # pad cube for border patches
        pad = patch_size // 2
        self.pad = pad
        self.padded_cube = np.pad(
            cube, ((pad, pad), (pad, pad), (0, 0)), mode="reflect"
        )

    # --------------------------------------------------------------------- #
    #  helper functions
    # --------------------------------------------------------------------- #
    @staticmethod
    def _read_mat(path: str) -> np.ndarray:
        key, = (k for k in scipy.io.loadmat(path).keys() if not k.startswith("__"))
        return scipy.io.loadmat(path)[key]

    def _compute_mean_std(self):
        # stream over pixels to keep memory modest
        n, mean, M2 = 0, np.zeros(self.B), np.zeros(self.B)
        for r, c in self.indices:
            v = self.cube[r, c]                 # (B,)
            n += 1
            delta = v - mean
            mean += delta / n
            M2   += delta * (v - mean)
        var = M2 / max(n - 1, 1)
        return mean, np.sqrt(var) + 1e-8

    def _get_patch(self, r: int, c: int):
        p = self.pad
        patch = self.padded_cube[r : r + 2*p + 1, c : c + 2*p + 1]
        return patch  # (k, k, B)
        
    def _augment(self, patch: np.ndarray):
        if np.random.rand() < 0.5:
            patch = np.fliplr(patch)
        if np.random.rand() < 0.5:
            patch = np.flipud(patch)
        patch = np.rot90(patch, k=np.random.randint(4))
        if np.random.rand() < 0.5:
            patch = patch + np.random.normal(0, 0.01, patch.shape)
        if np.random.rand() < 0.3:
            patch = patch * (0.9 + 0.2 * np.random.rand())  # random brightness
        return patch

    def __len__(self):       return len(self.indices)

    def __getitem__(self, i):
        r, c = self.indices[i]
        patch = self._get_patch(r, c)
        if self.augment:
            patch = self._augment(patch)

        patch = (patch - self.mean) / self.std         # normalise
        patch = np.transpose(patch, (2, 0, 1)).astype(np.float32)

        return {
            "patch": torch.from_numpy(patch),
            "label": torch.tensor(self.labels[i], dtype=torch.long),
            "coord": torch.tensor(self.indices[i], dtype=torch.long),  
        }