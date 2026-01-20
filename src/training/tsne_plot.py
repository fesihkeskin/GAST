#!/usr/bin/env python3
# src/training/tsne_plot.py

import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

from sklearn.manifold import TSNE

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

FIGURE_PATH = PROJECT_ROOT / "reports" / "figures"

from src.data.data_loader import load_dataset, DATASET_PATHS
from src.data.dataset_info import get_dataset_labels, DATASET_NAME_MAP
from src.data.dataset import HyperspectralDataset
from src.models.model_architecture import GAST
from src.utils.utils import set_seed

# Mapping from dataset name to model hyperparameters (from your CLI)
DATASET_MODEL_CONFIGS = {
    "Botswana": {
        "patch_size": 13, "embed_dim": 128, "gat_hidden_dim": 64, "gat_heads": 4, "gat_depth": 2,
        "transformer_heads": 8, "transformer_layers": 9, "dropout": 0.25, "fusion_mode": "gate"
    },
    "Houston13": {
        "patch_size": 13, "embed_dim": 128, "gat_hidden_dim": 128, "gat_heads": 4, "gat_depth": 2,
        "transformer_heads": 8, "transformer_layers": 4, "dropout": 0.15, "fusion_mode": "gate"
    },
    "Indian_Pines": {
        "patch_size": 7, "embed_dim": 128, "gat_hidden_dim": 32, "gat_heads": 2, "gat_depth": 8,
        "transformer_heads": 8, "transformer_layers": 6, "dropout": 0.1, "fusion_mode": "gate"
    },
    "Kennedy_Space_Center": {
        "patch_size": 9, "embed_dim": 256, "gat_hidden_dim": 64, "gat_heads": 10, "gat_depth": 6,
        "transformer_heads": 2, "transformer_layers": 4, "dropout": 0.25, "fusion_mode": "gate"
    },
    "Pavia_Centre": {
        "patch_size": 13, "embed_dim": 256, "gat_hidden_dim": 64, "gat_heads": 4, "gat_depth": 4,
        "transformer_heads": 16, "transformer_layers": 3, "dropout": 0.45, "fusion_mode": "gate"
    },
    "Pavia_University": {
        "patch_size": 11, "embed_dim": 64, "gat_hidden_dim": 32, "gat_heads": 4, "gat_depth": 4,
        "transformer_heads": 16, "transformer_layers": 9, "dropout": 0.2, "fusion_mode": "gate"
    },
    "Salinas": {
        "patch_size": 13, "embed_dim": 128, "gat_hidden_dim": 32, "gat_heads": 10, "gat_depth": 4,
        "transformer_heads": 2, "transformer_layers": 2, "dropout": 0.15, "fusion_mode": "gate"
    },
    "SalinasA": {
        "patch_size": 11, "embed_dim": 256, "gat_hidden_dim": 32, "gat_heads": 4, "gat_depth": 8,
        "transformer_heads": 16, "transformer_layers": 10, "dropout": 0.0, "fusion_mode": "gate"
    },
}



def extract_features(model, loader, device):
    model.eval()
    feats_list, labels_list = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting features"):
            x = batch["patch"].to(device)
            y = batch["label"].cpu().numpy()
            f = model.extract_features(x).cpu().numpy()
            feats_list.append(f)
            labels_list.append(y)
    features = np.concatenate(feats_list, axis=0)
    labels   = np.concatenate(labels_list, axis=0)
    return features, labels

def get_discrete_hsi_cmap(n_classes):
    """
    Returns a consistent ListedColormap and normalization for HSI plots.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import ListedColormap
    import matplotlib as mpl

    if n_classes - 1 <= 20:
        base_colors = plt.cm.tab20(np.linspace(0, 1, 20))
        colors = ["black"] + [base_colors[i] for i in range(n_classes - 1)]
    else:
        cmap_discrete = plt.cm.get_cmap("tab20", n_classes - 1)
        colors = ["black"] + [cmap_discrete(i) for i in range(n_classes - 1)]
    cmap = ListedColormap(colors)
    norm = mpl.colors.Normalize(vmin=0, vmax=n_classes - 1)
    return cmap, norm


def tsne_main():
    import argparse
    parser = argparse.ArgumentParser("t-SNE visualization for GAST features")
    parser.add_argument("--dataset",    required=True, choices=list(DATASET_PATHS))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", type=str, default=FIGURE_PATH)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed",       type=int, default=252)
    parser.add_argument(
        "--max_pixels",
        type=int,
        default=None,
        help="Maximum number of pixel samples to run t-SNE on (default: all)"
    )
    args = parser.parse_args()

    # Inject dataset-specific model parameters
    if args.dataset not in DATASET_MODEL_CONFIGS:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    model_cfg = DATASET_MODEL_CONFIGS[args.dataset]
    for k, v in model_cfg.items():
        setattr(args, k, v)

    if not hasattr(args, "max_pixels"):
        args.max_pixels = None

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Output path for t-SNE plot
    out_path = Path(args.output_dir) / f"tsne_{args.dataset}.png"
    if out_path.exists():
        print(f"t-SNE plot already exists at {out_path}, skipping.")
        return str(out_path)
    # --- Load test split indices ---
    split_dir = PROJECT_ROOT / "models" / "final" / "gast" / args.dataset / "splits"
    test_idx_path = split_dir / f"test_idx_seed_{args.seed}.npy"
    if not test_idx_path.exists():
        raise FileNotFoundError(f"Test split not found at {test_idx_path}")
    coords = np.load(test_idx_path)  # shape: (N_test, 2) with [row, col]

    # --- Load ground truth and get test labels ---
    _, gt_arr = load_dataset(args.dataset)
    labels_full = get_dataset_labels(args.dataset)  # index 0 = Undefined
    labels = gt_arr[coords[:, 0], coords[:, 1]] - 1  # zero-indexed, only for test

    # (Optionally subsample for large test splits)
    total_pixels = len(coords)
    print(f"Total test split pixels: {total_pixels}")

    n_pixels = total_pixels if args.max_pixels is None or args.max_pixels >= total_pixels else args.max_pixels
    if n_pixels < total_pixels:
        idx = np.random.choice(total_pixels, n_pixels, replace=False)
        coords = coords[idx]
        labels = labels[idx]
    print(f"Using {len(coords)} test pixels for t-SNE")


    # compute mean/std on all pixels
    cube, _ = load_dataset(args.dataset)
    mean = cube.reshape(-1, cube.shape[-1]).mean(axis=0)
    std  = cube.reshape(-1, cube.shape[-1]).std(axis=0) + 1e-8

    # dataset + loader
    test_ds = HyperspectralDataset(
        cube_path=str(PROJECT_ROOT / DATASET_PATHS[args.dataset]["image"]),
        gt_path=str(PROJECT_ROOT / DATASET_PATHS[args.dataset]["ground_truth"]),
        patch_size=getattr(args, "patch_size", 11),
        mode="test",
        indices=coords,
        mean=mean,
        std=std,
        augment=False,
        stride=1,
    )
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GAST(
        in_channels=test_ds.B,
        n_classes=len(labels_full),
        patch_size=args.patch_size,
        spec_dim=args.embed_dim,
        spat_dim=args.gat_hidden_dim,
        n_heads=args.gat_heads,
        n_layers=args.gat_depth,
        transformer_heads=args.transformer_heads,
        transformer_layers=args.transformer_layers,
        dropout=args.dropout,
        disable_spectral=getattr(args, "disable_spectral", False),
        disable_spatial=getattr(args, "disable_spatial", False),
        fusion_mode=args.fusion_mode,
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
    model.eval()

    # extract features
    feats, y = extract_features(model, loader, device)

    # t-SNE
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, random_state=args.seed, perplexity=30)
    feats_2d = tsne.fit_transform(feats)

    # Plot settings
    # --- Renk skalasını ground truth ile aynı yap ---
    # ... After extracting y, labels_full, etc.
    n_classes = len(labels_full)
    cmap, norm = get_discrete_hsi_cmap(n_classes)

    plt.figure(figsize=(8, 8))
    plt.scatter(
        feats_2d[:, 0], feats_2d[:, 1],
        c=y + 1,  # +1 if your data is 0-based and color 0 is background
        cmap=cmap,
        norm=norm,
        s=5,
        alpha=0.7
    )
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.title(f"t-SNE of {DATASET_NAME_MAP[args.dataset]} Features")
    plt.tight_layout()

    # Legend: only present classes, skip background
    unique, counts = np.unique(y, return_counts=True)
    handles = []
    for cls, cnt in zip(unique, counts):
        if cls < 0:  # skip background
            continue
        patch = mpatches.Patch(
            color=cmap(cls + 1),   # +1 so class 0 gets color 1, background always color 0 (black)
            label=f"{labels_full[cls + 1]} ({cnt})"
        )
        handles.append(patch)
    plt.legend(
        handles=handles,
        title="Classes",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        borderaxespad=0.
    )
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"t-SNE plot saved to {out_path}")

    return str(out_path)


if __name__ == "__main__":
    tsne_main()
