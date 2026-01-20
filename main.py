# main.py

import os
import argparse
import sys
from pathlib import Path

# Dynamically add the project root to PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(PROJECT_ROOT))

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from src.training.train import train
from src.data.data_loader import load_dataset, DATASET_PATHS
from src.utils.utils import set_seed
from src.training.test import test     
from src.data.dataset_info import get_imbalance_ratio
from src.training.tsne_plot import tsne_main

# -----------------------------------------------------------------------------#
def resolve_dataset_paths(dataset_name: str) -> tuple[Path, Path]:
    """Return absolute (cube_path, gt_path) from DATASET_PATHS entry."""
    cfg = DATASET_PATHS[dataset_name]
    return (PROJECT_ROOT / cfg["image"], PROJECT_ROOT / cfg["ground_truth"])
    
def main() -> None:
    p = argparse.ArgumentParser("GAST Hyperspectral Image Classification")

    # data
    p.add_argument("--dataset",      type=str, choices=list(DATASET_PATHS.keys()), default="SalinasA",
                   help="Dataset name (e.g., Indian_Pines, Botswana, etc.)")
    # model type
    p.add_argument("--model_type", type=str, default="gast",
                help="Type of model backend to use (e.g., gast, cnn)")

    p.add_argument("--mode", choices=["train", "test"], default="train")
    
    p.add_argument("--train_ratio",  type=float, default=0.05)
    p.add_argument("--val_ratio",    type=float, default=0.05)
    p.add_argument("--patch_size",   type=int,   default=11)
    p.add_argument("--stride",       type=int,   default=5,
                   help="Stride for patch extraction (default: 1, no overlap)")
    # optimisation
    p.add_argument("--batch_size",   type=int,   default=128)
    p.add_argument("--epochs",       type=int,   default=200)
    p.add_argument("--lr",           type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--dropout",      type=float, default=0.2)
    p.add_argument("--early_stop",   type=int,   default=20)

    # system / reproducibility
    p.add_argument("--num_workers",  type=int,   default=4)
    p.add_argument("--output_dir",   type=str,   default="./models/checkpoints")
    p.add_argument("--seed",         type=int,   default=242)

    # model hyper‑parameters
    p.add_argument("--embed_dim",      type=int, default=64)
    p.add_argument("--gat_hidden_dim", type=int, default=64)
    p.add_argument("--gat_heads",      type=int, default=8)
    p.add_argument("--gat_depth",      type=int, default=4)
    p.add_argument("--transformer_heads", type=int, default=8, help="Number of heads in transformer encoder")
    p.add_argument("--transformer_layers", type=int, default=2, help="Number of layers in transformer encoder")

    # ablation study
    p.add_argument("--disable_spectral", action="store_true")
    p.add_argument("--disable_spatial",  action="store_true")
    p.add_argument("--fusion_mode", choices=["gate", "concat", "spatial_only", "spectral_only"], default="gate")


    p.add_argument("--checkpoint", type=str,
                    help="required when --mode test")
    p.add_argument("--run_tsne", action="store_true", default=False, help="Run t-SNE after training or testing (default: False)")
    p.add_argument("--max_samples", type=int, default=3000, help="Max number of samples for t-SNE")

    # Loss function options
    p.add_argument('--force_wce', action='store_true', help='Force using weighted CrossEntropy loss')
    p.add_argument('--force_focal', action='store_true', help='Force using Focal loss')
    p.add_argument('--imbalance_threshold', type=float, default=5, help='Imbalance ratio threshold for adaptive loss selection')

    args = p.parse_args()

    # ── reproducibility ---------------------------------------------------- #
    set_seed(args.seed)

    # ── load arrays & resolve file paths ----------------------------------- #
    img_arr, gt_arr = load_dataset(args.dataset)       # just NumPy arrays
    args.raw_gt = gt_arr                               # used for splitting
    args.imbalance_ratio = get_imbalance_ratio(gt_arr)

    args.cube_path, args.gt_path = resolve_dataset_paths(args.dataset)

    # ── output dir --------------------------------------------------------- #
    os.makedirs(args.output_dir, exist_ok=True)

    # ----------------------------------------------- dispatch --------
    if args.mode == "train":
        train(args)
        # After training, optionally run t-SNE
        if args.run_tsne:
            # If checkpoint is not set, use the default best checkpoint path
            if not args.checkpoint:
                args.checkpoint = os.path.join(
                    args.output_dir, f"gast_best_{args.dataset}.pth"
                )
            tsne_main(args)
    else:  # test
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required for --mode test")
        test(args)
        # Optionally run t-SNE after test as well
        if args.run_tsne:
            tsne_main(args)


if __name__ == "__main__":
    main()


