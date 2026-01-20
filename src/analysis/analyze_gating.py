#!/usr/bin/env python3
# src/analysis/analyze_gating.py

"""
Script to analyze GAST gating weights (g) for Minority vs. Majority classes.
Extracts the gate value g from the fusion layer during inference and plots
the average gate value per class, sorted by class frequency.
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import HyperspectralDataset
from src.data.data_loader import DATASET_PATHS
from src.models.model_architecture import GAST
from src.utils.utils import set_seed
from src.data.dataset_info import get_dataset_labels

def analyze_gating(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Device: {device}")
    set_seed(args.seed)

    # 1. Load Dataset and Split Indices
    print(f"📂 Loading dataset: {args.dataset}")
    cube_path = PROJECT_ROOT / DATASET_PATHS[args.dataset]["image"]
    gt_path   = PROJECT_ROOT / DATASET_PATHS[args.dataset]["ground_truth"]
    
    # We need the test indices used during training to ensure valid analysis
    checkpoint_path = Path(args.checkpoint)
    split_dir = checkpoint_path.parent / "splits"
    test_idx_path = split_dir / f"test_idx_seed_{args.seed}.npy"
    
    if not test_idx_path.exists():
        if args.output_dir:
             split_dir = Path(args.output_dir) / "splits"
             test_idx_path = split_dir / f"test_idx_seed_{args.seed}.npy"
    
    if not test_idx_path.exists():
        raise FileNotFoundError(f"❌ Could not find test indices at {test_idx_path}. Please check path or seed.")
        
    print(f"📋 Loading test indices from: {test_idx_path}")
    test_idx = np.load(test_idx_path)

    # Create Dataset
    train_idx_path = split_dir / f"train_idx_seed_{args.seed}.npy"
    mean, std = None, None
    if train_idx_path.exists():
        train_idx = np.load(train_idx_path)
        train_ds_temp = HyperspectralDataset(cube_path, gt_path, patch_size=args.patch_size, mode="train", indices=train_idx)
        mean, std = train_ds_temp.mean, train_ds_temp.std
        del train_ds_temp

    test_ds = HyperspectralDataset(
        cube_path, gt_path,
        patch_size=args.patch_size,
        stride=1,
        mode="test",
        indices=test_idx,
        mean=mean,
        std=std,
        augment=False
    )

    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # 2. Load Model
    print(f"🤖 Loading model from: {args.checkpoint}")
    n_cls = len(np.unique(test_ds.gt))
    in_ch = test_ds.B
    
    model = GAST(
        in_channels=in_ch,
        n_classes=n_cls,
        patch_size=args.patch_size,
        spec_dim=args.spec_dim,
        spat_dim=args.spat_dim,
        n_heads=args.gat_heads,
        n_layers=args.gat_depth,
        transformer_heads=args.transformer_heads,
        transformer_layers=args.transformer_layers,
        dropout=args.dropout,
        fusion_mode="gate"
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # 3. Hook into the Gating Mechanism
    gating_values = []
    
    def get_gate_activation(name):
        def hook(model, input, output):
            # output is the result of gate_mlp (which includes Sigmoid in model_architecture.py)
            # So output is already in [0, 1]
            gate_val = output.detach().cpu()
            gating_values.append(gate_val)
        return hook

    # Register hook on the gate_mlp layer
    if hasattr(model, 'gate_mlp') and model.gate_mlp is not None:
        print("✅ Found 'gate_mlp' layer, registering hook.")
        model.gate_mlp.register_forward_hook(get_gate_activation('gate_mlp'))
    elif hasattr(model, 'gate_lin'):
        print("✅ Found 'gate_lin' layer, registering hook.")
        # Fallback for older versions where sigmoid wasn't in the layer
        def get_gate_activation_linear(name):
            def hook(model, input, output):
                gate_val = torch.sigmoid(output).detach().cpu()
                gating_values.append(gate_val)
            return hook
        model.gate_lin.register_forward_hook(get_gate_activation_linear('gate_lin'))
    else:
        print("❌ Error: Model does not have 'gate_mlp' or 'gate_lin' layer.")
        print("Available modules:", list(model._modules.keys()))
        return

    # 4. Run Inference
    print("⚡ Running inference to extract gating weights...")
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(loader):
            x = batch["patch"].to(device)
            y = batch["label"]
            _ = model(x)
            all_labels.append(y)

    # 5. Process Data
    if not gating_values:
        print("❌ No gating values captured. Check if fusion_mode='gate' was used.")
        return

    gates = torch.cat(gating_values, dim=0).numpy() # Shape: [N_samples, Hidden_Dim]
    labels = torch.cat(all_labels, dim=0).numpy()   # Shape: [N_samples]

    # Calculate mean gate value per pixel (scalar)
    mean_gate_per_pixel = np.mean(gates, axis=1) # Shape: [N_samples]

    # 6. Group by Class
    unique_classes = np.unique(labels)
    class_gate_means = {}
    class_counts = {}

    for cls in unique_classes:
        mask = (labels == cls)
        class_gate_means[cls] = np.mean(mean_gate_per_pixel[mask])
        class_counts[cls] = np.sum(mask)

    # 7. Sort Classes by Frequency (Minority -> Majority)
    sorted_classes = sorted(unique_classes, key=lambda c: class_counts[c])
    
    class_names_map = get_dataset_labels(args.dataset)
    sorted_names = [class_names_map[c] if c < len(class_names_map) else str(c) for c in sorted_classes]
    sorted_means = [class_gate_means[c] for c in sorted_classes]
    sorted_counts = [class_counts[c] for c in sorted_classes]

    print("\n📊 Analysis Results (Sorted by Sample Count):")
    print(f"{'Class Name':<25} | {'Count':<6} | {'Avg Gate':<8}")
    print("-" * 45)
    for n, c, g in zip(sorted_names, sorted_counts, sorted_means):
        print(f"{n:<25} | {c:<6} | {g:.4f}")

    # 8. Plotting
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 6))
    
    x_pos = np.arange(len(sorted_classes))
    bars = plt.bar(x_pos, sorted_means, color=sns.color_palette("viridis", len(sorted_classes)))
    
    plt.xlabel("Classes (Sorted: Minority → Majority)", fontsize=12, fontweight='bold')
    plt.ylabel(r"Average Gate Value $\mathbf{g}$ (0=Spectral, 1=Spatial)", fontsize=12, fontweight='bold')
    plt.title(f"Gating Mechanism Behavior by Class Imbalance ({args.dataset})", fontsize=14)
    
    plt.xticks(x_pos, sorted_names, rotation=45, ha='right', fontsize=10)
    plt.ylim(0, 1.0)
    
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                 f'n={sorted_counts[i]}',
                 ha='center', va='bottom', rotation=90, fontsize=9)

    plt.tight_layout()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / f"gating_analysis_{args.dataset}.eps"
    plt.savefig(save_path, dpi=300)
    print(f"\n💾 Plot saved to: {save_path}")
    
    csv_path = output_dir / f"gating_analysis_{args.dataset}.csv"
    with open(csv_path, "w") as f:
        f.write("ClassID,ClassName,Count,AvgGateValue\n")
        for cid, name, count, val in zip(sorted_classes, sorted_names, sorted_counts, sorted_means):
            f.write(f"{cid},{name},{count},{val:.6f}\n")
    print(f"💾 Data saved to: {csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze GAST Gating Weights")
    
    # Dataset & Paths
    parser.add_argument("--dataset", type=str, required=True, choices=list(DATASET_PATHS.keys()))
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained .pth model")
    parser.add_argument("--output_dir", type=str, default="./reports/analysis", help="Directory to save plots")
    
    # Model Hyperparameters
    parser.add_argument("--patch_size", type=int, default=11)
    parser.add_argument("--spec_dim", type=int, default=64)
    parser.add_argument("--spat_dim", type=int, default=64)
    parser.add_argument("--gat_heads", type=int, default=4)
    parser.add_argument("--gat_depth", type=int, default=2)
    parser.add_argument("--transformer_heads", type=int, default=8)
    parser.add_argument("--transformer_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    
    # System
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=242)

    args = parser.parse_args()
    
    analyze_gating(args)
    
# Example command lines for different datasets:
# python src/analysis/analyze_gating.py --dataset Botswana --checkpoint /home/fesih/Desktop/ubuntu_projects/GAST2/models/final/gast/Botswana/gast_best_Botswana.pth --output_dir /home/fesih/Desktop/ubuntu_projects/GAST2/reports/gating_analysis/Botswana --patch_size 13 --spec_dim 128 --spat_dim 64 --gat_heads 4 --gat_depth 2 --transformer_heads 8 --transformer_layers 9 --dropout 0.25 --batch_size 64 --seed 242 --num_workers 4

# python src/analysis/analyze_gating.py --dataset Houston13 --checkpoint /home/fesih/Desktop/ubuntu_projects/GAST2/models/final/gast/Houston13/gast_best_Houston13.pth --output_dir /home/fesih/Desktop/ubuntu_projects/GAST2/reports/gating_analysis/Houston13 --patch_size 13 --spec_dim 128 --spat_dim 128 --gat_heads 4 --gat_depth 2 --transformer_heads 8 --transformer_layers 4 --dropout 0.15 --batch_size 16 --seed 242 --num_workers 4

# python src/analysis/analyze_gating.py --dataset Indian_Pines --checkpoint /home/fesih/Desktop/ubuntu_projects/GAST2/models/final/gast/Indian_Pines/gast_best_Indian_Pines.pth --output_dir /home/fesih/Desktop/ubuntu_projects/GAST2/reports/gating_analysis/Indian_Pines --patch_size 7 --spec_dim 128 --spat_dim 32 --gat_heads 2 --gat_depth 8 --transformer_heads 8 --transformer_layers 6 --dropout 0.1 --batch_size 48 --seed 242 --num_workers 4

# python src/analysis/analyze_gating.py --dataset Kennedy_Space_Center --checkpoint /home/fesih/Desktop/ubuntu_projects/GAST2/models/final/gast/Kennedy_Space_Center/gast_best_Kennedy_Space_Center.pth --output_dir /home/fesih/Desktop/ubuntu_projects/GAST2/reports/gating_analysis/Kennedy_Space_Center --patch_size 9 --spec_dim 256 --spat_dim 64 --gat_heads 10 --gat_depth 6 --transformer_heads 2 --transformer_layers 4 --dropout 0.25 --batch_size 64 --seed 242 --num_workers 4

# python src/analysis/analyze_gating.py --dataset Pavia_Centre --checkpoint /home/fesih/Desktop/ubuntu_projects/GAST2/models/final/gast/Pavia_Centre/gast_best_Pavia_Centre.pth --output_dir /home/fesih/Desktop/ubuntu_projects/GAST2/reports/gating_analysis/Pavia_Centre --patch_size 13 --spec_dim 256 --spat_dim 64 --gat_heads 4 --gat_depth 4 --transformer_heads 16 --transformer_layers 3 --dropout 0.45 --batch_size 64 --seed 242 --num_workers 4

# python src/analysis/analyze_gating.py --dataset Pavia_University --checkpoint /home/fesih/Desktop/ubuntu_projects/GAST2/models/final/gast/Pavia_University/gast_best_Pavia_University.pth --output_dir /home/fesih/Desktop/ubuntu_projects/GAST2/reports/gating_analysis/Pavia_University --patch_size 11 --spec_dim 64 --spat_dim 32 --gat_heads 4 --gat_depth 4 --transformer_heads 16 --transformer_layers 9 --dropout 0.2 --batch_size 64 --seed 242 --num_workers 4

# python src/analysis/analyze_gating.py --dataset Salinas --checkpoint /home/fesih/Desktop/ubuntu_projects/GAST2/models/final/gast/Salinas/gast_best_Salinas.pth --output_dir /home/fesih/Desktop/ubuntu_projects/GAST2/reports/gating_analysis/Salinas --patch_size 13 --spec_dim 128 --spat_dim 32 --gat_heads 10 --gat_depth 4 --transformer_heads 2 --transformer_layers 2 --dropout 0.15 --batch_size 32 --seed 242 --num_workers 4

# python src/analysis/analyze_gating.py --dataset SalinasA --checkpoint /home/fesih/Desktop/ubuntu_projects/GAST2/models/final/gast/SalinasA/gast_best_SalinasA.pth --output_dir /home/fesih/Desktop/ubuntu_projects/GAST2/reports/gating_analysis/SalinasA --patch_size 11 --spec_dim 256 --spat_dim 32 --gat_heads 4 --gat_depth 8 --transformer_heads 16 --transformer_layers 10 --dropout 0.0 --batch_size 48 --seed 242 --num_workers 4