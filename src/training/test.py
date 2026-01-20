#!/usr/bin/env python3
# src/training/test.py

import os
import json
from pathlib import Path
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    classification_report,
)

import torch
from torch.utils.data import DataLoader
import matplotlib.patches as mpatches

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Project imports
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import HyperspectralDataset
from src.data.data_loader import DATASET_PATHS
from src.models.model_architecture import GAST
from src.utils.utils import set_seed
from src.data.dataset_info import get_dataset_labels
from src.utils.visualization import plot_gt_vs_pred_side_by_side


def _build_model(args, in_ch: int, n_cls: int):
    return GAST(
        in_channels=in_ch,
        n_classes=n_cls,
        patch_size=args.patch_size,
        spec_dim=args.embed_dim,
        spat_dim=args.gat_hidden_dim,
        n_heads=args.gat_heads,
        n_layers=args.gat_depth,
        transformer_heads=args.transformer_heads,
        transformer_layers=args.transformer_layers,
        dropout=args.dropout,
        disable_spectral=args.disable_spectral,
        disable_spatial=args.disable_spatial,
        fusion_mode=args.fusion_mode,
    )


def test(args):
    """Main evaluation routine with result saving."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n🚀 device:", device)
    print(f"Using dataset: {args.dataset}:\n")
    set_seed(args.seed)

    # resolve paths
    cube_path = PROJECT_ROOT / DATASET_PATHS[args.dataset]["image"]
    gt_path   = PROJECT_ROOT / DATASET_PATHS[args.dataset]["ground_truth"]

    # --- LOAD SPLIT INDICES ---
    split_dir = Path(args.output_dir) / "splits"
    train_idx_path = split_dir / f"train_idx_seed_{args.seed}.npy"
    test_idx_path  = split_dir / f"test_idx_seed_{args.seed}.npy"
    if not train_idx_path.exists() or not test_idx_path.exists():
        raise FileNotFoundError(f"Missing split indices in {split_dir}")
    train_idx = np.load(train_idx_path)
    test_idx  = np.load(test_idx_path)

    # build train_ds using the *exact* train_idx so mean/std match
    train_ds = HyperspectralDataset(
        cube_path, gt_path,
        patch_size=args.patch_size,
        stride=args.stride,  # stride for patch extraction       
        mode="train",           
        indices=train_idx,
        mean=None,
        std=None,
        augment=False
    )
    
    # build test_ds with saved test_idx and same normalization
    test_ds = HyperspectralDataset(
        cube_path, gt_path,
        patch_size=args.patch_size,
        stride=1,               
        mode="test",
        indices=test_idx,
        mean=train_ds.mean,
        std=train_ds.std,
        augment=False
    )

    loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # load model + checkpoint
    model = _build_model(args, in_ch=train_ds.B, n_cls=len(np.unique(args.raw_gt)))
    model.to(device)
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
    model.eval()

    # ── 1. inference + tek seferlik pred_map oluşturma ───────────────────
    y_true, y_pred = [], []
    
    # Önce train ve validation piksellerinin gerçek değerlerini pred_map'e ekleyelim
    pred_map = np.full(test_ds.gt.shape, -1, dtype=int)
    
    # Train split için gerçek değerleri ekle
    for r, c in train_idx:
        pred_map[r, c] = test_ds.gt[r, c]  # Gerçek sınıf değerlerini kullan
        
    # Val split için gerçek değerleri ekle (eğer val_idx varsa)
    val_idx_path = split_dir / f"val_idx_seed_{args.seed}.npy"
    if val_idx_path.exists():
        val_idx = np.load(val_idx_path)
        for r, c in val_idx:
            pred_map[r, c] = test_ds.gt[r, c]
    
    # Şimdi test split için tahminleri yapalım
    import time

    start_infer = time.time()
    n_samples = 0

    for batch in tqdm(loader, desc="testing"):
        x, labels = batch["patch"].to(device), batch["label"].cpu().numpy()
        coords = batch["coord"].cpu().numpy()
        with torch.no_grad():
            logits = model(x)
        preds = logits.argmax(1).cpu().numpy()
        n_samples += x.size(0)

        # ground-truth vs pred toplama
        mask = labels != -1
        y_true.extend(labels[mask])
        y_pred.extend(preds[mask])

        # Test tahminlerini ekle
        for i, (r, c) in enumerate(coords):
            if mask[i]:
                pred_map[r, c] = preds[i] + 1  # +1 ekleyerek ground truth ile aynı indekslemeyi sağla

    infer_time = time.time() - start_infer
    infer_time_per_sample = infer_time / n_samples
    print(f"Inference time per sample: {infer_time_per_sample:.6f} seconds")

    # remove background (0) and ignore_index (-1)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = (y_true >= 0)
    y_true_filt = y_true[mask]
    y_pred_filt = y_pred[mask]

    # present_labels ve isimleri
    present_labels = np.unique(y_true_filt)
    present_class_names = [str(lab) for lab in present_labels]

    # confusion matrix
    cm = confusion_matrix(y_true_filt, y_pred_filt, labels=present_labels)
    per_class_acc = (cm.diagonal() / cm.sum(1)).tolist()
    aa    = float(np.mean(per_class_acc))
    oa    = accuracy_score(y_true_filt, y_pred_filt)
    kappa = cohen_kappa_score(y_true_filt, y_pred_filt)

    # classification report
    cls_report = classification_report(
        y_true_filt, y_pred_filt,
        labels=present_labels,
        target_names=present_class_names,
        digits=4,
        zero_division=0,
        output_dict=True
    )
    print(classification_report(
        y_true_filt, y_pred_filt,
        labels=present_labels,
        target_names=present_class_names,
        digits=4,
        zero_division=0
    ))
    print(f"\nOA  {oa:.4f}  |  AA  {aa:.4f}  |  κ  {kappa:.4f}\n")

    # prepare results directory
    results_dir = Path(args.output_dir) / "test_results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # save metrics JSON
    metrics = {
        "Overall Accuracy": oa,
        "Average Accuracy": aa,
        "Kappa": kappa,
        "Per Class Accuracy": per_class_acc,
        "Classification Report": cls_report,
        "Inference Time Per Sample (seconds)": infer_time_per_sample,
        "Inference Time Total (seconds)": infer_time,
        "Number of Samples": n_samples
    }
    with open(results_dir / f"metrics_seed_{args.seed}.json", "w") as f:
        json.dump(metrics, f, indent=4)

    # plot confusion matrix with numeric ticks
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=present_class_names,
                yticklabels=present_class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(results_dir / f"confusion_matrix_seed_{args.seed}.png")
    plt.close()

    # Save confusion matrix as text file
    cm_txt_path = results_dir / f"confusion_matrix_seed_{args.seed}.txt"
    np.savetxt(cm_txt_path, cm, fmt="%d", delimiter="\t")

    # plot per-class accuracy bar chart
    plt.figure(figsize=(8,4))
    plt.bar(np.arange(len(per_class_acc)), per_class_acc, tick_label=present_class_names)
    plt.ylim(0,1)
    plt.title("Per-Class Accuracy")
    plt.xlabel("Class")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig(results_dir / f"per_class_accuracy_seed_{args.seed}.png")
    plt.close()
    
    # ── 2. sonuçları kaydet ───────────────────────────────────────────────
    np.save(results_dir / f"pred_map_seed_{args.seed}.npy", pred_map)

    # ── 3. yan yana karşılaştırma (tek Norm + discretize colormap) ──────
    import matplotlib as mpl
    labels_full = get_dataset_labels(args.dataset)
    n_colors    = len(labels_full)  # tüm label isimleri (arkaplan dahil)

    # shared Normalize ve discrete colormap
    norm = mpl.colors.Normalize(vmin=0, vmax=n_colors-1)
    cmap = mpl.cm.get_cmap("tab20", n_colors)

    # # ground truth ve pred aralığını, dataset label indeksine uyacak şekilde
    # gt_vis = test_ds.gt.copy()            # raw GT: 1..n_colors
    # pred_vis = pred_map.copy()            
    # pred_vis[pred_vis < 0] = 0            # background için 0

    # fig, axes = plt.subplots(1,3, figsize=(18,6))
    # im0 = axes[0].imshow(gt_vis, cmap=cmap, norm=norm)
    # axes[0].set_title("Ground Truth"); axes[0].axis("off")
    # im1 = axes[1].imshow(pred_vis, cmap=cmap, norm=norm)
    # axes[1].set_title("Prediction"); axes[1].axis("off")

    # # # legend
    # axes[2].axis("off")
    # patches = [mpatches.Patch(color=cmap(i), label=labels_full[i]) for i in range(n_colors)]
    # axes[2].legend( handles=patches, title=r"$\bf{Classes}$", bbox_to_anchor=(1.05, 1), borderaxespad=0. )

    # plt.savefig(results_dir / f"gt_vs_pred_seed_{args.seed}.png", dpi=200, bbox_inches='tight', pad_inches=0)
    # plt.close()

    # Save confusion matrix and per-class accuracy for later visualization
    np.save(results_dir / f"confusion_matrix_seed_{args.seed}.npy", cm)
    np.save(results_dir / f"per_class_accuracy_seed_{args.seed}.npy", np.array(per_class_acc))
    np.save(results_dir / f"pred_map_seed_{args.seed}.npy", pred_map)
    with open(results_dir / f"metrics_seed_{args.seed}.json", "w") as f:
        json.dump(metrics, f, indent=4)

    # Print summary
    print("Results saved to:", results_dir)
    print(f"Metrics JSON saved to: {results_dir / f'metrics_seed_{args.seed}.json'}")
    print(f"Confusion matrix (npy) saved to: {results_dir / f'confusion_matrix_seed_{args.seed}.npy'}")
    print(f"Per-class accuracy (npy) saved to: {results_dir / f'per_class_accuracy_seed_{args.seed}.npy'}")
    print(f"Ground truth vs prediction image saved to: {results_dir / f'gt_vs_pred_seed_{args.seed}.png'}")
    print(f"Prediction map saved to: {results_dir / f'pred_map_seed_{args.seed}.npy'}")
    print("✅ Test completed successfully.")

    plot_gt_vs_pred_side_by_side(
        gt=test_ds.gt,
        pred=pred_map,
        label_names=labels_full,
        dataset_name=args.dataset,
        save_path=results_dir / f"gt_vs_pred_side_by_side_seed_{args.seed}.png"
    )


if __name__ == "__main__":
    import argparse
    from src.data.data_loader import DATASET_PATHS

    parser = argparse.ArgumentParser(
        "Evaluate a GAST checkpoint and save results, including seed in filenames"
    )
    parser.add_argument("--dataset",           required=True, choices=list(DATASET_PATHS))
    parser.add_argument("--patch_size",        type=int, default=11)
    parser.add_argument("--embed_dim",         type=int, default=64)
    parser.add_argument("--gat_hidden_dim",    type=int, default=64)
    parser.add_argument("--gat_heads",         type=int, default=8)
    parser.add_argument("--gat_depth",         type=int, default=4)
    parser.add_argument("--dropout",           type=float, default=0.2)
    parser.add_argument("--transformer_heads", type=int, default=8)
    parser.add_argument("--transformer_layers",type=int, default=1)
    parser.add_argument("--checkpoint",        required=True,
                        help="Path to .pth checkpoint")
    parser.add_argument("--batch_size",        type=int, default=256)
    parser.add_argument("--num_workers",       type=int, default=4)
    parser.add_argument("--seed",              type=int, default=242)
    parser.add_argument("--disable_spectral",  action="store_true")
    parser.add_argument("--disable_spatial",   action="store_true")
    parser.add_argument("--fusion_mode",
                        choices=["gate","concat","spatial_only","spectral_only"],
                        default="gate")
    parser.add_argument("--output_dir",        type=str,
                        default="./models/checkpoints")

    args = parser.parse_args()
    set_seed(args.seed)
    test(args)
    print("✅ Test completed successfully.")