# summarize_metrics.py

import json
import csv
import os
import statistics
import sys
from pathlib import Path

# Dynamically add the project root to PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(PROJECT_ROOT))

def summarize_metrics(root_dir: str):
    """
    Summarize Overall Accuracy (OA), Average Accuracy (AA), and Kappa from
    metrics_seed_*.json files under each dataset's test_results directory.
    """
    root = Path(root_dir)
    datasets = [p for p in root.iterdir() if p.is_dir()]
    print(f"Found {len(datasets)} datasets in {root_dir}")
    print(f"Dataset names: {[ds.name for ds in datasets]}")
    print("Summarizing metrics...")
    # Initialize a dictionary to hold the summary statistics
    # for each dataset
    summary = {}

    for ds in datasets:
        print(f"Processing dataset: {ds.name}")
        test_results_dir = ds / "test_results"
        if not test_results_dir.is_dir():
            print(f"Skipping {ds.name}: No test_results directory found.")
            continue

        # Check if the directory contains JSON files
        json_files = list(test_results_dir.glob("metrics_seed_*.json"))
        if not json_files:
            print(f"Skipping {ds.name}: No metrics_seed_*.json files found.")
            continue

        # Initialize lists to hold the metrics for each dataset
        oa_vals, aa_vals, kappa_vals = [], [], []

        # Read the JSON files and extract metrics
        for json_file in json_files:
            with open(json_file, 'r') as f:
                data = json.load(f)
            oa_vals.append(data.get("Overall Accuracy", 0))
            aa_vals.append(data.get("Average Accuracy", 0))
            kappa_vals.append(data.get("Kappa", 0))

        if oa_vals:
            summary[ds.name] = {
                "OA_mean": statistics.mean(oa_vals),
                "OA_std": statistics.stdev(oa_vals) if len(oa_vals) > 1 else 0.0,
                "AA_mean": statistics.mean(aa_vals),
                "AA_std": statistics.stdev(aa_vals) if len(aa_vals) > 1 else 0.0,
                "Kappa_mean": statistics.mean(kappa_vals),
                "Kappa_std": statistics.stdev(kappa_vals) if len(kappa_vals) > 1 else 0.0
            }

    # Print summary table
    print(f"{'Dataset':20s} {'OA_mean':>8s} {'OA_std':>8s} {'AA_mean':>8s} {'AA_std':>8s} {'Kappa_mean':>12s} {'Kappa_std':>10s}")
    for ds, metrics in summary.items():
        print(f"{ds:20s} "
              f"{metrics['OA_mean']*100:8.2f}% {metrics['OA_std']*100:8.2f}% "
              f"{metrics['AA_mean']*100:8.2f}% {metrics['AA_std']*100:8.2f}% "
              f"{metrics['Kappa_mean']*100:12.2f}% {metrics['Kappa_std']*100:10.2f}%")

    # Print summary table (different formatting)
    print("=" * 70)
    print("\nSummary of metrics:")



    # Save summary as CSV
    csv_path = "reports/results/metrics_summary.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Dataset", "OA (mean ± std)", "AA (mean ± std)", "Kappa (mean ± std)"])
        for ds, metrics in summary.items():
            oa_str = f"{metrics['OA_mean']*100:.2f} ± {metrics['OA_std']*100:.2f}"
            aa_str = f"{metrics['AA_mean']*100:.2f} ± {metrics['AA_std']*100:.2f}"
            kappa_str = f"{metrics['Kappa_mean']*100:.2f} ± {metrics['Kappa_std']*100:.2f}"
            writer.writerow([ds, oa_str, aa_str, kappa_str])

    print(f"\nSummary table saved to {csv_path}")

    # Save summary in the requested format
    txt_path = "reports/results/metrics_summary.txt"
    os.makedirs(os.path.dirname(txt_path), exist_ok=True)
    with open(txt_path, "w") as f:
        f.write(f"{'Dataset':22s}{'OA':>16s}{'AA':>16s}{'Kappa':>16s}\n")
        for ds, metrics in summary.items():
            oa_str = f"{metrics['OA_mean']*100:.2f} ± {metrics['OA_std']*100:.2f}"
            aa_str = f"{metrics['AA_mean']*100:.2f} ± {metrics['AA_std']*100:.2f}"
            kappa_str = f"{metrics['Kappa_mean']*100:.2f} ± {metrics['Kappa_std']*100:.2f}"
            f.write(f"{ds:22s}{oa_str:>16s}{aa_str:>16s}{kappa_str:>16s}\n")

    print(f"\nSummary table saved to {txt_path}")

    # Save summary in LaTeX table format
    latex_path = "reports/results/metrics_summary_latex.txt"
    os.makedirs(os.path.dirname(latex_path), exist_ok=True)
    with open(latex_path, "w") as f:
        for ds, metrics in summary.items():
            oa_str = f"{metrics['OA_mean']*100:.2f} ± {metrics['OA_std']*100:.2f}"
            aa_str = f"{metrics['AA_mean']*100:.2f} ± {metrics['AA_std']*100:.2f}"
            kappa_str = f"{metrics['Kappa_mean']*100:.2f} ± {metrics['Kappa_std']*100:.2f}"
            # Replace underscores with spaces and capitalize for LaTeX style
            ds_fmt = ds.replace("_", " ")
            f.write(f"{ds_fmt:22s} & {oa_str:>12s} & {aa_str:>12s} & {kappa_str:>12s}   \\\\\n")

    print(f"\nLaTeX summary table saved to {latex_path}")

if __name__ == "__main__":
    # Adjust the root_dir path as needed
    summarize_metrics("./models/final/gast")

