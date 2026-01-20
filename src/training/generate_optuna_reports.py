# src/training/generate_optuna_reports.py

#!/usr/bin/env python3
"""
Visualize and report on Optuna studies (.pkl or DB).
Usage:
# For .pkl file:
python src/training/generate_optuna_reports.py --pkl_path /path/to/study_Pavia_University_fine.pkl --out_dir reports/optuna_figures
python src/training/generate_optuna_reports.py --pkl_path reports/results/20250611_215730/study_Kennedy_Space_Center_fine.pkl --out_dir reports/optuna_figures

# For DB studies:
python src/training/generate_optuna_reports.py --dataset Pavia_University --db_dir reports/optuna_db --out_dir reports/optuna_reports --top_n 10 --zip_name optuna_reports_study_Pavia_University.zip
python src/training/generate_optuna_reports.py --dataset Indian_Pines --db_dir reports/optuna_db --out_dir reports/optuna_reports --top_n 10 --zip_name optuna_reports_study_Indian_Pines.zip

python src/training/generate_optuna_reports.py --dataset Kennedy_Space_Center --db_dir reports/results/study_Kennedy_Space_Center_coarse.db --out_dir reports/optuna_reports --top_n 10 --zip_name optuna_reports_study_Kennedy_Space_Center.zip

# note that parameter importance for coarse or fine phases may vary slightly
"""

import argparse
import os
import json
import zipfile
from collections import Counter
from pathlib import Path

import optuna
import pandas as pd
import optuna.visualization as vis
import joblib

def visualize_study(study, out_dir, phase=""):
    os.makedirs(out_dir, exist_ok=True)
    suffix = f"_{phase}" if phase else ""
    # Optimization history
    fig1 = vis.plot_optimization_history(study)
    fig1.write_image(str(Path(out_dir) / f"optimization_history{suffix}.png"))

    # Parameter importances
    try:
        fig2 = vis.plot_param_importances(study)
        fig2.write_image(str(Path(out_dir) / f"param_importances{suffix}.png"))
    except ValueError:
        print("Not enough variation to plot importances, skipping.")
    # Parallel coordinate plot
    fig3 = vis.plot_parallel_coordinate(study)
    fig3.write_image(str(Path(out_dir) / f"parallel_coordinate{suffix}.png"))
    # Slice plot
    fig4 = vis.plot_slice(study)
    fig4.write_image(str(Path(out_dir) / f"slice_plot{suffix}.png"))
    print(f"Visualizations saved to {out_dir}")

def process_study(dataset: str, phase: str, db_dir: str, out_dir: str, top_n: int):
    db_path = os.path.join(db_dir, f"study_{dataset}_{phase}.db")
    storage = f"sqlite:///{db_path}"
    phase_dir = os.path.join(out_dir, phase)
    os.makedirs(phase_dir, exist_ok=True)
    study = optuna.load_study(study_name=phase, storage=storage)
    visualize_study(study, phase_dir, phase)
    # Export full trials table
    df = study.trials_dataframe(attrs=("number", "value", "state", "params"))
    csv_all = os.path.join(phase_dir, f"{phase}_all_trials.csv")
    df.to_csv(csv_all, index=False)
    # Top-N trials
    top_df = df.sort_values("value", ascending=False).head(top_n)
    csv_top = os.path.join(phase_dir, f"{phase}_top_{top_n}_trials.csv")
    top_df.to_csv(csv_top, index=False)
    # Trial-state summary
    states = Counter(df["state"])
    json_states = os.path.join(phase_dir, f"{phase}_state_summary.json")
    with open(json_states, "w") as f:
        json.dump(states, f, indent=2)
    # Robustness stats on top-N
    vals = top_df["value"]
    stats = {"mean_top": float(vals.mean()), "std_top": float(vals.std())}
    json_stats = os.path.join(phase_dir, f"{phase}_robustness_stats.json")
    with open(json_stats, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[{phase}] reports written to {phase_dir}")

def package_reports(out_dir: str, zip_name: str):
    zip_path = os.path.join(out_dir, zip_name)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(out_dir):
            for fn in files:
                if fn == zip_name:
                    continue
                full = os.path.join(root, fn)
                arc = os.path.relpath(full, start=out_dir)
                zf.write(full, arcname=arc)
    print(f"❏ Packaged all reports into {zip_path}")

def main():
    p = argparse.ArgumentParser(description="Visualize and report on Optuna studies (.pkl or DB).")
    p.add_argument("--pkl_path", type=str, help="Path to Optuna study .pkl file")
    p.add_argument("--dataset", type=str, help="Dataset name (for DB mode)")
    p.add_argument("--phases", nargs="+", default=["coarse", "fine"], help="Optuna study phases to process (DB mode)")
    p.add_argument("--db_dir", default="reports/optuna_db", help="Directory containing study_⟨dataset⟩_⟨phase⟩.db")
    p.add_argument("--out_dir", default="reports/optuna_figures", help="Where to write all reports/figures")
    p.add_argument("--top_n", type=int, default=10, help="How many top trials to include (DB mode)")
    p.add_argument("--zip_name", default="optuna_reports.zip", help="Name of the output ZIP (DB mode)")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.pkl_path:
        print(f"Loading Optuna study from: {args.pkl_path}")
        study = joblib.load(args.pkl_path)
        visualize_study(study, args.out_dir)
        print("✅ Visualizations done for .pkl study.")
    elif args.dataset:
        for phase in args.phases:
            process_study(args.dataset, phase, args.db_dir, args.out_dir, args.top_n)
        package_reports(args.out_dir, args.zip_name)
        print("✅ All DB-based reports and visualizations done.")
    else:
        print("❌ Please provide either --pkl_path or --dataset (with DB).")
        return 
    print(f"✅ All reports saved to {args.out_dir}")
if __name__ == "__main__":
    main()
    