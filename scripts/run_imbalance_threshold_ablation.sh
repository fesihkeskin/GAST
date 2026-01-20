#!/bin/bash
# run_imbalance_threshold_ablation.sh
# Efficient threshold study using dataset-specific thresholds to test both loss functions
# cd ./scripts/
# Usage: chmod +x run_imbalance_threshold_ablation.sh && ./run_imbalance_threshold_ablation.sh

set -e  # Stop if any command fails

# Get project root
PROJECT_ROOT="$(dirname "$(dirname "$(realpath "$0")")")"
echo "Project root: ${PROJECT_ROOT}"

# Go back to project root directory
cd "${PROJECT_ROOT}"

# ===== Configuration =====
# Datasets with imbalance ratios and dataset-specific thresholds
# Format: "dataset IR wce_threshold focal_threshold"
DATASETS=(
    "Botswana 3.00 2.5 3.5"
    "Houston13 3.94 3.5 4.0"
    "SalinasA 4.00 3.5 4.5"
    "Kennedy_Space_Center 9.20 9.0 10.0"
    "Salinas 12.51 12.0 13.0"
    "Pavia_University 19.83 19.0 20.0"
    "Pavia_Centre 24.61 24.5 25.0"
    "Indian_Pines 24.40 24.0 25.0"
)

# Base output directory
OUTPUT_DIR="models/threshold_tuning"
RESULTS_CSV="${OUTPUT_DIR}/threshold_tuning_results.csv"
SUMMARY_CSV="${OUTPUT_DIR}/threshold_summary.csv"

# ===== Create directories =====
mkdir -p "${OUTPUT_DIR}"
touch "${RESULTS_CSV}" # Create file if it doesn't exist
# Initialize header only if file is empty
if [ ! -s "${RESULTS_CSV}" ]; then
    echo "Dataset,ImbalanceRatio,Threshold,LossFunction,OA,AA,Kappa,TrainTime" > "${RESULTS_CSV}"
fi
echo "Dataset,IR,WCE_Threshold,WCE_OA,Focal_Threshold,Focal_OA,WCE_vs_Focal_Diff" > "${SUMMARY_CSV}"

# ===== Main function =====
run_experiment() {
    local dataset=$1
    local imbalance=$2
    local threshold=$3
    local output_subdir="${OUTPUT_DIR}/${dataset}/threshold_${threshold}"
    local test_log="${output_subdir}/test.log"
    
    # Determine loss function based on threshold comparison
    if (( $(echo "$imbalance <= $threshold" | bc -l) )); then
        LOSS_ARGS="--force_wce"
        LOSS_NAME="wCE"
    else
        LOSS_ARGS="--force_focal"  
        LOSS_NAME="Focal"
    fi
    
    # Check if this experiment has already completed (test.log exists)
    if [ -f "${test_log}" ]; then
        echo "✅ Experiment already completed for ${dataset} with threshold=${threshold}, using ${LOSS_NAME}"
        
        # Extract metrics from existing test log
        local oa=$(grep "Overall accuracy" "${test_log}" | awk '{print $3}')
        local aa=$(grep "Average accuracy" "${test_log}" | awk '{print $3}')
        local kappa=$(grep "Kappa" "${test_log}" | awk '{print $2}')
        local train_time="NA" # We don't have this for previously completed runs
        
        # Check if results for this dataset and threshold are already in the CSV
        if ! grep -q "${dataset},${imbalance},${threshold}" "${RESULTS_CSV}"; then
            echo "${dataset},${imbalance},${threshold},${LOSS_NAME},${oa},${aa},${kappa},${train_time}" >> "${RESULTS_CSV}"
            echo "  Added results to CSV: OA=${oa}, AA=${aa}, Kappa=${kappa}"
        else
            echo "  Results already in CSV, skipping"
        fi
    else
        echo "🔬 Running ${dataset} (IR=${imbalance}) with threshold=${threshold}, using ${LOSS_NAME}"
        mkdir -p "${output_subdir}"
        
        # Get best hyperparameters for this dataset
        BEST_ARGS=$(get_best_args_for_dataset "${dataset}")
        COMMON_ARGS="--train_ratio 0.05 --val_ratio 0.05 --epochs 500 --early_stop 50 --seed 242 --num_workers 4"
        
        # Run training
        echo "🚀 Training..."
        start_time=$(date +%s)
        python main.py --mode train \
            --dataset ${dataset} \
            --output_dir ${output_subdir} \
            ${COMMON_ARGS} \
            ${BEST_ARGS} \
            ${LOSS_ARGS} \
            | tee "${output_subdir}/train.log"
        end_time=$(date +%s)
        train_time=$((end_time - start_time))
            
        # Run testing
        echo "🧪 Testing..."
        python main.py --mode test \
            --dataset ${dataset} \
            --checkpoint ${output_subdir}/gast_best_${dataset}.pth \
            --output_dir ${output_subdir} \
            ${COMMON_ARGS} \
            ${BEST_ARGS} \
            ${LOSS_ARGS} \
            | tee "${test_log}"
            
        # Extract metrics from test log
        local oa=$(grep "Overall accuracy" "${test_log}" | awk '{print $3}')
        local aa=$(grep "Average accuracy" "${test_log}" | awk '{print $3}')
        local kappa=$(grep "Kappa" "${test_log}" | awk '{print $2}')
        
        # Append to results file
        echo "${dataset},${imbalance},${threshold},${LOSS_NAME},${oa},${aa},${kappa},${train_time}" >> "${RESULTS_CSV}"
    fi
}

# ===== Helper function to get dataset-specific best args =====
get_best_args_for_dataset() {
    local ds=$1
    case "$ds" in
        "Botswana")
            echo "--batch_size 64 --patch_size 13 --stride 3 --lr 0.0004224382351548547 --weight_decay 0.00011150299441886956 --dropout 0.25 --embed_dim 128 --gat_hidden_dim 64 --gat_heads 4 --gat_depth 2 --transformer_heads 8 --transformer_layers 9 --fusion_mode gate"
            ;;
        "Houston13")
            echo "--batch_size 16 --patch_size 13 --stride 2 --lr 0.00024092273071567313 --weight_decay 8.500203317456351e-08 --dropout 0.15 --embed_dim 128 --gat_hidden_dim 128 --gat_heads 4 --gat_depth 2 --transformer_heads 8 --transformer_layers 4 --fusion_mode gate"
            ;;
        "Indian_Pines")
            echo "--batch_size 48 --patch_size 7 --stride 3 --lr 0.0001906390094523524 --weight_decay 0.009021540956272492 --dropout 0.1 --embed_dim 128 --gat_hidden_dim 32 --gat_heads 2 --gat_depth 8 --transformer_heads 8 --transformer_layers 6 --fusion_mode gate"
            ;;
        "Kennedy_Space_Center")
            echo "--batch_size 64 --patch_size 9 --stride 8 --lr 0.0005940732289188326 --weight_decay 0.0008803995798938349 --dropout 0.25 --embed_dim 256 --gat_hidden_dim 64 --gat_heads 10 --gat_depth 6 --transformer_heads 2 --transformer_layers 4 --fusion_mode gate"
            ;;
        "Pavia_Centre")
            echo "--batch_size 64 --patch_size 13 --stride 4 --lr 0.0001236480816653455 --weight_decay 4.102919947676974e-07 --dropout 0.45 --embed_dim 256 --gat_hidden_dim 64 --gat_heads 4 --gat_depth 4 --transformer_heads 16 --transformer_layers 3 --fusion_mode gate"
            ;;
        "Pavia_University")
            echo "--batch_size 64 --patch_size 11 --stride 4 --lr 0.00031767281914492677 --weight_decay 0.00609658164739183 --dropout 0.2 --embed_dim 64 --gat_hidden_dim 32 --gat_heads 4 --gat_depth 4 --transformer_heads 16 --transformer_layers 9 --fusion_mode gate"
            ;;
        "Salinas")
            echo "--batch_size 32 --patch_size 13 --stride 4 --lr 0.00023614474992419862 --weight_decay 0.0007723031252522047 --dropout 0.15 --embed_dim 128 --gat_hidden_dim 32 --gat_heads 10 --gat_depth 4 --transformer_heads 2 --transformer_layers 2 --fusion_mode gate"
            ;;
        "SalinasA")
            echo "--batch_size 48 --patch_size 11 --stride 6 --lr 0.0003376670734565324 --weight_decay 8.662401208300589e-08 --dropout 0.0 --embed_dim 256 --gat_hidden_dim 32 --gat_heads 4 --gat_depth 8 --transformer_heads 16 --transformer_layers 10 --fusion_mode gate"
            ;;
        *)
            echo "Unknown dataset: $ds" >&2
            exit 1
            ;;
    esac
}

# ===== Main execution =====
echo "🔎 Starting efficient imbalance threshold study"
echo "Using dataset-specific thresholds to test both loss functions"
echo "Results will be saved to ${RESULTS_CSV}"

# Run experiments for all datasets with their specific thresholds
for dataset_info in "${DATASETS[@]}"; do
    # Parse dataset info
    read -r dataset imbalance wce_threshold focal_threshold <<< "$dataset_info"
    
    echo ""
    echo "==========================================="
    echo "📊 Dataset: ${dataset} (IR=${imbalance})"
    echo "    WCE Threshold: ${wce_threshold}"
    echo "    Focal Threshold: ${focal_threshold}"
    echo "==========================================="
    
    # Run WCE experiment (threshold below IR)
    run_experiment "${dataset}" "${imbalance}" "${wce_threshold}"
    
    # Run Focal experiment (threshold above IR)
    run_experiment "${dataset}" "${imbalance}" "${focal_threshold}"
done

# ===== Generate summary statistics =====
echo ""
echo "🏁 Experiments completed!"
echo "Generating summary statistics..."

# Use Python to analyze results and generate summary
python -c "
import pandas as pd
import numpy as np

# Read results
df = pd.read_csv('${RESULTS_CSV}')

# Prepare summary comparing WCE and Focal for each dataset
summary_data = []

for dataset, group in df.groupby('Dataset'):
    ir = group['ImbalanceRatio'].iloc[0]
    
    # Get wCE and Focal results
    wce_row = group[group['LossFunction'] == 'wCE']
    focal_row = group[group['LossFunction'] == 'Focal']
    
    if len(wce_row) > 0 and len(focal_row) > 0:
        wce_threshold = wce_row['Threshold'].iloc[0]
        wce_oa = wce_row['OA'].iloc[0]
        focal_threshold = focal_row['Threshold'].iloc[0]
        focal_oa = focal_row['OA'].iloc[0]
        diff = focal_oa - wce_oa
        
        summary_data.append({
            'Dataset': dataset,
            'IR': ir,
            'WCE_Threshold': wce_threshold,
            'WCE_OA': wce_oa,
            'Focal_Threshold': focal_threshold,
            'Focal_OA': focal_oa,
            'WCE_vs_Focal_Diff': diff
        })

# Create summary DataFrame and save
summary_df = pd.DataFrame(summary_data)
summary_df.to_csv('${SUMMARY_CSV}', index=False)

# Print summary table
print('\n=== LOSS FUNCTION COMPARISON BY DATASET ===')
summary_table = summary_df.sort_values('IR')
print(summary_table.round(4))

# Determine which loss function is better for different IR ranges
print('\n=== ANALYSIS BY IMBALANCE RATIO ===')
low_ir = summary_df[summary_df['IR'] <= 5]
mid_ir = summary_df[(summary_df['IR'] > 5) & (summary_df['IR'] <= 15)]
high_ir = summary_df[summary_df['IR'] > 15]

print(f'Low IR (≤5): {low_ir.shape[0]} datasets')
if low_ir.shape[0] > 0:
    low_better = low_ir[low_ir['WCE_vs_Focal_Diff'] < 0].shape[0]
    low_better_pct = (low_better / low_ir.shape[0]) * 100
    print(f'  WCE better in {low_better}/{low_ir.shape[0]} cases ({low_better_pct:.1f}%)')
    print(f'  Avg diff: {low_ir[\"WCE_vs_Focal_Diff\"].mean():.4f}')

print(f'Medium IR (5-15): {mid_ir.shape[0]} datasets')
if mid_ir.shape[0] > 0:
    mid_better = mid_ir[mid_ir['WCE_vs_Focal_Diff'] < 0].shape[0]
    mid_better_pct = (mid_better / mid_ir.shape[0]) * 100
    print(f'  WCE better in {mid_better}/{mid_ir.shape[0]} cases ({mid_better_pct:.1f}%)')
    print(f'  Avg diff: {mid_ir[\"WCE_vs_Focal_Diff\"].mean():.4f}')

print(f'High IR (>15): {high_ir.shape[0]} datasets')
if high_ir.shape[0] > 0:
    high_better = high_ir[high_ir['WCE_vs_Focal_Diff'] < 0].shape[0]
    high_better_pct = (high_better / high_ir.shape[0]) * 100
    print(f'  WCE better in {high_better}/{high_ir.shape[0]} cases ({high_better_pct:.1f}%)')
    print(f'  Avg diff: {high_ir[\"WCE_vs_Focal_Diff\"].mean():.4f}')

# Overall recommendation
print('\n=== RECOMMENDED THRESHOLD VALUE ===')
avg_diff_by_ir = summary_df.groupby(pd.cut(summary_df['IR'], bins=[0, 5, 15, 100], labels=['Low', 'Medium', 'High']))['WCE_vs_Focal_Diff'].mean()

print('Based on experimental results:')
for ir_range, avg_diff in avg_diff_by_ir.items():
    recommended = 'wCE' if avg_diff < 0 else 'Focal'
    print(f'  {ir_range} IR range: {recommended} recommended (avg diff: {avg_diff:.4f})')

# Create LaTeX table
latex_table = summary_df.sort_values('IR').round(4).to_latex(index=False)
with open('${OUTPUT_DIR}/threshold_comparison_latex.txt', 'w') as f:
    f.write(latex_table)
print(f'\nLaTeX table saved to: ${OUTPUT_DIR}/threshold_comparison_latex.txt')
"

echo ""
echo "Done! 🎉"