#!/bin/bash
# This script runs parameter sweeps for the GAST model on Pavia_University.
# It sweeps through various hyperparameters like learning rate, embedding dimension, GAT hidden dimension, etc.
# Each sweep will log the test OA for each parameter value.
# It will create a separate directory for each parameter sweep and save the results in JSON format.
# Make sure to run this script from the root of the project directory.
# cd scripts
# chmod +x run_parameter_sweep.sh
# ./run_parameter_sweep.sh

set -e   # Stop if any command fails

# ====== Sweep Ranges (shared for all datasets) ======
# These are the hyperparameters we will sweep through        
LR_LIST=(1e-8 1e-7 1e-6 1e-5 1e-4 5e-4 1e-3 5e-3)
EMBED_DIM_LIST=(32 64 128 256)
GAT_HIDDEN_DIM_LIST=(16 32 64 128)
GAT_HEADS_LIST=(2 3 4 5 6 7 8)
GAT_DEPTH_LIST=(4 6 8 10)
DROPOUT_LIST=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7)
TRANSFORMER_HEADS_LIST=(2 4 8 16)
TRANSFORMER_LAYERS_LIST=(2 4 8 12)
BATCH_SIZE_LIST=(8 16 24 32 48 64 96 128)
PATCH_SIZE_LIST=(7 9 11 13 15)
WEIGHT_DECAY_LIST=(1e-8 1e-7 1e-6 1e-5 1e-4 1e-3 1e-2)
STRIDE_LIST=(1 2 3 4 5 6 7)
FUSION_MODE_LIST=(gate)

# Dataset configs (from optimized parameters CLIs) 

declare -A PATCH_SIZE=( \
    [Botswana]=13 \
    [Indian_Pines]=7 \
    [Kennedy_Space_Center]=9 \
    [Pavia_Centre]=13 \
    [Pavia_University]=11 \
    [Salinas]=13 \
    [SalinasA]=11 \
)

declare -A BATCH_SIZE=( \
    [Botswana]=64 \
    [Indian_Pines]=48 \
    [Kennedy_Space_Center]=64 \
    [Pavia_Centre]=64 \
    [Pavia_University]=64 \
    [Salinas]=32 \
    [SalinasA]=48 \
)

declare -A EPOCHS=( \
    [Botswana]=500 \
    [Indian_Pines]=500 \
    [Kennedy_Space_Center]=500 \
    [Pavia_Centre]=500 \
    [Pavia_University]=200 \
    [Salinas]=500 \
    [SalinasA]=500 \
)

declare -A EARLY_STOP=( \
    [Botswana]=50 \
    [Indian_Pines]=50 \
    [Kennedy_Space_Center]=50 \
    [Pavia_Centre]=50 \
    [Pavia_University]=30 \
    [Salinas]=50 \
    [SalinasA]=50 \
)

declare -A LR=( \
    [Botswana]=0.0004224382351548547 \
    [Indian_Pines]=0.0001906390094523524 \
    [Kennedy_Space_Center]=0.0005940732289188326 \
    [Pavia_Centre]=0.0001236480816653455 \
    [Pavia_University]=0.00015871160049527858 \
    [Salinas]=0.00023614474992419862 \
    [SalinasA]=0.0003376670734565324 \
)

declare -A WEIGHT_DECAY=( \
    [Botswana]=0.00011150299441886956 \
    [Indian_Pines]=0.009021540956272492 \
    [Kennedy_Space_Center]=0.0008803995798938349 \
    [Pavia_Centre]=4.102919947676974e-07 \
    [Pavia_University]=0.0009538211768148025 \
    [Salinas]=0.0007723031252522047 \
    [SalinasA]=8.662401208300589e-08 \
)

declare -A DROPOUT=( \
    [Botswana]=0.25 \
    [Indian_Pines]=0.1 \
    [Kennedy_Space_Center]=0.25 \
    [Pavia_Centre]=0.45 \
    [Pavia_University]=0.2 \
    [Salinas]=0.15 \
    [SalinasA]=0.0 \
)

declare -A EMBED_DIM=( \
    [Botswana]=128 \
    [Indian_Pines]=128 \
    [Kennedy_Space_Center]=256 \
    [Pavia_Centre]=256 \
    [Pavia_University]=64 \
    [Salinas]=128 \
    [SalinasA]=256 \
)

declare -A GAT_HIDDEN_DIM=( \
    [Botswana]=64 \
    [Indian_Pines]=32 \
    [Kennedy_Space_Center]=64 \
    [Pavia_Centre]=64 \
    [Pavia_University]=32 \
    [Salinas]=32 \
    [SalinasA]=32 \
)

declare -A GAT_HEADS=( \
    [Botswana]=4 \
    [Indian_Pines]=2 \
    [Kennedy_Space_Center]=10 \
    [Pavia_Centre]=4 \
    [Pavia_University]=4 \
    [Salinas]=10 \
    [SalinasA]=4 \
)

declare -A GAT_DEPTH=( \
    [Botswana]=2 \
    [Indian_Pines]=8 \
    [Kennedy_Space_Center]=6 \
    [Pavia_Centre]=4 \
    [Pavia_University]=4 \
    [Salinas]=4 \
    [SalinasA]=8 \
)

declare -A TRANSFORMER_HEADS=( \
    [Botswana]=8 \
    [Indian_Pines]=8 \
    [Kennedy_Space_Center]=2 \
    [Pavia_Centre]=16 \
    [Pavia_University]=16 \
    [Salinas]=2 \
    [SalinasA]=16 \
)

declare -A TRANSFORMER_LAYERS=( \
    [Botswana]=9 \
    [Indian_Pines]=6 \
    [Kennedy_Space_Center]=4 \
    [Pavia_Centre]=3 \
    [Pavia_University]=10 \
    [Salinas]=2 \
    [SalinasA]=10 \
)

declare -A STRIDE=( \
    [Botswana]=3 \
    [Indian_Pines]=3 \
    [Kennedy_Space_Center]=8 \
    [Pavia_Centre]=4 \
    [Pavia_University]=4 \
    [Salinas]=4 \
    [SalinasA]=6 \
)

declare -A FUSION_MODE=( \
    [Botswana]=gate \
    [Indian_Pines]=gate \
    [Kennedy_Space_Center]=gate \
    [Pavia_Centre]=gate \
    [Pavia_University]=gate \
    [Salinas]=gate \
    [SalinasA]=gate \
)

# DATASETS=(Botswana Indian_Pines Kennedy_Space_Center Pavia_Centre Pavia_University Salinas SalinasA)

# If you want to sweep a single dataset, uncomment the line below
DATASETS=(Indian_Pines)

TRAIN_RATIO=0.05
VAL_RATIO=0.05
SEED=242
NUM_WORKERS=4
MODEL_TYPE="gast"

run_and_log() {
    local dataset=$1
    local param_name=$2
    local param_value=$3
    local log_file=$4
    local out_dir=$5

    echo ""
    echo "🚀 [$dataset] Training: $param_name = $param_value"
    python3 ../main.py --mode train \
        --dataset "$dataset" \
        --train_ratio "$TRAIN_RATIO" \
        --val_ratio "$VAL_RATIO" \
        --epochs "${EPOCHS[$dataset]}" \
        --early_stop "${EARLY_STOP[$dataset]}" \
        --batch_size "${BATCH_SIZE[$dataset]}" \
        --patch_size "${PATCH_SIZE[$dataset]}" \
        --stride "$STRIDE_VAL" \
        --lr "$LR_VAL" \
        --weight_decay "$WEIGHT_DECAY_VAL" \
        --dropout "$DROPOUT_VAL" \
        --embed_dim "$EMBED_DIM_VAL" \
        --gat_hidden_dim "$GAT_HIDDEN_DIM_VAL" \
        --gat_heads "$GAT_HEADS_VAL" \
        --gat_depth "$GAT_DEPTH_VAL" \
        --transformer_heads "$TRANSFORMER_HEADS_VAL" \
        --transformer_layers "$TRANSFORMER_LAYERS_VAL" \
        --fusion_mode "$FUSION_MODE_VAL" \
        --seed "$SEED" \
        --num_workers "$NUM_WORKERS" \
        --model_type "$MODEL_TYPE" \
        --output_dir "$out_dir"

    local hist_file="$out_dir/train_history_${dataset}_full_gast.json"
    if [[ -f "$hist_file" ]]; then
        local test_oa=$(jq '.test_OA' "$hist_file")
        echo "  📊 $param_name: $param_value | Test OA: $test_oa"
    else
        echo "❌ Missing history file: $hist_file"
        local test_oa="null"
    fi

    echo "  \"$param_value\": $test_oa" >> "$log_file"
}

for dataset in "${DATASETS[@]}"; do
    echo ""
    echo "==============================="
    echo "🔎 Starting parameter sweeps for $dataset"
    echo "==============================="

    # Set optimum values for this dataset
    LR_VAL=${LR[$dataset]}
    WEIGHT_DECAY_VAL=${WEIGHT_DECAY[$dataset]}
    DROPOUT_VAL=${DROPOUT[$dataset]}
    EMBED_DIM_VAL=${EMBED_DIM[$dataset]}
    GAT_HIDDEN_DIM_VAL=${GAT_HIDDEN_DIM[$dataset]}
    GAT_HEADS_VAL=${GAT_HEADS[$dataset]}
    GAT_DEPTH_VAL=${GAT_DEPTH[$dataset]}
    TRANSFORMER_HEADS_VAL=${TRANSFORMER_HEADS[$dataset]}
    TRANSFORMER_LAYERS_VAL=${TRANSFORMER_LAYERS[$dataset]}
    STRIDE_VAL=${STRIDE[$dataset]}
    FUSION_MODE_VAL=${FUSION_MODE[$dataset]}

    OUT_ROOT="../models/param_sweeps/${dataset}"
    mkdir -p "$OUT_ROOT"

    # Declare the parameters to sweep
    declare -A SWEEP_PARAMS=(
        [lr]="LR_LIST"
        [weight_decay]="WEIGHT_DECAY_LIST"
        [dropout]="DROPOUT_LIST"
        [embed_dim]="EMBED_DIM_LIST"
        [gat_hidden_dim]="GAT_HIDDEN_DIM_LIST"
        [gat_heads]="GAT_HEADS_LIST"
        [gat_depth]="GAT_DEPTH_LIST"
        [transformer_heads]="TRANSFORMER_HEADS_LIST"
        [transformer_layers]="TRANSFORMER_LAYERS_LIST"
        [batch_size]="BATCH_SIZE_LIST"
        [patch_size]="PATCH_SIZE_LIST"
        [stride]="STRIDE_LIST"
        [fusion_mode]="FUSION_MODE_LIST"
    )

    # Loop through each parameter to sweep
    for param in "${!SWEEP_PARAMS[@]}"; do
        echo ""
        echo "==============================="
        echo "🔎 [$dataset] Sweeping parameter: $param"
        echo "==============================="

        # Reset all params to optimum for this dataset
        LR_VAL=${LR[$dataset]}
        WEIGHT_DECAY_VAL=${WEIGHT_DECAY[$dataset]}
        DROPOUT_VAL=${DROPOUT[$dataset]}
        EMBED_DIM_VAL=${EMBED_DIM[$dataset]}
        GAT_HIDDEN_DIM_VAL=${GAT_HIDDEN_DIM[$dataset]}
        GAT_HEADS_VAL=${GAT_HEADS[$dataset]}
        GAT_DEPTH_VAL=${GAT_DEPTH[$dataset]}
        TRANSFORMER_HEADS_VAL=${TRANSFORMER_HEADS[$dataset]}
        TRANSFORMER_LAYERS_VAL=${TRANSFORMER_LAYERS[$dataset]}
        STRIDE_VAL=${STRIDE[$dataset]}
        FUSION_MODE_VAL=${FUSION_MODE[$dataset]}

        OUT_DIR="$OUT_ROOT/${param}_sweep"
        LOG_FILE="$OUT_DIR/results_${dataset}_${param}.json"
        mkdir -p "$OUT_DIR"
        echo "{" > "$LOG_FILE"

        param_list_name=${SWEEP_PARAMS[$param]}
        eval "values=(\"\${${param_list_name}[@]}\")"

        #
        for i in "${!values[@]}"; do
            value="${values[$i]}"
            OUT_PATH="$OUT_DIR/${param}_${value}"

            # Overwrite only the swept param
            case "$param" in
                lr) LR_VAL=$value ;;
                embed_dim) EMBED_DIM_VAL=$value ;;
                gat_hidden_dim) GAT_HIDDEN_DIM_VAL=$value ;;
                gat_heads) GAT_HEADS_VAL=$value ;;
                gat_depth) GAT_DEPTH_VAL=$value ;;
                dropout) DROPOUT_VAL=$value ;;
                transformer_heads) TRANSFORMER_HEADS_VAL=$value ;;
                transformer_layers) TRANSFORMER_LAYERS_VAL=$value ;;
                batch_size) BATCH_SIZE[$dataset]=$value ;;
                patch_size) PATCH_SIZE[$dataset]=$value ;;
                weight_decay) WEIGHT_DECAY_VAL=$value ;;
                stride) STRIDE_VAL=$value ;;
                fusion_mode) FUSION_MODE_VAL=$value ;;
            esac

            run_and_log "$dataset" "$param" "$value" "$LOG_FILE" "$OUT_PATH"

            if [ $i -lt $((${#values[@]} - 1)) ]; then
                echo "," >> "$LOG_FILE"
            fi
        done

        echo "" >> "$LOG_FILE"
        echo "}" >> "$LOG_FILE"
    done
done

echo ""
echo "✅ All parameter sweeps completed!"
echo "Results are saved in $OUT_ROOT"
