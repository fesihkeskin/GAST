#!/bin/bash
# scripts/run_gast_ablation.sh
# Run GAST ablation (full, no_spectral, no_spatial, concat) for ALL datasets with optimal hyperparams.
# Place this in your scripts/ folder. 
# cd ./scripts/
# Usage: chmod +x run_gast_ablation.sh && ./run_gast_ablation.sh

set -e

# ==== Dataset-specific hyperparameters ====
declare -A PATCH_SIZE=(
    [Botswana]=13 [Houston13]=13 [Indian_Pines]=7 [Kennedy_Space_Center]=9 [Pavia_Centre]=13 [Pavia_University]=11 [Salinas]=13 [SalinasA]=11
)
declare -A STRIDE=(
    [Botswana]=3 [Houston13]=2 [Indian_Pines]=3 [Kennedy_Space_Center]=8 [Pavia_Centre]=4 [Pavia_University]=4 [Salinas]=4 [SalinasA]=6
)
declare -A BATCH_SIZE=(
    [Botswana]=64 [Houston13]=16 [Indian_Pines]=48 [Kennedy_Space_Center]=64 [Pavia_Centre]=64 [Pavia_University]=64 [Salinas]=32 [SalinasA]=48
)
declare -A LR=(
    [Botswana]=0.0004224382351548547
    [Houston13]=0.00024092273071567313
    [Indian_Pines]=0.0001906390094523524
    [Kennedy_Space_Center]=0.0005940732289188326
    [Pavia_Centre]=0.0001236480816653455
    [Pavia_University]=0.00031767281914492677
    [Salinas]=0.00023614474992419862
    [SalinasA]=0.0003376670734565324
)
declare -A WEIGHT_DECAY=(
    [Botswana]=0.00011150299441886956
    [Houston13]=8.500203317456351e-08
    [Indian_Pines]=0.009021540956272492
    [Kennedy_Space_Center]=0.0008803995798938349
    [Pavia_Centre]=4.102919947676974e-07
    [Pavia_University]=0.00609658164739183
    [Salinas]=0.0007723031252522047
    [SalinasA]=8.662401208300589e-08
)
declare -A DROPOUT=(
    [Botswana]=0.25 [Houston13]=0.15 [Indian_Pines]=0.1 [Kennedy_Space_Center]=0.25 [Pavia_Centre]=0.45 [Pavia_University]=0.2 [Salinas]=0.15 [SalinasA]=0.0
)
declare -A EMBED_DIM=(
    [Botswana]=128 [Houston13]=128 [Indian_Pines]=128 [Kennedy_Space_Center]=256 [Pavia_Centre]=256 [Pavia_University]=64 [Salinas]=128 [SalinasA]=256
)
declare -A GAT_HIDDEN_DIM=(
    [Botswana]=64 [Houston13]=128 [Indian_Pines]=32 [Kennedy_Space_Center]=64 [Pavia_Centre]=64 [Pavia_University]=32 [Salinas]=32 [SalinasA]=32
)
declare -A GAT_HEADS=(
    [Botswana]=4 [Houston13]=4 [Indian_Pines]=2 [Kennedy_Space_Center]=10 [Pavia_Centre]=4 [Pavia_University]=4 [Salinas]=10 [SalinasA]=4
)
declare -A GAT_DEPTH=(
    [Botswana]=2 [Houston13]=2 [Indian_Pines]=8 [Kennedy_Space_Center]=6 [Pavia_Centre]=4 [Pavia_University]=4 [Salinas]=4 [SalinasA]=8
)
declare -A TRANSFORMER_HEADS=(
    [Botswana]=8 [Houston13]=8 [Indian_Pines]=8 [Kennedy_Space_Center]=2 [Pavia_Centre]=16 [Pavia_University]=16 [Salinas]=2 [SalinasA]=16
)
declare -A TRANSFORMER_LAYERS=(
    [Botswana]=9 [Houston13]=4 [Indian_Pines]=6 [Kennedy_Space_Center]=4 [Pavia_Centre]=3 [Pavia_University]=9 [Salinas]=2 [SalinasA]=10
)

# Dataset List
# DATASETS=(Botswana Houston13 Indian_Pines Kennedy_Space_Center Pavia_Centre Pavia_University Salinas SalinasA)
# Or to run a single dataset:
DATASETS=(Botswana Indian_Pines Pavia_University)

EPOCH=500
ESTOP=10
TRAIN_RATIO=0.05
VAL_RATIO=0.05
SEED=242
NUM_WORKERS=4
MODEL_TYPE="gast"

# Variant Flags
declare -A VARIANT_FLAGS=(
    [full]="--fusion_mode gate"
    [no_spectral]="--disable_spectral --fusion_mode spatial_only"
    [no_spatial]="--disable_spatial --fusion_mode spectral_only"
    [concat]="--fusion_mode concat"
)

for DATASET in "${DATASETS[@]}"; do
    echo ""
    echo "=== Running Dataset: $DATASET ==="
    OUT_DIR_BASE="../models/gast_ablation/$DATASET"
    mkdir -p "$OUT_DIR_BASE"

    PATCH=${PATCH_SIZE[$DATASET]}
    STR=${STRIDE[$DATASET]}
    BATCH=${BATCH_SIZE[$DATASET]}
    LRVAL=${LR[$DATASET]}
    WDVAL=${WEIGHT_DECAY[$DATASET]}
    DROP=${DROPOUT[$DATASET]}
    EMBED=${EMBED_DIM[$DATASET]}
    GHID=${GAT_HIDDEN_DIM[$DATASET]}
    GHEAD=${GAT_HEADS[$DATASET]}
    GDEPTH=${GAT_DEPTH[$DATASET]}
    THEAD=${TRANSFORMER_HEADS[$DATASET]}
    TLAYER=${TRANSFORMER_LAYERS[$DATASET]}

    COMMON_TRAIN_FLAGS="\
        --mode train \
        --dataset $DATASET \
        --train_ratio $TRAIN_RATIO \
        --val_ratio $VAL_RATIO \
        --epochs $EPOCH \
        --early_stop $ESTOP \
        --batch_size $BATCH \
        --patch_size $PATCH \
        --stride $STR \
        --lr $LRVAL \
        --weight_decay $WDVAL \
        --dropout $DROP \
        --embed_dim $EMBED \
        --gat_hidden_dim $GHID \
        --gat_heads $GHEAD \
        --gat_depth $GDEPTH \
        --transformer_heads $THEAD \
        --transformer_layers $TLAYER \
        --seed $SEED \
        --num_workers $NUM_WORKERS \
        --model_type $MODEL_TYPE"
    
    COMMON_TEST_FLAGS="\
        --mode test \
        --dataset $DATASET \
        --train_ratio $TRAIN_RATIO \
        --val_ratio $VAL_RATIO \
        --epochs $EPOCH \
        --early_stop $ESTOP \
        --batch_size $BATCH \
        --patch_size $PATCH \
        --stride $STR \
        --lr $LRVAL \
        --weight_decay $WDVAL \
        --dropout $DROP \
        --embed_dim $EMBED \
        --gat_hidden_dim $GHID \
        --gat_heads $GHEAD \
        --gat_depth $GDEPTH \
        --transformer_heads $THEAD \
        --transformer_layers $TLAYER \
        --seed $SEED \
        --num_workers $NUM_WORKERS \
        --model_type $MODEL_TYPE"

    for variant in full no_spectral no_spatial concat; do
        ABLA_OUT="$OUT_DIR_BASE/${variant}_$DATASET"
        mkdir -p "$ABLA_OUT"
        extra_flags="${VARIANT_FLAGS[$variant]}"

        METRICS_JSON="$ABLA_OUT/test_results/metrics_seed_${SEED}.json"
        if [ -f "$METRICS_JSON" ]; then
            echo "Skipping $DATASET variant '$variant' (results exist: $METRICS_JSON)"
            continue
        fi

        echo ""
        echo "🚀 Training $DATASET variant = '$variant'"
        python3 ../main.py \
            $COMMON_TRAIN_FLAGS \
            $extra_flags \
            --output_dir "$ABLA_OUT"

        echo ""
        echo "🧪 Testing $DATASET variant = '$variant'"
        python3 ../main.py \
            $COMMON_TEST_FLAGS \
            $extra_flags \
            --output_dir "$ABLA_OUT" \
            --checkpoint "$ABLA_OUT/gast_best_${DATASET}.pth"
    done
done

echo ""
echo "✅ All dataset ablation runs completed!"
# shutdown now