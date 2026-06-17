#!/usr/bin/env bash
set -euo pipefail    # fail loudly on any error — required by submission guide

# Accept arguments with sensible defaults
DATA_DIR="${1:-./data}"
MODEL_PATH="${2:-./pickle/model.pkl}"
OUTPUT_PATH="${3:-./output/predictions.csv}"

echo "=== AIgnition 3.0 Revenue Forecast Engine ==="
echo "Data dir:    $DATA_DIR"
echo "Model path:  $MODEL_PATH"
echo "Output path: $OUTPUT_PATH"

# Create output directory
mkdir -p "$(dirname "$OUTPUT_PATH")"

# Step 1: Generate features from raw CSVs
echo "[1/3] Generating features from $DATA_DIR..."
python src/generate_features.py \
    --data-dir "$DATA_DIR" \
    --out features.pkl

# Step 2: Retrain model on new data (P1.1)
echo "[2/3] Retraining model on incoming data..."
python src/train_model.py \
    --data-dir "$DATA_DIR"

# Step 3: Load model and generate predictions
echo "[3/3] Generating probabilistic forecasts..."
python src/predict.py \
    --features features.pkl \
    --model    "$MODEL_PATH" \
    --output   "$OUTPUT_PATH"

# Copy pre-generated AI insights alongside predictions
cp output/insights.json "$(dirname "$OUTPUT_PATH")/insights.json" 2>/dev/null || true

echo "Done. Predictions written to $OUTPUT_PATH"
echo "Rows in output: $(wc -l < "$OUTPUT_PATH")"
