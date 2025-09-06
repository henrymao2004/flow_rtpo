#!/bin/bash

# Flow-RTPO Held-Out Evaluation Runner
# Usage: ./run_heldout_evaluation.sh [checkpoint_path] [config_path] [optional_args]

set -e  # Exit on any error

# Default values
CHECKPOINT_PATH="${1:-./checkpoints/best_model}"
CONFIG_PATH="${2:-config/flow_rtpo.py}"
MAX_PROMPTS="${3:-100}"
SAVE_DIR="${4:-./heldout_evaluation_results}"
DEVICE="${5:-cuda}"

# Check if checkpoint exists
if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint directory not found: $CHECKPOINT_PATH"
    echo "Usage: $0 <checkpoint_path> <config_path> [max_prompts] [save_dir] [device]"
    exit 1
fi

# Check if config exists  
if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Config file not found: $CONFIG_PATH"
    echo "Usage: $0 <checkpoint_path> <config_path> [max_prompts] [save_dir] [device]"
    exit 1
fi

echo "========================================"
echo "Flow-RTPO Held-Out Evaluation"
echo "========================================"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Config: $CONFIG_PATH"
echo "Max Prompts: $MAX_PROMPTS"
echo "Save Directory: $SAVE_DIR"
echo "Device: $DEVICE"
echo "========================================"

# Create save directory
mkdir -p "$SAVE_DIR"

# Check for Perspective API key
if [ -z "$PERSPECTIVE_API_KEY" ]; then
    echo "Warning: PERSPECTIVE_API_KEY environment variable not set."
    echo "Perspective API metrics will return zero scores."
    echo "To enable Perspective API, set: export PERSPECTIVE_API_KEY=your_key"
fi

# Check dependencies
echo "Checking dependencies..."
if ! python -c "import openai" 2>/dev/null; then
    echo "Installing OpenAI package for DeepSeek API..."
    pip install openai>=1.0.0
fi

if ! python -c "import detoxify" 2>/dev/null; then
    echo "Installing Detoxify package..."
    pip install detoxify
fi

if ! python -c "import sentence_transformers" 2>/dev/null; then
    echo "Installing Sentence Transformers..."
    pip install sentence-transformers
fi

# Run evaluation
echo "Starting evaluation..."
python scripts/evaluate_flow_rtpo_heldout.py \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --config_path "$CONFIG_PATH" \
    --max_prompts "$MAX_PROMPTS" \
    --save_dir "$SAVE_DIR" \
    --device "$DEVICE" \
    --save_images \
    --perspective_api_key "$PERSPECTIVE_API_KEY"

echo "========================================"
echo "Evaluation completed!"
echo "Results saved to: $SAVE_DIR"
echo "========================================"

# Show summary if available
if [ -f "$SAVE_DIR/evaluation_summary.txt" ]; then
    echo ""
    echo "EVALUATION SUMMARY:"
    echo "==================="
    cat "$SAVE_DIR/evaluation_summary.txt"
fi