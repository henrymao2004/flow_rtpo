#!/bin/bash

# Temporal Intervention Test Runner
# This script runs the temporal intervention experiment with a sample configuration

echo "🧪 Starting Temporal Intervention Experiment"
echo "============================================="

# Set up paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG_PATH="$PROJECT_ROOT/config/flow_rtpo.py"
PROMPTS_PATH="$PROJECT_ROOT/dataset/prompts.jsonl"
RESULTS_DIR="$PROJECT_ROOT/results/temporal_intervention_experiment"

# Create results directory
mkdir -p "$RESULTS_DIR"

echo "📁 Project root: $PROJECT_ROOT"
echo "⚙️ Config: $CONFIG_PATH"
echo "📝 Prompts: $PROMPTS_PATH"
echo "💾 Results: $RESULTS_DIR"

# Check if required files exist
if [ ! -f "$CONFIG_PATH" ]; then
    echo "❌ Config file not found: $CONFIG_PATH"
    exit 1
fi

if [ ! -f "$PROMPTS_PATH" ]; then
    echo "❌ Prompts file not found: $PROMPTS_PATH"
    exit 1
fi

echo ""
echo "🚀 Launching temporal intervention experiment..."
echo "   - Testing 5 prompts"
echo "   - Vanilla SD3 baseline"
echo "   - Early intervention (steps 5-10)"
echo "   - Late intervention (steps 28-38)"
echo "   - Mixed intervention patterns"
echo "   - Noise injection test"
echo ""

# Run the experiment
cd "$PROJECT_ROOT"
python scripts/test_temporal_intervention.py \
    --config_path "$CONFIG_PATH" \
    --prompts_file "$PROMPTS_PATH" \
    --num_prompts 5 \
    --save_dir "$RESULTS_DIR" \
    --device cuda

echo ""
echo "✅ Experiment completed!"
echo "📊 Check results in: $RESULTS_DIR"
echo "   - intervention_results.json (detailed results)"
echo "   - experiment_summary.json (summary statistics)"
echo "   - experiment_summary_plots.png (visualization)"
echo "   - prompt_XX_*.png (generated images)"
