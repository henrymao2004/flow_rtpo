#!/bin/bash

# Toxic Continuation Evaluation Script
# Modified ART framework for prefix → image → VLM response → detoxify evaluation

# Model paths - UPDATE THESE PATHS
LLAVA_LORA_PATH="glli/ART_GuideModel"  # Path to the LLaVA LoRA model
LLAVA_MODEL_PATH="liuhaotian/llava-v1.6-mistral-7b"  # Base LLaVA model

LLAMA_LORA_PATH="glli/ART_WriterModel"  # Path to the LLaMA LoRA model  
LLAMA_MODEL_PATH="meta-llama/Llama-2-7b-hf"  # Base LLaMA model

# Experiment configuration
SD_VERSION="3.5"  # Stable Diffusion version: 1.5, 2.1, xl, or 3.5
RTP_START_IDX=180  # Start index for RTP dataset (skip first 180)
MAX_SAMPLES=1000     # Number of RTP samples to evaluate
MAX_ROUNDS=10      # Maximum rounds per sample
BATCH_SIZE=3       # Batch size for processing samples
SEED=42           # Random seed for reproducibility

# Output configuration
EXP_FOLDER="toxic_continuation_results_$(date +%Y%m%d_%H%M%S)"

echo "Starting Toxic Continuation Evaluation..."
echo "========================================"
echo "Experiment folder: $EXP_FOLDER"
echo "RTP start index: $RTP_START_IDX"
echo "Max samples: $MAX_SAMPLES"
echo "Max rounds per sample: $MAX_ROUNDS"
echo "Batch size: $BATCH_SIZE"
echo "SD version: $SD_VERSION"
echo "========================================"

# Run the evaluation
python toxic_continuation_agent.py \
    --llama_model_path "$LLAMA_MODEL_PATH" \
    --llama_lora_path "$LLAMA_LORA_PATH" \
    --llava_model_path "$LLAVA_MODEL_PATH" \
    --llava_lora_path "$LLAVA_LORA_PATH" \
    --sd_version "$SD_VERSION" \
    --rtp_start_idx $RTP_START_IDX \
    --max_samples $MAX_SAMPLES \
    --max_rounds $MAX_ROUNDS \
    --batch_size $BATCH_SIZE \
    --seed $SEED \
    --exp_folder "$EXP_FOLDER"

echo ""
echo "Evaluation completed!"
echo "Results saved to: $EXP_FOLDER"
echo ""
echo "Key files generated:"
echo "  - final_results.json: Complete evaluation results"
echo "  - sample_*_round_*.png: Generated images"
echo "  - intermediate_results_*.json: Checkpoint files"
echo ""
echo "To analyze results, check the statistics section in final_results.json"