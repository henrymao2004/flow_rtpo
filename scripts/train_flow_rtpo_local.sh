#!/bin/bash

# Flow-RTPO Training Script for Local Models
# Multi-GPU training for Flow-RTPO with local checkpoints and datasets

# Configuration
CONFIG_NAME="flow_rtpo_sd3_local"  # Use local configuration
ACCELERATE_CONFIG="scripts/accelerate_configs/multi_gpu.yaml"

# Environment setup
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # Adjust based on your GPU setup
export WANDB_PROJECT="flow_rtpo_local"
export WANDB_MODE="online"

# Create output directory with timestamp
OUTPUT_DIR="logs/flow_rtpo/$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUTPUT_DIR

echo "Starting Flow-RTPO Multi-GPU Training with Local Models..."
echo "Configuration: $CONFIG_NAME"
echo "Output directory: $OUTPUT_DIR"
echo "Using local models and datasets"

# Check if local models exist
echo "Checking local models..."
if [ ! -d "models/local/stable-diffusion-3.5-medium" ]; then
    echo "ERROR: Local SD3 model not found at models/local/stable-diffusion-3.5-medium"
    echo "Please run scripts/setup_local_models.py first"
    exit 1
fi

if [ ! -d "models/local/llava-v1.6-mistral-7b-hf" ]; then
    echo "ERROR: Local LLaVA model not found at models/local/llava-v1.6-mistral-7b-hf"
    echo "Please run scripts/setup_local_models.py first"
    exit 1
fi

if [ ! -d "models/local/gtr-base-t5" ]; then
    echo "ERROR: Local GTR-T5 model not found at models/local/gtr-base-t5"
    echo "Please run scripts/setup_local_models.py first"
    exit 1
fi

if [ ! -d "models/local/all-MiniLM-L6-v2" ]; then
    echo "ERROR: Local SBERT model not found at models/local/all-MiniLM-L6-v2"
    echo "Please run scripts/setup_local_models.py first"
    exit 1
fi

if [ ! -d "dataset/local/real-toxicity-prompts" ]; then
    echo "ERROR: Local RTP dataset not found at dataset/local/real-toxicity-prompts"
    echo "Please run scripts/setup_local_models.py first"
    exit 1
fi

echo "All local models and datasets found!"

# Run training with accelerate
accelerate launch --config_file $ACCELERATE_CONFIG \
    --main_process_port 29500 \
    scripts/train_flow_rtpo.py \
    --config config/flow_rtpo_local.py:$CONFIG_NAME \
    --output_dir $OUTPUT_DIR \
    --logging_dir $OUTPUT_DIR/logs \
    --mixed_precision fp16 \
    --gradient_accumulation_steps 1 \
    --max_grad_norm 1.0 \
    --save_steps 1000 \
    --save_total_limit 5 \
    --learning_rate 1e-5 \
    --max_train_steps 10000 \
    --lr_scheduler_type cosine \
    --lr_warmup_steps 500 \
    --seed 42 \
    --dataloader_num_workers 4 \
    --preprocessing_num_workers 4 \
    --checkpointing_steps 1000 \
    --validation_steps 500 \
    --eval_steps 500 \
    --logging_steps 10 \
    --report_to wandb \
    --run_name "flow_rtpo_local_$(date +%Y%m%d_%H%M%S)"

echo "Training completed!"
echo "Results saved to: $OUTPUT_DIR" 