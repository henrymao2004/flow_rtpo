# Flow-RTPO Held-Out Evaluation

This directory contains scripts for evaluating trained Flow-RTPO models on the held-out challenging subset of the Real Toxicity Prompts dataset.

## Files

- `evaluate_flow_rtpo_heldout.py` - Main evaluation script
- `run_heldout_evaluation.sh` - Bash wrapper for easy execution
- `evaluation_requirements.txt` - Python dependencies
- `EVALUATION_README.md` - This documentation

## Quick Start

### 1. Install Dependencies

```bash
pip install -r scripts/evaluation_requirements.txt
```

### 2. Set API Keys (Optional)

```bash
# For Perspective API (optional)
export PERSPECTIVE_API_KEY="your_google_cloud_api_key"
```

**Note**: DeepSeek API key is already embedded in the script for LLM-as-judge evaluation.

### 3. Run Evaluation

```bash
# Using the bash wrapper (recommended)
./scripts/run_heldout_evaluation.sh /path/to/checkpoint config/flow_rtpo.py

# Or directly with Python
python scripts/evaluate_flow_rtpo_heldout.py \
    --checkpoint_path /path/to/checkpoint \
    --config_path config/flow_rtpo.py \
    --max_prompts 100 \
    --save_dir ./results \
    --save_images
```

## Evaluation Metrics

### Toxicity Metrics

1. **Detoxify (6 dimensions)**:
   - Toxicity, Severe Toxicity, Obscene, Threat, Insult, Identity Attack
   - Plus: Any toxic (>0.5) rate and average score

2. **Perspective API (6 dimensions)**:
   - Identity Attack, Profanity, Severe Toxicity, Sexually Explicit, Threat, Toxicity
   - Plus: Any toxic (>0.5) rate and average score

### Alignment Metrics

1. **Intent Preservation**: SBERT cosine similarity between original and modified prompts
2. **Faithfulness**: CLIP score between generated image and modified prompt
3. **Toxic Continuation Alignment**: LLM judge (DeepSeek-V3) rating (1-10) of VLM toxic description alignment with modified prompt

## Checkpoint Structure

The evaluation script expects checkpoints saved by `train_flow_rtpo.py`:

```
checkpoint_dir/
├── transformer_lora/     # LoRA weights for flow controller
├── prompt_editor.pt      # Prompt editor state dict
└── training_state.pt     # Training configuration
```

## Output Files

Results are saved to the specified directory:

- `aggregated_metrics.json` - Summary statistics for all metrics
- `detailed_results.json` - Per-sample detailed results
- `evaluation_summary.txt` - Human-readable summary report
- `failed_samples.json` - Information about any failed evaluations
- `heldout_XXXX.png` - Generated images (if --save_images is used)

## Example Usage

```bash
# Evaluate best model checkpoint with 50 prompts
./scripts/run_heldout_evaluation.sh \
    ./checkpoints/best_model \
    config/flow_rtpo.py \
    50 \
    ./evaluation_results_best \
    cuda

# Evaluate specific checkpoint with all available prompts
python scripts/evaluate_flow_rtpo_heldout.py \
    --checkpoint_path ./checkpoints/checkpoint_1000 \
    --config_path config/flow_rtpo.py \
    --save_dir ./full_evaluation \
    --save_images \
    --device cuda
```

## API Configuration

### DeepSeek API
- **Model**: `deepseek-chat`
- **API Key**: Embedded in script (sk-03644cd3f1f64cac9c7df4ee1da2ea8d)
- **Usage**: LLM-as-judge for toxic continuation alignment scoring

### Perspective API
- **Optional**: Set `PERSPECTIVE_API_KEY` environment variable
- **Fallback**: Returns zero scores if API key not available
- **Usage**: Alternative toxicity classification

## Performance Notes

- **GPU Memory**: Evaluation requires ~8-12GB GPU memory
- **Time**: ~30-60 seconds per prompt (depending on generation steps)
- **Batch Processing**: Processes one prompt at a time to avoid memory issues
- **Error Handling**: Continues evaluation even if individual samples fail

## Troubleshooting

1. **CUDA Out of Memory**: Reduce `num_inference_steps` in config or use smaller batch sizes
2. **API Failures**: Check internet connection and API keys
3. **Missing Dependencies**: Run `pip install -r scripts/evaluation_requirements.txt`
4. **Checkpoint Not Found**: Verify checkpoint path and structure

## Results Interpretation

- **Higher toxicity scores** = More successful toxic image generation
- **Higher intent preservation** = Better semantic similarity between original/modified prompts  
- **Higher faithfulness** = Better image-prompt alignment
- **Higher toxic continuation alignment** = Better alignment between VLM description and modified prompt

The evaluation provides comprehensive assessment of Flow-RTPO's ability to generate toxic content while maintaining prompt fidelity and semantic coherence.