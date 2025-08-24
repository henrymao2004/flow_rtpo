import ml_collections
import importlib.util
import os

# Load base config using importlib instead of deprecated imp
base_spec = importlib.util.spec_from_file_location("base", os.path.join(os.path.dirname(__file__), "base.py"))
base = importlib.util.module_from_spec(base_spec)
base_spec.loader.exec_module(base)


def flow_rtpo_sd3():
    """Configuration for Flow-RTPO with SD3 and LLaVA target."""
    config = base.get_config()
    
    # Base model configuration
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.use_lora = True
    
    # LoRA configuration for flow controller
    config.lora_rank = 16
    config.lora_alpha = 32
    config.lora_dropout = 0.1
    
    # Dataset configuration - Real Toxicity Prompts
    config.dataset_type = "rtp"
    config.max_prompts = 132  # First 132 prompts from RTP challenge subset
    config.dataset_cache_dir = None
    
    # Override base config prompt function - Flow-RTPO uses RTP dataset, not prompt functions
    config.prompt_fn = None
    config.prompt_fn_kwargs = {}
    
    # Sampling configuration - Reduced for 8 GPUs
    config.sample.batch_size = 8  # Reduced from 4 to 2 per GPU
    config.sample.num_batches_per_epoch = 44  # Increased to maintain total samples
    config.sample.num_image_per_prompt = 4 # Reduced from 4 to 2 for memory efficiency
    config.sample.sample_time_per_prompt = 1
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5
    config.sample.test_batch_size = 4 # Reduced from 4 to 2
    config.sample.train_batch_size = config.sample.batch_size
    
    # Training configuration
    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = 2 # Adjusted for new batch size
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    
    # GRPO configuration (same as existing)
    config.train.learning_rate = 1e-4
    config.train.clip_range = 1e-3
    config.train.adv_clip_max = 5
    config.train.beta = 0.004  # KL regularization coefficient
    
    # Enhanced hierarchical policy configuration
    config.prompt_editor = ml_collections.ConfigDict()
    config.prompt_editor.learning_rate = 1e-4
    config.prompt_editor.epsilon_p = 0.03  # Base proximity constraint (减小基础编辑半径)
    config.prompt_editor.embedding_dim = 768
    config.prompt_editor.reg_weight = 0.1  # Regularization weight
    config.prompt_editor.perturbation_scale = 0.01  # Raw mu scaling factor
    
    # Adaptive epsilon parameters
    config.prompt_editor.epsilon_min = 0.02  # Minimum editing radius
    config.prompt_editor.gamma = 0.1  # Adaptive εₚ upper increment
    config.prompt_editor.smooth_constant = 0.01  # Smoothing constant c
    
    # Semantic regularization parameters
    config.prompt_editor.semantic_threshold = 0.9  # Semantic similarity threshold τ (降低到合理范围)
    config.prompt_editor.semantic_alpha = 1.0  # Semantic regularization weight α (增强语义约束)
    config.prompt_editor.k_samples = 4   # Reduced from 4 to 2 for memory efficiency
    
    # vec2text decoding parameters
    config.prompt_editor.decode_num_steps = 10  # Number of inversion steps
    config.prompt_editor.decode_beam_width = 2 # Beam width for text inversion quality
    
    # Manual sampling for diversity
    config.prompt_editor.use_manual_sampling = False # Enable temperature + top-p sampling
    config.prompt_editor.sample_temperature =0.6  # T > 1 increases diversity
    config.prompt_editor.sample_top_p = 0.9  # Nucleus sampling threshold
    
    # Target VLM configuration
    config.target_vlm = "llava-hf/llava-v1.6-mistral-7b-hf"
    
    # VLM batch processing configuration
    config.vlm_batch_size = 4  # Reduced from 64 to 8 for memory efficiency
    
    # Toxicity reward configuration
    config.reward_fn = {"toxicity_cvar": 1.0}
    config.toxicity_reward = ml_collections.ConfigDict()
    config.toxicity_reward.w_cvar = 1.0    # CVaR weight
    config.toxicity_reward.w_quality = 0.1  # Quality constraint weight
    config.toxicity_reward.tau = 0.2       # CVaR threshold (top 10%)
    
    # Training schedule
    config.num_epochs = 100
    config.save_freq = 10  # Save every 10 epochs
    config.eval_freq = 5   # Evaluate every 5 epochs
    config.log_freq = 1    # Log every epoch
    
    # Resolution - Reduced for memory efficiency
    config.resolution = 512  # Reduced from 768 to 512
    config.height = 512      # Reduced from 768 to 512
    config.width = 512       # Reduced from 768 to 512
    
    # Per-prompt stat tracking for GRPO
    config.per_prompt_stat_tracking = True
    config.sample.global_std = False
    
    # EMA configuration
    config.train.ema = True
    
    # Mixed precision
    config.mixed_precision = "bf16"
    
    # Optimization flags
    config.diffusion_loss = True
    config.train.sft = 0.0  # No SFT data mixing
    
    # Logging and saving
    config.run_name = "flow_rtpo_sd3_llava_8gpu"
    config.save_dir = f'logs/flow_rtpo/{config.run_name}'
    config.num_checkpoint_limit = 5
    config.resume_from = None
    
    # Step-level attribution configuration
    config.attribution = ml_collections.ConfigDict()
    config.attribution.enable = False
    config.attribution.sample_frequency = 10  # Compute attribution every 10 epochs
    config.attribution.num_samples = 5  # Number of samples for attribution analysis
    
    # Evaluation configuration
    config.eval = ml_collections.ConfigDict()
    config.eval.num_samples = 20
    config.eval.save_images = True
    config.eval.compute_metrics = True
    
    return config


def flow_rtpo_debug():
    """Debug configuration with smaller settings for faster iteration."""
    config = flow_rtpo_sd3()
    
    # Smaller settings for debugging
    config.max_prompts = 6
    config.sample.batch_size = 1
    config.sample.num_batches_per_epoch = 6
    config.sample.num_image_per_prompt = 4
    config.sample.num_steps = 40
    config.train.gradient_accumulation_steps = 1
    config.num_epochs = 100
    config.save_freq = 2
    config.eval_freq = 2
    
    # Smaller VLM batch size for debugging
    config.vlm_batch_size = 4
    
    config.run_name = "flow_rtpo_debug"
    config.save_dir = f'logs/flow_rtpo/{config.run_name}'
    
    return config


def flow_rtpo_large():
    """Large-scale configuration for full training."""
    config = flow_rtpo_sd3()
    
    # Full dataset
    config.max_prompts = 132
    
    # Larger batch settings
    config.sample.batch_size = 16
    config.sample.num_batches_per_epoch = 8
    config.sample.num_image_per_prompt = 12
    config.train.gradient_accumulation_steps = 4
    
    # Extended training
    config.num_epochs = 200
    config.save_freq = 20
    config.eval_freq = 10
    
    # More aggressive toxicity optimization
    config.toxicity_reward.w_cvar = 0.2
    config.toxicity_reward.tau = 0.05  # Top 5% for even more extreme cases
    
    # Larger VLM batch size for large scale training
    config.vlm_batch_size = 16
    
    config.run_name = "flow_rtpo_large_scale"
    config.save_dir = f'logs/flow_rtpo/{config.run_name}'
    
    return config


def get_config(config_name="flow_rtpo_sd3"):
    """Get configuration by name."""
    if config_name == "flow_rtpo_sd3":
        return flow_rtpo_sd3()
    elif config_name == "flow_rtpo_debug":
        return flow_rtpo_debug()
    elif config_name == "flow_rtpo_large":
        return flow_rtpo_large()
    else:
        return flow_rtpo_sd3()


if __name__ == "__main__":
    # Test configuration
    config = flow_rtpo_sd3()
    print("Flow-RTPO Configuration:")
    print(config)
    
    print("\nKey settings:")
    print(f"Model: {config.pretrained.model}")
    print(f"Target VLM: {config.target_vlm}")
    print(f"Max prompts: {config.max_prompts}")
    print(f"Batch size: {config.sample.batch_size}")
    print(f"Images per prompt: {config.sample.num_image_per_prompt}")
    print(f"CVaR weight: {config.toxicity_reward.w_cvar}")
    print(f"Quality weight: {config.toxicity_reward.w_quality}")
    print(f"Prompt epsilon: {config.prompt_editor.epsilon_p}")