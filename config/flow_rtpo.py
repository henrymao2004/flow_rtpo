import ml_collections
import importlib.util
import os

# Load base config using importlib instead of deprecated imp
base_path = os.path.join(os.path.dirname(__file__), "base.py")
spec = importlib.util.spec_from_file_location("base", base_path)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


def flow_rtpo_sd3():
    """Configuration for Flow-RTPO with SD3 and LLaVA target."""
    config = base.get_config()
    
    # Base model configuration
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.use_lora = True
    
    # Model loading configuration - default to HuggingFace
    config.model_loading = ml_collections.ConfigDict()
    config.model_loading.use_local = False  # Default to HuggingFace
    config.model_loading.local_base_path = ""
    
    # Local model paths (when use_local=True) - absolute paths
    config.model_loading.local_models = ml_collections.ConfigDict()
    config.model_loading.local_models.sd3 = "/mnt/data/group/zhaoliangjie/ICLR-work/stable-diffusion-3.5-medium"
    config.model_loading.local_models.llava = "/mnt/data/group/zhaoliangjie/ICLR-work/llava-v1.6-mistral-7b-hf"
    config.model_loading.local_models.clip = "/mnt/data/group/zhaoliangjie/ICLR-work/clip-vit-large-patch14-336"
    config.model_loading.local_models.sbert = "/mnt/data/group/zhaoliangjie/ICLR-work/all-MiniLM-L6-v2"
    config.model_loading.local_models.gtr = "/mnt/data/group/zhaoliangjie/ICLR-work/gtr-t5-base"
    config.model_loading.local_models.vec2text = "/mnt/data/group/zhaoliangjie/ICLR-work/gtr-base"
    config.model_loading.local_models.detoxify = "/mnt/data/group/zhaoliangjie/ICLR-work/detoxify-original"
    
    # HuggingFace model names (when use_local=False)
    config.model_loading.hf_models = ml_collections.ConfigDict()
    config.model_loading.hf_models.sd3 = "stabilityai/stable-diffusion-3.5-medium"
    config.model_loading.hf_models.llava = "llava-hf/llava-v1.6-mistral-7b-hf"
    config.model_loading.hf_models.clip = "openai/clip-vit-large-patch14-336"
    config.model_loading.hf_models.sbert = "sentence-transformers/all-MiniLM-L6-v2"
    config.model_loading.hf_models.gtr = "sentence-transformers/gtr-t5-base"
    config.model_loading.hf_models.vec2text = "gtr-base"
    config.model_loading.hf_models.detoxify = "original"
    
    # Dataset loading configuration
    config.dataset_loading = ml_collections.ConfigDict()
    config.dataset_loading.use_local = False  # Default to HuggingFace
    config.dataset_loading.local_base_path = "/mnt/data/group/zhaoliangjie/ICLR-work/"
    config.dataset_loading.local_datasets = ml_collections.ConfigDict()
    config.dataset_loading.local_datasets.rtp = "/mnt/data/group/zhaoliangjie/ICLR-work/real-toxicity-prompts"
    config.dataset_loading.hf_datasets = ml_collections.ConfigDict()
    config.dataset_loading.hf_datasets.rtp = "allenai/real-toxicity-prompts"
    
    # Save directory configuration - default to local
    config.save_loading = ml_collections.ConfigDict()
    config.save_loading.use_local = True  # Default to local saving
    config.save_loading.local_base_path = "/mnt/data/group/zhaoliangjie/ICLR-work/logs/"
    config.save_loading.default_base_path = "./logs/"  # Default local path when not using remote storage
    
    # LoRA configuration for flow controller
    config.lora_rank = 8
    config.lora_alpha = 16
    config.lora_dropout = 0.1
    
    # Dataset configuration - Real Toxicity Prompts
    config.dataset_type = "rtp"
    config.max_prompts = 180  # First 132 prompts from RTP challenge subset
    config.dataset_cache_dir = None
    
    # Override base config prompt function - Flow-RTPO uses RTP dataset, not prompt functions
    config.prompt_fn = None
    config.prompt_fn_kwargs = {}
    
    # Sampling configuration
    config.sample.batch_size = 2
    config.sample.num_batches_per_epoch = 8  # For 8 GPUs: 132 prompts / (8 GPUs * 2 batch_size / 4 images_per_prompt) = 132 / 4 = 33, but we use 8 for stability
    config.sample.num_image_per_prompt = 4  # Multiple samples per prompt for ranking
    config.sample.sample_time_per_prompt = 1
    config.sample.num_steps = 40
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5
    config.sample.test_batch_size = 6
    config.sample.train_batch_size = config.sample.batch_size
    
    # Test split configuration
    config.test_ratio = 0.2  # 20% of prompts for testing
    config.distributed_eval = False  # Distribute test set across GPUs to reduce computation
    
    # Training configuration
    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch // 2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.cfg = True  # Enable CFG (like SD3)
    
    # GRPO configuration (same as existing)
    config.train.learning_rate = 5e-6
    config.train.clip_range = 1e-4
    config.train.adv_clip_max = 5
    config.train.beta = 0.04  # KL regularization coefficient
    
    # Default sampling strategy configuration (set early)
    config.use_grpo_sampling = True # Default to Flow-RTPO sampling
    
    # Enhanced hierarchical policy configuration
    config.prompt_editor = ml_collections.ConfigDict()
    config.prompt_editor.learning_rate = 5e-6
    config.prompt_editor.epsilon_p = 0.02  # Base proximity constraint (减小基础编辑半径)
    config.prompt_editor.embedding_dim = 768
    config.prompt_editor.reg_weight = 0.2  # Regularization weight
    config.prompt_editor.perturbation_scale = 0.01  # Raw mu scaling factor
    
    # Adaptive epsilon parameters
    config.prompt_editor.epsilon_min = 0.04  # Minimum editing radius
    config.prompt_editor.gamma = 0.1  # Adaptive εₚ upper increment
    config.prompt_editor.smooth_constant = 0.01  # Smoothing constant c
    
    # Semantic regularization parameters
    config.prompt_editor.semantic_threshold = 0.8  # Semantic similarity threshold τ (降低到合理范围)
    config.prompt_editor.semantic_alpha = 1.0  # Semantic regularization weight α (增强语义约束)
    config.prompt_editor.k_samples = 2  # Number of editing variations per prompt for GRPO comparison
    
    # vec2text decoding parameters
    config.prompt_editor.decode_num_steps = 20  # Number of inversion steps
    config.prompt_editor.decode_beam_width = 4 # Beam width for text inversion quality
    
    # Manual sampling for diversity
    config.prompt_editor.use_manual_sampling = False # Enable temperature + top-p sampling
    config.prompt_editor.sample_temperature =0.6  # T > 1 increases diversity
    config.prompt_editor.sample_top_p = 0.9  # Nucleus sampling threshold
    
    # Target VLM configuration
    config.target_vlm = "llava-hf/llava-v1.6-mistral-7b-hf"
    
    # Toxicity reward configuration
    config.reward_fn = {"toxicity_cvar": 1.0}
    config.toxicity_reward = ml_collections.ConfigDict()
    config.toxicity_reward.w_cvar = 0    # CVaR weight
    config.toxicity_reward.w_quality = 0.2 # Quality constraint weight
    config.toxicity_reward.tau = 0.1       # CVaR threshold (top 10%)
    
    # Training schedule
    config.num_epochs = 100
    config.save_freq = 5  # Save every 10 epochs
    config.eval_freq = 1   # Evaluate every 5 epochs
    config.log_freq = 1    # Log every epoch
    
    # Evaluation configuration
    config.distributed_eval = False  # If False, only main process evaluates test set
    
    # Resolution
    config.resolution = 768
    config.height = 768
    config.width = 768
    
    # Per-prompt stat tracking for GRPO
    config.per_prompt_stat_tracking = True
    config.sample.global_std = False
    
    # EMA configuration
    config.train.ema = True
    
    # Mixed precision
    config.mixed_precision = "fp16"
    
    # Optimization flags
    config.diffusion_loss = True
    config.train.sft = 0.0  # No SFT data mixing
    
    # Logging and saving
    config.run_name = "flow_rtpo_sd3_llava"

    # Set save directory based on configuration
    if config.save_loading.use_local:
        config.save_dir = f'{config.save_loading.local_base_path}flow_rtpo/{config.run_name}'
    else:
        config.save_dir = f'{config.save_loading.default_base_path}flow_rtpo/{config.run_name}'
    
    config.num_checkpoint_limit = 100
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
    
    # Debug mode: Use local loading for faster iteration
    config.model_loading.use_local =False
    config.dataset_loading.use_local = False
    
    # Enable GRPO sampling for consistency with main training
    config.use_grpo_sampling = True
    
    # Smaller settings for debugging
    config.max_prompts = 8  # Increase to ensure enough prompts for 2 GPUs 
    config.sample.batch_size = 2 # Reduce to ensure balanced distribution
    config.sample.num_image_per_prompt = 1  # Keep multiple samples for GRPO grouping
    config.sample.num_steps = 20
    config.num_epochs = 10
    config.save_freq = 2
    config.eval_freq = 1
    
    # Evaluation configuration
    config.distributed_eval = False  # Debug mode: only main process evaluates for speed
    
    # Debug-specific prompt editor settings (fewer variations for faster debug)
    config.prompt_editor.k_samples = 2  # Reduced for debug, but still allow GRPO comparison
    
    # GRPO sampling configuration for debug
    gpu_number = 2 # Assume debug runs on fewer GPUs
    config.sample.train_batch_size = config.sample.batch_size
    config.sample.grpo_k = config.sample.num_image_per_prompt * config.prompt_editor.k_samples  # K-repeat factor
    config.sample.grpo_num_batches = int(8/(gpu_number*config.sample.train_batch_size/config.sample.grpo_k))  # Updated for 8 prompts
    config.sample.num_batches_per_epoch = max(1, config.sample.grpo_num_batches)
    
    # Gradient accumulation for debug
    config.train.gradient_accumulation_steps = max(1, config.sample.num_batches_per_epoch // 2)
    config.train.batch_size = config.sample.batch_size
    config.train.cfg = True  # Enable CFG (like SD3)
    
    config.run_name = "flow_rtpo_debug"
    
    # Set save directory based on configuration
    if config.save_loading.use_local:
        config.save_dir = f'{config.save_loading.local_base_path}flow_rtpo/{config.run_name}'
    else:
        config.save_dir = f'{config.save_loading.default_base_path}flow_rtpo/{config.run_name}'
    
    return config


def flow_rtpo_large():
    """Large-scale configuration for full training."""
    config = flow_rtpo_sd3()
    
    # Large mode: Use local loading for better performance
    config.model_loading.use_local = True
    config.dataset_loading.use_local = True
    
    # Full dataset
    config.max_prompts = 180
    
    # Sampling strategy configuration
    config.use_grpo_sampling = True  # Flag to switch between sampling modes - GRPO MODE ENABLED
    
    # Keep batch size as requested but increase batches for GRPO grouping
    config.sample.batch_size = 8 # Changed to 2 as requested
    config.sample.num_image_per_prompt = 8 # Multiple samples per prompt for ranking
    
    # GRPO sampling configuration for large scale
    gpu_number = 32
    config.sample.train_batch_size = config.sample.batch_size  # For GRPO compatibility
    config.sample.grpo_k = config.sample.num_image_per_prompt * config.prompt_editor.k_samples  # Total K-repeat
    
    # Calculate batches based on GRPO grouping requirements
    # Use full prompt set but optimize batch distribution
    config.sample.grpo_num_batches = int(32/(gpu_number*config.sample.train_batch_size/config.sample.grpo_k))
    config.sample.num_batches_per_epoch = max(1, config.sample.grpo_num_batches) 
    
    # Set gradient accumulation steps to half of num_batches_per_epoch
    # This ensures gradients are updated twice per epoch for stability
    # Use max(1, ...) to ensure at least 1 gradient accumulation step
    config.train.gradient_accumulation_steps = max(1, config.sample.num_batches_per_epoch // 2)
    config.train.batch_size = config.sample.batch_size
    config.train.cfg = True  # Enable CFG (like SD3)
    # Extended training
    config.num_epochs = 10000
    config.save_freq = 3
    config.eval_freq = 1
    
    # Evaluation configuration  
    config.distributed_eval = False  # Large scale: only main process evaluates to save GPU resources
    
    # More aggressive toxicity optimization
    config.toxicity_reward.w_cvar = 0
    config.toxicity_reward.tau = 0.2  
    
    config.run_name = "flow_rtpo_large"
    
    # Set save directory based on configuration
    if config.save_loading.use_local:
        config.save_dir = f'{config.save_loading.local_base_path}flow_rtpo/{config.run_name}'
    else:
        config.save_dir = f'{config.save_loading.default_base_path}flow_rtpo/{config.run_name}'
    
    return config


def flow_rtpo_memory_optimized():
    """Memory-optimized configuration to prevent CUDA OOM errors."""
    config = flow_rtpo_sd3()
    
    # Memory optimized mode: Use HuggingFace loading for flexibility
    config.model_loading.use_local = False
    config.dataset_loading.use_local = False
    
    # Reduce batch sizes to minimize memory usage
    config.sample.batch_size = 1  # Reduce from 2 to 1
    config.sample.num_batches_per_epoch = 44  # Increase to maintain same total samples
    config.sample.num_image_per_prompt = 2  # Reduce from 4 to 2
    config.sample.test_batch_size = 1  # Reduce from 4 to 1
    config.sample.train_batch_size = config.sample.batch_size
    
    # Reduce resolution to save memory
    config.resolution = 512  # Reduce from 768 to 512
    config.height = 512
    config.width = 512
    
    # Reduce sampling steps
    config.sample.num_steps = 8  # Reduce from 10 to 8
    config.sample.eval_num_steps = 40  # Reduce from 40 to 20
    
    # Enable gradient checkpointing to save memory
    config.gradient_checkpointing = True
    
    # Reduce LoRA rank to save memory
    config.lora_rank = 8 # Reduce from 16 to 8
    config.lora_alpha = 16  # Reduce from 32 to 16
    
    # Reduce prompt editor parameters
    config.prompt_editor.embedding_dim = 512  # Reduce from 768 to 512
    config.prompt_editor.decode_num_steps = 10  # Reduce from 20 to 10
    config.prompt_editor.decode_beam_width = 2  # Reduce from 4 to 2
    config.prompt_editor.k_samples = 2  # Reduce from 4 to 2
    
    # Memory management settings
    config.mixed_precision = "fp16"  # Use fp16 instead of bf16 for better memory efficiency
    config.allow_tf32 = False  # Disable tf32 to save memory
    
    # Reduce dataset size for testing
    config.max_prompts = 10  # Reduce for testing
    
    # Training adjustments
    config.train.gradient_accumulation_steps = 22  # Adjust for new batch size
    config.train.num_inner_epochs = 1
    config.train.cfg = True  # Enable CFG (like SD3)
    
    # Evaluation settings
    config.eval.num_samples = 10  # Reduce from 20 to 10
    
    config.run_name = "flow_rtpo_memory_optimized"
    
    # Set save directory based on configuration
    if config.save_loading.use_local:
        config.save_dir = f'{config.save_loading.local_base_path}flow_rtpo/{config.run_name}'
    else:
        config.save_dir = f'{config.save_loading.default_base_path}flow_rtpo/{config.run_name}'
    
    return config


def get_config(config_name="flow_rtpo_sd3"):
    """Get configuration by name."""
    if config_name == "flow_rtpo_sd3":
        return flow_rtpo_sd3()
    elif config_name == "flow_rtpo_debug":
        return flow_rtpo_debug()
    elif config_name == "flow_rtpo_large":
        return flow_rtpo_large()
    elif config_name == "flow_rtpo_memory_optimized":
        return flow_rtpo_memory_optimized()
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
    print(f"Use local models: {config.model_loading.use_local}")
    print(f"Use local datasets: {config.dataset_loading.use_local}")
    print(f"Use local save directory: {config.save_loading.use_local}")
    print(f"Save directory: {config.save_dir}")