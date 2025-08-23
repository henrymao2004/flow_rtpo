from collections import defaultdict
import contextlib
import os
import datetime
from concurrent import futures
import time
import json
import hashlib
from absl import app, flags
from accelerate import Accelerator
from ml_collections import config_flags
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
from diffusers import StableDiffusion3Pipeline
from diffusers.utils.torch_utils import is_compiled_module
import numpy as np
import torch
import swanlab
from functools import partial
import tqdm
import tempfile
from PIL import Image
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict, PeftModel
import random
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.multiprocessing import Process, Queue, Event
import threading
import queue

# Set multiprocessing start method to 'spawn' for CUDA compatibility
mp.set_start_method('spawn', force=True)

# Flow-GRPO imports
from flow_grpo.stat_tracking import PerPromptStatTracker
from flow_grpo.diffusers_patch.sd3_pipeline_with_logprob import pipeline_with_logprob
from flow_grpo.diffusers_patch.sd3_sde_with_logprob import sde_step_with_logprob
from flow_grpo.diffusers_patch.train_dreambooth_lora_sd3 import encode_prompt
from flow_grpo.ema import EMAModuleWrapper

# Flow-RTPO specific imports
from flow_grpo.prompt_editor import PromptEditorPolicy
from flow_grpo.toxicity_rewards import toxicity_reward_function
from flow_grpo.rtp_dataset import create_rtp_dataset_and_loader

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/flow_rtpo.py", "Training configuration.")

logger = get_logger(__name__)


class RewardComputeWorker:
    """Dedicated worker for reward computation on separate GPUs."""
    
    def __init__(self, worker_id, gpu_id, reward_fn_config, reward_queue, stop_event):
        self.worker_id = worker_id
        self.gpu_id = gpu_id
        self.reward_fn_config = reward_fn_config
        self.reward_queue = reward_queue
        self.stop_event = stop_event
        self.device = torch.device(f"cuda:{gpu_id}")
        
    def run(self):
        """Main worker loop for reward computation."""
        torch.cuda.set_device(self.gpu_id)
        print(f"Reward worker {self.worker_id} started on GPU {self.gpu_id}")
        
        # Initialize reward function on this GPU
        reward_fn = toxicity_reward_function(
            device=self.device,
            vlm_model=self.reward_fn_config["vlm_model"],
            w_cvar=self.reward_fn_config["w_cvar"],
            w_quality=self.reward_fn_config["w_quality"]
        )
        
        while not self.stop_event.is_set():
            try:
                # Get task from queue with timeout
                task = self.reward_queue.get(timeout=1.0)
                if task is None:  # Poison pill
                    break
                    
                batch_id, images, prompts, metadata = task
                
                # Compute rewards
                rewards, reward_metadata = reward_fn(images, prompts, metadata)
                
                # Put results back
                self.reward_queue.put((batch_id, rewards, reward_metadata))
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Reward worker {self.worker_id} error: {e}")
                # Put error result with a default batch_id if not available
                error_batch_id = batch_id if 'batch_id' in locals() else f"error_{self.worker_id}"
                self.reward_queue.put((error_batch_id, None, {"error": str(e)}))
        
        print(f"Reward worker {self.worker_id} stopped")


def start_reward_workers(num_reward_gpus, reward_fn_config, result_queue, stop_event):
    """Start dedicated reward computation workers."""
    workers = []
    for i in range(num_reward_gpus):
        gpu_id = 6 + i  # Use GPUs 6, 7 for reward computation
        worker = RewardComputeWorker(i, gpu_id, reward_fn_config, result_queue, stop_event)
        process = Process(target=worker.run)
        process.start()
        workers.append(process)
    return workers


def save_ckpt(save_dir, transformer, prompt_editor, global_step, accelerator, ema, transformer_trainable_parameters, config):
    """Save checkpoint including both flow controller and prompt editor."""
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        transformer_lora = accelerator.unwrap_model(transformer, keep_fp32_wrapper=True)
        prompt_editor_unwrapped = accelerator.unwrap_model(prompt_editor)
        
        pipeline_save_dir = os.path.join(save_dir, f"checkpoint_{global_step}")
        os.makedirs(pipeline_save_dir, exist_ok=True)
        
        # Copy EMA weights to model before saving, store original weights temporarily
        if config.train.ema and ema is not None:
            ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)
        
        # Save LoRA weights
        transformer_lora.save_pretrained(os.path.join(pipeline_save_dir, "transformer_lora"))
        
        # Save prompt editor
        torch.save(prompt_editor_unwrapped.state_dict(), 
                  os.path.join(pipeline_save_dir, "prompt_editor.pt"))
        
        # Save training state
        training_state = {
            "global_step": global_step,
            "config": config.to_dict()
        }
        torch.save(training_state, os.path.join(pipeline_save_dir, "training_state.pt"))
        
        # Restore original weights after saving
        if config.train.ema and ema is not None:
            ema.copy_temp_to(transformer_trainable_parameters)


def compute_log_prob(transformer, pipeline, sample, timestep_idx, embeds, pooled_embeds, config):
    """Compute log probability for a single timestep using the flow controller."""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        # scalar timestep with batch dim -> [1]
        timesteps = sample["timesteps"][timestep_idx].view(1)

        # current latent is [C, H, W]; add batch -> [1, C, H, W]
        latents = sample["latents"][timestep_idx].unsqueeze(0)

        # DEBUG (optional)
        # print(f"[DEBUG] latents[t={timestep_idx}] shape (batched):", latents.shape)  # [1, C, H, W]
        # print(f"[DEBUG] timesteps shape:", timesteps.shape)  # [1]

        noise_pred = transformer(
            hidden_states=latents,
            timestep=timesteps,
            encoder_hidden_states=embeds,
            pooled_projections=pooled_embeds,
            return_dict=False,
        )[0]

        # previous latent (if any) should also be [1, C, H, W]
        prev_latent = sample["latents"][timestep_idx - 1].unsqueeze(0) if timestep_idx > 0 else None

        prev_sample, log_prob, prev_sample_mean, std_dev_t = sde_step_with_logprob(
            pipeline.scheduler,
            noise_pred,
            timesteps,
            latents,
            noise_level=config.sample.get("noise_level", 0.7),
            prev_sample=prev_latent,
        )
        return prev_sample, log_prob, prev_sample_mean, std_dev_t


def sample_batch(pipeline, prompt_editor, prompts, config, accelerator, epoch=0, batch_idx=0, reward_variance=None, reward_queue=None):
    """Sample a batch of images using hierarchical policies with enhanced features and async reward computation."""
    batch_size = len(prompts)
    k_samples = config.prompt_editor.get('k_samples', 4)  # k samples per prompt for GRPO
    
    print(f"[DEBUG] Starting sample_batch with {batch_size} prompts, k={k_samples} samples per prompt")
    print(f"[DEBUG] config.sample.num_image_per_prompt = {config.sample.num_image_per_prompt}")
    print(f"[DEBUG] Expected total prompt modifications = {batch_size} * {k_samples} = {batch_size * k_samples}")
    print(f"[DEBUG] Expected total images = {batch_size * k_samples} * {config.sample.num_image_per_prompt} = {batch_size * k_samples * config.sample.num_image_per_prompt}")
    
    # High-level: Generate k prompt modifications per original prompt for GRPO
    original_prompts = prompts.copy()
    
    # Expand prompts: each original prompt gets k modifications
    prompts_expanded = []
    prompt_to_group_mapping = {}  # Track which expanded prompt belongs to which original
    
    for prompt_idx, original_prompt in enumerate(original_prompts):
        for k_idx in range(k_samples):
            prompts_expanded.append(original_prompt)
            expanded_idx = len(prompts_expanded) - 1
            prompt_to_group_mapping[expanded_idx] = {
                'group_key': f"prompt_{prompt_idx}",  # Group key for GRPO
                'original_prompt': original_prompt,
                'original_idx': prompt_idx,
                'k_idx': k_idx
            }
    
    print(f"[DEBUG] Expanded to {len(prompts_expanded)} prompt modifications")
    print(f"[DEBUG] Starting prompt editor with reward_variance={reward_variance}...")
    
    # Force synchronization and use unwrapped model to avoid DDP buffer sync issues during sampling
    if accelerator.num_processes > 1:
        accelerator.wait_for_everyone()
        # Use unwrapped version for forward pass to avoid buffer sync issues
        prompt_editor_unwrapped = accelerator.unwrap_model(prompt_editor)
        with torch.no_grad():
            modified_prompts, prompt_deltas, original_embeddings, policy_info = prompt_editor_unwrapped(prompts_expanded, reward_variance)
    else:
        # Single GPU - use regular prompt_editor
        with torch.no_grad():
            modified_prompts, prompt_deltas, original_embeddings, policy_info = prompt_editor(prompts_expanded, reward_variance)
    
    print(f"[DEBUG] Prompt editor completed. Modified prompts: {[p[:50] + '...' if len(p) > 50 else p for p in modified_prompts[:2]]}")
    print(f"[DEBUG] Length mismatch check: prompts_expanded={len(prompts_expanded)}, modified_prompts={len(modified_prompts)}")
    
    # Safety check: ensure modified_prompts has the same length as prompts_expanded
    if len(modified_prompts) != len(prompts_expanded):
        print(f"[ERROR] Length mismatch! prompts_expanded has {len(prompts_expanded)} items but modified_prompts has {len(modified_prompts)} items")
        # Pad or truncate modified_prompts to match prompts_expanded
        if len(modified_prompts) < len(prompts_expanded):
            print(f"[FIX] Padding modified_prompts with original prompts")
            for i in range(len(modified_prompts), len(prompts_expanded)):
                modified_prompts.append(prompts_expanded[i])  # Use original prompt as fallback
        else:
            print(f"[FIX] Truncating modified_prompts to match prompts_expanded")
            modified_prompts = modified_prompts[:len(prompts_expanded)]
    
    all_samples = []
    sample_counter = 0  # Global counter for all samples in this batch
    batch_images = []  # Collect images for async reward computation
    batch_prompts = []  # Collect prompts for async reward computation
    
    # Process each expanded prompt (k modifications per original prompt)
    for expanded_idx in range(len(prompts_expanded)):
        modified_prompt = modified_prompts[expanded_idx]
        mapping_info = prompt_to_group_mapping[expanded_idx]
        original_prompt = mapping_info['original_prompt']
        group_key = mapping_info['group_key']
        original_idx = mapping_info['original_idx']
        k_idx = mapping_info['k_idx']
        
        print(f"[DEBUG] Processing expanded prompt {expanded_idx+1}/{len(prompts_expanded)} (group {group_key}, k={k_idx}):")
        print(f"  Original: \"{original_prompt[:80]}{'...' if len(original_prompt) > 80 else ''}\"")
        print(f"  Modified: \"{modified_prompt[:80]}{'...' if len(modified_prompt) > 80 else ''}\"")
        
        # Encode prompts for SD3
        prompt_embeds, pooled_prompt_embeds = encode_prompt(
            text_encoders=[pipeline.text_encoder, pipeline.text_encoder_2, pipeline.text_encoder_3],
            tokenizers=[pipeline.tokenizer, pipeline.tokenizer_2, pipeline.tokenizer_3],
            prompt=modified_prompt,
            max_sequence_length=256,
            device=accelerator.device,
            num_images_per_prompt=1
        )
        
        # Low-level: Flow sampling with controller
        samples_for_prompt = []
        
        print(f"[DEBUG] Generating {config.sample.num_image_per_prompt} images for expanded prompt {expanded_idx+1}")
        for img_idx in range(config.sample.num_image_per_prompt):
            print(f"[REALTIME] Generating image {img_idx+1}/{config.sample.num_image_per_prompt} for expanded prompt {expanded_idx+1} (group {group_key})")
            print(f"[REALTIME] Modified prompt: \"{modified_prompt[:100]}{'...' if len(modified_prompt) > 100 else ''}\"")
            
            start_time = time.time()
            with torch.no_grad():
                # Sample using pipeline with log probabilities
                final_images, latents_list, log_probs = pipeline_with_logprob(
                    pipeline,
                    prompt=modified_prompt,
                    height=config.height,
                    width=config.width,
                    num_inference_steps=config.sample.num_steps,
                    guidance_scale=config.sample.guidance_scale,
                    num_images_per_prompt=1,
                    generator=torch.Generator(device=accelerator.device).manual_seed(
                        random.randint(0, 2**32 - 1)
                    ),
                )
                
                # Debug: Check shapes of latents from pipeline
                if latents_list:
                    print(f"[DEBUG] Number of latents in list: {len(latents_list)}")
                    print(f"[DEBUG] Individual latent shape: {latents_list[0].shape}")
                    print(f"[DEBUG] Shape after stacking: {torch.stack(latents_list).shape}")
            generation_time = time.time() - start_time
            print(f"[REALTIME] Image {img_idx+1} generated in {generation_time:.2f}s")
            
            # Extract the first (and only) image from the list
            final_image = final_images[0] if isinstance(final_images, list) else final_images
            
            # Debug: Check image type
            print(f"[DEBUG] final_image type: {type(final_image)}")
            if hasattr(final_image, 'mode'):
                print(f"[DEBUG] final_image mode: {final_image.mode}")
            elif isinstance(final_image, torch.Tensor):
                print(f"[DEBUG] final_image shape: {final_image.shape}")
            
            # Ensure final_image is PIL for VLM evaluation
            if isinstance(final_image, torch.Tensor):
                # Convert tensor to PIL
                import torchvision.transforms as T
                to_pil = T.ToPILImage()
                final_image = to_pil(final_image.cpu())
                print(f"[DEBUG] Converted tensor to PIL: {type(final_image)}")
            
            # Create sample ID for immediate saving
            sample_id = f"epoch_{epoch}_batch_{batch_idx}_expanded_{expanded_idx}_img_{img_idx}"
            
            sample = {
                "original_prompt": original_prompt,
                "modified_prompt": modified_prompt,
                "prompt_delta": prompt_deltas[expanded_idx],
                "original_embedding": original_embeddings[expanded_idx],
                "policy_info": policy_info,  # Add policy info for GRPO
                "group_key": group_key,  # GRPO group key
                "original_idx": original_idx,  # Original prompt index
                "k_idx": k_idx,  # k-th modification of this prompt
                "expanded_idx": expanded_idx,  # Index in expanded prompt list
                "latents": torch.stack([x.squeeze(0) if x.dim() == 4 and x.shape[0] == 1 else x
                                        for x in latents_list]),  # [T, C, H, W]
                "log_probs": torch.stack([lp.squeeze() for lp in log_probs]),  # [T]
                "timesteps": pipeline.scheduler.timesteps,  # Use actual scheduler timesteps [T]
                "final_image": final_image,
                "prompt_embeds": prompt_embeds,
                "pooled_prompt_embeds": pooled_prompt_embeds,
                "sample_id": sample_id,  # Add sample ID to the sample
                "generation_time": generation_time,
            }
            
            # IMMEDIATE SAVING: Save image and basic info right after generation
            if accelerator.is_main_process:
                # Create directory for immediate saves
                immediate_save_dir = os.path.join(config.save_dir, "immediate_saves", f"epoch_{epoch}", f"batch_{batch_idx}")
                os.makedirs(immediate_save_dir, exist_ok=True)
                
                # Save image immediately
                image_path = os.path.join(immediate_save_dir, f"{sample_id}.png")
                final_image.save(image_path)
                
                # Save basic sample data immediately (without reward info)
                basic_sample_data = {
                    "sample_id": sample_id,
                    "epoch": epoch,
                    "batch": batch_idx,
                    "prompt_idx": prompt_idx,
                    "img_idx": img_idx,
                    "original_prompt": sample["original_prompt"],
                    "modified_prompt": sample["modified_prompt"],
                    "generation_time": generation_time,
                    "timestamp_generated": datetime.datetime.now().isoformat(),
                    "image_path": image_path,
                    "status": "generated_awaiting_reward"  # Status indicator
                }
                
                # Save basic JSON data immediately
                json_path = os.path.join(immediate_save_dir, f"{sample_id}_basic.json")
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(basic_sample_data, f, indent=2, ensure_ascii=False)
                
                print(f"[IMMEDIATE SAVE] Image and basic data saved: {image_path}")
                print(f"[IMMEDIATE SAVE] Sample ID: {sample_id}")
            
            # Collect images and prompts for async reward computation
            batch_images.append(final_image)
            batch_prompts.append(modified_prompt)
            
            samples_for_prompt.append(sample)
            sample_counter += 1
            

        
        all_samples.extend(samples_for_prompt)
    
    # Submit batch for async reward computation if reward queue is available
    if reward_queue is not None and accelerator.is_main_process:
        batch_id = f"epoch_{epoch}_batch_{batch_idx}"
        reward_queue.put((batch_id, batch_images, batch_prompts, [{}] * len(batch_images)))
        print(f"[ASYNC REWARD] Submitted batch {batch_id} for reward computation")
    
    return all_samples


def compute_attribution(sample, transformer, pipeline, config, accelerator):
    """Compute step-level Δ-Attribution for interpretability."""
    original_toxicity = sample.get("final_toxicity", 0.0)
    if original_toxicity == 0.0:
        return torch.zeros(len(sample["timesteps"]))  # Use len for timestep dimension
    
    attributions = []
    
    for step_idx in range(len(sample["timesteps"])):  # Use len for timestep dimension
        try:
            # Create modified sample with nullified control at step_idx
            modified_sample = sample.copy()
            
            # Simulate rollout with u_k = 0 at step_idx
            # This is a simplified version - full implementation would require 
            # re-running generation with specific control nullification
            
            # For now, use gradient-based approximation
            latents = sample["latents"][step_idx].clone().requires_grad_(True)  # Access single timestep
            timesteps = sample["timesteps"][step_idx:step_idx+1]  # Access single timestep
            
            with torch.enable_grad():
                # Forward through transformer
                # Use keyword arguments to avoid argument conflicts
                noise_pred = transformer(
                    hidden_states=latents.unsqueeze(0),
                    timestep=timesteps,
                    encoder_hidden_states=sample["prompt_embeds"],
                    pooled_projections=sample["pooled_prompt_embeds"],
                    return_dict=False,
                )[0]
                
                # Approximate attribution via gradient
                if latents.grad is not None:
                    latents.grad.zero_()
                
                loss = noise_pred.sum()
                loss.backward()
                
                attribution = torch.norm(latents.grad).item() if latents.grad is not None else 0.0
                attributions.append(attribution)
                
        except Exception as e:
            logger.warning(f"Attribution computation failed at step {step_idx}: {e}")
            attributions.append(0.0)
    
    return torch.tensor(attributions)


def main(_):
    """Main training function for Flow-RTPO with Policy Gradient and multi-GPU support."""
    # Ensure sample_batch function is accessible
    global sample_batch
    
    # Basic Accelerate and logging setup
    config = FLAGS.config
    
    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    if not config.run_name:
        config.run_name = unique_id
    else:
        config.run_name += "_" + unique_id
    
    # Number of timesteps within each trajectory to train on
    num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction)
    
    # Multi-GPU configuration
    num_training_gpus = 6  # 6 GPUs for training
    num_reward_gpus = 2    # 2 GPUs for reward computation
    total_gpus = num_training_gpus + num_reward_gpus
    
    # Set CUDA device mapping
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(total_gpus)))
    
    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.save_dir),
        automatic_checkpoint_naming=True,
        total_limit=config.num_checkpoint_limit,
    )
    
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps * num_train_timesteps,
    )
    
    if accelerator.is_main_process:
        swanlab.init(
            project="flow_rtpo_pg",  # PG for Policy Gradient
            experiment_name=config.run_name,
            config=config.to_dict()
        )
    
    logger.info(f"\n{config}")
    logger.info(f"Training on {num_training_gpus} GPUs, reward computation on {num_reward_gpus} GPUs")
    
    # Set seed (device_specific is very important to get different prompts on different devices)
    set_seed(config.seed, device_specific=True)
    
    # Initialize reward computation infrastructure
    reward_queue = None
    reward_workers = []
    stop_event = Event()
    
    if accelerator.is_main_process:
        # Initialize reward function for workers
        reward_fn_config = {
            "vlm_model": config.target_vlm,
            "w_cvar": config.toxicity_reward.w_cvar,
            "w_quality": config.toxicity_reward.w_quality
        }
        
        # Start reward computation workers
        reward_queue = Queue()
        reward_workers = start_reward_workers(num_reward_gpus, reward_fn_config, reward_queue, stop_event)
        logger.info(f"Started {num_reward_gpus} reward computation workers")
    
    # Load SD3 pipeline
    pipeline = StableDiffusion3Pipeline.from_pretrained(config.pretrained.model)
    
    # Freeze base model components
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.text_encoder_2.requires_grad_(False)
    pipeline.text_encoder_3.requires_grad_(False)
    pipeline.transformer.requires_grad_(not config.use_lora)
    
    # Disable safety checker
    pipeline.safety_checker = None
    pipeline.set_progress_bar_config(disable=True)
    
    # Apply LoRA to transformer (flow controller)
    if config.use_lora:
        lora_config = LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            lora_dropout=config.lora_dropout,
        )
        pipeline.transformer = get_peft_model(pipeline.transformer, lora_config)
        pipeline.transformer.train()
    
    # Initialize enhanced prompt editor with adaptive constraints and semantic regularization
    prompt_editor = PromptEditorPolicy(
        embedding_dim=config.prompt_editor.embedding_dim,
        epsilon_p=config.prompt_editor.epsilon_p,
        device=accelerator.device,
        perturbation_scale=config.prompt_editor.get('perturbation_scale', 0.02),
        # Adaptive epsilon parameters
        epsilon_min=config.prompt_editor.get('epsilon_min', 0.02),
        gamma=config.prompt_editor.get('gamma', 0.1),
        smooth_constant=config.prompt_editor.get('smooth_constant', 0.01),
        # Semantic regularization parameters
        semantic_threshold=config.prompt_editor.get('semantic_threshold', 0.9),
        semantic_alpha=config.prompt_editor.get('semantic_alpha', 0.5),
        k_samples=config.prompt_editor.get('k_samples', 4),
        # vec2text decoding parameters
        decode_num_steps=config.prompt_editor.get('decode_num_steps', 40),
        decode_beam_width=config.prompt_editor.get('decode_beam_width', 4),
        # Manual sampling for diversity
        use_manual_sampling=config.prompt_editor.get('use_manual_sampling', False),
        sample_temperature=config.prompt_editor.get('sample_temperature', 0.6),
        sample_top_p=config.prompt_editor.get('sample_top_p', 0.9)
    )
    
    # Initialize optimizers
    transformer_trainable_parameters = list(
        filter(lambda p: p.requires_grad, pipeline.transformer.parameters())
    )
    optimizer = torch.optim.AdamW(
        transformer_trainable_parameters,
        lr=config.train.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.01,
        eps=1e-8,
    )
    
    prompt_optimizer = torch.optim.AdamW(
        prompt_editor.parameters(),
        lr=config.prompt_editor.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.01,
        eps=1e-8,
    )
    
    # EMA
    ema = None
    if config.train.ema:
        ema = EMAModuleWrapper(
            transformer_trainable_parameters,
            decay=0.999,
            update_step_interval=8,
            device=accelerator.device
        )
    
    # Load dataset
    dataset, train_dataloader = create_rtp_dataset_and_loader(
        max_prompts=config.max_prompts,
        batch_size=config.sample.batch_size,
        shuffle=True,
        cache_dir=config.get("dataset_cache_dir", None)
    )
    
    if accelerator.is_main_process:
        logger.info(f"Dataset stats: {dataset.get_prompt_stats()}")
    
    # Per-prompt stat tracking for GRPO
    if config.per_prompt_stat_tracking:
        stat_tracker = PerPromptStatTracker(
            global_std=config.sample.global_std
        )
    
    # Prepare models with accelerator
    (pipeline.transformer, prompt_editor, optimizer, prompt_optimizer, 
     train_dataloader) = accelerator.prepare(
        pipeline.transformer, prompt_editor, optimizer, prompt_optimizer,
        train_dataloader
    )
    
    # Move pipeline components to device
    pipeline.vae.to(accelerator.device)
    pipeline.text_encoder.to(accelerator.device)
    pipeline.text_encoder_2.to(accelerator.device)
    pipeline.text_encoder_3.to(accelerator.device)
    
    global_step = 0
    
    #################### TRAINING LOOP ####################
    for epoch in range(config.num_epochs):
        epoch_start_time = time.time()
        
        #################### SAMPLING ####################
        pipeline.transformer.eval()
        prompt_editor.eval()
        
        epoch_samples = []
        epoch_prompts = []
        epoch_metadata = []
        
        # Get current reward variance for adaptive epsilon
        if config.per_prompt_stat_tracking:
            # Use per-prompt variance from stat_tracker if available
            if epoch > 0:  # stat_tracker needs at least one epoch of data
                try:
                    group_size, trained_prompt_num = stat_tracker.get_stats()
                    if trained_prompt_num > 0:
                        # Use the average variance across all tracked prompts
                        current_reward_variance = getattr(stat_tracker, 'global_variance', 0.01)
                    else:
                        current_reward_variance = 0.01
                except:
                    current_reward_variance = 0.01
            else:
                current_reward_variance = 0.01  # Default for first epoch
        else:
            # Fall back to prompt editor's internal tracking
            current_reward_variance = getattr(prompt_editor.module if hasattr(prompt_editor, 'module') else prompt_editor, 
                                             'reward_variance_tracker', {}).get('current', 0.01)
            if current_reward_variance is None:
                current_reward_variance = 0.01
        
        for batch_idx, (prompts, metadata) in enumerate(train_dataloader):
            if batch_idx >= config.sample.num_batches_per_epoch:
                break
            
            logger.info(f"Sampling batch {batch_idx + 1}/{config.sample.num_batches_per_epoch}")
            logger.info(f"Using reward variance: {current_reward_variance:.6f}")
            
            # Sample images using enhanced hierarchical policies with async reward computation
            batch_samples = sample_batch(pipeline, prompt_editor, prompts, config, accelerator, 
                                       epoch, batch_idx, current_reward_variance, reward_queue)
            
            # Wait for async reward computation results
            if batch_samples and accelerator.is_main_process:
                batch_id = f"epoch_{epoch}_batch_{batch_idx}"
                logger.info(f"Waiting for reward computation results for batch {batch_id}...")
                
                # Wait for reward results with timeout
                try:
                    result_batch_id, batch_rewards, batch_reward_metadata = reward_queue.get(timeout=300)  # 5 minute timeout
                    
                    if result_batch_id == batch_id and batch_rewards is not None:
                        # Assign rewards to samples
                        for i, sample in enumerate(batch_samples):
                            sample["reward"] = batch_rewards[i]
                            sample["final_toxicity"] = batch_reward_metadata["toxicity_scores"]["primary"][i]
                        
                        logger.info(f"Received reward results for batch {batch_id}")
                        
                        # Real-time logging for each sample in the batch
                        logger.info("=" * 80)
                        logger.info(f"BATCH {batch_idx + 1} REAL-TIME RESULTS:")
                        logger.info("=" * 80)
                        
                        for i, sample in enumerate(batch_samples):
                            vlm_response = batch_reward_metadata["vlm_responses"][i]
                            
                            # Complete toxicity breakdown with all 6 categories
                            toxicity_breakdown = {
                                "toxicity": batch_reward_metadata["toxicity_scores"]["primary"][i],
                                "severe_toxicity": batch_reward_metadata["toxicity_scores"]["severe_toxicity"][i],
                                "obscene": batch_reward_metadata["toxicity_scores"]["obscene"][i],
                                "threat": batch_reward_metadata["toxicity_scores"]["threat"][i],
                                "insult": batch_reward_metadata["toxicity_scores"]["insult"][i],
                                "identity_attack": batch_reward_metadata["toxicity_scores"]["identity_attack"][i],
                            }
                            
                            quality_scores = {
                                "clip": batch_reward_metadata["quality_scores"]["clip_similarity"][i],
                            }
                            cvar_bonus = batch_reward_metadata["cvar_bonus"][i]
                            
                            logger.info(f"Sample {i+1}/{len(batch_samples)}:")
                            logger.info(f"  original_prompt: \"{sample['original_prompt'][:80]}{'...' if len(sample['original_prompt']) > 80 else ''}\"")
                            logger.info(f"  modified_prompt: \"{sample['modified_prompt'][:80]}{'...' if len(sample['modified_prompt']) > 80 else ''}\"")
                            logger.info(f"  vlm_response: \"{vlm_response[:80]}{'...' if len(vlm_response) > 80 else ''}\"")
                            logger.info(f"  reward: {sample['reward']:.6f}")
                            logger.info(f"  toxicity_scores:")
                            for tox_type, score in toxicity_breakdown.items():
                                logger.info(f"    {tox_type}: {score:.6f}")
                            logger.info(f"  cvar_bonus: {cvar_bonus:.6f}")
                            logger.info(f"  quality_scores: clip={quality_scores['clip']:.6f}")
                            logger.info("-" * 60)
                        
                        # Batch statistics and immediate saving
                        batch_reward_mean = np.mean(batch_rewards)
                        batch_reward_std = np.std(batch_rewards)
                        batch_toxicity_mean = np.mean([score for score in batch_reward_metadata["toxicity_scores"]["primary"]])
                        batch_toxicity_max = max([score for score in batch_reward_metadata["toxicity_scores"]["primary"]])
                        
                        logger.info(f"BATCH {batch_idx + 1} STATISTICS:")
                        logger.info(f"  reward_mean: {batch_reward_mean:.6f}")
                        logger.info(f"  reward_std: {batch_reward_std:.6f}")
                        logger.info(f"  toxicity_mean: {batch_toxicity_mean:.6f}")
                        logger.info(f"  toxicity_max: {batch_toxicity_max:.6f}")
                        logger.info("=" * 80)
                        
                        # Update the immediately saved files with reward information
                        immediate_save_dir = os.path.join(config.save_dir, "immediate_saves", f"epoch_{epoch}", f"batch_{batch_idx}")
                        
                        for i, sample in enumerate(batch_samples):
                            sample_id = sample["sample_id"]
                            
                            # Read the existing basic data
                            basic_json_path = os.path.join(immediate_save_dir, f"{sample_id}_basic.json")
                            if os.path.exists(basic_json_path):
                                with open(basic_json_path, 'r', encoding='utf-8') as f:
                                    existing_data = json.load(f)
                            else:
                                existing_data = {}
                            
                            # Update with reward information
                            reward_data = {
                                "reward": sample["reward"],
                                "toxicity": sample["final_toxicity"],
                                "vlm_response": batch_reward_metadata["vlm_responses"][i],
                                "toxicity_breakdown": {
                                    "toxicity": batch_reward_metadata["toxicity_scores"]["primary"][i],
                                    "severe_toxicity": batch_reward_metadata["toxicity_scores"]["severe_toxicity"][i],
                                    "obscene": batch_reward_metadata["toxicity_scores"]["obscene"][i],
                                    "threat": batch_reward_metadata["toxicity_scores"]["threat"][i],
                                    "insult": batch_reward_metadata["toxicity_scores"]["insult"][i],
                                    "identity_attack": batch_reward_metadata["toxicity_scores"]["identity_attack"][i],
                                },
                                "cvar_bonus": batch_reward_metadata["cvar_bonus"][i],
                                "quality_scores": {
                                    "clip": batch_reward_metadata["quality_scores"]["clip_similarity"][i],
                                },
                                "timestamp_reward_computed": datetime.datetime.now().isoformat(),
                                "status": "completed_with_reward"
                            }
                            
                            # Merge existing data with reward data
                            complete_sample_data = {**existing_data, **reward_data}
                            
                            # Save complete data (replace the basic file)
                            complete_json_path = os.path.join(immediate_save_dir, f"{sample_id}_complete.json")
                            with open(complete_json_path, 'w', encoding='utf-8') as f:
                                json.dump(complete_sample_data, f, indent=2, ensure_ascii=False)
                            
                            # Optionally remove the basic file now that we have complete data
                            if os.path.exists(basic_json_path):
                                os.remove(basic_json_path)
                        
                        logger.info(f"Batch {batch_idx + 1} reward data updated in: {immediate_save_dir}")
                        logger.info(f"Batch {batch_idx + 1} mean reward: {np.mean(batch_rewards):.4f}")
                        
                    else:
                        logger.error(f"Error in reward computation for batch {batch_id}")
                        # Assign default rewards
                        for sample in batch_samples:
                            sample["reward"] = 0.0
                            sample["final_toxicity"] = 0.0
                            
                except queue.Empty:
                    logger.error(f"Timeout waiting for reward computation for batch {batch_id}")
                    # Assign default rewards
                    for sample in batch_samples:
                        sample["reward"] = 0.0
                        sample["final_toxicity"] = 0.0
            
            epoch_samples.extend(batch_samples)
            epoch_prompts.extend([s["modified_prompt"] for s in batch_samples])
            epoch_metadata.extend([{} for _ in batch_samples])
        
        logger.info(f"Generated {len(epoch_samples)} samples for epoch {epoch}")
        
        #################### REWARD AGGREGATION ####################
        # Extract rewards that were already computed during batch processing
        all_rewards = [sample["reward"] for sample in epoch_samples]
        all_toxicity_scores = [sample["final_toxicity"] for sample in epoch_samples]
        
        # Create reward metadata from aggregated batch results
        reward_metadata = {
            "statistics": {
                "mean_toxicity": np.mean(all_toxicity_scores),
                "max_toxicity": max(all_toxicity_scores),
                "cvar_mean": np.mean(all_rewards),  # Approximate CVaR from rewards
                "quality_mean": 0.0  # Will be computed from individual samples if needed
            },
            "cvar_threshold": np.percentile(all_rewards, 10) if len(all_rewards) > 0 else 0.0,
            "toxicity_scores": {"primary": all_toxicity_scores}
        }
        
        logger.info(f"Mean reward (from batch computations): {np.mean(all_rewards):.4f}")
        logger.info(f"CVaR threshold: {reward_metadata['cvar_threshold']:.4f}")
        
        #################### ADVANTAGE COMPUTATION ####################
        if config.per_prompt_stat_tracking:
            advantages = stat_tracker.update(epoch_prompts, np.array(all_rewards))
        else:
            advantages = (np.array(all_rewards) - np.mean(all_rewards)) / (np.std(all_rewards) + 1e-4)
        
        # Assign advantages to samples (broadcast to timesteps)
        for i, sample in enumerate(epoch_samples):
            sample["advantages"] = torch.tensor(advantages[i]).repeat(len(sample["timesteps"])).to(accelerator.device)
        
        #################### TRAINING ####################
        pipeline.transformer.train()
        prompt_editor.train()
        
        # Batch samples for training
        samples_batched = [
            epoch_samples[i:i + config.train.batch_size] 
            for i in range(0, len(epoch_samples), config.train.batch_size)
        ]
        
        train_info = defaultdict(list)
        
        for inner_epoch in range(config.train.num_inner_epochs):
            logger.info(f"Training inner epoch {inner_epoch + 1}/{config.train.num_inner_epochs}")
            
            # Training loop over timesteps (Flow Controller)
            for i, samples_batch in enumerate(samples_batched):
                train_timesteps = list(range(num_train_timesteps))
                
                for j in tqdm(
                    train_timesteps,
                    desc=f"Timestep (Batch {i+1}/{len(samples_batched)})",
                    disable=not accelerator.is_local_main_process,
                ):
                    for sample in samples_batch:
                        with accelerator.accumulate(pipeline.transformer):
                            # Compute log probabilities
                            prev_sample, log_prob, prev_sample_mean, std_dev_t = compute_log_prob(
                                pipeline.transformer, pipeline, sample, j, 
                                sample["prompt_embeds"], sample["pooled_prompt_embeds"], config
                            )
                            
                            # Reference log probs for KL regularization
                            if config.train.beta > 0:
                                with torch.no_grad():
                                    with pipeline.transformer.module.disable_adapter():
                                        _, _, prev_sample_mean_ref, _ = compute_log_prob(
                                            pipeline.transformer, pipeline, sample, j,
                                            sample["prompt_embeds"], sample["pooled_prompt_embeds"], config
                                        )
                            
                            # GRPO loss computation
                            advantages = torch.clamp(
                                sample["advantages"][j:j+1],
                                -config.train.adv_clip_max,
                                config.train.adv_clip_max,
                            )
                            
                            ratio = torch.exp(log_prob - sample["log_probs"][j])
                            unclipped_loss = -advantages * ratio
                            clipped_loss = -advantages * torch.clamp(
                                ratio,
                                1.0 - config.train.clip_range,
                                1.0 + config.train.clip_range,
                            )
                            policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))
                            
                            # KL regularization
                            if config.train.beta > 0:
                                kl_loss = ((prev_sample_mean - prev_sample_mean_ref) ** 2).mean() / (2 * std_dev_t ** 2)
                                kl_loss = torch.mean(kl_loss)
                                flow_loss = policy_loss + config.train.beta * kl_loss
                            else:
                                flow_loss = policy_loss
                                kl_loss = torch.tensor(0.0)
                            
                            # Backward pass
                            accelerator.backward(flow_loss)
                            
                            train_info["flow_policy_loss"].append(policy_loss.item())
                            train_info["kl_loss"].append(kl_loss.item())
                
                # Optimizer step
                optimizer.step()
                optimizer.zero_grad()
                
                if config.train.ema:
                    ema.step(transformer_trainable_parameters, global_step)
            
            # Training for Prompt Editor using Policy Gradient (REINFORCE)
            if len(epoch_samples) > 0:
                # Collect trajectories for policy gradient training with GRPO grouping
                trajectories = []
                for sample in epoch_samples:
                    # Extract proper log_prob for this specific sample
                    if 'log_prob' in sample["policy_info"]:
                        if isinstance(sample["policy_info"]['log_prob'], torch.Tensor):
                            if sample["policy_info"]['log_prob'].numel() > 1:
                                # Multiple log_probs, extract the one for this expanded sample
                                sample_log_prob = sample["policy_info"]['log_prob'][sample["expanded_idx"]:sample["expanded_idx"]+1]
                            else:
                                sample_log_prob = sample["policy_info"]['log_prob']
                        else:
                            sample_log_prob = torch.tensor([sample["policy_info"]['log_prob']], device=accelerator.device, requires_grad=True)
                    else:
                        sample_log_prob = torch.zeros(1, device=accelerator.device, requires_grad=True)
                    
                    trajectory = {
                        'group_key': sample["group_key"],  # GRPO group key
                        'prompts': [sample["original_prompt"]],
                        'states': sample["original_embedding"].unsqueeze(0),
                        'actions': sample["prompt_delta"].unsqueeze(0),
                        'rewards': torch.tensor([sample["reward"]], device=accelerator.device),
                        'log_probs': sample_log_prob,
                        'modified_prompts': [sample["modified_prompt"]],
                        'policy_info': sample["policy_info"],
                        'reward_metadata': {}
                    }
                    trajectories.append(trajectory)
                
                # Compute baseline value (moving average of rewards for variance reduction)
                current_rewards = [sample["reward"] for sample in epoch_samples]
                baseline_value = np.mean(current_rewards)
                
                # Use enhanced policy gradient training method
                prompt_metrics = prompt_editor.module.update_policy(
                    trajectories, prompt_optimizer, baseline_value
                )
                
                # Log enhanced metrics
                train_info["prompt_policy_loss"].append(prompt_metrics.get('policy_loss', 0.0))
                train_info["prompt_reg_loss"].append(prompt_metrics.get('regularization_loss', 0.0))
                train_info["total_prompt_loss"].append(prompt_metrics.get('total_loss', 0.0))
                train_info["prompt_mean_advantage"].append(prompt_metrics.get('mean_advantage', 0.0))
                train_info["prompt_baseline_value"].append(prompt_metrics.get('baseline_value', 0.0))
                
                # Enhanced metrics from adaptive and semantic components
                train_info["reward_variance"].append(prompt_metrics.get('reward_variance', 0.0))
                train_info["epsilon_adaptive"].append(prompt_metrics.get('epsilon_adaptive', 0.0))
                train_info["num_groups"].append(prompt_metrics.get('grpo_num_groups', 0))
                train_info["reg_proximity_reg"].append(prompt_metrics.get('reg_proximity_reg', 0.0))
                train_info["reg_semantic_reg"].append(prompt_metrics.get('reg_semantic_reg', 0.0))
                train_info["reg_reconstruction"].append(prompt_metrics.get('reg_reconstruction', 0.0))
                train_info["reg_epsilon_current"].append(prompt_metrics.get('reg_epsilon_current', 0.0))
                train_info["reg_mean_semantic_sim"].append(prompt_metrics.get('reg_mean_semantic_sim', 0.0))
                
                # Warmup factor tracking
                if len(epoch_samples) > 0 and 'policy_info' in epoch_samples[0]:
                    sample_policy_info = epoch_samples[0]['policy_info']
                    train_info["prompt_warmup_factor"].append(sample_policy_info.get('warmup_factor', 1.0))
        
        global_step += 1
        
        #################### LOGGING AND SAVING ####################
        epoch_time = time.time() - epoch_start_time
        
        # Log training metrics
        if accelerator.is_main_process:
            log_data = {
                "epoch": epoch,
                "global_step": global_step,
                "epoch_time": epoch_time,
                "num_samples": len(epoch_samples),
                "reward_mean": np.mean(all_rewards),
                "reward_std": np.std(all_rewards),
                "toxicity_mean": reward_metadata["statistics"]["mean_toxicity"],
                "toxicity_max": reward_metadata["statistics"]["max_toxicity"],
                "cvar_mean": reward_metadata["statistics"]["cvar_mean"],
                "cvar_threshold": reward_metadata["cvar_threshold"],
                "quality_mean": reward_metadata["statistics"]["quality_mean"],
                "flow_policy_loss": np.mean(train_info["flow_policy_loss"]),
                "kl_loss": np.mean(train_info["kl_loss"]),
                "prompt_policy_loss": np.mean(train_info["prompt_policy_loss"]),
                "prompt_reg_loss": np.mean(train_info["prompt_reg_loss"]),
                "total_prompt_loss": np.mean(train_info["total_prompt_loss"]),
                "prompt_mean_advantage": np.mean(train_info.get("prompt_mean_advantage", [0])),
                "prompt_baseline_value": np.mean(train_info.get("prompt_baseline_value", [0])),
                # Enhanced adaptive and semantic metrics
                "reward_variance": np.mean(train_info.get("reward_variance", [0])),
                "epsilon_adaptive": np.mean(train_info.get("epsilon_adaptive", [0])),
                "num_groups": np.mean(train_info.get("num_groups", [0])),
                "reg_proximity_reg": np.mean(train_info.get("reg_proximity_reg", [0])),
                "reg_semantic_reg": np.mean(train_info.get("reg_semantic_reg", [0])),
                "reg_reconstruction": np.mean(train_info.get("reg_reconstruction", [0])),
                "reg_epsilon_current": np.mean(train_info.get("reg_epsilon_current", [0])),
                "reg_mean_semantic_sim": np.mean(train_info.get("reg_mean_semantic_sim", [0])),
                # Warmup tracking
                "prompt_warmup_factor": np.mean(train_info.get("prompt_warmup_factor", [1])),
            }
            
            swanlab.log(log_data)
            logger.info(f"Epoch {epoch}: {log_data}")
        
        # Save results for each processed prompt
        if accelerator.is_main_process:
            epoch_results = {
                "epoch": epoch,
                "timestamp": datetime.datetime.now().isoformat(),
                "samples": [],
                "statistics": reward_metadata["statistics"],
                "training_metrics": {k: np.mean(v) for k, v in train_info.items()}
            }
            
            for sample in epoch_samples:
                # Compute attribution if enabled
                attribution = None
                if config.attribution.enable and epoch % config.attribution.sample_frequency == 0:
                    attribution = compute_attribution(sample, pipeline.transformer, pipeline, config, accelerator)
                
                sample_result = {
                    "original_prompt": sample["original_prompt"],
                    "modified_prompt": sample["modified_prompt"],
                    "reward": sample["reward"],
                    "toxicity": sample["final_toxicity"],
                    "advantage": sample["advantages"].mean().item(),
                    "attribution": attribution.tolist() if attribution is not None else None
                }
                epoch_results["samples"].append(sample_result)
                
                # Note: Image saving removed to prevent I/O blocking
                # Images are not saved during training for performance
            
            # Save epoch results
            results_path = os.path.join(config.save_dir, f"epoch_{epoch}_results.json")
            with open(results_path, 'w') as f:
                json.dump(epoch_results, f, indent=2)
        
        # Save checkpoint
        if (epoch + 1) % config.save_freq == 0:
            save_ckpt(config.save_dir, pipeline.transformer, prompt_editor, global_step, 
                     accelerator, ema, transformer_trainable_parameters, config)
    
    # Cleanup reward workers
    if accelerator.is_main_process:
        stop_event.set()
        for worker in reward_workers:
            worker.join(timeout=10)
        logger.info("Reward computation workers stopped")
    
    # Final save
    save_ckpt(config.save_dir, pipeline.transformer, prompt_editor, global_step, 
             accelerator, ema, transformer_trainable_parameters, config)
    
    logger.info("Training completed!")


if __name__ == "__main__":
    app.run(main)
