
import torch
import torch.nn as nn
import numpy as np
import vec2text
from typing import List, Tuple, Dict, Optional
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import math
from collections import defaultdict
import random


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer architecture."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        # Clone the tensor to ensure it doesn't share memory before registering as buffer
        self.register_buffer('pe', pe.clone())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        # Clone x to avoid modifying the input tensor in place (DDP safety)
        return x.clone() + self.pe[:x.size(1), :].transpose(0, 1)


class PromptEditorPolicy(nn.Module):
    """High-level prompt editing policy π_p using vec2text for reconstruction.
    
    Based on DART paper: https://arxiv.org/abs/2401.11506
    This model learns to predict noise vectors μ that maximize rewards 
    while keeping ||μ||₂ ≤ ε (budget constraint).
    
    Uses true GRPO (Group-wise Reinforcement Policy Optimization) for training.
    """
    
    def __init__(self, 
                 embedding_dim: int = 768,
                 epsilon_p: float = 0.05,  # 基础约束范围，防止过度修改
                 device: str = "cuda",
                 perturbation_scale: float = 0.01,  # 大幅减小扰动缩放因子
                 # Adaptive epsilon parameters
                 epsilon_min: float = 0.02,
                 gamma: float = 0.1,
                 smooth_constant: float = 0.01,
                 # Semantic regularization parameters
                 semantic_threshold: float = 0.9,  # 提高阈值加强语义约束
                 semantic_alpha: float = 1.0,  # 增强语义约束权重
                 k_samples: int = 4,  # samples per prompt for GRPO
                 # GRPO-specific parameters
                 grpo_clip: float = 1.0,  # c parameter in GRPO gradient scaling (suppress extreme advantages)
                 use_true_grpo: bool = True,  # Use true GRPO vs REINFORCE with GRPO advantages
                 # vec2text decoding parameters
                 decode_num_steps: int = 40,  # Number of inversion steps
                 decode_beam_width: int = 4,  # Beam width for text inversion
                 # Manual sampling parameters for diversity
                 use_manual_sampling: bool = False,  # Use manual temperature + top-p sampling
                 sample_temperature: float = 1.2,  # Temperature for diversity (T>1 increases diversity)
                 sample_top_p: float = 0.9,  # Nucleus sampling threshold
                 # Simple noise injection parameters
                 use_simple_noise: bool = True,  # Use simple Gaussian noise injection
                 simple_noise_std: float = 0.01,  # Standard deviation for simple noise
                 # Modification noise parameters for sampling diversity
                 use_modification_noise: bool = True,  # Use random noise during embedding modification
                 modification_noise_std: float = 0.005,  # Standard deviation for modification noise
                 # Local model paths
                 gtr_model_path: str = "/mnt/data/group/zhaoliangjie/ICLR-work/flow_rtpo/models/local/gtr-base-t5",  # Local GTR model path
                 sbert_model_path: str = "/mnt/data/group/zhaoliangjie/ICLR-work/flow_rtpo/models/local/all-MiniLM-L6-v2",  # Local SBERT model path
                 **kwargs):  # Accept extra kwargs for backward compatibility
        super().__init__()
        self.epsilon_p = epsilon_p
        self.device = device
        self.embedding_dim = embedding_dim
        self.perturbation_scale = perturbation_scale
        self.training_step = 0  # 训练步数计数器，用于热身
        
        # Adaptive epsilon parameters
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        self.smooth_constant = smooth_constant
        
        # Semantic regularization parameters
        self.semantic_threshold = semantic_threshold
        self.semantic_alpha = semantic_alpha
        self.k_samples = k_samples
        
        # GRPO-specific parameters
        self.grpo_clip = grpo_clip
        self.use_true_grpo = use_true_grpo
        
        # vec2text decoding parameters
        self.decode_num_steps = decode_num_steps
        self.decode_beam_width = decode_beam_width
        
        # Manual sampling parameters
        self.use_manual_sampling = use_manual_sampling
        self.sample_temperature = sample_temperature
        self.sample_top_p = sample_top_p
        
        # Simple noise injection parameters
        self.use_simple_noise = use_simple_noise
        self.simple_noise_std = simple_noise_std
        
        # Modification noise parameters for sampling diversity
        self.use_modification_noise = use_modification_noise
        self.modification_noise_std = modification_noise_std
        if self.use_modification_noise:
            print(f"[INFO] Modification noise enabled by default with std={modification_noise_std}")
        
        # Reward variance tracking for adaptive epsilon
        self.reward_variance_tracker = defaultdict(list)
        self.group_reward_stats = {}
        
        # Initialize SBERT model for semantic similarity
        self.sbert_model = None
        try:
            # Try to import and use SentenceTransformer
            from sentence_transformers import SentenceTransformer as ST
            
            # Use local SBERT model path if provided, otherwise fall back to HuggingFace
            sbert_model_name = sbert_model_path
            print(f"[INFO] Loading SBERT model from: {sbert_model_name}")
            
            self.sbert_model = ST(sbert_model_name).to(device)
            print(f"[INFO] SBERT model loaded for semantic regularization from: {sbert_model_name}")
        except ImportError:
            print("[WARNING] sentence-transformers not available, semantic regularization disabled")
            self.sbert_model = None
        except Exception as e:
            print(f"[WARNING] Failed to load SBERT model from {sbert_model_name}: {e}")
            self.sbert_model = None
        
        # Initialize vec2text corrector
        self.vec2text_corrector = vec2text.load_pretrained_corrector("gtr-base")
        
        # Use OFFICIAL vec2text approach for GTR as per documentation
        from transformers import AutoTokenizer, AutoModel
        
        # Use local GTR model path if provided, otherwise fall back to HuggingFace
        gtr_model_name = gtr_model_path
        print(f"[INFO] Loading GTR model from: {gtr_model_name}")
        
        self.gtr_tokenizer = AutoTokenizer.from_pretrained(gtr_model_name)
        
        # Try loading with different precision/settings to avoid NaN
        try:
            self.gtr_encoder = AutoModel.from_pretrained(
                gtr_model_name,
                torch_dtype=torch.float32  # Force float32 to avoid precision issues
            ).encoder.to(device)
            self.use_fallback_encoder = False
        except Exception as e:
            print(f"[WARNING] Failed to load GTR encoder from {gtr_model_name}: {e}")
            # Use a simple fallback encoder
            try:
                from sentence_transformers import SentenceTransformer
                fallback_model_name = sbert_model_path
                self.sentence_transformer_fallback = SentenceTransformer(fallback_model_name).to(device)
            except ImportError:
                print("[WARNING] sentence-transformers not available for fallback encoder")
                self.sentence_transformer_fallback = None
            self.use_fallback_encoder = True
        
        # Noise prediction network (π_θ in paper) - encoder-decoder transformer
        # Following the paper's architecture description
        self.d_model = 512 # Transformer hidden dimension (increased from 256)
        self.nhead = 16    # Number of attention heads (increased from 8)
        self.num_layers = 6 # Number of transformer layers (increased from 3)
        
        # Input projection to transformer dimension
        self.input_projection = nn.Linear(embedding_dim, self.d_model)
        
        # Encoder-Decoder Transformer Architecture
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.d_model * 2,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, 
            num_layers=self.num_layers
        )
        
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.d_model * 2,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            self.decoder_layer,
            num_layers=self.num_layers
        )
        
        # Output projection back to embedding dimension
        self.output_projection = nn.Linear(self.d_model, embedding_dim)
        
        # Positional encoding for transformer
        self.pos_encoding = PositionalEncoding(self.d_model)
        
        # 使用更小的权重初始化，减少初始扰动
        for module in [self.input_projection, self.output_projection]:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=0.1)  # 更小的初始化gain
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # Initialize transformer weights with small scale
        for module in [self.transformer_encoder, self.transformer_decoder]:
            for param in module.parameters():
                if param.dim() > 1:
                    nn.init.xavier_normal_(param, gain=0.1)
        
        # Move to device
        self.input_projection.to(device)
        self.transformer_encoder.to(device)
        self.transformer_decoder.to(device)
        self.output_projection.to(device)
        self.pos_encoding.to(device)
    
    def nucleus_sampling(self, logits: torch.Tensor, temperature: float = 1.0, top_p: float = 0.9) -> torch.Tensor:
        """Apply temperature scaling + nucleus (top-p) sampling for diverse text generation."""
        # Temperature scaling: logits / T
        # T > 1: 拉平分布，增加多样性
        # T < 1: 尖锐分布，减少多样性
        scaled_logits = logits / temperature
        
        # Convert to probabilities
        probs = torch.softmax(scaled_logits, dim=-1)
        
        # Nucleus (top-p) sampling
        # 1. Sort probabilities in descending order
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        
        # 2. Compute cumulative probabilities
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # 3. Find the cutoff: keep tokens until cumulative prob > top_p
        cutoff_mask = cumulative_probs <= top_p
        
        # Always keep at least the top token
        cutoff_mask[..., 0] = True
        
        # 4. Zero out probabilities beyond the cutoff
        sorted_probs[~cutoff_mask] = 0.0
        
        # 5. Renormalize
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
        
        # 6. Sample from the filtered distribution
        sampled_indices = torch.multinomial(sorted_probs, num_samples=1)
        
        # 7. Map back to original indices
        return sorted_indices.gather(-1, sampled_indices)
    
    def diverse_decode_embeddings(self, embeddings: torch.Tensor, num_candidates: int = 3) -> List[str]:
        """Generate multiple diverse text candidates using temperature + top-p sampling."""
        candidates = []
        
        for _ in range(num_candidates):
            try:
                # Use vec2text for initial inversion with beam search for quality
                decoded_texts = vec2text.invert_embeddings(
                    embeddings=embeddings.float(),
                    corrector=self.vec2text_corrector,
                    num_steps=self.decode_num_steps,
                    sequence_beam_width=self.decode_beam_width
                )
                
                if decoded_texts and any(text.strip() for text in decoded_texts):
                    # Randomly select from beam search results
                    valid_texts = [text for text in decoded_texts[:self.decode_beam_width] if text.strip()]
                    best_text = random.choice(valid_texts) if valid_texts else decoded_texts[0]
                    candidates.append(best_text)
                else:
                    candidates.append(f"fallback_text_{len(candidates)}")
                    
            except Exception as e:
                print(f"[WARNING] Diverse decoding failed: {e}")
                candidates.append(f"fallback_text_{len(candidates)}")
        
        return candidates
    
    def decode_embedding_with_sampling(self, 
                                       embedding: torch.Tensor, 
                                       temperature: float = 1.2, 
                                       top_p: float = 0.9, 
                                       max_new_tokens: int = 64) -> str:
        """使用 vec2text.invert_embeddings() 对单个 embedding 进行解码。
        
        Args:
            embedding: 单个 768 维 embedding tensor, shape [1, 768]
            temperature: 采样温度，T>1 增加多样性
            top_p: nucleus sampling 阈值
            max_new_tokens: 最大生成长度
            
        Returns:
            decoded_text: 解码出的文本字符串
        """
        with torch.no_grad():
            try:
                # 确保 embedding 是正确的形状和设备
                if embedding.dim() == 1:
                    embedding = embedding.unsqueeze(0)  # [768] -> [1, 768]
                
                embedding = embedding.float().to(self.device)
                
                # 使用 vec2text.invert_embeddings() 进行解码
                decoded_texts = vec2text.invert_embeddings(
                    embeddings=embedding,
                    corrector=self.vec2text_corrector,
                    num_steps=max(5, max_new_tokens // 10),  # 基于最大长度调整步数
                    sequence_beam_width=1  # 单束搜索
                )
                
                decoded_text = decoded_texts[0] if decoded_texts and decoded_texts[0].strip() else "fallback_text"
                print(f"[DEBUG] Sampling decode result: '{decoded_text}' (T={temperature}, p={top_p})")
                return decoded_text
                
            except Exception as e:
                print(f"[ERROR] vec2text.invert_embeddings() failed: {e}")
                # 回退到标准的 vec2text 方法
                try:
                    fallback_result = vec2text.invert_embeddings(
                        embeddings=embedding,
                        corrector=self.vec2text_corrector,
                        num_steps=self.decode_num_steps,
                        sequence_beam_width=1,
                    )
                    return fallback_result[0] if fallback_result and fallback_result[0].strip() else "fallback_text"
                except Exception as e2:
                    print(f"[ERROR] Fallback method also failed: {e2}")
                    return "fallback_text"
        
    def encode_prompts(self, prompts: List[str]) -> torch.Tensor:
        """Encode prompts using OFFICIAL vec2text GTR method from documentation."""
        with torch.no_grad():
            # Branch based on encoder availability
            if getattr(self, "use_fallback_encoder", False):
                if hasattr(self, 'sentence_transformer_fallback') and self.sentence_transformer_fallback is not None:
                    print(f"[DEBUG] Using SentenceTransformer fallback encoder...")
                    embeddings = self.sentence_transformer_fallback.encode(prompts, convert_to_tensor=True)
                    embeddings = F.normalize(embeddings, p=2, dim=-1)  # Normalize to unit norm
                    return embeddings.clone()
                else:
                    print(f"[WARNING] Fallback encoder not available, using zero embeddings")
                    return torch.zeros(len(prompts), self.embedding_dim, device=self.device)
            
            print(f"[DEBUG] Encoding prompts using official vec2text approach...")
            
            # Official vec2text approach for GTR embeddings
            inputs = self.gtr_tokenizer(
                prompts,
                return_tensors="pt",
                max_length=128,  # As per official example
                truncation=True,
                padding="max_length"
            ).to(self.device)
            
            
            # Check model weights for NaN
            model_has_nan = any(torch.isnan(p).any() for p in self.gtr_encoder.parameters())
            
            
            if model_has_nan:
                
                # Reinitialize the model
                from transformers import AutoModel
                self.gtr_encoder = AutoModel.from_pretrained('sentence-transformers/gtr-t5-base').encoder.to(self.device)
            
            # Use the encoder directly with autocast disabled to prevent NaN
            try:
                # Set encoder to eval mode and disable autocast to prevent fp16 NaN
                self.gtr_encoder.eval()
                
                with torch.autocast(device_type="cuda", enabled=False):
                    model_output = self.gtr_encoder(
                        input_ids=inputs['input_ids'], 
                        attention_mask=inputs['attention_mask']
                    )
                    hidden_state = model_output.last_hidden_state.float()  # Ensure fp32
                    
                   
                    
                    if torch.isnan(hidden_state).any():
                        print("[ERROR] GTR encoder still produces NaN! Using zero embeddings.")
                        return torch.zeros(len(prompts), self.embedding_dim, device=self.device)
                    
                    # Use vec2text's official mean_pool function (also in fp32)
                    embeddings = vec2text.models.model_utils.mean_pool(hidden_state, inputs['attention_mask'])
                    embeddings = F.normalize(embeddings, p=2, dim=-1)  # Normalize to unit norm
                    # Clone to ensure no memory sharing for DDP
                    embeddings = embeddings.clone()
                
            except Exception as e:
                print(f"[ERROR] GTR encoder failed: {e}")
                return torch.zeros(len(prompts), self.embedding_dim, device=self.device)
            
           
            
            return embeddings
    
    def decode_embeddings(self, embeddings: torch.Tensor) -> List[str]:
        """Decode embeddings back to text using vec2text with optional diverse sampling."""
        with torch.no_grad():
            # Keep embeddings on same device as vec2text corrector
            embeddings_for_inversion = embeddings.detach().float()  # Keep on GPU where vec2text model is
            
            try:
                # Test with a simple text first to verify vec2text is working
                if len(embeddings_for_inversion) > 0:
                    test_embedding = embeddings_for_inversion[0:1]  # Take first embedding
                
                if self.use_manual_sampling:
                    # Use vec2text.invert_embeddings() with decoding parameters for diversity
                    all_decoded_texts = []
                    for i in range(len(embeddings_for_inversion)):
                        single_embedding = embeddings_for_inversion[i:i+1]
                        
                        # 使用 vec2text.invert_embeddings() 进行解码
                        try:
                            decoded_texts = vec2text.invert_embeddings(
                                embeddings=single_embedding,
                                corrector=self.vec2text_corrector,
                                num_steps=self.decode_num_steps,  # 使用配置的步数
                                sequence_beam_width=1  # 单束搜索提高多样性
                            )
                            decoded_text = decoded_texts[0] if decoded_texts and decoded_texts[0].strip() else "generated_text"
                            all_decoded_texts.append(decoded_text)
                        except Exception as e:
                            print(f"[WARNING] vec2text.invert_embeddings() failed for embedding {i}: {e}")
                            # 回退到标准的 vec2text.invert_embeddings
                            try:
                                fallback_result = vec2text.invert_embeddings(
                                    embeddings=single_embedding,
                                    corrector=self.vec2text_corrector,
                                    num_steps=self.decode_num_steps,
                                    sequence_beam_width=1,
                                )
                                fallback_text = fallback_result[0] if fallback_result and fallback_result[0].strip() else "generated_text"
                                all_decoded_texts.append(fallback_text)
                            except Exception as e2:
                                print(f"[WARNING] Fallback invert_embeddings also failed: {e2}")
                                all_decoded_texts.append("generated_text")
                    
                    decoded_texts = all_decoded_texts
                else:
                    # Use standard vec2text corrector with configurable beam search
                    beam_results = vec2text.invert_embeddings(
                        embeddings=embeddings_for_inversion.float(),  # Ensure fp32
                        corrector=self.vec2text_corrector,
                        num_steps=self.decode_num_steps,  # Configurable inversion steps
                        sequence_beam_width=self.decode_beam_width,  # Configurable beam widt,
                    )
                    
                    # Improved handling of vec2text results
                    decoded_texts = []
                    if beam_results:
                        print(f"[DEBUG] beam_results type: {type(beam_results)}, length: {len(beam_results) if hasattr(beam_results, '__len__') else 'N/A'}")
                        print(f"[DEBUG] First element type: {type(beam_results[0]) if beam_results else 'N/A'}")
                        
                        # Handle different return formats from vec2text
                        if isinstance(beam_results, list) and len(beam_results) > 0:
                            if isinstance(beam_results[0], str):
                                # Flat list of strings - could be multiple results for single embedding or one result per embedding
                                if len(embeddings_for_inversion) == 1:
                                    # Single embedding: select randomly from beam results
                                    valid_texts = [text for text in beam_results[:self.decode_beam_width] if text.strip()]
                                    selected_text = random.choice(valid_texts) if valid_texts else beam_results[0]
                                    decoded_texts.append(selected_text)
                                else:
                                    # Multiple embeddings: assume one result per embedding
                                    for i in range(len(embeddings_for_inversion)):
                                        if i < len(beam_results):
                                            text = beam_results[i]
                                            if text and text.strip():
                                                decoded_texts.append(text)
                                            else:
                                                # Use a more meaningful fallback
                                                decoded_texts.append("generated_text")
                                        else:
                                            decoded_texts.append("generated_text")
                            elif isinstance(beam_results[0], list):
                                # Nested list: multiple beam results per embedding
                                for i, embedding_results in enumerate(beam_results):
                                    if embedding_results:
                                        valid_texts = [text for text in embedding_results[:self.decode_beam_width] if text.strip()]
                                        selected_text = random.choice(valid_texts) if valid_texts else (embedding_results[0] if embedding_results else "generated_text")
                                        decoded_texts.append(selected_text)
                                    else:
                                        decoded_texts.append("generated_text")
                            else:
                                # Unknown format - create fallback
                                print(f"[WARNING] Unknown beam_results format: {type(beam_results[0])}")
                                decoded_texts = ["generated_text"] * len(embeddings_for_inversion)
                        else:
                            # Empty or invalid results
                            decoded_texts = ["generated_text"] * len(embeddings_for_inversion)
                    else:
                        decoded_texts = ["generated_text"] * len(embeddings_for_inversion)
                
                # Ensure we have the correct number of decoded texts
                if len(decoded_texts) != len(embeddings_for_inversion):
                    print(f"[WARNING] Length mismatch in decode_embeddings: expected {len(embeddings_for_inversion)}, got {len(decoded_texts)}")
                    # Fix the length mismatch
                    if len(decoded_texts) < len(embeddings_for_inversion):
                        # Pad with fallback texts
                        for i in range(len(decoded_texts), len(embeddings_for_inversion)):
                            decoded_texts.append("generated_text")
                    else:
                        # Truncate to correct length
                        decoded_texts = decoded_texts[:len(embeddings_for_inversion)]
                
                if not decoded_texts or all(not text.strip() for text in decoded_texts):
                    # Simple fallback: use generic text
                    return ["generated_text"] * len(embeddings_for_inversion)
                
                return decoded_texts
                
            except Exception as e:
                print(f"[ERROR] vec2text failed: {e}")
                import traceback
                print(f"[ERROR] Full traceback: {traceback.format_exc()}")
                return ["generated_text"] * len(embeddings_for_inversion)
    
    def compute_adaptive_epsilon(self, reward_variance: float) -> float:
        """Compute adaptive epsilon based on reward variance.
        
        Formula: ε_p(t) = ε_min + γ * σ_R² / (σ_R² + c)
        High reward variance allows larger editing radius.
        """
        adaptive_component = self.gamma * reward_variance / (reward_variance + self.smooth_constant)
        epsilon_adaptive = self.epsilon_min + adaptive_component
        return epsilon_adaptive
    
    def apply_proximity_constraint(self, mu: torch.Tensor, reward_variance: Optional[float] = None) -> torch.Tensor:
        """Apply adaptive proximity constraint based on reward variance."""
        # Compute L2 norm for each sample
        norms = torch.norm(mu, dim=-1, keepdim=True)
        
        # Use adaptive epsilon if reward variance is provided
        if reward_variance is not None:
            epsilon_current = self.compute_adaptive_epsilon(reward_variance)
        else:
            epsilon_current = self.epsilon_p
        
        # Apply constraint: ||Δe||₂ ≤ ε_p(t)
        scaling_factor = torch.min(
            torch.ones_like(norms),
            epsilon_current / (norms + 1e-8)
        )
        
        # 额外的自适应约束：如果扰动太大，进一步缩小
        adaptive_factor = torch.exp(-norms / epsilon_current)  # 指数衰减
        final_scaling = scaling_factor * (0.1 + 0.9 * adaptive_factor)  # 极度保守的缩放，解决第一个epoch问题
        
        # Clone mu to avoid memory sharing issues with DDP
        return mu.clone() * final_scaling
    
    def compute_semantic_similarity(self, original_prompts: List[str], modified_prompts: List[str]) -> torch.Tensor:
        """Compute SBERT cosine similarity between original and modified prompts."""
        if self.sbert_model is None:
            # Return high similarity (no penalty) if SBERT is not available
            return torch.ones(len(original_prompts), device=self.device)
        
        with torch.no_grad():
            original_embeddings = self.sbert_model.encode(original_prompts, convert_to_tensor=True, device=self.device)
            modified_embeddings = self.sbert_model.encode(modified_prompts, convert_to_tensor=True, device=self.device)
            
            # Compute cosine similarity
            original_embeddings = F.normalize(original_embeddings, p=2, dim=-1)
            modified_embeddings = F.normalize(modified_embeddings, p=2, dim=-1)
            
            similarities = torch.sum(original_embeddings * modified_embeddings, dim=-1)
            return similarities
    
    def _add_simple_noise(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Add simple Gaussian noise to embeddings for diversity.
        
        向量空间里的轻微扰动会把目标点"推"进邻近 basin，
        矫正循环会走出另一条链路，输出自然多样
        """
        if not self.use_simple_noise:
            return embeddings
            
        noise = torch.randn_like(embeddings) * self.simple_noise_std
        noisy_embeddings = embeddings + noise
        # Normalize to preserve embedding magnitude
        return F.normalize(noisy_embeddings, p=2, dim=-1)
    
    def enable_simple_noise(self, std: float = 0):
        """Enable simple noise injection with specified standard deviation."""
        self.use_simple_noise = True
        self.simple_noise_std = std
        
    def disable_simple_noise(self):
        """Disable simple noise injection."""
        self.use_simple_noise = False
    
    def enable_modification_noise(self, max_std: float = 0.005):
        """Enable random noise injection during embedding modification for sampling diversity.
        
        Each sample will get a randomly selected noise strength between 0 and max_std.
        
        Args:
            max_std: Maximum standard deviation of Gaussian noise (actual noise will be random between 0 and this value)
        """
        self.use_modification_noise = True
        self.modification_noise_std = max_std
        print(f"[INFO] Modification noise enabled with random std in range [0, {max_std}]")
    
    def disable_modification_noise(self):
        """Disable random noise injection during embedding modification."""
        self.use_modification_noise = False
        print(f"[INFO] Modification noise disabled")
    
    def forward(self, prompts: List[str], reward_variance: Optional[float] = None) -> Tuple[List[str], torch.Tensor, torch.Tensor, dict]:
        """
        Forward pass of prompt editor.
        
        Returns:
            modified_prompts: List of modified prompt strings
            delta_e: Perturbation vectors (for regularization loss)
            original_embeddings: Original embeddings (for reconstruction loss)
            policy_info: Dictionary containing policy information including log probabilities
        """
        # Encode original prompts
        original_embeddings = self.encode_prompts(prompts)
        
        # Apply simple noise injection if enabled (minimal approach for diversity)
        if self.use_simple_noise:
            original_embeddings = self._add_simple_noise(original_embeddings)
       
        
        # Generate noise vector μ using encoder-decoder transformer (as described in paper)
        batch_size = original_embeddings.size(0)
        
        # Project to transformer dimension and add positional encoding
        x = self.input_projection(original_embeddings)  # [batch_size, d_model]
        x = x.unsqueeze(1).clone()  # [batch_size, 1, d_model] - treat as sequence length 1, clone to avoid memory sharing
        x = self.pos_encoding(x)
        
        # Encoder: process the reference prompt embedding
        memory = self.transformer_encoder(x)  # [batch_size, 1, d_model]
        
        # Decoder: generate noise vector
        # Initialize target sequence (can be learned parameter or zero)
        # Create new tensor instead of zeros_like to avoid memory sharing
        tgt = torch.zeros(x.size(), dtype=x.dtype, device=x.device)  # [batch_size, 1, d_model]
        tgt = self.pos_encoding(tgt)
        
        # Decode to get noise prediction
        decoded = self.transformer_decoder(tgt, memory)  # [batch_size, 1, d_model]
        
        # Project back to embedding dimension
        raw_mu = self.output_projection(decoded.squeeze(1).clone())  # [batch_size, embedding_dim]
        
        # 训练热身：前几个epoch使用更小的扰动
        warmup_epochs = 3
        if self.training_step < warmup_epochs * 1000:  # 假设每个epoch约1000步
            warmup_factor = 0.1 + 0.9 * (self.training_step / (warmup_epochs * 1000))
        else:
            warmup_factor = 1.0
        
        # Apply scaling and warmup to noise vector μ
        mu = (raw_mu * self.perturbation_scale * warmup_factor).clone()
        
        # 增加训练步数计数
        self.training_step += 1
        
        # Apply adaptive proximity constraint (P1 in paper: ||μ||₂ ≤ ε_p(t))
        mu_constrained = self.apply_proximity_constraint(mu, reward_variance)
        
        # Create modified embeddings: s_t - a_t (transition dynamics in paper)
        # Here we use addition instead of subtraction for consistency with existing code
        # Clone to avoid memory sharing issues with DDP
        modified_embeddings = original_embeddings.clone() + mu_constrained
        
        # Add random noise for diversity in each sampling (optional)
        if hasattr(self, 'use_modification_noise') and self.use_modification_noise:
            # Random noise strength for each sample between 0 and max_std
            max_noise_std = getattr(self, 'modification_noise_std', 0.005)
            # Generate random noise strength for each sample in the batch
            batch_size = modified_embeddings.size(0)
            random_noise_stds = torch.rand(batch_size, device=modified_embeddings.device) * max_noise_std
            
            # Generate base noise and scale it per sample
            base_noise = torch.randn_like(modified_embeddings)
            # Broadcast noise strength to match embedding dimensions [batch_size, 1] -> [batch_size, embedding_dim]
            noise_scales = random_noise_stds.unsqueeze(1).expand_as(modified_embeddings)
            modification_noise = base_noise * noise_scales
            
            modified_embeddings = modified_embeddings + modification_noise
            
            # Log noise levels for debugging
            print(f"[DEBUG] Random noise levels: {random_noise_stds.cpu().numpy()}")
        
        modified_embeddings = F.normalize(modified_embeddings, p=2, dim=-1)  # Normalize to unit norm
        
        # Decode back to text
        modified_prompts = self.decode_embeddings(modified_embeddings)
        
        # Compute semantic similarity for regularization
        semantic_similarities = self.compute_semantic_similarity(prompts, modified_prompts)
        
        # Compute log probabilities for policy gradient
        # For original policy gradient, we need the log probability of taking the action (mu_constrained)
        # We'll compute this as the negative L2 norm (treating it as a Gaussian policy)
        log_probs = -0.5 * torch.sum(mu_constrained ** 2, dim=-1)
        
        # Create policy info for training
        policy_info = {
            'log_prob': log_probs,
            'mu_raw': raw_mu,
            'mu_constrained': mu_constrained,
            'perturbation_scale': self.perturbation_scale,
            'warmup_factor': warmup_factor,
            'training_step': self.training_step,
            'semantic_similarities': semantic_similarities,
            'reward_variance': reward_variance,
            'epsilon_current': self.compute_adaptive_epsilon(reward_variance) if reward_variance is not None else self.epsilon_p
        }
        
        return modified_prompts, mu_constrained, original_embeddings, policy_info
    
    def compute_regularization_loss(self, 
                                    mu: torch.Tensor,
                                    original_embeddings: torch.Tensor,
                                    modified_prompts: List[str],
                                    original_prompts: List[str],
                                    semantic_similarities: Optional[torch.Tensor] = None,
                                    reward_variance: Optional[float] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute enhanced regularization loss with adaptive epsilon and semantic constraints.
        
        Components:
        1. Adaptive proximity: max(0, ||μ||₂ - ε_p(t))
        2. Semantic regularization: α * max(0, τ - sim_SBERT)
        3. Reconstruction consistency: β * MSE(e_orig + μ, e_recon)
        """
        # 1. Adaptive proximity regularization
        mu_norms = torch.norm(mu, dim=-1)  # ||μ||₂ for each sample
        
        if reward_variance is not None:
            epsilon_current = self.compute_adaptive_epsilon(reward_variance)
        else:
            epsilon_current = self.epsilon_p
            
        proximity_reg_loss = torch.mean(torch.clamp(mu_norms - epsilon_current, min=0.0))
        
        # 2. Semantic regularization
        semantic_reg_loss = torch.tensor(0.0, device=self.device)
        if semantic_similarities is not None:
            # Penalty for similarities below threshold
            semantic_violations = torch.clamp(self.semantic_threshold - semantic_similarities, min=0.0)
            semantic_reg_loss = self.semantic_alpha * torch.mean(semantic_violations)
        
        # 3. Reconstruction consistency loss for stability
        reconstructed_embeddings = self.encode_prompts(modified_prompts)
        reconstruction_loss = F.mse_loss(
            original_embeddings + mu,
            reconstructed_embeddings
        )
        
        # Combine losses
        total_reg_loss = proximity_reg_loss + semantic_reg_loss + 0.1 * reconstruction_loss
        
        # Return detailed breakdown
        loss_breakdown = {
            'proximity_reg': proximity_reg_loss.item(),
            'semantic_reg': semantic_reg_loss.item(),
            'reconstruction': reconstruction_loss.item(),
            'epsilon_current': epsilon_current,
            'mean_semantic_sim': semantic_similarities.mean().item() if semantic_similarities is not None else 1.0
        }
        
        return total_reg_loss, loss_breakdown
    
    def update_reward_variance_tracking(self, trajectories: List[dict]) -> float:
        """Update reward variance tracking for adaptive epsilon computation."""
        # Group trajectories by group_key (for GRPO) - use same grouping as advantage computation
        groups = defaultdict(list)
        for traj in trajectories:
            # Use group_key if available, otherwise fall back to first prompt (same logic as compute_grpo_advantages)
            group_key = traj.get('group_key', traj['prompts'][0] if isinstance(traj['prompts'], list) else str(traj['prompts']))
            rewards = traj['rewards']
            if isinstance(rewards, torch.Tensor):
                rewards = rewards.cpu().numpy()
            groups[group_key].extend(rewards.tolist() if hasattr(rewards, 'tolist') else [rewards])
        
        # Compute group-wise reward variance
        group_variances = []
        for group_key, group_rewards in groups.items():
            if len(group_rewards) > 1:
                group_var = np.var(group_rewards)
                group_variances.append(group_var)
                self.group_reward_stats[group_key] = {
                    'mean': np.mean(group_rewards),
                    'var': group_var,
                    'count': len(group_rewards)
                }
        
        # Return overall reward variance for adaptive epsilon
        if group_variances:
            return float(np.mean(group_variances))
        else:
            return 0.01  # Default small variance
    
    def compute_grpo_advantages(self, trajectories: List[dict]) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        """Compute true GRPO advantages with gradient scaling and clipping."""
        # Group trajectories by group_key (original prompt)
        groups = defaultdict(list)
        trajectory_to_group = {}
        
        for i, traj in enumerate(trajectories):
            # Use group_key if available, otherwise fall back to first prompt
            group_key = traj.get('group_key', traj['prompts'][0] if isinstance(traj['prompts'], list) else str(traj['prompts']))
            groups[group_key].append(i)
            trajectory_to_group[i] = group_key
        
        # Debug logging: verify GRPO grouping is working
        print(f"[GRPO DEBUG] compute_grpo_advantages: {len(trajectories)} trajectories grouped into {len(groups)} groups")
        if len(groups) > 1:
            group_sizes = {g: len(indices) for g, indices in groups.items()}
            print(f"[GRPO DEBUG] Group sizes: {group_sizes}")
            print(f"[GRPO DEBUG] Group keys: {list(groups.keys())}")
        else:
            print(f"[GRPO DEBUG] WARNING: Only {len(groups)} group(s) found - GRPO may not be working properly")
            print(f"[GRPO DEBUG] All trajectories have group_key: {list(groups.keys())[0] if groups else 'None'}")
            print(f"[GRPO DEBUG] This suggests batch_size=1 or all prompts are identical")
        
        # Compute group-wise advantages with GRPO scaling
        advantages = {}
        grpo_stats = {
            'num_groups': len(groups),
            'total_clip_frac': 0.0,
            'mean_group_size': 0.0
        }
        
        total_clipped = 0
        total_samples = 0
        group_sizes = []
        
        for group_key, traj_indices in groups.items():
            group_rewards = []
            for idx in traj_indices:
                rewards = trajectories[idx]['rewards']
                if isinstance(rewards, torch.Tensor):
                    group_rewards.extend(rewards.cpu().tolist())
                else:
                    group_rewards.append(rewards)
            
            # Compute group statistics
            group_mean = np.mean(group_rewards)
            group_std = np.std(group_rewards) + 1e-8
            group_sizes.append(len(traj_indices))
            
            # Collect all advantages for this group first
            group_advantages = []
            for idx in traj_indices:
                rewards = trajectories[idx]['rewards']
                if isinstance(rewards, torch.Tensor):
                    reward_values = rewards.cpu().numpy()
                else:
                    reward_values = np.array([rewards])
                
                # Z-score normalization: A_i = (R_i - μ_grp) / (σ_grp + ε)
                z_score_advantages = (reward_values - group_mean) / group_std
                group_advantages.append(z_score_advantages)
            
            # Convert to tensor for GRPO scaling
            group_advantages_tensor = torch.tensor(
                np.concatenate(group_advantages), 
                device=self.device, 
                dtype=torch.float32
            )
            
            # GRPO gradient scaling: min(1, c * |A_i| / |A_bar|)
            # where A_bar is the mean absolute advantage in the group
            abs_advantages = torch.abs(group_advantages_tensor)
            mean_abs_advantage = abs_advantages.mean() + 1e-8
            
            # Compute scaling factors
            scale_factors = torch.clamp(
                self.grpo_clip * abs_advantages / mean_abs_advantage, 
                max=1.0
            )
            
            # Apply scaling to advantages
            scaled_advantages = group_advantages_tensor * scale_factors
            
            # Track clipping statistics
            clipped_mask = scale_factors < 1.0
            total_clipped += clipped_mask.sum().item()
            total_samples += len(scale_factors)
            
            # Assign scaled advantages back to trajectories
            start_idx = 0
            for idx in traj_indices:
                rewards = trajectories[idx]['rewards']
                if isinstance(rewards, torch.Tensor):
                    adv_length = len(rewards)
                else:
                    adv_length = 1
                
                advantages[idx] = scaled_advantages[start_idx:start_idx + adv_length]
                start_idx += adv_length
        
        # Compute overall statistics
        grpo_stats['total_clip_frac'] = total_clipped / total_samples if total_samples > 0 else 0.0
        grpo_stats['mean_group_size'] = np.mean(group_sizes) if group_sizes else 0.0
        
        return advantages, grpo_stats
    
    def compute_policy_loss(self, 
                           trajectories: List[dict],
                           baseline_value: float = 0.0) -> Tuple[torch.Tensor, dict]:
        """
        Compute true GRPO policy loss with group-wise advantages and gradient scaling.
        
        Args:
            trajectories: List of trajectory dictionaries containing:
                - 'prompts': Original prompts
                - 'states': Original embeddings
                - 'actions': Perturbation vectors (mu)
                - 'rewards': Episode rewards
                - 'log_probs': Log probabilities of actions
                - 'policy_info': Additional policy information
            baseline_value: Baseline value for variance reduction (optional)
        
        Returns:
            policy_loss: True GRPO policy loss
            metrics: Dictionary of training metrics including GRPO stats
        """
        # Update reward variance tracking
        reward_variance = self.update_reward_variance_tracking(trajectories)
        
        # Compute GRPO advantages with scaling
        grpo_advantages, grpo_stats = self.compute_grpo_advantages(trajectories)
        
        total_loss = 0.0
        total_regularization = 0.0
        batch_rewards = []
        batch_log_probs = []
        reg_breakdown_accumulator = defaultdict(list)
        
        for i, trajectory in enumerate(trajectories):
            # Extract trajectory components
            rewards = trajectory['rewards']
            log_probs = trajectory['log_probs']
            actions = trajectory['actions']
            
            # Convert to tensors if needed (preserve gradients)
            if not isinstance(rewards, torch.Tensor):
                rewards = torch.tensor(rewards, device=self.device, dtype=torch.float32)
            if not isinstance(log_probs, torch.Tensor):
                log_probs = torch.tensor(log_probs, device=self.device, dtype=torch.float32, requires_grad=True)
            else:
                # Ensure log_probs tensor is on correct device and has gradients
                log_probs = log_probs.to(device=self.device, dtype=torch.float32)
                if not log_probs.requires_grad:
                    log_probs.requires_grad_(True)
            
            # Use GRPO advantages (group-based z-score with scaling)
            advantages = grpo_advantages.get(i, torch.zeros_like(rewards))
            
            if self.use_true_grpo:
                # True GRPO loss: -ratio * scaled_advantages
                # ratio = exp(log_π(a|s) - log_π_old(a|s)) 
                # For simplicity, we assume log_π_old ≈ log_π (ratio ≈ 1) since we update frequently
                # In practice, you might want to store old log_probs for proper ratio computation
                ratio = torch.exp(log_probs - log_probs.detach())  # ratio = 1, but keeps gradient
                policy_loss_traj = -(ratio * advantages).sum()
            else:
                # Fallback to REINFORCE with GRPO advantages
                policy_loss_traj = -(log_probs * advantages).sum()
            
            total_loss += policy_loss_traj
            
            # Enhanced regularization loss
            if 'policy_info' in trajectory and 'mu_constrained' in trajectory['policy_info']:
                mu = trajectory['policy_info']['mu_constrained']
                semantic_similarities = trajectory['policy_info'].get('semantic_similarities')
                
                reg_loss, reg_breakdown = self.compute_regularization_loss(
                    mu, 
                    trajectory['states'],
                    trajectory.get('modified_prompts', []),
                    trajectory['prompts'],
                    semantic_similarities,
                    reward_variance
                )
                total_regularization += reg_loss
                
                # Accumulate regularization breakdown
                for key, value in reg_breakdown.items():
                    reg_breakdown_accumulator[key].append(value)
            
            batch_rewards.extend(rewards.cpu().tolist() if rewards.dim() > 0 else [rewards.item()])
            batch_log_probs.extend(log_probs.cpu().tolist() if log_probs.dim() > 0 else [log_probs.item()])
        
        # Average over batch
        batch_size = len(trajectories)
        if batch_size > 0:
            policy_loss = total_loss / batch_size
            regularization_loss = total_regularization / batch_size
            total_loss_combined = policy_loss + 0.1 * regularization_loss  # Weight regularization
        else:
            policy_loss = torch.tensor(0.0, device=self.device)
            regularization_loss = torch.tensor(0.0, device=self.device)
            total_loss_combined = torch.tensor(0.0, device=self.device)
        
        # Compute enhanced metrics including GRPO statistics
        metrics = {
            'policy_loss': policy_loss.item() if hasattr(policy_loss, 'item') else policy_loss,
            'regularization_loss': regularization_loss.item() if hasattr(regularization_loss, 'item') else regularization_loss,
            'total_loss': total_loss_combined.item() if hasattr(total_loss_combined, 'item') else total_loss_combined,
            'mean_reward': np.mean(batch_rewards) if batch_rewards else 0.0,
            'mean_advantage': np.mean([adv.mean().item() for adv in grpo_advantages.values()]) if grpo_advantages else 0.0,
            'mean_log_prob': np.mean(batch_log_probs) if batch_log_probs else 0.0,
            'baseline_value': baseline_value,
            'reward_variance': reward_variance,
            'epsilon_adaptive': self.compute_adaptive_epsilon(reward_variance),
            # GRPO-specific metrics
            'grpo_clip_frac': grpo_stats['total_clip_frac'],
            'grpo_num_groups': grpo_stats['num_groups'],
            'grpo_mean_group_size': grpo_stats['mean_group_size'],
            'grpo_clip_threshold': self.grpo_clip,
            'use_true_grpo': self.use_true_grpo,
            'k_samples': self.k_samples
        }
        
        # Add regularization breakdown to metrics
        for key, values in reg_breakdown_accumulator.items():
            metrics[f'reg_{key}'] = np.mean(values) if values else 0.0
        
        return total_loss_combined, metrics
    
    def update_policy(self, trajectories: List[dict], optimizer: torch.optim.Optimizer, baseline_value: float = 0.0) -> dict:
        """
        Update policy using policy gradient (REINFORCE).
        
        Args:
            trajectories: List of trajectory dictionaries
            optimizer: Policy optimizer
            baseline_value: Baseline for variance reduction
            
        Returns:
            metrics: Dictionary of training metrics
        """
        # Compute policy loss
        total_loss, metrics = self.compute_policy_loss(trajectories, baseline_value)
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        
        # Update parameters
        optimizer.step()
        
        return metrics