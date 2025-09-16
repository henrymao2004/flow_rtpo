# Flow-RTPO 模型和数据集配置指南

## 📋 完整模型和数据集清单

### 🤖 **模型列表**

| 模型类型 | 用途 | HuggingFace 名称 | 本地路径（use_local=True） | 加载方式 |
|---------|------|-----------------|------------------------|---------|
| **SD3** | 主要扩散模型 | `stabilityai/stable-diffusion-3.5-medium` | `stable-diffusion-3.5-medium` | `StableDiffusion3Pipeline.from_pretrained()` |
| **LLaVA** | 视觉语言模型（VLM） | `llava-hf/llava-v1.6-mistral-7b-hf` | `llava-v1.6-mistral-7b-hf` | `LlavaNextForConditionalGeneration.from_pretrained()` |
| **CLIP** | 图像-文本相似度 | `openai/clip-vit-large-patch14-336` | `clip-vit-large-patch14-336` | `CLIPModel.from_pretrained()` |
| **SBERT** | 语义相似度 | `sentence-transformers/all-MiniLM-L6-v2` | `all-MiniLM-L6-v2` | `SentenceTransformer()` |
| **GTR-T5** | 文本嵌入编码器 | `sentence-transformers/gtr-t5-base` | `gtr-t5-base` | `AutoModel.from_pretrained().encoder` |
| **vec2text** | 文本反演模型 | `gtr-base` | `gtr-base` | `vec2text.load_pretrained_corrector()` + 环境变量 |
| **Detoxify** | 毒性检测 | `original` | `detoxify-original` | `Detoxify()` + 环境变量 | checkpoint + bert

### 📚 **数据集列表**

| 数据集 | 用途 | HuggingFace 名称 | 本地路径（use_local=True） | 加载方式 |
|-------|------|-----------------|------------------------|---------|
| **Real Toxicity Prompts** | 训练数据集 | `allenai/real-toxicity-prompts` | `real-toxicity-prompts` | `load_dataset()` 或 JSON文件 |

---

## 🗂️ **本地目录结构**

### **基础路径配置**
```python
# 在 config/flow_rtpo.py 中配置
config.model_loading.local_base_path = "/mnt/data/group/zhaoliangjie/ICLR-work/"
config.model_loading.use_local = True
```

### **完整目录结构**
```
/mnt/data/group/zhaoliangjie/ICLR-work/
├── stable-diffusion-3.5-medium/           # SD3 扩散模型
│   ├── model_index.json
│   ├── scheduler/
│   │   └── scheduler_config.json
│   ├── text_encoder/
│   │   ├── config.json
│   │   └── pytorch_model.bin
│   ├── text_encoder_2/
│   │   ├── config.json
│   │   └── pytorch_model.bin
│   ├── text_encoder_3/
│   │   ├── config.json
│   │   └── pytorch_model.bin
│   ├── tokenizer/
│   │   ├── tokenizer_config.json
│   │   ├── tokenizer.json
│   │   └── vocab.json
│   ├── tokenizer_2/
│   ├── tokenizer_3/
│   ├── transformer/
│   │   ├── config.json
│   │   └── diffusion_pytorch_model.safetensors
│   └── vae/
│       ├── config.json
│       └── diffusion_pytorch_model.safetensors
│
├── llava-v1.6-mistral-7b-hf/              # LLaVA 视觉语言模型
│   ├── config.json
│   ├── generation_config.json
│   ├── preprocessor_config.json
│   ├── pytorch_model-00001-of-00004.bin
│   ├── pytorch_model-00002-of-00004.bin
│   ├── pytorch_model-00003-of-00004.bin
│   ├── pytorch_model-00004-of-00004.bin
│   ├── pytorch_model.bin.index.json
│   ├── special_tokens_map.json
│   ├── tokenizer.json
│   ├── tokenizer.model
│   └── tokenizer_config.json
│
├── clip-vit-large-patch14-336/             # CLIP 模型
│   ├── config.json
│   ├── preprocessor_config.json
│   ├── pytorch_model.bin
│   ├── tokenizer_config.json
│   ├── tokenizer.json
│   └── vocab.json
│
├── all-MiniLM-L6-v2/                       # SBERT 模型
│   ├── config.json
│   ├── config_sentence_transformers.json
│   ├── pytorch_model.bin
│   ├── sentence_bert_config.json
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   ├── tokenizer.json
│   └── vocab.txt
│
├── gtr-t5-base/                            # GTR-T5 编码器
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── special_tokens_map.json
│   ├── spiece.model
│   ├── tokenizer_config.json
│   └── tokenizer.json
│
├── gtr-base/                               # vec2text 模型缓存
│   ├── models--ielabgroup--vec2text_gtr-base-st_corrector/
│   │   ├── blobs/
│   │   ├── refs/
│   │   └── snapshots/
│   │       └── [commit_hash]/
│   │           ├── config.json
│   │           ├── pytorch_model.bin
│   │           ├── tokenizer_config.json
│   │           ├── tokenizer.json
│   │           └── vocab.json
│   └── models--ielabgroup--vec2text_gtr-base-st_inversion/
│       ├── blobs/
│       ├── refs/
│       └── snapshots/
│           └── [commit_hash]/
│               ├── config.json
│               ├── pytorch_model.bin
│               └── ...
│
├── detoxify-original/                      # Detoxify 模型缓存
│   ├── hub/
│   │   └── checkpoints/
│   │       └── toxic_original-c1212f89.ckpt
│   └── huggingface_cache/
│       └── models--unitary--toxic-bert-base/
│           ├── blobs/
│           ├── refs/
│           └── snapshots/
│               └── [commit_hash]/
│                   ├── config.json
│                   ├── pytorch_model.bin
│                   └── ...
│
└── real-toxicity-prompts/                  # RTP 数据集
    ├── dataset_info.json
    ├── state.json
    └── data/
        └── train-00000-of-00001.parquet
    # 或者 JSON 格式：
    # └── data.json
```

---

## 🔧 **本地加载机制详解**

### **1. 标准HuggingFace模型加载（SD3, LLaVA, CLIP, SBERT, GTR）**
```python
def _get_model_path(self, model_type: str, default_path: str) -> str:
    if self.use_local:
        # 从配置获取本地模型名称
        local_name = self.local_models.get(model_type, default_path.split('/')[-1])
        # 组合基础路径和模型名称
        return os.path.join(self.local_base_path, local_name)
    else:
        return self.hf_models.get(model_type, default_path)

# 使用方式
model = AutoModel.from_pretrained(model_path)
```

### **2. vec2text 环境变量加载**
```python
# 设置环境变量指向本地缓存
if self.use_local:
    # 从配置获取vec2text本地路径
    # 实际值："/mnt/data/group/zhaoliangjie/ICLR-work/gtr-base"
    vec2text_local_path = self.local_models.get('vec2text', 'gtr-base')
    
    # 如果不是绝对路径，则与基础路径组合
    if not vec2text_local_path.startswith('/'):
        vec2text_local_path = os.path.join(self.local_base_path, vec2text_local_path)
    
    # 设置HuggingFace缓存环境变量
    os.environ['HF_HOME'] = vec2text_local_path
    os.environ['HF_HUB_CACHE'] = vec2text_local_path
    os.environ['TRANSFORMERS_CACHE'] = vec2text_local_path

# 使用标准API（会从环境变量指定的缓存加载）
self.vec2text_corrector = vec2text.load_pretrained_corrector('gtr-base')

# 恢复环境变量（确保不影响其他模块）
```

### **3. Detoxify 双重缓存加载**
```python
# PyTorch Hub 缓存（存储.ckpt检查点）
torch_hub_base = "/mnt/data/group/zhaoliangjie/ICLR-work/detoxify-original"
os.environ['TORCH_HOME'] = torch_hub_base

# HuggingFace 缓存（存储BERT模型）
hf_cache_dir = "/mnt/data/group/zhaoliangjie/ICLR-work/detoxify-original/huggingface_cache"
os.environ['HF_HOME'] = hf_cache_dir
os.environ['TRANSFORMERS_CACHE'] = hf_cache_dir

# 使用标准API
self.detoxify = Detoxify('original', device=self.device)
```

### **4. RTP数据集加载**
```python
# 方式1：HuggingFace datasets格式
dataset = load_dataset(local_dataset_path, cache_dir=self.cache_dir)

# 方式2：JSON文件格式
json_path = os.path.join(local_dataset_path, "data.json")
with open(json_path, 'r') as f:
    train_data = json.load(f)
```

---

## ⚠️ **特殊注意事项**

### **1. vec2text 特殊处理**
- `vec2text.load_pretrained_corrector()` 只接受模型名称，不接受路径
- 通过设置环境变量 `HF_HOME`, `HF_HUB_CACHE`, `TRANSFORMERS_CACHE` 重定向缓存
- 需要包含两个模型：`corrector` 和 `inversion`

### **2. Detoxify 双重缓存**
- PyTorch Hub 缓存：存储 `.ckpt` 检查点文件
- HuggingFace 缓存：存储 BERT 模型文件
- 需要正确设置 `TORCH_HOME` 和 `HF_HOME` 环境变量

### **3. 模型文件完整性检查**
```bash
# 检查关键文件是否存在
ls -la /mnt/data/group/zhaoliangjie/ICLR-work/stable-diffusion-3.5-medium/model_index.json
ls -la /mnt/data/group/zhaoliangjie/ICLR-work/llava-v1.6-mistral-7b-hf/config.json
ls -la /mnt/data/group/zhaoliangjie/ICLR-work/detoxify-original/hub/checkpoints/toxic_original-c1212f89.ckpt
```

### **4. 回退机制**
所有模型都实现了回退机制：
1. 本地路径不存在 → 回退到 HuggingFace
2. 本地加载失败 → 回退到 HuggingFace
3. HuggingFace 加载失败 → 抛出异常或使用 mock 数据

---

## 🚀 **快速部署检查清单**

### **配置文件设置**
- [ ] `config.model_loading.use_local = True`
- [ ] `config.model_loading.local_base_path = "/mnt/data/group/zhaoliangjie/ICLR-work/"`
- [ ] 所有 `config.model_loading.local_models.*` 路径正确

### **目录结构验证**
- [ ] 基础目录存在：`/mnt/data/group/zhaoliangjie/ICLR-work/`
- [ ] 所有模型子目录存在且包含必要文件
- [ ] Detoxify 检查点存在：`detoxify-original/hub/checkpoints/toxic_original-c1212f89.ckpt`
- [ ] vec2text 缓存目录结构正确

### **权限检查**
- [ ] 目录读取权限
- [ ] 模型文件读取权限
- [ ] 环境变量设置权限

### **测试验证**
```python
# 测试各模型加载
from flow_grpo.prompt_editor import PromptEditorPolicy
from flow_grpo.toxicity_rewards import ToxicityRewardSystem
from flow_grpo.rtp_dataset import RealToxicityPromptsDataset

# 验证模型加载
policy = PromptEditorPolicy(use_local=True, local_base_path="/mnt/data/group/zhaoliangjie/ICLR-work/")
reward_system = ToxicityRewardSystem(use_local=True, local_base_path="/mnt/data/group/zhaoliangjie/ICLR-work/")
dataset = RealToxicityPromptsDataset(use_local=True, local_base_path="/mnt/data/group/zhaoliangjie/ICLR-work/")
```

---

## 📞 **故障排除**

### **常见问题**
1. **模型文件不存在**：检查路径配置和文件完整性
2. **权限不足**：确保读取权限正确设置
3. **环境变量冲突**：检查是否有其他程序设置了相同环境变量
4. **版本兼容性**：确保本地模型版本与代码兼容

### **调试日志**
启动训练时注意观察以下日志：
```
[VEC2TEXT] Using local cache directory: /mnt/data/group/zhaoliangjie/ICLR-work/gtr-base
[DETOXIFY INIT] Found checkpoint: /mnt/data/group/zhaoliangjie/ICLR-work/detoxify-original/hub/checkpoints/toxic_original-c1212f89.ckpt
[CLIP LOCAL] Successfully loaded CLIP model from local path
[LOCAL] Successfully loaded local dataset with XXX samples
```

这个配置指南涵盖了 Flow-RTPO 项目所需的所有模型和数据集的本地部署方案。