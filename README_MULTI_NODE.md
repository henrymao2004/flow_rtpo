# 多机多卡训练说明

## 修复内容

已修复以下多机多卡训练的关键问题：

1. **奖励广播**: 使用 `torch.distributed.broadcast` 将主进程计算的奖励广播到所有进程
2. **全局统计**: 使用 `accelerator.gather()` 收集所有进程的奖励和毒性分数，计算全局统计
3. **优势计算**: 基于全局统计计算优势函数，确保所有进程使用相同的归一化参数
4. **奖励函数并行**: 在分布式训练中关闭奖励函数的多GPU并行，避免资源冲突

## 启动方式

### 单机多卡
```bash
accelerate launch --num_processes 8 --mixed_precision bf16 scripts/train_flow_rtpo.py --config config/flow_rtpo.py
```

### 多机多卡 (2节点，每节点8卡)

**节点0 (主节点)**:
```bash
accelerate launch \
  --num_processes 8 \
  --num_machines 2 \
  --machine_rank 0 \
  --main_process_ip <node0_ip> \
  --main_process_port 29500 \
  --mixed_precision bf16 \
  scripts/train_flow_rtpo.py \
  --config config/flow_rtpo.py
```

**节点1 (工作节点)**:
```bash
accelerate launch \
  --num_processes 8 \
  --num_machines 2 \
  --machine_rank 1 \
  --main_process_ip <node0_ip> \
  --main_process_port 29500 \
  --mixed_precision bf16 \
  scripts/train_flow_rtpo.py \
  --config config/flow_rtpo.py
```

## 关键修复点

### 1. 奖励广播
```python
# 修复前: 非主进程使用0值
# 修复后: 使用torch.distributed.broadcast广播奖励
import torch.distributed as dist
dist.broadcast(rewards_tensor, src=0)
dist.broadcast(toxicity_tensor, src=0)
```

### 2. 全局统计收集
```python
# 修复前: 使用本地统计
# 修复后: 收集全局统计
rewards_global = accelerator.gather(rewards_local)
toxicity_global = accelerator.gather(toxicity_local)
```

### 3. 优势计算
```python
# 修复前: 基于本地均值/方差
# 修复后: 基于全局统计
global_mean = float(rewards_global.mean())
global_std = float(rewards_global.std(unbiased=False).clamp_min(1e-4))
advantages = (np.array(all_rewards) - global_mean) / global_std
```

### 4. 奖励函数配置
```python
# 修复前: 可能启用多GPU并行
# 修复后: 在分布式训练中关闭
reward_fn = toxicity_reward_function(
    device=accelerator.device,
    vlm_model=config.target_vlm,
    w_cvar=config.toxicity_reward.w_cvar,
    w_quality=config.toxicity_reward.w_quality,
    use_local_models=config.get('use_local_models', False),
    clip_model_path=config.get('clip_model', None),
    use_multi_gpu=False,      # ★ 推荐在分布式训练中关闭
    chunk_size=8
)
```

## 注意事项

1. **网络配置**: 确保所有节点可以通过指定端口通信
2. **环境一致性**: 所有节点应使用相同的Python环境和依赖版本
3. **数据分片**: `accelerator.prepare(dataloader)` 会自动处理数据分片
4. **日志同步**: 只有主进程会保存日志和检查点，其他进程会等待同步

## 性能优化

- 奖励计算只在主进程进行，避免重复计算
- 使用 `torch.distributed.broadcast` 高效同步奖励
- 全局统计使用 `accelerator.gather()` 收集，确保训练信号一致
- 关闭奖励函数的多GPU并行，避免与训练并行冲突 