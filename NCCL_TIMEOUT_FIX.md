# NCCL超时问题关键修复

## 问题描述
在Flow-RTPO多GPU训练中遇到NCCL超时错误：
```
NCCL error in: /opt/conda/lib/python3.10/site-packages/torch/lib/libtorch_cuda.so, unhandled cuda error, NCCL version 2.18.3
ncclInternalError: Internal check failed.
Last error:
NCCL error in: /opt/conda/lib/python3.10/site-packages/torch/lib/libtorch_cuda.so, unhandled cuda error, NCCL version 2.18.3
ncclTimeoutError: Call to NCCL function timed out, NCCL version 2.18.3
```

## 根本原因
NCCL的`ALLREDUCE`集体通信操作在推理阶段超时，主要原因是：
1. **推理阶段不需要梯度同步**：采样时只需要前向推理，不需要反向传播
2. **DDP集体通信是多余的**：在推理时进行参数同步是不必要的开销
3. **网络通信压力**：多GPU间的同步操作导致超时

## 关键修复

### 1. 在推理时禁用DDP同步（最重要）
```python
# 修复前
with torch.no_grad():
    modified_prompts, prompt_deltas, original_embeddings, policy_info = prompt_editor(prompts_expanded, reward_variance)

# 修复后 - 关键修复
with torch.no_grad():
    with prompt_editor.no_sync():  # 关键修复：禁用DDP同步
        modified_prompts, prompt_deltas, original_embeddings, policy_info = prompt_editor(prompts_expanded, reward_variance)
```

### 2. KL正则化部分的修复
```python
# 修复前
with torch.no_grad():
    with pipeline.transformer.module.disable_adapter():
        _, _, prev_sample_mean_ref, _ = compute_log_prob(...)

# 修复后
with torch.no_grad():
    with pipeline.transformer.no_sync():  # 避免DDP同步问题
        with pipeline.transformer.module.disable_adapter():
            _, _, prev_sample_mean_ref, _ = compute_log_prob(...)
```

### 3. NCCL环境变量配置
在`scripts/single_node/flow_rtpo.sh`中添加：
```bash
# NCCL Configuration - Fix timeout issues
export NCCL_TIMEOUT=1800  # 30 minutes timeout
export NCCL_IB_TIMEOUT=1800
export NCCL_IB_RETRY_CNT=7
export NCCL_DEBUG=INFO
export NCCL_BUFFSIZE=2097152
export NCCL_NTHREADS=8
export NCCL_RINGS=4
```

## 为什么这个修复最关键

1. **直接解决根本问题**：`no_sync()`跳过DDP的集体通信操作
2. **推理阶段不需要同步**：采样时只需要前向推理，不需要参数同步
3. **避免不必要的网络开销**：减少GPU间的通信压力
4. **保持训练阶段同步**：只在推理时禁用同步，训练时仍然正常同步

## 修复效果
- 消除NCCL超时错误
- 提高推理速度
- 减少网络通信开销
- 保持训练的正确性

## 注意事项
- 只在推理阶段使用`no_sync()`
- 训练阶段仍然需要正常的梯度同步
- 确保所有进程在关键点同步（使用`accelerator.wait_for_everyone()`） 