# Configuration Guide

## Training Parameters

### Basic Parameters
```python
batch_size = 32
block_size = 1024
max_iters = 600000
learning_rate = 3e-4
min_lr = learning_rate/10
warmup_iters = 2000
grad_clip = 1.0
grad_accum = 4
```

### Model Architecture
```python
num_experts = 8  # Number of experts in MoE layers
expert_capacity = 1.25  # Capacity factor for load balancing
num_tokens_predict = 2  # Multi-token prediction
```

## Hardware-Specific Settings

### CUDA Configuration
- FP8 training for H100
- Mixed precision for other GPUs
- Memory efficient attention

### Apple Silicon
- MPS backend optimization
- Metal performance shaders
- Float16 precision

### CPU Training
- Thread optimization
- Memory-efficient attention
- BFloat16 support
