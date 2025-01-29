# Hardware Requirements

## Supported Platforms

### NVIDIA GPUs
- H100 (optimal performance with FP8 training)
- Consumer GPUs (RTX 30/40 series)
- Older GPUs (with reduced features)

### Apple Silicon
- M1/M2/M3 series chips
- Uses MPS backend
- Optimized for Metal performance

### CPU Training
- Supports multi-threading
- Uses bfloat16 when available
- Reduced performance compared to GPU

## Memory Requirements

Minimum requirements per configuration:
- H100: 32GB+ VRAM
- Consumer GPU: 8GB+ VRAM
- Apple Silicon: 16GB+ unified memory
- CPU: 32GB+ RAM recommended

## Storage

- 20GB+ free space for model checkpoints
- Additional space for training data
