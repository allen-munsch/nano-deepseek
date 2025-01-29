# Training Guide

## Basic Training

The training script supports various hardware configurations and includes optimizations for different platforms.

### Default Training

```bash
python train.py
```

### Distributed Training

For multi-GPU training:

```bash
torchrun --nproc_per_node=NUM_GPUS train.py
```

## Configuration Options

Key parameters in train.py:

- `batch_size`: Default 32
- `block_size`: Context window size (1024)
- `learning_rate`: Initial learning rate (3e-4)
- `max_iters`: Maximum training iterations
- `grad_clip`: Gradient clipping value
- `eval_interval`: Evaluation frequency
- `num_tokens_predict`: Multi-token prediction count

## Training Data

Place your training data in the `data` directory:
- `train.bin`: Training dataset
- `val.bin`: Validation dataset

## Checkpoints

Checkpoints are saved to the `out` directory. Resume training using:

```bash
python train.py --init_from=resume
```

## Monitoring

The training script outputs:
- Loss metrics
- Training speed (tokens/second)
- GPU utilization
- Memory usage
