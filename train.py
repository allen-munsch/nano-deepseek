import os
import time
import math
import pickle
from contextlib import nullcontext
import platform

import numpy as np
import torch
from torch.nn import functional as F
import torch.backends.mps
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import tiktoken
from torch.cuda.amp import autocast
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

# Training hyperparameters
batch_size = 32
block_size = 1024
max_iters = 600000
learning_rate = 3e-4
min_lr = learning_rate/10
warmup_iters = 2000
grad_clip = 1.0
grad_accum = 4
dtype = 'float16'
# Determine device and precision based on hardware
if torch.cuda.is_available():
    device_type = 'cuda'
    # Check if we have H100
    has_h100 = torch.cuda.get_device_capability()[0] >= 9
    # Default to FP16 for consumer GPUs
    default_dtype = torch.float8 if has_h100 else torch.float16
elif torch.backends.mps.is_available():
    device_type = 'mps'  # Apple Silicon
    default_dtype = torch.float16
else:
    device_type = 'cpu'
    # Use bfloat16 for CPU as it's more numerically stable
    default_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32

ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=default_dtype)

# Enable memory efficient attention if available
if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
    torch.backends.cuda.enable_flash_sdp(True)

def setup_device():
    """Setup the device for training"""
    global device
    if device_type == 'cuda':
        device = torch.device('cuda')
        # Enable TF32 for better performance on Ampere+ GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    elif device_type == 'mps':
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
        # Enable memory efficient attention for CPU training
        if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            torch.set_num_threads(os.cpu_count())

def create_config():
    """Create model configuration"""
    return {
        'n_layer': 32,
        'n_head': 32,
        'n_embd': 4096,
        'vocab_size': 32000,
        'block_size': block_size,
        'dropout': 0.0,
    }

def forward(idx, targets, config):
    """Forward pass of the model"""
    # This is just a placeholder - actual implementation would depend on your model architecture
    B, T = idx.shape
    tok_emb = torch.randn((B, T, config['n_embd']), device=device)
    logits = torch.randn((B, T, config['vocab_size']), device=device)
    if targets is None:
        return logits, None
    else:
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return logits, loss

# -----------------------------------------------------------------------------
# Training settings from DeepSeek paper
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'

# Model architecture settings
num_experts = 8  # Number of experts in MoE layers
expert_capacity = 1.25  # Capacity factor for load balancing
num_tokens_predict = 2  # Multi-token prediction

# DDP settings
backend = 'nccl'

# setup distributed training
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    
    # Configure precision and optimizations based on hardware
    if device_type == 'cuda':
        torch.set_float32_matmul_precision('high')
        if torch.cuda.get_device_capability()[0] >= 9:  # H100
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
            torch.backends.cuda.matmul.allow_fp8_training = True
        else:  # Consumer GPUs
            # Enable AMP for better memory efficiency
            torch.cuda.empty_cache()
            torch.cuda.memory.set_per_process_memory_fraction(0.95)
else:
    master_process = True
    ddp_world_size = 1
    setup_device()

# -----------------------------------------------------------------------------
tokens_per_iter = grad_accum * ddp_world_size * batch_size * block_size
if master_process:
    print(f"tokens per iteration will be: {tokens_per_iter:,}")
    os.makedirs(out_dir, exist_ok=True)

torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# -----------------------------------------------------------------------------
# Data loading
class DataLoader:
    def __init__(self, split='train'):
        self.split = split
        # Walk through data directory to find .bin files
        data_dir = 'data'
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file == f'{split}.bin':
                    filename = os.path.join(root, file)
                    self.data = np.memmap(filename, dtype=np.uint16, mode='r')
                    return
        raise FileNotFoundError(f"Could not find {split}.bin in {data_dir} directory tree")
    
    def get_batch(self):
        ix = torch.randint(len(self.data) - block_size - num_tokens_predict, (batch_size,))
        x = torch.stack([torch.from_numpy((self.data[i:i+block_size]).astype(np.int64)) for i in ix])
        # Get multiple next tokens for MTP training
        y = torch.stack([torch.from_numpy((self.data[i+1:i+1+block_size+num_tokens_predict]).astype(np.int64)) 
                        for i in ix])
        if device_type == 'cuda':
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x = x.to(device)
            y = y.to(device)
        return x, y

# -----------------------------------------------------------------------------
# Initialize the model
config = create_config()
if init_from == 'scratch':
    print("Initializing a new model from scratch")
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # Load the checkpoint
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    config.update(checkpoint['model'])
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']

# Convert config dict to nn.Module for DDP
class ModelWrapper(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        # Create actual trainable parameters
        self.token_embedding = torch.nn.Embedding(config['vocab_size'], config['n_embd'])
        self.position_embedding = torch.nn.Parameter(torch.zeros(1, config['block_size'], config['n_embd']))
        self.dropout = torch.nn.Dropout(config['dropout'])
        self.ln_f = torch.nn.LayerNorm(config['n_embd'])
        self.head = torch.nn.Linear(config['n_embd'], config['vocab_size'], bias=False)
        self.config = config
    
    def forward(self, idx, targets=None):
        B, T = idx.shape
        # Get token embeddings
        tok_emb = self.token_embedding(idx)
        # Add positional embeddings
        pos_emb = self.position_embedding[:, :T, :]
        x = self.dropout(tok_emb + pos_emb)
        # Final layernorm and head
        x = self.ln_f(x)
        logits = self.head(x)
        
        if targets is None:
            return logits, None
        else:
            # Ensure targets are same length as input
            targets = targets[:, :T]  # Truncate targets to match input length
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            return logits, loss

model = ModelWrapper(config)
model.to(device)

# Wrap model in DDP
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# Create optimizer with settings from paper
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    betas=(0.9, 0.95),  # Paper's recommended values
    weight_decay=0.1,
    fused=True  # Enable fused optimizer for better performance
)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])

# Initialize scaler for mixed precision training
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# -----------------------------------------------------------------------------
# helper functions

def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        loader = DataLoader(split)
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = loader.get_batch()
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# -----------------------------------------------------------------------------
# Training loop
print(f"Starting training...")
train_loader = DataLoader('train')
iter_num = 0
best_val_loss = float('inf')

t0 = time.time()
while True:
    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': {k: v for k, v in model.module.config.items() if isinstance(v, torch.Tensor)},
                    'optimizer': optimizer.state_dict(),
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))

    if iter_num == 0 and eval_only:
        break

    # forward backward update with MTP and MoE handling
    model.train()
    X, Y = train_loader.get_batch()
    
    # Zero grad and accumulate gradients
    optimizer.zero_grad(set_to_none=True)
    for micro_step in range(grad_accum):
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum - 1)
        
        with autocast(dtype=default_dtype):
            # Multi-token prediction
            logits, aux_loss = model(X, Y)
            # Main loss for multiple token predictions
            main_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y[:, :num_tokens_predict].contiguous().view(-1))
            # Combine losses
            loss = (main_loss + aux_loss) / grad_accum
        
        # immediately async prefetch next batch while model is computing
        X, Y = train_loader.get_batch()
        
        # backward pass
        scaler.scale(loss).backward()
    
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * grad_accum
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")

    iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
