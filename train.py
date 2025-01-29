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
from torch.utils.checkpoint import checkpoint
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
use_checkpoint = True  # Enable activation checkpointing
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
eval_interval = 500  # More frequent evaluations
log_interval = 1
eval_iters = 200
eval_only = False
always_save_checkpoint = True
checkpoint_interval = 1000  # Save checkpoints every 1000 iterations
save_last_n_checkpoints = 5  # Keep last N checkpoints
init_from = 'scratch'

# Early stopping settings
early_stopping_patience = 5  # Number of evaluations to wait for improvement
early_stopping_threshold = 0.001  # Minimum improvement required
early_stopping_history = []  # Track validation losses

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
    print("\n=== Training Configuration ===")
    print(f"Device type: {device_type}")
    print(f"Batch size: {batch_size}")
    print(f"Block size: {block_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Gradient accumulation steps: {grad_accum}")
    print(f"Tokens per iteration: {tokens_per_iter:,}")
    print(f"Max iterations: {max_iters}")
    print(f"Warmup iterations: {warmup_iters}")
    print(f"Using dtype: {dtype}")
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
        data_size = len(self.data)
        max_index = data_size - block_size - num_tokens_predict
        print(f"\nDataLoader {self.split}: Using {data_size:,} tokens, max_index={max_index:,}")
        ix = torch.randint(max_index, (batch_size,))
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
        
        # Multiple dropout layers for Monte Carlo dropout
        self.embed_dropout = torch.nn.Dropout(config['dropout'])
        self.attn_dropout = torch.nn.Dropout(config['dropout'])
        self.proj_dropout = torch.nn.Dropout(config['dropout'])
        self.final_dropout = torch.nn.Dropout(config['dropout'])
        
        # Attention projections
        self.q_proj = torch.nn.Linear(config['n_embd'], config['n_embd'])
        self.k_proj = torch.nn.Linear(config['n_embd'], config['n_embd'])
        self.v_proj = torch.nn.Linear(config['n_embd'], config['n_embd'])
        
        self.ln_f = torch.nn.LayerNorm(config['n_embd'])
        self.head = torch.nn.Linear(config['n_embd'], config['vocab_size'], bias=False)
        self.config = config
    
    def monte_carlo_attention(self, q, k, v, num_samples=64):
        # q, k, v shape: (batch, seq_len, dim)
        B, L, D = q.shape
        
        # Compute attention scores for a subset of random keys
        indices = torch.randint(L, (num_samples,), device=q.device)
        k_sampled = k[:, indices, :]  # (B, num_samples, D)
        v_sampled = v[:, indices, :]  # (B, num_samples, D)
        
        # Compute attention scores
        scores = torch.matmul(q, k_sampled.transpose(-2, -1)) / math.sqrt(D)  # (B, L, num_samples)
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply attention
        out = torch.matmul(attn_weights, v_sampled)  # (B, L, D)
        return out

    def _forward_impl(self, x, pos_emb):
        # Apply dropouts even during inference for Monte Carlo dropout
        x = self.embed_dropout(x + pos_emb)
        
        # Project input to queries, keys, and values with dropout
        q = self.proj_dropout(self.q_proj(x))
        k = self.proj_dropout(self.k_proj(x))
        v = self.proj_dropout(self.v_proj(x))
        
        # Apply Monte Carlo attention with dropout
        x = self.attn_dropout(self.monte_carlo_attention(q, k, v))
        
        x = self.ln_f(x)
        x = self.final_dropout(x)
        return self.head(x)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # Get token embeddings
        tok_emb = self.token_embedding(idx)
        # Add positional embeddings
        pos_emb = self.position_embedding[:, :T, :]
        
        # Use checkpointing for the forward pass
        if use_checkpoint and self.training:
            logits = checkpoint(self._forward_impl, tok_emb, pos_emb)
        else:
            logits = self._forward_impl(tok_emb, pos_emb)
        
        if targets is None:
            return logits, None
        else:
            # Ensure targets are same length as input and properly reshaped
            targets = targets[:, :T].reshape(-1)  # Flatten targets after truncating
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets, ignore_index=-1)
            return logits, loss

model = ModelWrapper(config)
model.to(device)

# Wrap model in DDP
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# LAMB optimizer implementation
class LAMB(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, trust_clip=True):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, trust_clip=trust_clip)
        super(LAMB, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None if closure is None else closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                
                # Update step
                state['step'] += 1
                
                # Decay the first and second moment running average coefficient
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                
                # Weight decay
                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])
                
                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Compute bias-corrected moments
                step = state['step']
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                
                # Compute adam update
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                update = (exp_avg / bias_correction1) / denom
                
                # Compute trust ratio
                if group['trust_clip']:
                    weight_norm = p.norm()
                    update_norm = update.norm()
                    trust_ratio = weight_norm / (update_norm + 1e-7)
                    trust_ratio = torch.clamp(trust_ratio, 0.0, 10.0)
                else:
                    trust_ratio = 1.0
                
                # Update weights
                p.add_(update, alpha=-group['lr'] * trust_ratio)
        
        return loss

# Create optimizer with LAMB
optimizer = LAMB(
    model.parameters(),
    lr=learning_rate,
    betas=(0.9, 0.95),  # Same beta values as before
    weight_decay=0.1,
    trust_clip=True
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
    # Don't set model.eval() to keep dropout active for Monte Carlo
    for split in ['train', 'val']:
        loader = DataLoader(split)
        losses = torch.zeros(eval_iters)
        predictions = []
        
        # Multiple forward passes for uncertainty estimation
        num_mc_samples = 5
        for k in range(eval_iters):
            X, Y = loader.get_batch()
            mc_losses = []
            mc_logits = []
            
            # Monte Carlo sampling with dropout
            for _ in range(num_mc_samples):
                with ctx:
                    logits, loss = model(X, Y)
                    mc_losses.append(loss.item())
                    mc_logits.append(logits.softmax(dim=-1))
            
            # Average losses and compute uncertainty
            losses[k] = sum(mc_losses) / num_mc_samples
            mean_probs = torch.stack(mc_logits).mean(dim=0)
            uncertainty = -(mean_probs * torch.log(mean_probs + 1e-8)).sum(dim=-1).mean().item()
            predictions.append({'mean_probs': mean_probs, 'uncertainty': uncertainty})
            
        out[split] = {
            'loss': losses.mean().item(),
            'uncertainty': sum(p['uncertainty'] for p in predictions) / len(predictions)
        }
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
        print(f"\nStep {iter_num}:")
        print(f"Train loss: {losses['train']:.4f}")
        print(f"Val loss: {losses['val']:.4f}")
        print(f"Learning rate: {lr:.6f}")
        
        # Early stopping check
        early_stopping_history.append(losses['val'])
        if len(early_stopping_history) > early_stopping_patience:
            recent_best = min(early_stopping_history[-early_stopping_patience:])
            if losses['val'] > recent_best - early_stopping_threshold:
                print(f"\nEarly stopping triggered! No improvement in validation loss for {early_stopping_patience} evaluations.")
                print(f"Best val loss: {best_val_loss:.4f}")
                print(f"Current val loss: {losses['val']:.4f}")
                break
            early_stopping_history.pop(0)  # Remove oldest loss
            
        # Save best model checkpoint
        if losses['val'] < best_val_loss:
            improvement = best_val_loss - losses['val']
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': {k: v for k, v in model.module.config.items() if isinstance(v, torch.Tensor)},
                    'optimizer': optimizer.state_dict(),
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                }
                print(f"Saving best checkpoint to {out_dir} (improvement: {improvement:.6f})")
                torch.save(checkpoint, os.path.join(out_dir, 'best_ckpt.pt'))

        # Save periodic checkpoints
        if iter_num % checkpoint_interval == 0 and iter_num > 0:
            checkpoint = {
                'model': {k: v for k, v in model.module.config.items() if isinstance(v, torch.Tensor)},
                'optimizer': optimizer.state_dict(),
                'iter_num': iter_num,
                'best_val_loss': best_val_loss,
            }
            ckpt_path = os.path.join(out_dir, f'ckpt_{iter_num:07d}.pt')
            print(f"Saving periodic checkpoint to {ckpt_path}")
            torch.save(checkpoint, ckpt_path)
            
            # Remove old checkpoints if needed
            if save_last_n_checkpoints > 0:
                checkpoints = sorted([f for f in os.listdir(out_dir) if f.startswith('ckpt_')])
                if len(checkpoints) > save_last_n_checkpoints:
                    for ckpt in checkpoints[:-save_last_n_checkpoints]:
                        os.remove(os.path.join(out_dir, ckpt))

    if iter_num == 0 and eval_only:
        break

    # forward backward update with MTP and MoE handling
    model.train()
    if iter_num % log_interval == 0:
        print(f"\n=== Training Iteration {iter_num} ===")
        print(f"Learning rate: {lr:.6e}")
    
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
        if iter_num % log_interval == 0:
            print(f"Micro-step {micro_step + 1}/{grad_accum}, Loss: {loss.item():.4f}")
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
        tokens_processed = iter_num * tokens_per_iter
        print(f"\n=== Training Stats ===")
        print(f"Iteration: {iter_num}/{max_iters}")
        print(f"Loss: {lossf:.4f}")
        print(f"Time per iter: {dt*1000:.2f}ms")
        print(f"Tokens processed: {tokens_processed:,}")
        print(f"Training speed: {tokens_per_iter/dt:,.0f} tokens/sec")
        print(f"Memory used: {torch.cuda.max_memory_allocated()/1e9:.2f}GB") if device_type == 'cuda' else None

    iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
