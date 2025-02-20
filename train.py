from functools import lru_cache
import os
import time
import math
import pickle
from datetime import timedelta
from contextlib import nullcontext
import platform
from typing import Dict, Any, List, Tuple

from probabilistic_layer import Network_DQNN, QuantumExpert, quantum_loss
from stochastic_optimizer import QuantumAdam, QuantumParticleOptimizer
from quantum_network import Network_DQNN as QuantumProcessor

import numpy as np
import torch
from torch.nn import functional as F
import torch.distributed as dist
import torch.backends.mps
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.checkpoint import checkpoint
import tiktoken
from torch.cuda.amp import autocast
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.nn import functional as F
from fairscale.nn.moe import MOELayer, Top2Gate
from fairscale.nn import MoE
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import wrap

# Model configuration
def get_default_config():
    """Get default configuration"""
    return {
        'batch_size': 4,     # Much smaller batch size
        'block_size': 64,    # Smaller context window
        'max_iters': 100,    # Fewer iterations for initial testing
        'learning_rate': 3e-4,
        'min_lr': 3e-5,      # learning_rate/10
        'warmup_iters': 20,  # Shorter warmup
        'grad_clip': 1.0,
        'grad_accum': 4,
        'dtype': 'float16',
        'use_checkpoint': True  # Enable activation checkpointing
    }

# Training hyperparameters from config
config = get_default_config()
batch_size = config['batch_size']
block_size = config['block_size']
max_iters = config['max_iters']
learning_rate = config['learning_rate']
min_lr = config['min_lr']
warmup_iters = config['warmup_iters']
grad_clip = config['grad_clip']
grad_accum = config['grad_accum']
dtype = config['dtype']
use_checkpoint = config['use_checkpoint']

# Global variables for distributed training
ddp = False
ddp_local_rank = 0
tokens_per_iter = grad_accum * batch_size * block_size
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

def get_device():
    """Setup the device for training"""
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
    return device
def create_config():
    """Create model configuration"""
    return {
        'n_layer': 32,
        'n_head': 8,
        'n_embd': 512,
        'vocab_size': 50257,
        'block_size': block_size,
        'dropout': 0.1,
    }

def forward(idx, targets, config):
    """Forward pass of the model"""
    B, T = idx.shape
    
    # Token embeddings
    tok_emb = torch.nn.Embedding(config['vocab_size'], config['n_embd'])(idx)
    
    # Position embeddings
    pos = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0)
    pos_emb = torch.nn.Embedding(config['block_size'], config['n_embd'])(pos)
    
    # Combine embeddings
    x = tok_emb + pos_emb
    
    # Apply dropout
    x = F.dropout(x, p=config['dropout'], training=True)
    
    # Multi-head attention
    head_size = config['n_embd'] // config['n_head']
    
    # Split into heads
    x = x.view(B, T, config['n_head'], head_size)
    
    # Self attention
    q = x @ torch.nn.Linear(head_size, head_size, device=device).weight.T
    k = x @ torch.nn.Linear(head_size, head_size, device=device).weight.T
    v = x @ torch.nn.Linear(head_size, head_size, device=device).weight.T
    
    # Scaled dot product attention
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    att = F.softmax(att, dim=-1)
    att = F.dropout(att, p=config['dropout'], training=True)
    
    # Apply attention to values
    x = att @ v
    
    # Merge heads
    x = x.transpose(1, 2).contiguous().view(B, T, config['n_embd'])
    
    # Final linear layer and layer norm
    x = F.layer_norm(x, [config['n_embd']])
    logits = torch.nn.Linear(config['n_embd'], config['vocab_size'], device=device)(x)
    
    # Loss calculation
    if targets is None:
        return logits, None
    else:
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return logits, loss

def save_checkpoint(checkpoint: Dict[str, Any], path: str, use_atomic: bool = True) -> None:
    """Save a checkpoint atomically to avoid corruption
    
    Args:
        checkpoint: Dictionary containing checkpoint data
        path: Path to save the checkpoint to
        use_atomic: Whether to use atomic save with temporary file
    """
    print(f"\nSaving checkpoint to {path}")
    
    # Make sure output directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"Created/verified output directory: {os.path.dirname(path)}")
    
    try:
        if use_atomic:
            # Save with a temporary file first
            tmp_path = path + '.tmp'
            print(f"Saving to temporary path: {tmp_path}")
            torch.save(checkpoint, tmp_path)
            print("Successfully saved to temporary file")
            
            # Atomic rename to final path
            os.replace(tmp_path, path)
            print(f"Successfully renamed to final path: {path}")
        else:
            # Direct save without atomic operation
            torch.save(checkpoint, path)
            print(f"Successfully saved checkpoint directly to: {path}")
            
    except Exception as e:
        print(f"Error saving checkpoint: {str(e)}")
        # Clean up temp file if it exists
        if use_atomic and os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

def log_training_stats(loss, dt, iter_num):
    """Log training statistics"""
    lossf = loss.item() * grad_accum
    tokens_processed = iter_num * tokens_per_iter
    print("\n=== Training Stats ===")
    print(f"Iteration: {iter_num}/{max_iters}")
    print(f"Loss: {lossf:.4f}")
    print(f"Time per iter: {dt*1000:.2f}ms")
    print(f"Tokens processed: {tokens_processed:,}")
    print(f"Training speed: {tokens_per_iter/dt:,.0f} tokens/sec")
    if device_type == 'cuda':
        print(f"Memory used: {torch.cuda.max_memory_allocated()/1e9:.2f}GB")


# -----------------------------------------------------------------------------
# Training settings from DeepSeek paper
out_dir = 'out'
eval_interval = 10   # Evaluate every 10 iterations
log_interval = 1
eval_iters = 200
eval_only = False
always_save_checkpoint = True
checkpoint_interval = 2  # Save checkpoints every 2 iterations
save_last_n_checkpoints = 5  # Keep last N checkpoints
init_from = 'scratch'

# Early stopping settings
early_stopping_patience = 5  # Number of evaluations to wait for improvement
early_stopping_threshold = 0.001  # Minimum improvement required
early_stopping_history = []  # Track validation losses

# Model architecture settings from DeepSeek paper
num_experts = 32  # Increased number of experts
expert_capacity = 2.0  # Higher capacity factor
num_tokens_predict = 4  # More tokens predicted at once
sparse_top_k = 32  # For sparse attention

# DDP settings
backend = 'nccl' if torch.cuda.is_available() else 'gloo'
world_size = int(os.environ.get('WORLD_SIZE', 1))
world_rank = int(os.environ.get('RANK', 0))
local_rank = int(os.environ.get('LOCAL_RANK', 0))


@lru_cache
def setup_training():
    """Setup distributed training and device configuration with proper synchronization"""
    device = None
    
    # Initialize DDP variables with defaults
    ddp = world_rank != -1
    master_process = world_rank == 0
    
    # Setup distributed training if enabled
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        if ddp:
            # Set device before any other CUDA operations
            torch.cuda.set_device(device)
            
            # Initialize process group with timeout and proper synchronization
            timeout = timedelta(minutes=30)
            init_process_group(
                backend=backend,
                rank=world_rank,
                world_size=world_size,
                timeout=timeout
            )
            
            # Ensure all processes are ready before creating new group
            dist.barrier()
            
            # Create process group with proper synchronization
            group = dist.new_group(
                list(range(world_size)),
                timeout=timeout
            )
            
            # Wait for group creation to complete
            dist.barrier(group)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        if ddp:
            init_process_group(backend="gloo", rank=world_rank,
                             world_size=world_size)
            group = dist.new_group(list(range(world_size)))
    else:
        device = torch.device("cpu")
        if ddp:
            init_process_group(backend="gloo", rank=world_rank,
                             world_size=world_size)
            group = dist.new_group(list(range(world_size)))
        
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
            device = get_device()

    # Print configuration if master process
    if master_process:
        global tokens_per_iter
        tokens_per_iter = grad_accum * world_size * batch_size * block_size
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

    # Set random seed and enable optimizations
    torch.manual_seed(1337)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    return device, master_process, world_size, ddp, ddp_local_rank, tokens_per_iter, group
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
        print(f"\nDataLoader {self.split}:")
        print(f"- Data size: {data_size:,} tokens")
        print(f"- Max index: {max_index:,}")
        print(f"- Block size: {block_size}")
        print(f"- Batch size: {batch_size}")
        ix = torch.randint(max_index, (batch_size,))
        print(f"- Sample indices range: {ix.min().item():,} to {ix.max().item():,}")
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
# Initialize device first
device = get_device()

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

def apply_rotary_emb(q, k, seq_len, dim, base=10000.0):
    """
    Apply rotary embeddings (sinusoidal positional encoding) to queries and keys.
    
    Parameters:
    - q (Tensor): Query tensor of shape (batch_size, seq_len, dim)
    - k (Tensor): Key tensor of shape (batch_size, seq_len, dim)
    - seq_len (int): Length of the sequence (L)
    - dim (int): Dimension of the embeddings (D)
    - base (float): Scaling factor for the sinusoidal function (optional, default 10000.0)
    
    Returns:
    - q_rot (Tensor): Rotated query tensor
    - k_rot (Tensor): Rotated key tensor
    """
    device = q.device  # Get the device of the input tensor (q or k)
    
    # Calculate the position indices
    position = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(1)  # shape: (seq_len, 1)
    div_term = torch.exp(torch.arange(0., dim, 2.0, device=device) * -(math.log(base) / dim))  # shape: (dim/2,)
    
    # Apply the sinusoidal encoding (cosine and sine)
    pos_enc = torch.zeros(seq_len, dim, device=device)  # shape: (seq_len, dim)
    pos_enc[:, 0::2] = torch.sin(position * div_term)  # even indices (sine)
    pos_enc[:, 1::2] = torch.cos(position * div_term)  # odd indices (cosine)

    # Reshape pos_enc to match the batch size and apply to q and k
    pos_enc = pos_enc.unsqueeze(0).expand(q.size(0), -1, -1)  # shape: (batch_size, seq_len, dim)

    # Apply rotary embedding by multiplying q and k with positional encoding
    q_rot = q * pos_enc  # element-wise multiplication (rotating q)
    k_rot = k * pos_enc  # element-wise multiplication (rotating k)

    return q_rot, k_rot


class DeepSeekQNN(torch.nn.Module):
    def __init__(self, config, group=None):
        super().__init__()
        # QNN architecture from example.py
        self.qnn_arch = [config['n_embd']] * (config['n_layer'] + 1)
        
        # Embeddings with quantum processing
        self.token_embedding = torch.nn.Embedding(config['vocab_size'], config['n_embd'])
        self.position_embedding = torch.nn.Parameter(torch.zeros(1, config['block_size'], config['n_embd']))
        
        # Quantum network from example.py
        self.quantum_network = Network_DQNN(
            qnn_arch=self.qnn_arch,
            fidelity_measurement_method='big_swap'
        )
        
        # Quantum experts
        self.quantum_experts = torch.nn.ModuleList([
            QuantumProcessor(n_qubits=int(np.log2(config['n_embd'])), 
                           qnn_arch=self.qnn_arch)
            for _ in range(num_experts)
        ])
        
        # Enhanced MoE layers with capacity factor and load balancing
        expert_capacity_factor = 2.0  # From DeepSeek paper
        self.moe_layers = torch.nn.ModuleList([
            MoE(
                input_size=config['n_embd'],
                num_experts=num_experts,
                hidden_size=4 * config['n_embd'],
                activation=torch.nn.GELU(),
                capacity_factor=expert_capacity_factor,
                drop_tokens=True,  # Enable token dropping
                use_expert_choice=True,  # Enable expert choice routing
                expert_capacity=int(expert_capacity_factor * (config['block_size'] / num_experts)),
                group=group,
                # Add quantum experts
                experts=torch.nn.ModuleList([
                    QuantumExpert(
                        input_dim=config['n_embd'],
                        output_dim=config['n_embd'],
                        n_qubits=int(np.log2(config['n_embd']))
                    ) for _ in range(num_experts)
                ])
            ) for _ in range(config['n_layer'])
        ])
        
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
    
    def sparse_attention(self, q, k, v, mask=None):
        """Sparse attention mechanism from DeepSeek paper"""
        B, L, D = q.shape
        
        # Apply rotary embeddings
        q, k = apply_rotary_emb(q, k, L, D)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Get top-k attention for each query
        topk_scores, topk_idx = scores.topk(sparse_top_k, dim=-1)
        
        # Create sparse attention pattern
        sparse_scores = torch.zeros_like(scores).scatter_(-1, topk_idx, topk_scores)
        
        # Normalize scores
        attn_weights = F.softmax(sparse_scores, dim=-1)
        attn_weights = F.dropout(attn_weights, p=0.1, training=self.training)
        
        # Apply attention
        out = torch.matmul(attn_weights, v)
        return out

    def monte_carlo_attention(self, q, k, v, num_samples=64, num_mc_samples=8):
        """Monte Carlo attention with adaptive temperature and noise"""
        B, L, D = q.shape
        q, k = apply_rotary_emb(q, k, L, D)
        
        # Adaptive temperature annealing
        if not hasattr(self, 'forward_count'):
            self.forward_count = 0
        self.forward_count += 1
        
        temp = max(0.1, 1.0 / math.sqrt(self.forward_count))  # Decay temperature
        noise_scale = max(0.05, 0.1 / math.sqrt(self.forward_count))  # Decay noise
        print(f"\nMonte Carlo Attention:")
        print(f"- Batch size: {B}")
        print(f"- Sequence length: {L}")
        print(f"- Hidden dimension: {D}")
        print(f"- Number of key samples: {num_samples}")
        print(f"- Number of MC samples: {num_mc_samples}")
        
        # Multiple Monte Carlo sampling rounds
        outputs = []
        for mc_round in range(num_mc_samples):
            # Sample different keys/values each round
            indices = torch.randint(L, (num_samples,), device=q.device)
            k_sampled = k[:, indices, :]  # (B, num_samples, D)
            v_sampled = v[:, indices, :]  # (B, num_samples, D)
            
            # Add Gaussian noise for exploration
            k_noise = torch.randn_like(k_sampled) * 0.1
            v_noise = torch.randn_like(v_sampled) * 0.1
            k_sampled = k_sampled + k_noise
            v_sampled = v_sampled + v_noise
            
            # Compute attention scores with temperature scaling
            temp = 1.0 / math.sqrt(D)
            scores = torch.matmul(q, k_sampled.transpose(-2, -1)) * temp
            
            # Gumbel-Softmax for stochastic attention
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(scores) + 1e-10))
            scores = scores + gumbel_noise * 0.1
            attn_weights = F.softmax(scores, dim=-1)
            
            # Dropout for regularization
            attn_weights = F.dropout(attn_weights, p=0.1, training=self.training)
            
            # Apply attention
            out = torch.matmul(attn_weights, v_sampled)  # (B, L, D)
            outputs.append(out)
        
        # Combine MC samples with uncertainty estimation
        stacked_outputs = torch.stack(outputs)  # (num_mc_samples, B, L, D)
        mean_output = torch.mean(stacked_outputs, dim=0)  # (B, L, D)
        std_output = torch.std(stacked_outputs, dim=0)  # (B, L, D)
        
        # Add uncertainty information to output
        output = mean_output + std_output * torch.randn_like(mean_output) * 0.1
        
        print(f"- Attention uncertainty: {std_output.mean().item():.4f}")
        return output

    def _forward_impl(self, x, pos_emb, targets=None):
        """Forward implementation with enhanced MoE routing and load balancing"""
        batch_size = x.size(0)
        feature_dim = x.size(2)
        
        # Track expert usage for load balancing
        expert_counts = torch.zeros(num_experts, device=x.device)
        aux_loss = 0.0

        # Check if x and pos_emb need padding/trimming
        if x.size(1) > pos_emb.size(1):
            # Pad pos_emb to match x's length
            padding = x.size(1) - pos_emb.size(1)
            pos_emb = torch.cat([pos_emb, torch.zeros(batch_size, padding, feature_dim).to(pos_emb.device)], dim=1)
        elif pos_emb.size(1) > x.size(1):
            # Pad x to match pos_emb's length
            padding = pos_emb.size(1) - x.size(1)
            x = torch.cat([x, torch.zeros(batch_size, padding, feature_dim).to(x.device)], dim=1)

        x = self.embed_dropout(x + pos_emb)
        
        # Project input to queries, keys, and values with dropout
        q = self.proj_dropout(self.q_proj(x))
        k = self.proj_dropout(self.k_proj(x))
        v = self.proj_dropout(self.v_proj(x))
        
        # Apply sparse attention with dropout
        x = self.attn_dropout(self.sparse_attention(q, k, v))
        
        # Apply MoE layers with load balancing and quantum experts
        moe_loss = 0
        for i, moe_layer in enumerate(self.moe_layers):
            # Get expert outputs and auxiliary losses
            x, aux_l, expert_mask = moe_layer(x)
            moe_loss += aux_l
            
            # Track expert usage
            expert_counts += expert_mask.sum(dim=0)
            
            # Add load balancing loss from DeepSeek paper
            expert_probs = expert_counts / expert_counts.sum()
            target_probs = torch.ones_like(expert_probs) / num_experts
            load_balance_loss = F.kl_div(
                expert_probs.log(), target_probs,
                reduction='batchmean'
            )
            moe_loss += 0.01 * load_balance_loss
            
        x = self.ln_f(x)
        x = self.final_dropout(x)
        
        # Multi-token prediction
        logits = self.head(x)
        
        # Return MoE auxiliary loss
        return logits, moe_loss

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # Track forward pass count
        if not hasattr(self, 'forward_count'):
            self.forward_count = 0
        self.forward_count += 1
        
        print(f"\nForward Pass #{self.forward_count}:")
        print(f"- Input shape: {idx.shape}")
        print(f"- Target shape: {targets.shape if targets is not None else None}")
        
        # Get token embeddings
        tok_emb = self.token_embedding(idx)
        print(f"- Token embedding shape: {tok_emb.shape}")
        
        # Add positional embeddings, truncating to input sequence length
        pos_emb = self.position_embedding[:, :min(T, self.config['block_size']), :]
        pos_emb = pos_emb.expand(B, min(T, self.config['block_size']), -1)
        print(f"- Position embedding shape: {pos_emb.shape}")
        
        # Use checkpointing for the forward pass
        if use_checkpoint and self.training:
            logits = torch.utils.checkpoint.checkpoint(self._forward_impl, tok_emb, pos_emb)
        else:
            logits = self._forward_impl(tok_emb, pos_emb)
        
        if targets is None:
            return logits, None
        else:
            # Ensure targets are same length as input and properly reshaped
            targets = targets[:, :T].reshape(-1)  # Flatten targets after truncating
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets, ignore_index=-1)
            return logits, loss
*x, group = setup_training()
device = get_device()
model = DeepSeekQNN(config, group=group)
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
        print("\nLAMB Optimizer Step:")
        
        for group_idx, group in enumerate(self.param_groups):
            print(f"- Parameter group {group_idx}:")
            print(f"  - Learning rate: {group['lr']:.6e}")
            print(f"  - Weight decay: {group['weight_decay']}")
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

# Create quantum-enhanced optimizer
optimizer = QuantumAdam(
    model.parameters(),
    lr=learning_rate,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01,
    quantum_factor=0.1
)

# Add quantum-inspired optimizer for exploration
# Quantum optimizer removed since QuantumInspiredOptimizer is not defined
# and not needed since we're using QuantumParticleOptimizer
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])

# Initialize scaler for mixed precision training
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# -----------------------------------------------------------------------------
# helper functions

def get_lr(it):
    # DeepSeek learning rate schedule
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_iters:
        return min_lr
    # 3) cosine decay with longer tail
    decay_ratio = (it - warmup_iters) / max(1, max_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    # Add small constant to prevent learning rate from going too low
    return max(min_lr, min_lr + coeff * (learning_rate - min_lr)) + 1e-7

# Number of Monte Carlo samples for evaluation
num_mc_samples = 2  # Reduced samples for faster evaluation

@torch.no_grad()
def estimate_loss():
    print("\nEstimating Loss:")
    print(f"- Evaluation iterations: {eval_iters}")
    print(f"- Monte Carlo samples: {num_mc_samples}")
    
    out = {}
    # Don't set model.eval() to keep dropout active for Monte Carlo
    for split in ['train', 'val']:
        print(f"\nEvaluating {split} split:")
        loader = DataLoader(split)
        losses = torch.zeros(eval_iters)
        predictions = []
        
        # Multiple forward passes for uncertainty estimation
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


def train_iteration(model, optimizer, train_loader, scaler, iter_num, best_val_loss, master_process):
    """Execute a single training iteration"""
    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) 
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"\nEvaluation at step {iter_num}")
        print(f"Train loss: {losses.get('train', {}).get('loss', 0.0):.4f}")
        print(f"Val loss: {losses.get('val', {}).get('loss', 0.0):.4f}")
        print(f"Learning rate: {lr:.6f}")
        
        # Save checkpoints and handle early stopping
        val_loss = losses.get('val', {}).get('loss', float('inf'))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if iter_num > 0:
                save_checkpoint({
                    'model_state_dict': model.state_dict() if not ddp else model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': model.module.config if ddp else model.config,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                }, os.path.join(out_dir, 'best_ckpt.pt'))

        # Periodic checkpoints
        if iter_num % checkpoint_interval == 0 and iter_num > 0:
            save_checkpoint({
                'model_state_dict': model.state_dict() if not ddp else model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': model.module.config if ddp else model.config,
                'iter_num': iter_num,
                'best_val_loss': best_val_loss,
            }, os.path.join(out_dir, f'ckpt_{iter_num:07d}.pt'))

    # Training step
    model.train()
    if iter_num % log_interval == 0:
        print(f"\n=== Training Iteration {iter_num} ===")
        print(f"Learning rate: {lr:.6e}")
    
    X, Y = train_loader.get_batch()
    t0 = time.time()
    
    # Execute training step
    loss = execute_training_step(model, optimizer, X, Y, scaler, iter_num)
    
    # Timing and logging
    t1 = time.time()
    dt = t1 - t0
    
    if iter_num % log_interval == 0 and master_process:
        log_training_stats(loss, dt, iter_num)
        
    return best_val_loss

def calculate_loss(logits, targets, aux_loss, quantum_states: List[torch.Tensor]):
    """Calculate loss with Monte Carlo sampling and quantum coherence"""
    num_mc_samples = 10
    mc_losses = []
    mc_predictions = []
    
    # Track quantum states for coherence calculation
    quantum_coherence_states = []
    
    for _ in range(num_mc_samples):
        # Multiple sampling strategies
        
        # 1. Gumbel-Softmax sampling
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10))
        gumbel_logits = (logits + gumbel_noise) / 0.1
        
        # 2. Temperature sampling
        temp_scaled = logits / 0.8  # Temperature parameter
        
        # 3. Top-k sampling
        top_k = 40
        top_k_logits = torch.topk(logits, top_k, dim=-1)
        top_k_mask = torch.zeros_like(logits).scatter_(-1, top_k_logits.indices, 1.0)
        
        # Combine sampling strategies
        combined_logits = (gumbel_logits + temp_scaled) * top_k_mask
        sampled_probs = F.softmax(combined_logits, dim=-1)
        
        # Add exploration noise
        exploration_noise = torch.randn_like(sampled_probs) * 0.05
        sampled_probs = F.softmax(torch.log(sampled_probs + 1e-10) + exploration_noise, dim=-1)
        
        # Calculate losses with different metrics
        ce_loss = F.cross_entropy(
            sampled_probs.view(-1, sampled_probs.size(-1)),
            targets[:, :sampled_probs.size(1)].contiguous().view(-1)
        )
        
        # KL divergence loss for regularization
        kl_loss = F.kl_div(
            F.log_softmax(logits, dim=-1),
            sampled_probs,
            reduction='batchmean'
        )
        
        # Combine losses
        combined_loss = ce_loss + 0.1 * kl_loss
        mc_losses.append(combined_loss)
        mc_predictions.append(sampled_probs)
    
    # Compute statistics across MC samples
    mc_losses = torch.stack(mc_losses)
    mc_predictions = torch.stack(mc_predictions)
    
    # Mean and uncertainty estimation
    main_loss = mc_losses.mean()
    uncertainty = mc_losses.std()
    
    # Prediction diversity penalty
    pred_mean = mc_predictions.mean(dim=0)
    pred_std = mc_predictions.std(dim=0)
    diversity_penalty = -0.1 * pred_std.mean()  # Encourage diverse predictions
    
    # Calculate quantum loss
    q_loss = quantum_loss(pred_mean, targets, quantum_coherence_states)
    
    # Combine all loss components including quantum loss
    return (main_loss + aux_loss + 0.1 * uncertainty + diversity_penalty + 0.01 * q_loss) / grad_accum

def execute_training_step(model, optimizer, X, Y, scaler, iter_num):
    """Execute core training logic for a single step with proper gradient accumulation"""
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0
    
    # Accumulate gradients over multiple forward passes
    for micro_step in range(grad_accum):
        # Get a different batch slice for each micro step
        batch_slice = slice(micro_step * X.size(0) // grad_accum,
                          (micro_step + 1) * X.size(0) // grad_accum)
        X_micro, Y_micro = X[batch_slice], Y[batch_slice]
        
        if ddp:
            # Only synchronize on last micro-step
            with model.no_sync() if micro_step < grad_accum - 1 else nullcontext():
                with autocast(dtype=default_dtype):
                    logits, aux_loss = model(X_micro, Y_micro)
                    loss = calculate_loss(logits, Y_micro, aux_loss)
                    # Scale loss by grad_accum to maintain correct magnitude
                    loss = loss / grad_accum
                    
                scaler.scale(loss).backward()
        else:
            with autocast(dtype=default_dtype):
                logits, aux_loss = model(X_micro, Y_micro)
                loss = calculate_loss(logits, Y_micro, aux_loss)
                # Scale loss by grad_accum to maintain correct magnitude
                loss = loss / grad_accum
                
            scaler.scale(loss).backward()
            
        total_loss += loss.item() * grad_accum
        
        if iter_num % log_interval == 0:
            print(f"Micro-step {micro_step + 1}/{grad_accum}, "
                  f"Loss: {loss.item() * grad_accum:.4f}")
    
    # Clip gradients after accumulation
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    
    # Optimizer step
    scaler.step(optimizer)
    scaler.update()
    
    return total_loss

def train_model(model, optimizer, master_process):
    """Main training loop"""
    print(f"Starting training...")
    train_loader = DataLoader('train')
    iter_num = 0
    best_val_loss = float('inf')

    # Main training loop
    while True:
        best_val_loss = train_iteration(model, optimizer, train_loader, scaler, iter_num, best_val_loss, master_process)
        
        if iter_num == 0 and eval_only:
            break
            
        iter_num += 1
        
        if iter_num > max_iters:
            break
            
    return best_val_loss

if __name__ == '__main__':
    try:
        # Setup training environment
        device, master_process, ddp_world_size, ddp, ddp_local_rank, tokens_per_iter, group = setup_training()
        
        # Initialize model and optimizer
        model = DeepSeekQNN(create_config(), group=group)
        model.to(device)
        
        if ddp:
            model = DDP(model, device_ids=[ddp_local_rank])
        
        optimizer = QuantumParticleOptimizer(
            model.parameters(), 
            lr=learning_rate,
            momentum=0.9,
            n_particles=10,
            exploration_rate=0.1
        )
        
        # Run training
        final_val_loss = train_model(model, optimizer, master_process)
        print(f"Training completed with final validation loss: {final_val_loss:.4f}")
        
    finally:
        if ddp:
            destroy_process_group()
def calculate_loss(logits, targets, aux_loss):
    """Calculate loss with Monte Carlo sampling"""
    num_mc_samples = 10
    mc_losses = []
    mc_predictions = []
    
    for _ in range(num_mc_samples):
        # Multiple sampling strategies
        
        # 1. Gumbel-Softmax sampling
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10))
        gumbel_logits = (logits + gumbel_noise) / 0.1
        
        # 2. Temperature sampling
        temp_scaled = logits / 0.8  # Temperature parameter
        
        # 3. Top-k sampling
        top_k = 40
        top_k_logits = torch.topk(logits, top_k, dim=-1)
        top_k_mask = torch.zeros_like(logits).scatter_(-1, top_k_logits.indices, 1.0)
        
        # Combine sampling strategies
        combined_logits = (gumbel_logits + temp_scaled) * top_k_mask
        sampled_probs = F.softmax(combined_logits, dim=-1)
        
        # Add exploration noise
        exploration_noise = torch.randn_like(sampled_probs) * 0.05
        sampled_probs = F.softmax(torch.log(sampled_probs + 1e-10) + exploration_noise, dim=-1)
        
        # Calculate losses with different metrics
        ce_loss = F.cross_entropy(
            sampled_probs.view(-1, sampled_probs.size(-1)),
            targets[:, :sampled_probs.size(1)].contiguous().view(-1)
        )
        
        # KL divergence loss for regularization
        kl_loss = F.kl_div(
            F.log_softmax(logits, dim=-1),
            sampled_probs,
            reduction='batchmean'
        )
        
        # Combine losses
        combined_loss = ce_loss + 0.1 * kl_loss
        mc_losses.append(combined_loss)
        mc_predictions.append(sampled_probs)
    
    # Compute statistics across MC samples
    mc_losses = torch.stack(mc_losses)
    mc_predictions = torch.stack(mc_predictions)
    
    # Mean and uncertainty estimation
    main_loss = mc_losses.mean()
    uncertainty = mc_losses.std()
    
    # Prediction diversity penalty
    pred_mean = mc_predictions.mean(dim=0)
    pred_std = mc_predictions.std(dim=0)
    diversity_penalty = -0.1 * pred_std.mean()  # Encourage diverse predictions
    
    # Combine all loss components
    return (main_loss + aux_loss + 0.1 * uncertainty + diversity_penalty) / grad_accum
