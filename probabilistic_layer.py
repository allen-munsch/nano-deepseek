import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple

class ProbabilisticLayer(nn.Module):
    """Neural network layer using probabilistic computations and Monte Carlo sampling.
    
    This layer implements classical probabilistic algorithms including:
    - Stochastic state preparation
    - Monte Carlo sampling
    - Noise-robust measurements
    - Uncertainty estimation"""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Trainable parameters
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        
        # Dropout layers for MC sampling
        self.dropout = nn.Dropout(0.1)
        
    def monte_carlo_attention(self, q, k, v, num_samples=64):
        """Monte Carlo attention with proper uncertainty estimation"""
        batch_size = q.size(0)
        
        # Multiple sampling rounds
        outputs = []
        for _ in range(num_samples):
            # Apply dropout for sampling
            q_sample = self.dropout(q)
            k_sample = self.dropout(k)
            v_sample = self.dropout(v)
            
            # Compute attention scores
            scores = torch.matmul(q_sample, k_sample.transpose(-2, -1))
            scores = scores / np.sqrt(self.hidden_dim)
            
            # Apply temperature sampling
            temp = 0.1
            scores = scores / temp
            
            # Gumbel-softmax sampling
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(scores) + 1e-10))
            scores = scores + gumbel_noise
            
            # Compute attention weights
            weights = F.softmax(scores, dim=-1)
            
            # Apply attention
            output = torch.matmul(weights, v_sample)
            outputs.append(output)
            
        # Compute mean and uncertainty
        outputs = torch.stack(outputs)
        mean = outputs.mean(dim=0)
        std = outputs.std(dim=0)
        
        return mean, std
        
    def forward(self, x):
        # Project input
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)
        
        # Apply MC attention
        output, uncertainty = self.monte_carlo_attention(q, k, v)
        
        return output, uncertainty

class StochasticExpert(nn.Module):
    """Expert module using stochastic computation"""
    
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 4 * input_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(4 * input_dim, output_dim)
        )
        
    def forward(self, x):
        return self.net(x)

def uncertainty_loss(predictions: torch.Tensor, targets: torch.Tensor, 
                    uncertainties: List[torch.Tensor], epsilon: float = 1e-10) -> torch.Tensor:
    """Loss function incorporating uncertainty estimation"""
    # Standard cross entropy with label smoothing
    ce_loss = F.cross_entropy(predictions, targets, label_smoothing=0.1)
    
    # Uncertainty regularization
    uncertainty_loss = 0
    for uncertainty in uncertainties:
        # Penalize high uncertainty
        uncertainty_loss += uncertainty.mean()
        
    # Combine losses
    total_loss = ce_loss + 0.1 * uncertainty_loss
    return total_loss
