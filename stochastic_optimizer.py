import torch
import torch.nn as nn
from typing import List, Dict, Any
import numpy as np

class QuantumParticleOptimizer(torch.optim.Optimizer):
    """Quantum-enhanced particle swarm optimization
    
    A hybrid quantum-classical optimization algorithm using quantum circuits
    for exploration and classical PSO for exploitation."""
    
    def __init__(self, params, lr=1e-3, momentum=0.9, 
                 n_particles=10, exploration_rate=0.1,
                 cognitive_coeff=1.5, social_coeff=1.5):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            n_particles=n_particles,
            exploration_rate=exploration_rate,
            cognitive_coeff=cognitive_coeff,
            social_coeff=social_coeff,
            positions=[],
            velocities=[],
            local_best_positions=[],
            local_best_values=[],
            global_best_position=None,
            global_best_value=float('inf')
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step respecting unitarity constraints"""
        loss = None if closure is None else closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                
                # Initialize state if needed
                if len(state) == 0:
                    state['step'] = 0
                    state['momentum_buffer'] = torch.zeros_like(p)
                    state['velocity_buffer'] = torch.zeros_like(p)

                momentum_buffer = state['momentum_buffer']
                velocity_buffer = state['velocity_buffer']
                
                # Update momentum with damping to maintain unitarity
                momentum_buffer.mul_(group['momentum']).add_(
                    p.grad, alpha=1 - group['momentum']
                )
                
                # Update velocity considering quantum phase
                phase = torch.angle(p + 1j * momentum_buffer)
                velocity_buffer.mul_(1 - group['damping']).add_(
                    torch.sin(phase) * p.grad, alpha=group['damping']
                )
                
                # Apply update while preserving unitarity
                update = momentum_buffer + velocity_buffer
                unitary_update = update / (1 + torch.norm(update))
                
                p.add_(unitary_update, alpha=-group['lr'])
                
                state['step'] += 1

        return loss

class AdaptiveAdam(torch.optim.Adam):
    """Adam optimizer with enhanced exploration and adaptation
    
    Extends Adam with additional exploration mechanisms and adaptive
    learning strategies for better optimization."""
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, exploration_factor=0.1, 
                 noise_decay=0.999, min_noise=0.01):
        super().__init__(params, lr, betas, eps, weight_decay)
        self.exploration_factor = exploration_factor
        self.noise_decay = noise_decay
        self.min_noise = min_noise
        self.step_count = 0
    
    @torch.no_grad()
    def step(self, closure=None):
        loss = super().step(closure)
        self.step_count += 1
        
        # Calculate adaptive noise scale
        noise_scale = max(
            self.min_noise,
            self.exploration_factor * (self.noise_decay ** self.step_count)
        )
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                state = self.state[p]
                
                # Initialize exploration state if needed
                if 'exploration_momentum' not in state:
                    state['exploration_momentum'] = torch.zeros_like(p)
                    state['best_position'] = p.clone()
                    state['best_value'] = float('inf')
                
                # Quantum phase estimation using QPE algorithm
                # Convert parameter to quantum state
                quantum_state = torch.exp(1j * p)
                
                # Quantum Fourier transform with unitarity preservation
                def unitary_qft(state):
                    # Normalize input state
                    state = state / (torch.norm(state) + 1e-8)
                    
                    # Apply QFT while preserving unitarity
                    n = state.size(-1)
                    omega = torch.exp(2j * torch.pi / n)
                    indices = torch.arange(n, device=state.device)
                    qft_matrix = omega ** (indices.view(-1, 1) * indices)
                    qft_matrix = qft_matrix / torch.sqrt(torch.tensor(n, dtype=torch.float))
                    
                    # Ensure matrix is unitary
                    qft_matrix = 0.5 * (qft_matrix + qft_matrix.conj().transpose(-2, -1))
                    U, _, V = torch.linalg.svd(qft_matrix)
                    unitary_qft = U @ V
                    
                    return state @ unitary_qft.conj()

                # Apply unitary QFT
                qft_state = unitary_qft(quantum_state)
                phase = torch.angle(qft_state)
                
                # Update phase with unitarity preservation
                new_phase = phase / (torch.norm(phase) + 1e-8)
                state['quantum_phase'] = 0.9 * state['quantum_phase'] + 0.1 * new_phase
                
                # Calculate energy preserving quantum mechanical constraints
                energy = torch.abs(qft_state) ** 2
                energy = energy / (energy.sum() + 1e-8)  # Normalize probabilities
                state['quantum_energy'] = 0.9 * state['quantum_energy'] + 0.1 * energy
                
                # Quantum momentum with phase and energy
                quantum_momentum = (
                    self.quantum_factor * 
                    torch.sin(state['quantum_phase']) * 
                    torch.sqrt(state['quantum_energy'])
                )
                
                # Update quantum momentum with uncertainty principle
                state['quantum_momentum'] = (
                    0.9 * state['quantum_momentum'] + 
                    0.1 * quantum_momentum
                )
                
                # Apply quantum correction
                quantum_update = (
                    state['quantum_momentum'] * torch.cos(state['quantum_phase']) +
                    quantum_momentum * torch.sin(state['quantum_phase'])
                )
                
                # Update parameter with quantum effects
                p.add_(quantum_update)
                
                # Update quantum state
                state['quantum_state'] = torch.exp(1j * p)
        
        return loss
