import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple

class QuantumLayer(nn.Module):
    """Quantum layer that simulates quantum operations"""
    
    def __init__(self, n_qubits: int, n_rotations: int):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_rotations = n_rotations
        
        # Trainable rotation parameters
        self.rx_params = nn.Parameter(torch.randn(n_qubits, n_rotations))
        self.ry_params = nn.Parameter(torch.randn(n_qubits, n_rotations))
        self.rz_params = nn.Parameter(torch.randn(n_qubits, n_rotations))
        
        # Decoherence parameters
        self.decoherence_rate = 0.01
        
    def _apply_rotation(self, state: torch.Tensor, params: torch.Tensor, axis: str) -> torch.Tensor:
        """Apply rotation gates along specified axis"""
        for qubit in range(self.n_qubits):
            for rot in range(self.n_rotations):
                angle = params[qubit, rot]
                if axis == 'x':
                    state = self._rx_gate(state, qubit, angle) 
                elif axis == 'y':
                    state = self._ry_gate(state, qubit, angle)
                else:  # z axis
                    state = self._rz_gate(state, qubit, angle)
        return state
    
    def _rx_gate(self, state: torch.Tensor, qubit: int, angle: float) -> torch.Tensor:
        """Rotation around X axis"""
        cos = torch.cos(angle/2)
        sin = torch.sin(angle/2)
        matrix = torch.tensor([[cos, -1j*sin], [-1j*sin, cos]], dtype=torch.complex64)
        return self._apply_single_qubit_gate(state, qubit, matrix)
        
    def _ry_gate(self, state: torch.Tensor, qubit: int, angle: float) -> torch.Tensor:
        """Rotation around Y axis"""
        cos = torch.cos(angle/2) 
        sin = torch.sin(angle/2)
        matrix = torch.tensor([[cos, -sin], [sin, cos]], dtype=torch.complex64)
        return self._apply_single_qubit_gate(state, qubit, matrix)
        
    def _rz_gate(self, state: torch.Tensor, qubit: int, angle: float) -> torch.Tensor:
        """Rotation around Z axis"""
        exp = torch.exp(-1j * angle/2)
        matrix = torch.tensor([[exp, 0], [0, exp.conj()]], dtype=torch.complex64)
        return self._apply_single_qubit_gate(state, qubit, matrix)
    
    def _apply_single_qubit_gate(self, state: torch.Tensor, qubit: int, matrix: torch.Tensor) -> torch.Tensor:
        """Apply a single qubit gate"""
        # Reshape state for gate application
        state = state.reshape(-1, 2**self.n_qubits)
        # Apply gate
        for i in range(state.shape[0]):
            # Get amplitudes for |0⟩ and |1⟩ states of target qubit
            even_state = state[i, ::2]
            odd_state = state[i, 1::2]
            # Apply matrix
            state[i, ::2] = matrix[0,0] * even_state + matrix[0,1] * odd_state
            state[i, 1::2] = matrix[1,0] * even_state + matrix[1,1] * odd_state
        return state.reshape(-1, 2**self.n_qubits)
    
    def _apply_noise(self, state: torch.Tensor) -> torch.Tensor:
        """Simulate quantum noise/decoherence"""
        noise = torch.randn_like(state) * self.decoherence_rate
        state = state + noise
        # Renormalize
        state = state / torch.norm(state, dim=-1, keepdim=True)
        return state
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through quantum layer"""
        batch_size = x.shape[0]
        # Initialize quantum state
        state = torch.zeros((batch_size, 2**self.n_qubits), dtype=torch.complex64)
        state[:, 0] = 1  # Initialize to |0⟩ state
        
        # Apply rotation gates
        state = self._apply_rotation(state, self.rx_params, 'x')
        state = self._apply_rotation(state, self.ry_params, 'y')
        state = self._apply_rotation(state, self.rz_params, 'z')
        
        # Apply noise
        state = self._apply_noise(state)
        
        # Project back to real space
        return torch.abs(state)

class QuantumExpert(nn.Module):
    """Expert module that processes information through quantum channels"""
    
    def __init__(self, input_dim: int, output_dim: int, n_qubits: int = 4):
        super().__init__()
        self.pre_quantum = nn.Linear(input_dim, 2**n_qubits)
        self.quantum_layer = QuantumLayer(n_qubits=n_qubits, n_rotations=3)
        self.post_quantum = nn.Linear(2**n_qubits, output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre_quantum(x)
        x = self.quantum_layer(x)
        return self.post_quantum(x)

def quantum_loss(predictions: torch.Tensor, targets: torch.Tensor, quantum_states: List[torch.Tensor]) -> torch.Tensor:
    """Loss function incorporating quantum coherence"""
    # Standard cross entropy
    ce_loss = nn.functional.cross_entropy(predictions, targets)
    
    # Quantum coherence regularization
    coherence_loss = 0
    for state in quantum_states:
        # Calculate von Neumann entropy
        eigenvals = torch.linalg.eigvals(state @ state.conj().transpose(-2, -1))
        entropy = -torch.sum(eigenvals * torch.log(eigenvals + 1e-10))
        coherence_loss += entropy
        
    # Combine losses
    total_loss = ce_loss + 0.1 * coherence_loss
    return total_loss
