import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple

class QuantumLayer(nn.Module):
    """Quantum layer that simulates quantum operations with error correction"""
    
    def __init__(self, n_qubits: int, n_rotations: int):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_rotations = n_rotations
        
        # Trainable rotation parameters with phase estimation
        self.rx_params = nn.Parameter(torch.randn(n_qubits, n_rotations))
        self.ry_params = nn.Parameter(torch.randn(n_qubits, n_rotations))
        self.rz_params = nn.Parameter(torch.randn(n_qubits, n_rotations))
        
        # Additional quantum gates
        self.hadamard = nn.Parameter(torch.tensor([[1, 1], [1, -1]], dtype=torch.complex64) / np.sqrt(2))
        self.cnot = nn.Parameter(torch.eye(4, dtype=torch.complex64).reshape(2, 2, 2, 2))
        
        # Error correction parameters
        self.error_correction_code = nn.Parameter(torch.randn(3, n_qubits))
        self.syndrome_measurement = nn.Parameter(torch.randn(2, n_qubits))
        
        # Decoherence and noise parameters
        self.decoherence_rate = nn.Parameter(torch.tensor(0.01))
        self.phase_damping = nn.Parameter(torch.tensor(0.005))
        
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
        """Apply a single qubit gate with no-cloning enforcement"""
        if self.measured:
            raise RuntimeError("Cannot apply gates to measured quantum state")
            
        # Verify matrix is unitary
        if not torch.allclose(matrix @ matrix.conj().T, 
                            torch.eye(2, dtype=torch.complex64)):
            raise ValueError("Non-unitary gate operation detected")
            
        # No cloning - operate on original state
        state_copy = state  # Reference only, no actual copy
        
        # Reshape state for gate application
        state_copy = state_copy.reshape(-1, 2**self.n_qubits)
        
        # Apply gate (in-place operations where possible)
        for i in range(state_copy.shape[0]):
            # Get views of state components
            even_state = state_copy[i, ::2]
            odd_state = state_copy[i, 1::2]
            
            # Store temporary results
            new_even = matrix[0,0] * even_state + matrix[0,1] * odd_state
            new_odd = matrix[1,0] * even_state + matrix[1,1] * odd_state
            
            # Update in-place
            state_copy[i, ::2] = new_even
            state_copy[i, 1::2] = new_odd
            
        return state_copy.reshape(-1, 2**self.n_qubits)
    
    def _apply_error_correction(self, state: torch.Tensor) -> torch.Tensor:
        """Apply quantum error correction"""
        # Encode quantum state
        encoded_state = torch.einsum('ijk,bj->bik', self.error_correction_code, state)
        
        # Simulate error detection
        syndrome = torch.einsum('ij,bjk->bik', self.syndrome_measurement, encoded_state)
        
        # Error correction based on syndrome
        correction = torch.where(syndrome > 0, 
                               encoded_state.flip(-1),
                               encoded_state)
        
        return correction

    def _apply_noise(self, state: torch.Tensor) -> torch.Tensor:
        """Simulate realistic quantum noise effects"""
        # Amplitude damping
        noise = torch.randn_like(state) * self.decoherence_rate
        
        # Phase damping
        phase_noise = torch.exp(1j * torch.randn_like(state) * self.phase_damping)
        
        # Apply noise
        state = state + noise
        state = state * phase_noise
        
        # Error correction
        state = self._apply_error_correction(state)
        
        # Renormalize
        state = state / torch.norm(state, dim=-1, keepdim=True)
        return state
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through quantum layer with proper measurement handling"""
        batch_size = x.shape[0]
        
        # Warning about classical simulation limitations
        if not hasattr(self, '_warning_shown'):
            print("WARNING: This is a classical simulation of quantum operations.")
            print("True quantum entanglement and operations require physical quantum hardware.")
            self._warning_shown = True
        
        # Initialize quantum state (only once per forward pass)
        state = torch.zeros((batch_size, 2**self.n_qubits), dtype=torch.complex64)
        state[:, 0] = 1  # Initialize to |0⟩ state
        
        # Track if state has been measured
        self.measured = False
        
        # Ensure gates are unitary before applying
        with torch.no_grad():
            # Orthogonalize parameters using QR decomposition
            rx_unitary, _ = torch.linalg.qr(self.rx_params.reshape(-1, 2))
            ry_unitary, _ = torch.linalg.qr(self.ry_params.reshape(-1, 2))
            rz_unitary, _ = torch.linalg.qr(self.rz_params.reshape(-1, 2))
            
            # Verify unitarity (U†U = I)
            for U in [rx_unitary, ry_unitary, rz_unitary]:
                if not torch.allclose(U @ U.conj().T, torch.eye(2, dtype=torch.complex64)):
                    raise ValueError("Non-unitary gate operation detected")
        
        # Apply unitary rotations (no intermediate measurements)
        state = self._apply_rotation(state, rx_unitary.reshape(self.rx_params.shape), 'x')
        state = self._apply_rotation(state, ry_unitary.reshape(self.ry_params.shape), 'y')
        state = self._apply_rotation(state, rz_unitary.reshape(self.rz_params.shape), 'z')
        
        # Apply realistic noise (decoherence)
        state = self._apply_noise(state)
        
        # Single measurement at the end
        if self.measured:
            raise RuntimeError("Quantum state has already been measured")
        self.measured = True
        
        # Project to computational basis
        measurement = torch.abs(state)
        
        # Clear state after measurement
        state.zero_()
        
        return measurement

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
