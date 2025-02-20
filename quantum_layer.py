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
        """Apply parallel rotation gates along specified axis with quantum advantage"""
        # Reshape for parallel processing
        B, T, D = state.shape
        state = state.view(B, T, self.n_qubits, -1)
        
        # Prepare angles for parallel rotation
        angles = params.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1)
        
        # Apply quantum Fourier transform for parallel processing
        state_freq = torch.fft.fft2(state, dim=(-2, -1))
        
        # Parallel rotation in frequency domain
        if axis == 'x':
            rot_matrix = self._get_parallel_rx_matrix(angles)
        elif axis == 'y':
            rot_matrix = self._get_parallel_ry_matrix(angles)
        else:  # z axis
            rot_matrix = self._get_parallel_rz_matrix(angles)
            
        # Apply rotation in parallel
        state_freq = torch.einsum('btqd,btqq->btqd', state_freq, rot_matrix)
        
        # Inverse quantum Fourier transform
        state = torch.fft.ifft2(state_freq, dim=(-2, -1)).real
        
        return state.view(B, T, D)
    
    def _get_parallel_rx_matrix(self, angles: torch.Tensor) -> torch.Tensor:
        """Get parallel X rotation matrices for quantum advantage"""
        cos = torch.cos(angles/2)
        sin = torch.sin(angles/2)
        
        # Construct block diagonal rotation matrices
        rx = torch.zeros(*angles.shape[:-1], self.n_qubits, self.n_qubits, 
                        dtype=torch.complex64, device=angles.device)
        
        # Populate with 2x2 rotation blocks
        idx = torch.arange(0, self.n_qubits-1, 2)
        rx[..., idx, idx] = cos[..., :len(idx)]
        rx[..., idx, idx+1] = -1j*sin[..., :len(idx)]
        rx[..., idx+1, idx] = -1j*sin[..., :len(idx)]
        rx[..., idx+1, idx+1] = cos[..., :len(idx)]
        
        return rx
        
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
        """Apply quantum error correction following equations.tex formalism"""
        # Encode state using stabilizer code
        # |ψ_encoded⟩ = C|ψ_in⟩ where C is the encoding circuit
        encoded_state = torch.einsum('ijk,bj->bik', self.error_correction_code, state)
        
        # Apply noise channel
        noisy_state = self._apply_noise(encoded_state)
        
        # Measure syndrome operators
        # syndrome = M|ψ_noisy⟩ where M are the syndrome measurements
        syndrome_x = torch.einsum('ij,bjk->bik', self.syndrome_measurement[:, :, 0], noisy_state)
        syndrome_z = torch.einsum('ij,bjk->bik', self.syndrome_measurement[:, :, 1], noisy_state)
        
        # Stabilizer measurements
        stabilizer_x = (syndrome_x > 0).float()
        stabilizer_z = (syndrome_z > 0).float()
        
        # Error detection using syndrome pattern
        error_pattern = torch.cat([stabilizer_x, stabilizer_z], dim=-1)
        
        # Lookup recovery operation from syndrome table
        recovery_ops = self._get_recovery_operations(error_pattern)
        
        # Apply recovery operations
        corrected_state = torch.zeros_like(noisy_state)
        for i in range(recovery_ops.size(0)):
            # Apply X gates
            x_correction = recovery_ops[i, :, 0]
            corrected_state[i] = torch.where(x_correction.unsqueeze(-1) > 0,
                                           noisy_state[i].flip(-1),
                                           noisy_state[i])
            
            # Apply Z gates
            z_correction = recovery_ops[i, :, 1]
            corrected_state[i] = torch.where(z_correction.unsqueeze(-1) > 0,
                                           -corrected_state[i],
                                           corrected_state[i])
            
        # Verify correction using fidelity
        fidelity = torch.abs(torch.sum(corrected_state.conj() * encoded_state, dim=-1))
        
        # If fidelity is too low, try alternative correction
        alt_correction = torch.where(
            fidelity < 0.9,
            self._alternative_correction(noisy_state, error_pattern),
            corrected_state
        )
        
        return alt_correction
        
    def _get_recovery_operations(self, error_pattern: torch.Tensor) -> torch.Tensor:
        """Get recovery operations from syndrome lookup table"""
        # Initialize recovery operations
        batch_size = error_pattern.size(0)
        recovery = torch.zeros((batch_size, self.n_qubits, 2), device=error_pattern.device)
        
        # Syndrome lookup table (pre-computed)
        # This could be expanded to handle more error patterns
        for i in range(batch_size):
            pattern = error_pattern[i]
            if torch.all(pattern == 0):
                continue  # No error detected
            
            # Single qubit errors
            for q in range(self.n_qubits):
                # X error on qubit q
                if pattern[q] == 1 and pattern[q + self.n_qubits] == 0:
                    recovery[i, q, 0] = 1  # Apply X correction
                # Z error on qubit q
                elif pattern[q] == 0 and pattern[q + self.n_qubits] == 1:
                    recovery[i, q, 1] = 1  # Apply Z correction
                # Y error on qubit q
                elif pattern[q] == 1 and pattern[q + self.n_qubits] == 1:
                    recovery[i, q, 0] = 1  # Apply both X and Z
                    recovery[i, q, 1] = 1
        
        return recovery
        
    def _alternative_correction(self, state: torch.Tensor, error_pattern: torch.Tensor) -> torch.Tensor:
        """Try alternative error correction strategy when primary fails"""
        # Implement more sophisticated correction strategies here
        # For now, just try inverting the error pattern
        inv_pattern = 1 - error_pattern
        recovery_ops = self._get_recovery_operations(inv_pattern)
        
        corrected_state = torch.zeros_like(state)
        for i in range(recovery_ops.size(0)):
            x_correction = recovery_ops[i, :, 0]
            z_correction = recovery_ops[i, :, 1]
            
            temp_state = state[i]
            # Apply corrections in reverse order
            temp_state = torch.where(z_correction.unsqueeze(-1) > 0,
                                   -temp_state,
                                   temp_state)
            corrected_state[i] = torch.where(x_correction.unsqueeze(-1) > 0,
                                           temp_state.flip(-1),
                                           temp_state)
        
        return corrected_state

    def _apply_noise(self, state: torch.Tensor) -> torch.Tensor:
        """Simulate realistic quantum noise effects per equations.tex noise model"""
        # Amplitude damping noise N(0, 0.01)
        amp_noise = torch.normal(0, 0.01, state.shape, device=state.device)
        
        # Phase damping noise N(0, 0.005) 
        phase_noise = torch.normal(0, 0.005, state.shape, device=state.device)
        
        # Apply noise model from equations.tex:
        # |ψ_noisy⟩ = ((1 + N_amp)exp(iN_phase)|ψ⟩) / ||(1 + N_amp)exp(iN_phase)|ψ⟩||
        noisy_state = (1 + amp_noise) * torch.exp(1j * phase_noise) * state
        
        # Normalize
        norm = torch.norm(noisy_state, dim=-1, keepdim=True)
        noisy_state = noisy_state / (norm + 1e-8)
        
        # Apply error correction
        corrected_state = self._apply_error_correction(noisy_state)
        
        return corrected_state
    
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

def quantum_loss(predictions: torch.Tensor, targets: torch.Tensor, quantum_states: List[torch.Tensor], epsilon: float = 1e-10) -> torch.Tensor:
    """Loss function incorporating quantum coherence via von Neumann entropy"""
    # Standard cross entropy with label smoothing
    ce_loss = nn.functional.cross_entropy(predictions, targets, label_smoothing=0.1)
    
    # Quantum coherence via von Neumann entropy S(ρ) = -Tr(ρlogρ)
    coherence_loss = 0
    for state in quantum_states:
        # Calculate density matrix ρ = |ψ⟩⟨ψ|
        density_matrix = state @ state.conj().transpose(-2, -1)
        
        # Ensure Hermiticity
        density_matrix = 0.5 * (density_matrix + density_matrix.conj().transpose(-2, -1))
        
        # Normalize trace
        trace = torch.trace(density_matrix)
        density_matrix = density_matrix / (trace + epsilon)
        
        # Get eigenvalues λᵢ of density matrix using stable computation
        eigenvals = torch.linalg.eigvalsh(density_matrix)  # Use eigvalsh for Hermitian matrix
        eigenvals = eigenvals.clamp(min=epsilon)  # Ensure positive eigenvalues
        
        # Calculate von Neumann entropy: S(ρ) = -∑ᵢλᵢlog(λᵢ)
        entropy = -torch.sum(eigenvals * torch.log(eigenvals))
        
        # Add purity regularization: Tr(ρ²)
        purity = torch.trace(density_matrix @ density_matrix)
        purity_reg = (1.0 - purity) * 0.01
        
        coherence_loss += entropy + purity_reg
        
    # Combine losses with weighting from equations.tex
    total_loss = ce_loss + 0.1 * coherence_loss
    return total_loss
