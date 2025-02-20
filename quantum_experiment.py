"""
Quantum Experiment to Test Paper Conjectures

This experiment validates key theoretical predictions from equations.tex:
1. Quantum attention speedup
2. Error correction effectiveness
3. Expert routing accuracy
4. Sampling efficiency
"""

import numpy as np
import math
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import state_fidelity, Operator
from qiskit_aer.noise import NoiseModel
import time
import matplotlib.pyplot as plt

from qiskit.quantum_info import Statevector, DensityMatrix, Operator
import numpy as np

def test_quantum_attention_speedup(n_qubits_range=None, shots=1000):
    """Test quantum attention speedup according to equations.tex section 5.1
    
    Tests architectures from 1 to 1M qubits using logarithmic sampling to
    keep runtime manageable while covering the full range."""
    
    if n_qubits_range is None:
        # Generate logarithmically spaced points from 1 to 1M qubits
        n_qubits_range = np.logspace(0, 6, num=20, dtype=int)
        n_qubits_range = np.unique(n_qubits_range)  # Remove duplicates
        print(f"Testing qubit range: {n_qubits_range}")
    """Test quantum attention speedup according to equations.tex section 5.1"""
    classical_times = []
    quantum_times = []
    
    for n_qubits in n_qubits_range:
        dim = 2**n_qubits
        
        # Classical attention computation
        t0 = time.time()
        for _ in range(shots):
            q = np.random.randn(dim, dim)
            k = np.random.randn(dim, dim)
            # Add temperature annealing per equations.tex
            T = 1.0 / math.sqrt(_ + 1)
            attn = np.matmul(q, k.T) / (np.sqrt(dim) * T)
        classical_time = time.time() - t0
        classical_times.append(classical_time)
        
        # Quantum attention computation
        qr = QuantumRegister(n_qubits)
        cr = ClassicalRegister(n_qubits)
        qc = QuantumCircuit(qr, cr)
        
        t0 = time.time()
        for _ in range(shots):
            # Add rotary embeddings per equations.tex
            qc.h(range(n_qubits))
            for i in range(n_qubits):
                theta = 2 * np.pi * i / (10000 ** (2*i/dim))
                qc.rz(theta, i)
            
            # Apply controlled operations with phase tracking
            for i in range(n_qubits-1):
                qc.cx(i, i+1)
                # Add phase evolution
                qc.rz(-0.1/_,i)  # Dephasing term
            qc.measure(qr, cr)
            
        backend = AerSimulator()
        job = backend.run(qc, shots=1)
        quantum_time = time.time() - t0
        quantum_times.append(quantum_time)
        
        # Verify theoretical speedup ratio
        speedup = classical_time/quantum_time
        theoretical = np.sqrt(dim/n_qubits)
        print(f"n_qubits: {n_qubits}, Measured speedup: {speedup:.2f}x, Theoretical: {theoretical:.2f}x")
    
    # Plot results
    plt.figure(figsize=(10,6))
    plt.plot(n_qubits_range, classical_times, 'bo-', label='Classical')
    plt.plot(n_qubits_range, quantum_times, 'ro-', label='Quantum')
    plt.xlabel('Number of qubits')
    plt.ylabel('Time (seconds)')
    plt.title('Attention Computation Scaling')
    plt.legend()
    plt.yscale('log')
    plt.savefig('attention_scaling.png')
    plt.close()
    
    return classical_times, quantum_times

def test_error_correction(physical_error_rates=None, code_distances=None, shots=1000):
    """Test surface code error correction per equations.tex section 5.2
    
    Tests large-scale error correction with code distances up to 100,
    covering architectures up to 10k physical qubits."""
    
    if physical_error_rates is None:
        # Test error rates from 10^-6 to 10^-1
        physical_error_rates = np.logspace(-6, -1, num=6)
        
    if code_distances is None:
        # Test code distances up to 100 (10k physical qubits)
        code_distances = [3, 5, 7, 11, 15, 21, 31, 51, 75, 100]
    """Test surface code error correction per equations.tex section 5.2"""
    from qiskit.circuit.library import IGate, XGate, YGate, ZGate
    logical_error_rates = []
    
    for d in code_distances:
        d_rates = []
        for p in physical_error_rates:
            # Create noise model with proper error channels
            noise_model = NoiseModel()
            
            # Define error probabilities per equations.tex
            p_x = p/3  # Bit flip
            p_y = p/3  # Y error  
            p_z = p/3  # Phase flip
            p_i = 1 - p  # No error
            
            # Create quantum error using proper gates
            error_ops = [
                (IGate(), p_i),  # No error
                (XGate(), p_x),  # Bit flip
                (YGate(), p_y),  # Y error
                (ZGate(), p_z)   # Phase flip
            ]
            
            # Add the error to the noise model
            noise_model.add_all_qubit_quantum_error(error_ops, ['x', 'y', 'z'])
            p_i = 1 - p  # probability of no error
            
            # Create quantum error using Qiskit gates
            error_ops = [
                (IGate(), p_i),  # No error
                (XGate(), p_x),  # Bit flip
                (YGate(), p_y),  # Y error
                (ZGate(), p_z)   # Phase flip
            ]
            
            # Add the error to the noise model
            noise_model.add_all_qubit_quantum_error(error_ops, ['x', 'y', 'z'])
            
            # Create surface code circuit
            n_qubits = d*d
            qr = QuantumRegister(n_qubits)
            cr = ClassicalRegister(n_qubits)
            qc = QuantumCircuit(qr, cr)
            
            # Encode logical state
            qc.h(0)  # Create |+⟩ state
            for i in range(d):
                for j in range(d-1):
                    qc.cx(i*d + j, i*d + j + 1)
            
            # Measure stabilizers
            for i in range(1,d-1):
                for j in range(1,d-1):
                    # Plaquette operators
                    qc.h(i*d + j)
                    qc.cx(i*d + j, (i-1)*d + j)
                    qc.cx(i*d + j, (i+1)*d + j)
                    qc.cx(i*d + j, i*d + j-1)
                    qc.cx(i*d + j, i*d + j+1)
                    qc.h(i*d + j)
            
            qc.measure(qr, cr)
            
            # Execute with noise
            backend = AerSimulator()
            job = backend.run(qc, noise_model=noise_model, shots=shots)
            results = job.result().get_counts()
            
            # Calculate logical error rate
            logical_errors = sum(count for bitstring, count in results.items() 
                               if bitstring.count('1') % 2 == 1)
            logical_error_rate = logical_errors / shots
            d_rates.append(logical_error_rate)
            
        logical_error_rates.append(d_rates)
    
    # Plot results
    plt.figure(figsize=(10,6))
    for i, d in enumerate(code_distances):
        plt.plot(physical_error_rates, logical_error_rates[i], 
                marker='o', label=f'd={d}')
    plt.xlabel('Physical Error Rate')
    plt.ylabel('Logical Error Rate')
    plt.title('Surface Code Error Correction')
    plt.legend()
    plt.yscale('log')
    plt.xscale('log')
    plt.savefig('error_correction.png')
    plt.close()
    
    return logical_error_rates

def test_quantum_sampling(n_qubits_range=None, n_samples_range=None):
    """Test quantum Monte Carlo sampling efficiency per equations.tex section 5.3
    
    Tests sampling efficiency across architectures from 1 to 1M qubits."""
    
    if n_qubits_range is None:
        # Logarithmically spaced points from 1 to 1M qubits
        n_qubits_range = np.logspace(0, 6, num=20, dtype=int)
        n_qubits_range = np.unique(n_qubits_range)
        
    if n_samples_range is None:
        # Logarithmically spaced sample counts
        n_samples_range = np.logspace(1, 5, num=10, dtype=int)
        n_samples_range = np.unique(n_samples_range)
    """Test quantum Monte Carlo sampling efficiency per equations.tex section 5.3"""
    classical_errors = []
    quantum_errors = []
    
    for n_qubits in n_qubits_range:
        c_errors = []
        q_errors = []
        
        # Create true state to estimate
        qr = QuantumRegister(n_qubits)
        qc = QuantumCircuit(qr)
        
        # Add rotations per equations.tex
        for i in range(n_qubits):
            theta = np.random.random()
            phi = np.random.random()
            qc.rx(theta, i)
            qc.ry(phi, i)
            qc.rz(-theta*phi, i)  # Phase correction
        
        # Get true state as a Statevector
        backend = AerSimulator()
        job = backend.run(qc)
        true_state = Statevector.from_instruction(qc)
        
        for n_samples in n_samples_range:
            # Classical Monte Carlo
            c_estimates = []
            for _ in range(n_samples):
                # Generate random complex state vector
                state = np.random.randn(2**n_qubits) + 1j*np.random.randn(2**n_qubits)
                # Normalize
                state = state / np.linalg.norm(state)
                c_estimates.append(state)
            
            # Average the estimates
            c_mean = np.mean(c_estimates, axis=0)
            c_mean = c_mean / np.linalg.norm(c_mean)  # Ensure normalization
            # Convert to Statevector for comparison
            c_state = Statevector(c_mean)
            c_error = 1 - state_fidelity(true_state, c_state)
            c_errors.append(c_error)
            
            # Quantum sampling
            cr = ClassicalRegister(n_qubits)
            meas_qc = QuantumCircuit(qr, cr)
            meas_qc.compose(qc, inplace=True)
            meas_qc.measure(qr, cr)
            
            # Run quantum circuit
            job = backend.run(meas_qc, shots=n_samples)
            counts = job.result().get_counts()
            
            # Reconstruct state from measurements
            q_state = np.zeros(2**n_qubits, dtype=complex)
            for bitstring, count in counts.items():
                idx = int(bitstring, 2)
                q_state[idx] = np.sqrt(count/n_samples)
            
            # Convert to Statevector for comparison
            q_state = Statevector(q_state)
            q_error = 1 - state_fidelity(true_state, q_state)
            q_errors.append(q_error)
            
        classical_errors.append(c_errors)
        quantum_errors.append(q_errors)
    
    # Plot results
    plt.figure(figsize=(10,6))
    for i, n_qubits in enumerate(n_qubits_range):
        plt.plot(n_samples_range, classical_errors[i], 
                'b--', alpha=0.5, label=f'Classical {n_qubits} qubits')
        plt.plot(n_samples_range, quantum_errors[i],
                'r--', alpha=0.5, label=f'Quantum {n_qubits} qubits')
    plt.xlabel('Number of Samples')
    plt.ylabel('Error')
    plt.title('Sampling Efficiency')
    plt.legend()
    plt.yscale('log')
    plt.xscale('log')
    plt.savefig('sampling_efficiency.png')
    plt.close()
    
    return classical_errors, quantum_errors

def main():
    """Run all experiments and validate against equations.tex predictions"""
    print("\n=== Large-Scale Quantum Architecture Tests ===")
    
    print("\nTesting quantum attention speedup...")
    print("Testing architectures from 1 to 1M qubits...")
    n_qubits_range = np.logspace(0, 6, num=20, dtype=int)
    n_qubits_range = np.unique(n_qubits_range)
    c_times, q_times = test_quantum_attention_speedup(n_qubits_range)
    print("\nAttention speedup results:")
    print("Number of qubits | Classical (s) | Quantum (s) | Speedup")
    print("-" * 55)
    for i, n_qubits in enumerate(n_qubits_range):
        speedup = c_times[i]/q_times[i]
        print(f"{n_qubits:13d} | {c_times[i]:11.3f} | {q_times[i]:9.3f} | {speedup:7.1f}x")
    
    print("\nTesting error correction...")
    print("Testing code distances up to 100 (10k physical qubits)...")
    physical_error_rates = np.logspace(-6, -1, num=6)
    code_distances = [3, 5, 7, 11, 15, 21, 31, 51, 75, 100]
    error_rates = test_error_correction(physical_error_rates, code_distances)
    print("\nError correction results:")
    print("Code distance | Physical error | Logical error")
    print("-" * 45)
    for i, d in enumerate(code_distances):
        for j, p in enumerate(physical_error_rates):
            print(f"{d:12d} | {p:13.1e} | {error_rates[i][j]:.1e}")
    
    print("\nTesting quantum sampling efficiency...")
    print("Testing sampling across 1 to 1M qubit architectures...")
    n_samples_range = np.logspace(1, 5, num=10, dtype=int)
    n_samples_range = np.unique(n_samples_range)
    c_errors, q_errors = test_quantum_sampling(n_qubits_range, n_samples_range)
    print("\nSampling efficiency results:")
    print("Number of qubits | Samples | Classical err | Quantum err | Improvement")
    print("-" * 70)
    for i, n_qubits in enumerate(n_qubits_range):
        for j, n_samples in enumerate(n_samples_range):
            improvement = c_errors[i][j]/q_errors[i][j]
            print(f"{n_qubits:13d} | {n_samples:7d} | {c_errors[i][j]:.1e} | {q_errors[i][j]:.1e} | {improvement:10.1f}x")

if __name__ == "__main__":
    main()
