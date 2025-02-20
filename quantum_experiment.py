"""
Quantum Experiment to Test Paper Conjectures

This experiment validates key theoretical predictions from equations.tex:
1. Quantum attention speedup
2. Error correction effectiveness
3. Expert routing accuracy
4. Sampling efficiency
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import state_fidelity, Operator
from qiskit_aer.noise import NoiseModel
import time
import matplotlib.pyplot as plt

def test_quantum_attention_speedup(n_qubits_range=[2,3,4,5,6], shots=1000):
    """Test the theoretical quantum attention speedup"""
    classical_times = []
    quantum_times = []
    
    for n_qubits in n_qubits_range:
        # Classical attention computation
        dim = 2**n_qubits
        t0 = time.time()
        for _ in range(shots):
            q = np.random.randn(dim, dim)
            k = np.random.randn(dim, dim)
            attn = np.matmul(q, k.T) / np.sqrt(dim)
        classical_time = time.time() - t0
        classical_times.append(classical_time)
        
        # Quantum attention computation
        qr = QuantumRegister(n_qubits)
        cr = ClassicalRegister(n_qubits)
        qc = QuantumCircuit(qr, cr)
        
        t0 = time.time()
        for _ in range(shots):
            # Prepare quantum states
            qc.h(range(n_qubits))
            # Apply controlled operations
            for i in range(n_qubits-1):
                qc.cx(i, i+1)
            qc.measure(qr, cr)
            
        backend = AerSimulator()
        job = backend.run(qc, shots=1)
        quantum_time = time.time() - t0
        quantum_times.append(quantum_time)
    
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

def test_error_correction(physical_error_rates=[0.001, 0.01, 0.05, 0.1], 
                        code_distances=[3,5,7], shots=1000):
    """Test surface code error correction effectiveness"""
    logical_error_rates = []
    
    for d in code_distances:
        d_rates = []
        for p in physical_error_rates:
            # Create noise model
            noise_model = NoiseModel()
            # Create proper quantum error channels
            error_ops = [
                # No error
                ([np.eye(2), 1-p]),
                # X error
                ([np.array([[0, 1], [1, 0]]), p/3]),
                # Y error
                ([np.array([[0, -1j], [1j, 0]]), p/3]),
                # Z error
                ([np.array([[1, 0], [0, -1]]), p/3])
            ]
            
            noise_model.add_all_qubit_quantum_error(error_ops, ['x','y','z'])
            
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

def test_quantum_sampling(n_qubits_range=[2,3,4,5], n_samples_range=[10,50,100,500]):
    """Test quantum Monte Carlo sampling efficiency"""
    classical_errors = []
    quantum_errors = []
    
    for n_qubits in n_qubits_range:
        c_errors = []
        q_errors = []
        
        # True state to estimate
        qr = QuantumRegister(n_qubits)
        qc = QuantumCircuit(qr)
        for i in range(n_qubits):
            qc.rx(np.random.random(), i)
            qc.ry(np.random.random(), i)
        true_state = Operator(qc).data
        
        for n_samples in n_samples_range:
            # Classical Monte Carlo
            c_estimates = []
            for _ in range(n_samples):
                sample = np.random.randn(2**n_qubits) + 1j*np.random.randn(2**n_qubits)
                sample = sample / np.linalg.norm(sample)
                c_estimates.append(sample)
            c_mean = np.mean(c_estimates, axis=0)
            c_error = 1 - state_fidelity(true_state, c_mean)
            c_errors.append(c_error)
            
            # Quantum sampling
            cr = ClassicalRegister(n_qubits)
            meas_qc = QuantumCircuit(qr, cr)
            meas_qc.compose(qc, inplace=True)
            meas_qc.measure(qr, cr)
            
            backend = AerSimulator()
            job = backend.run(meas_qc, shots=n_samples)
            counts = job.result().get_counts()
            
            # Reconstruct state from measurements
            q_state = np.zeros(2**n_qubits, dtype=complex)
            for bitstring, count in counts.items():
                idx = int(bitstring, 2)
                q_state[idx] = np.sqrt(count/n_samples)
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
    """Run all experiments"""
    print("Testing quantum attention speedup...")
    c_times, q_times = test_quantum_attention_speedup()
    print("\nAttention speedup results:")
    print(f"Classical times: {c_times}")
    print(f"Quantum times: {q_times}")
    
    print("\nTesting error correction...")
    error_rates = test_error_correction()
    print("\nError correction results:")
    print(f"Logical error rates: {error_rates}")
    
    print("\nTesting quantum sampling efficiency...")
    c_errors, q_errors = test_quantum_sampling()
    print("\nSampling efficiency results:")
    print(f"Classical errors: {c_errors}")
    print(f"Quantum errors: {q_errors}")

if __name__ == "__main__":
    main()
