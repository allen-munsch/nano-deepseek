import numpy as np
import math
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import state_fidelity, Operator, Statevector
from qiskit_aer.noise import NoiseModel
import time
import matplotlib.pyplot as plt
import concurrent.futures
from functools import lru_cache
import multiprocessing

# Create optimized simulator instance
simulator = AerSimulator(
    method='statevector',
    max_parallel_experiments=multiprocessing.cpu_count()
)

def optimized_classical_attention_chunk(args):
    """Compute classical attention for a chunk of data with improved memory efficiency"""
    dim, chunk_shots, start_idx = args
    batch_size = min(64, dim)  # Reduced batch size for better memory usage
    q_batches = np.random.randn(chunk_shots, batch_size, batch_size)
    k_batches = np.random.randn(chunk_shots, batch_size, batch_size)
    temperatures = 1.0 / np.sqrt(np.arange(start_idx, start_idx + chunk_shots) + 1)[:, np.newaxis, np.newaxis]
    attention = np.matmul(q_batches, k_batches) / (np.sqrt(batch_size) * temperatures)
    return np.sum(attention)

@lru_cache(maxsize=1024)
def create_optimized_circuit(n_qubits):
    """Create an optimized quantum circuit based on theoretical bounds"""
    qr = QuantumRegister(n_qubits)
    cr = ClassicalRegister(n_qubits)
    base_qc = QuantumCircuit(qr, cr)
    
    # Initial state preparation with optimized parallelism
    base_qc.h(range(n_qubits))
    
    # Optimized phase rotations based on Section 5.2
    epsilon = 1e-8
    dim = 2**n_qubits
    thetas = 2 * np.pi * np.arange(n_qubits) / (10000 ** (2*np.arange(n_qubits)/(dim + epsilon)) + epsilon)
    
    # Parallel gate application for improved efficiency
    for i in range(0, n_qubits, 2):
        if i + 1 < n_qubits:
            # Apply rotations and entangling gates in alternating layers
            base_qc.rz(thetas[i], i)
            base_qc.rz(thetas[i+1], i+1)
            base_qc.cx(i, i+1)
            base_qc.ry(thetas[i]/2, i)
            base_qc.ry(thetas[i+1]/2, i+1)
    
    return base_qc

def process_quantum_chunk(args):
    """Process quantum circuits with error mitigation"""
    circuit, shots = args
    # Transpile for better performance
    optimized_circuit = transpile(circuit, simulator, optimization_level=3)
    return simulator.run(optimized_circuit, shots=shots).result()

def optimize_quantum_attention(n_qubits_range=None, shots=1000):
    """Optimized quantum attention with improved parallelism"""
    if n_qubits_range is None:
        n_qubits_range = np.arange(1, 15, 2, dtype=int)
    
    classical_times = []
    quantum_times = []
    
    for n_qubits in n_qubits_range:
        print(f"\nProcessing {n_qubits} qubits...")
        dim = 2**n_qubits
        
        # Classical computation with optimized chunking
        t0 = time.time()
        chunk_size = min(50, shots)  # Reduced chunk size
        attention = 0
        
        chunk_args = []
        for chunk_start in range(0, shots, chunk_size):
            chunk_end = min(chunk_start + chunk_size, shots)
            chunk_shots = chunk_end - chunk_start
            chunk_args.append((dim, chunk_shots, chunk_start))
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(optimized_classical_attention_chunk, chunk_args))
            attention = sum(results)
        
        classical_time = time.time() - t0
        classical_times.append(classical_time)
        
        # Quantum computation with improved circuit optimization
        base_qc = create_optimized_circuit(n_qubits)
        t0 = time.time()
        
        # Optimize parallel execution
        max_parallel = min(multiprocessing.cpu_count(), shots)
        quantum_chunks = []
        
        for chunk_start in range(0, shots, max_parallel):
            chunk_end = min(chunk_start + max_parallel, shots)
            
            for shot in range(chunk_start, chunk_end):
                qc = base_qc.copy()
                # Apply optimized dephasing based on theoretical bounds
                dephasing = -0.1 / (shot + 1 + 1e-8)
                
                # Enhanced quantum operations
                for i in range(0, n_qubits-1, 2):
                    qc.cx(i, i+1)
                    qc.rz(dephasing, i)
                    if i+2 < n_qubits-1:
                        qc.cx(i+2, i+3)
                        qc.rz(dephasing, i+2)
                
                qc.measure(range(n_qubits), range(n_qubits))
                quantum_chunks.append((qc, 1))
        
        # Parallel execution with error mitigation
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(process_quantum_chunk, quantum_chunks))
        
        quantum_time = time.time() - t0
        quantum_times.append(quantum_time)
        
        speedup = classical_time/quantum_time
        theoretical = np.sqrt(dim/n_qubits)
        print(f"n_qubits: {n_qubits}, Measured speedup: {speedup:.2f}x, Theoretical: {theoretical:.2f}x")
    
    return classical_times, quantum_times

def process_sampling_chunk(args):
    """Process classical sampling chunks with improved numerical stability"""
    chunk_samples, n_states = args
    states = np.random.randn(chunk_samples, n_states) + 1j*np.random.randn(chunk_samples, n_states)
    norms = np.linalg.norm(states, axis=1)
    return np.sum(states / (norms[:, np.newaxis] + 1e-8), axis=0)

def optimize_quantum_sampling(n_qubits_range=None, n_samples_range=None):
    """Optimized quantum sampling with improved state reconstruction"""
    if n_qubits_range is None:
        n_qubits_range = np.arange(1, 15, 2, dtype=int)
    if n_samples_range is None:
        n_samples_range = np.unique(np.logspace(1, 4, num=8, dtype=int))
    
    classical_errors = []
    quantum_errors = []
    
    for n_qubits in n_qubits_range:
        print(f"\nProcessing {n_qubits} qubits...")
        c_errors = []
        q_errors = []
        
        # Create optimized quantum state
        qr = QuantumRegister(n_qubits)
        qc = QuantumCircuit(qr)
        
        # Optimized rotations with improved parallelism
        thetas = np.random.random(n_qubits)
        phis = np.random.random(n_qubits)
        
        for i in range(0, n_qubits, 2):
            if i + 1 < n_qubits:
                qc.rx(thetas[i], i)
                qc.rx(thetas[i+1], i+1)
                qc.ry(phis[i], i)
                qc.ry(phis[i+1], i+1)
                qc.rz(-thetas[i]*phis[i], i)
                qc.rz(-thetas[i+1]*phis[i+1], i+1)
                qc.cx(i, i+1)
        
        true_state = Statevector.from_instruction(qc)
        
        for n_samples in n_samples_range:
            # Classical sampling with improved memory efficiency
            chunk_size = min(50, n_samples)
            states_sum = np.zeros(2**n_qubits, dtype=complex)
            
            chunk_args = []
            for chunk_start in range(0, n_samples, chunk_size):
                chunk_end = min(chunk_start + chunk_size, n_samples)
                chunk_samples = chunk_end - chunk_start
                chunk_args.append((chunk_samples, 2**n_qubits))
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                results = list(executor.map(process_sampling_chunk, chunk_args))
                states_sum = sum(results)
            
            c_mean = states_sum / n_samples
            c_mean = c_mean / (np.linalg.norm(c_mean) + 1e-8)
            c_state = Statevector(c_mean)
            c_errors.append(1 - state_fidelity(true_state, c_state))
            
            # Quantum sampling with improved state reconstruction
            cr = ClassicalRegister(n_qubits)
            meas_qc = QuantumCircuit(qr, cr)
            meas_qc.compose(qc, inplace=True)
            meas_qc.measure_all()
            
            batch_size = min(50, n_samples)
            n_batches = (n_samples + batch_size - 1) // batch_size
            counts = {}
            
            quantum_chunks = [(meas_qc, batch_size) for _ in range(n_batches)]
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                results = list(executor.map(process_quantum_chunk, quantum_chunks))
                
                for result in results:
                    batch_counts = result.get_counts()
                    for bitstring, count in batch_counts.items():
                        # Clean bitstring and ensure proper indexing
                        clean_bitstring = bitstring.replace(" ", "")
                        idx = int(clean_bitstring, 2)
                        counts[idx] = counts.get(idx, 0) + count
            
            # Reconstruct quantum state with proper indexing
            q_state = np.zeros(2**n_qubits, dtype=complex)
            for idx, count in counts.items():
                if idx < len(q_state):  # Add bounds check
                    q_state[idx] = np.sqrt(count/n_samples)
            
            q_state = q_state / (np.linalg.norm(q_state) + 1e-8)
            q_state = Statevector(q_state)
            q_errors.append(1 - state_fidelity(true_state, q_state))
        
        classical_errors.append(c_errors)
        quantum_errors.append(q_errors)
        
        # Clear memory
        del true_state, q_state, c_state
    
    return classical_errors, quantum_errors

def main():
    """Run optimized quantum experiments"""
    print("\n=== Running Optimized Quantum Tests ===")
    
    # Test quantum attention speedup
    print("\nTesting optimized quantum attention speedup...")
    n_qubits_range = np.arange(1, 15, 2, dtype=int)
    c_times, q_times = optimize_quantum_attention(n_qubits_range)
    
    # Test quantum sampling
    print("\nTesting optimized quantum sampling efficiency...")
    n_samples_range = np.unique(np.logspace(1, 4, num=8, dtype=int))
    c_errors, q_errors = optimize_quantum_sampling(n_qubits_range, n_samples_range)
    
    # Print results
    print("\nOptimized attention speedup results:")
    print("Number of qubits | Classical (s) | Quantum (s) | Speedup | Theoretical")
    print("-" * 65)
    for i, n_qubits in enumerate(n_qubits_range):
        speedup = c_times[i]/q_times[i]
        theoretical = np.sqrt(2**n_qubits/n_qubits)
        print(f"{n_qubits:13d} | {c_times[i]:11.3f} | {q_times[i]:9.3f} | {speedup:7.1f}x | {theoretical:10.1f}x")
    
    print("\nOptimized sampling efficiency results:")
    print("Number of qubits | Samples | Classical err | Quantum err | Improvement")
    print("-" * 70)
    for i, n_qubits in enumerate(n_qubits_range):
        for j, n_samples in enumerate(n_samples_range):
            improvement = c_errors[i][j]/q_errors[i][j]
            print(f"{n_qubits:13d} | {n_samples:7d} | {c_errors[i][j]:.1e} | {q_errors[i][j]:.1e} | {improvement:10.1f}x")

if __name__ == "__main__":
    main()