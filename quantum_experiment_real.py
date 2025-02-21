# Import required libraries
import numpy as np
from qiskit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
from qiskit import QuantumCircuit, transpile
import timeit
import os

# Classical Attention Mechanism
def classical_attention(Q, K, V):
    """
    Classical dot-product attention mechanism.
    Args:
        Q: Query matrix (n x d)
        K: Key matrix (n x d)
        V: Value matrix (n x d)
    Returns:
        output: Weighted sum of values based on attention weights
    """
    # Scaled dot-product attention
    scores = np.dot(Q, K.T) / np.sqrt(K.shape[1])  # Scaled dot-product
    weights = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)  # Softmax
    output = np.dot(weights, V)  # Weighted sum
    return output

def quantum_attention_hardware(Q, K, V, n_qubits, backend):
    # Create a quantum circuit
    qc = QuantumCircuit(n_qubits)
    
    # Apply Hadamard gates for superposition
    for i in range(n_qubits):
        qc.h(i)
    
    # Encode Q and K as rotation angles
    for i in range(n_qubits):
        qc.ry(Q[i], i)  # Encode Q
        qc.rz(K[i], i)  # Encode K
    
    # Entangle qubits to model attention weights
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)
    
    # Measure all qubits
    qc.measure_all()
    
    # Transpile the quantum circuit
    transpiled_qc = transpile(qc, backend)
    
    # Use the Sampler primitive to execute the transpiled circuit on the backend
    sampler = Sampler(backend)
    job = sampler.run([transpiled_qc], shots=1024)
    result = job.result()[0]  # Get the result for the first (and only) circuit
    
    # Post-process results to compute attention weights
    weights = np.zeros(n_qubits)
    for state, prob in result.data.meas.get_counts().items():
        for i in range(n_qubits):
            if state[i] == '1':  # Check if the i-th bit is 1
                weights[i] += prob
    weights /= np.sum(weights)  # Normalize
    
    # Compute output as weighted sum of V
    output = np.dot(weights, V)
    return output

# Main Experiment
if __name__ == "__main__":
    # Define inputs
    n_qubits = 127
    d = 2  # Dimension of Q, K, V

    # Set up IBM Quantum account (replace with your token)
    service = QiskitRuntimeService(channel='ibm_quantum', token=os.environ.get('IBM_TOKEN'))
    backend = service.backend("ibm_brisbane")  # Use a device with 127 qubits

    print(f"Running Classical Attention: n_qubits: {n_qubits}")
    Q = np.random.rand(n_qubits, d)  # Query matrix
    K = np.random.rand(n_qubits, d)  # Key matrix
    V = np.random.rand(n_qubits, d)  # Value matrix

    # Run Classical Attention
    classical_time = timeit.timeit(
        lambda: classical_attention(Q, K, V), number=10
    ) / 10  # Average over 10 runs
    classical_output = classical_attention(Q, K, V)
    print(f"Classical Output shape: {classical_output.shape}")
    print(f"Classical Time (avg): {classical_time:.6f} seconds\n")

    # Run Quantum Attention (Hardware)
    print("Running Quantum Attention (Hardware)...")
    Q_flat = Q.flatten()
    K_flat = K.flatten()
    quantum_time_hw = timeit.timeit(
        lambda: quantum_attention_hardware(Q_flat, K_flat, V, n_qubits, backend), number=1
    )  # Single run (hardware is slower)
    quantum_output_hw = quantum_attention_hardware(Q_flat, K_flat, V, n_qubits, backend)
    print(f"Quantum Output (Hardware) shape: {quantum_output_hw.shape}")
    print(f"Quantum Time (Hardware): {quantum_time_hw:.6f} seconds\n")

    # Compare Results
    print("Comparison:")
    print(f"Classical Output shape: {classical_output.shape}")
    print(f"Quantum Output (Hardware) shape: {quantum_output_hw.shape}")
    print(f"Classical Time (avg): {classical_time:.6f} seconds")
    print(f"Quantum Time (Hardware): {quantum_time_hw:.6f} seconds")