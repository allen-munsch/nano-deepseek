from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.providers.aer import AerSimulator
from qiskit.quantum_info import Operator, DensityMatrix
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.fake_provider import FakeManila
from qiskit.transpiler import CouplingMap
from qiskit.transpiler.passes import SabreLayout, SabreSwap
import numpy as np

class QuantumProcessor:
    """Quantum processor using real quantum circuits via Qiskit"""
    
    def __init__(self, n_qubits: int, qnn_arch: Optional[List[int]] = None):
        # Validate against hardware constraints
        self.device_backend = FakeManila()
        max_qubits = self.device_backend.configuration().n_qubits
        if n_qubits > max_qubits:
            raise ValueError(f"Requested {n_qubits} qubits exceeds device maximum of {max_qubits}")
            
        # Store QNN architecture if provided
        self.qnn_arch = qnn_arch
            
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits)
        self.cr = ClassicalRegister(n_qubits)
        
        # Get hardware constraints
        self.basis_gates = self.device_backend.configuration().basis_gates
        self.coupling_map = self.device_backend.configuration().coupling_map
        self.noise_model = NoiseModel.from_backend(self.device_backend)
        
        # Get detailed device noise characteristics
        properties = self.device_backend.properties()
        
        # Build comprehensive noise model
        self.noise_model = NoiseModel()
        
        # Add readout errors
        for qubit in range(self.n_qubits):
            readout_error = properties.readout_error(qubit)
            if readout_error > 0:
                self.noise_model.add_readout_error(
                    [readout_error, 1-readout_error],
                    [qubit]
                )
        
        # Add gate errors with crosstalk
        for gate in self.basis_gates:
            for qubits in properties.gates_with_type(gate):
                error_param = properties.gate_error(gate, qubits)
                if error_param > 0:
                    self.noise_model.add_quantum_error(
                        error_param,
                        gate,
                        qubits
                    )
        
        # Initialize simulator with detailed noise model
        self.simulator = AerSimulator(
            noise_model=self.noise_model,
            basis_gates=self.basis_gates,
            coupling_map=self.coupling_map,
            max_parallel_experiments=0,  # Disable parallel for better error tracking
            max_parallel_shots=1
        )
        
        # For visualization
        from qiskit.visualization import circuit_drawer
        self.visualizer = circuit_drawer
        
    def prepare_quantum_state(self, classical_data):
        """Hardware-efficient quantum state preparation with error tracking"""
        from qiskit.circuit.library import StatePreparation, ZZFeatureMap
        from qiskit.transpiler import PassManager
        from qiskit.transpiler.passes import (
            BasicSwap, CXCancellation, Optimize1qGates,
            CommutativeCancellation, OptimizeSwapBeforeMeasure,
            FixedPoint, ConsolidateBlocks, Optimize1qGatesDecomposition
        )
        from qiskit.quantum_info import Statevector, partial_trace
        
        # Create optimization passes
        pm = PassManager([
            BasicSwap(self.coupling_map),
            CXCancellation(),
            Optimize1qGates(),
            CommutativeCancellation(),
            OptimizeSwapBeforeMeasure()
        ])
        
        # Normalize and prepare complex amplitudes
        amplitudes = classical_data.astype(np.complex128)
        norm = np.sqrt(np.sum(np.abs(amplitudes) ** 2))
        if norm > 1e-8:  # Check for zero vector
            amplitudes = amplitudes / norm
        else:
            amplitudes = np.zeros_like(amplitudes)
            amplitudes[0] = 1.0
            
        # Create circuit with efficient state preparation
        circuit = QuantumCircuit(self.qr, self.cr)
        state_prep = StatePreparation(
            amplitudes,
            normalize=True,
            insert_barriers=True
        )
        circuit.compose(state_prep, inplace=True)
        
        # Get hardware constraints
        coupling_map = CouplingMap(self.coupling_map)
        basis_gates = self.basis_gates

        # Optimize circuit for hardware
        # Use SABRE for better qubit mapping
        layout_pass = SabreLayout(
            coupling_map,
            max_iterations=5,
            seed=42
        )
        swap_pass = SabreSwap(
            coupling_map,
            heuristic='basic',
            seed=42
        )
        
        # Transpile with optimization
        transpiled = transpile(
            circuit,
            backend=self.device_backend,
            basis_gates=basis_gates,
            coupling_map=coupling_map,
            layout_method='sabre',
            routing_method='sabre',
            optimization_level=3,
            seed_transpiler=42
        )
        
        # Get device properties
        properties = self.device_backend.properties()
        coherence_times = [
            properties.qubit_property(q, 'T2') 
            for q in range(self.n_qubits)
        ]
        gate_times = {
            gate: properties.gate_property(gate, 'gate_length')
            for gate in self.basis_gates
        }
        
        # Calculate total circuit time
        total_time = self._calculate_circuit_time(transpiled, gate_times)
        min_coherence = min(coherence_times)
        
        if total_time > min_coherence * 0.8:  # Allow 80% of coherence time
            raise ValueError(
                f"Circuit time {total_time*1e6:.1f}μs exceeds "
                f"device coherence {min_coherence*1e6:.1f}μs"
            )
            
        # Validate against hardware constraints
        self._validate_circuit(transpiled)
                
        return circuit
    
    def apply_quantum_operation(self, circuit, operation_type, params=None):
        """Apply quantum operations while preserving coherence"""
        if operation_type == 'hadamard':
            circuit.h(range(self.n_qubits))
            
        elif operation_type == 'rotation':
            for i in range(self.n_qubits):
                circuit.rx(params['rx'][i], self.qr[i])
                circuit.ry(params['ry'][i], self.qr[i])
                circuit.rz(params['rz'][i], self.qr[i])
                
        elif operation_type == 'entangle':
            for i in range(self.n_qubits-1):
                circuit.cx(self.qr[i], self.qr[i+1])
                
        return circuit
    
    def measure_state(self, circuit, shots=1000):
        """Perform quantum measurement with error mitigation"""
        # Add measurements
        circuit.measure(self.qr, self.cr)
        
        # Run with noise model
        job = self.simulator.run(
            circuit, 
            noise_model=self.noise_model,
            shots=shots
        )
        result = job.result()
        
        # Get counts and convert to probabilities
        counts = result.get_counts()
        total_shots = sum(counts.values())
        probabilities = {k: v/total_shots for k, v in counts.items()}
        
        # Error mitigation
        if hasattr(self, 'calibration_matrix'):
            # Apply readout error correction
            probabilities = self._mitigate_readout_errors(probabilities)
            
        return probabilities
    
    def _mitigate_readout_errors(self, measured_probs, circuit):
        """Apply comprehensive error mitigation with correlation tracking"""
        from qiskit.ignis.mitigation.measurement import (
            complete_meas_cal, CompleteMeasFitter, TensoredMitigation
        )
        from qiskit.result import Result
        
        # Generate calibration circuits with correlation tracking
        cal_circuits = []
        # Single qubit calibration
        single_cal, single_labels = complete_meas_cal(
            qubit_list=list(range(self.n_qubits)),
            qr=self.qr,
            cr=self.cr,
            circlabel='single_qubit_cal'
        )
        cal_circuits.extend(single_cal)
        
        # Two-qubit correlation calibration
        for i in range(self.n_qubits-1):
            for j in range(i+1, self.n_qubits):
                if [i,j] in self.coupling_map:
                    corr_cal, corr_labels = complete_meas_cal(
                        qubit_list=[i,j],
                        qr=self.qr,
                        cr=self.cr,
                        circlabel=f'corr_cal_{i}_{j}'
                    )
                    cal_circuits.extend(corr_cal)
        
        # Transpile calibration circuits with crosstalk mitigation
        from qiskit.transpiler.passes import CrosstalkAdaptiveSchedule
        
        # Create crosstalk-aware transpiler pass
        crosstalk_pass = CrosstalkAdaptiveSchedule(
            self.device_backend.properties(),
            self.coupling_map
        )
        
        # Transpile with crosstalk awareness
        cal_circuits = [
            transpile(
                circ,
                backend=self.device_backend,
                basis_gates=self.basis_gates,
                coupling_map=self.coupling_map,
                optimization_level=3,
                scheduling_method='alap',
                instruction_durations=self.device_backend.properties().gate_length,
                additional_optimization_passes=[crosstalk_pass]
            ) for circ in cal_circuits
        ]
        
        # Verify circuit depths against coherence times
        for circ in cal_circuits:
            depth = circ.depth()
            max_time = self._calculate_circuit_time(circ)
            min_coherence = min(
                self.device_backend.properties().qubit_property(q, 'T2')
                for q in range(self.n_qubits)
            )
            if max_time > 0.8 * min_coherence:
                raise ValueError(
                    f"Circuit time {max_time*1e6:.1f}μs exceeds "
                    f"80% of coherence time {min_coherence*1e6:.1f}μs"
                )
        
        # Execute calibration circuits with noise model
        job = self.simulator.run(
            cal_circuits,
            noise_model=self.noise_model,
            shots=8192  # More shots for better calibration
        )
        cal_results = job.result()
        
        # Build calibration matrix
        meas_fitter = CompleteMeasFitter(
            cal_results,
            single_labels,  # Use labels from single qubit calibration
            circlabel='measurement_calibration'
        )
        
        # Print calibration error info
        print(f"Calibration matrix:\n{meas_fitter.cal_matrix}")
        print(f"Readout error rate: {meas_fitter.readout_fidelity()}")
        
        # Apply correction
        mitigated_results = meas_fitter.filter.apply(measured_probs)
        
        return mitigated_results
    
    def get_density_matrix(self, circuit):
        """Get density matrix representation of quantum state"""
        # Simulate without measurement
        simulator = AerSimulator(method='density_matrix')
        result = simulator.run(circuit).result()
        return DensityMatrix(result.data()['density_matrix'])
