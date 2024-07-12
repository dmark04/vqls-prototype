from typing import Union, List, Callable, Tuple, Dict

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from qiskit.primitives import Estimator
from qiskit_aer.primitives import EstimatorV2 as aer_EstimatorV2
from qiskit_aer.primitives import Estimator as aer_Estimator
from qiskit_ibm_runtime import Estimator as ibm_runtime_Estimator
from qiskit_ibm_runtime import EstimatorV2 as ibm_runtime_EstimatorV2

EstimatorValidType = Union[
    Estimator,
    aer_Estimator,
    aer_EstimatorV2,
    ibm_runtime_Estimator,
    ibm_runtime_EstimatorV2
]


class EstimatorRunBuilder:
    """Docs.."""
    def __init__(
        self,
        estimator: EstimatorValidType,
        circuits: List[QuantumCircuit],
        observables: List[SparsePauliOp],
        parameter_sets: List,
        options: Dict
    ):
        """Docs..."""
        self.estimator = estimator
        self.provenance = self.find_estimator_provenance()

        self.circuits = circuits
        self.observables = observables
        self.parameter_sets = parameter_sets

        self.shots = options.pop("shots", None)

    def find_estimator_provenance(self) -> Tuple[str, str]:
        """XXXX"""
        return (
            self.estimator.__class__.__module__.split('.')[0],
            self.estimator.__class__.__name__
        )

    def build_run(self) -> Callable:
        """XXX"""
        if self.provenance == ('qiskit', 'Estimator'):
            return self._build_qiskit_estimator_run()

        if self.provenance == ('qiskit_aer', 'EstimatorV2'):
            return self._build_aer_qiskit_estimatorV2_run()

        raise NotImplementedError(
            f"'EstimatorRunBuilder' not compatible with {self.provenance}."
        )

    def _build_qiskit_estimator_run(self) -> Callable:
        """XXX"""
        return self.estimator.run(
            self.circuits,
            self.observables,
            self.parameter_sets,
            shots=self.shots,
        )

    def _build_aer_qiskit_estimatorV2_run(self) -> Callable:
        """XXX"""
        pubs = []
        pm = generate_preset_pass_manager(optimization_level=1, backend=self.estimator._backend)
        for qc, obs, param in zip(self.circuits, self.observables, self.parameter_sets):
            isa_circuit = pm.run(qc)
            isa_obs = obs.apply_layout(isa_circuit.layout)
            pubs.append((isa_circuit, isa_obs, param))
        return self.estimator.run(pubs)
