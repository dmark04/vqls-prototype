from typing import Union, List, Callable, Tuple, Dict, Any

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.primitives import Estimator, PrimitiveJob

from qiskit_aer.primitives import Estimator as aer_Estimator
from qiskit_aer.primitives import EstimatorV2 as aer_EstimatorV2

from qiskit_ibm_runtime import Estimator as ibm_runtime_Estimator
from qiskit_ibm_runtime import EstimatorV2 as ibm_runtime_EstimatorV2
from qiskit_ibm_runtime import RuntimeJobV2

EstimatorValidType = Union[
    Estimator,
    aer_Estimator,
    aer_EstimatorV2,
    ibm_runtime_Estimator,
    ibm_runtime_EstimatorV2,
]


class EstimatorRunBuilder:
    """
    A class to build and configure estimator runs based on their provenance and options.

    Attributes:
        estimator (EstimatorValidType): The quantum estimator instance.
        circuits (List[QuantumCircuit]): List of quantum circuits.
        observables (List[SparsePauliOp]): List of observables.
        parameter_sets (List[List[float]]): List of parameter sets.
    """

    def __init__(
        self,
        estimator: EstimatorValidType,
        circuits: List[QuantumCircuit],
        observables: List[SparsePauliOp],
        parameter_sets: List[List[float]],
        options: Dict[str, Any],
    ):
        """
        Initializes the EstimatorRunBuilder with the given estimator, circuits, observables,
        parameter sets, and options.

        Args:
            estimator (EstimatorValidType): The estimator to use for runs.
            circuits (List[QuantumCircuit]): The quantum circuits to run.
            observables (List[SparsePauliOp]): The observables to measure.
            parameter_sets (List[List[float]]): The parameters to vary in the circuits.
            options (Dict[str, Any]): Configuration options such as number of shots.
        """
        self.estimator = estimator
        self.circuits = circuits
        self.observables = observables
        self.parameter_sets = parameter_sets
        self.shots = options.pop("shots", None)
        self.seed = options.pop("seed", None)
        self.provenance = self.find_estimator_provenance()

    def find_estimator_provenance(self) -> Tuple[str, str]:
        """Determines the provenance of the estimator based on its class and module."""
        return (
            self.estimator.__class__.__module__.split(".")[0],
            self.estimator.__class__.__name__,
        )

    def build_run(self) -> Union[PrimitiveJob, RuntimeJobV2]:
        """
        Configures and returns estimator runs based on its provenance.

        Raises:
            NotImplementedError: If the estimator's provenance is not supported.

        Returns:
            Union[PrimitiveJob, RuntimeJobV2]: A configured callable function to execute the estimator run.
        """
        builder_function = self._select_run_builder()
        return builder_function()

    def _select_run_builder(self) -> Union[PrimitiveJob, RuntimeJobV2]:
        """Selects the appropriate builder function based on the estimator's provenance."""
        builders = {
            ("qiskit", "Estimator"): self._build_native_qiskit_estimator_run,
            ("qiskit_aer", "EstimatorV2"): self._build_estimatorv2_run,
            ("qiskit_aer", "Estimator"): self._build_estimatorv1_run,
            ("qiskit_ibm_runtime", "EstimatorV2"): self._build_estimatorv2_run,
            ("qiskit_ibm_runtime", "EstimatorV1"): self._build_estimatorv1_run,
        }
        try:
            return builders[self.provenance]
        except KeyError as err:
            raise NotImplementedError(
                f"{self.__class__.__name__} not compatible with {self.provenance}."
            ) from err

    def _build_native_qiskit_estimator_run(self) -> PrimitiveJob:
        """Builds a run function for a standard qiskit Estimator."""
        return self.estimator.run(
            self.circuits,
            self.observables,
            self.parameter_sets,
            shots=self.shots,
            seed=self.seed,
        )

    def _build_estimatorv2_run(self) -> Union[PrimitiveJob, RuntimeJobV2]:
        """Builds a run function for qiskit-aer and qiskit-ibm-runtime EstimatorV2."""
        backend = self.estimator._backend  # pylint: disable=protected-access
        optimization_level = 1
        pm = generate_preset_pass_manager(optimization_level, backend)
        pubs = []
        for qc, obs, param in zip(self.circuits, self.observables, self.parameter_sets):
            isa_circuit = pm.run(qc)
            isa_obs = obs.apply_layout(isa_circuit.layout)
            pubs.append((isa_circuit, isa_obs, param))
        return self.estimator.run(pubs)

    def _build_estimatorv1_run(self):
        """
        Attempts to build a run function for EstimatorV1, which will be soon deprecated.

        Raises:
            NotImplementedError:
                Indicates that EstimatorV1 will be soon deprecated and
                suggests using EstimatorV2 instead.
        """
        raise NotImplementedError(
            "EstimatorV1 will be soon deprecated. Please, use EstimatorV2 implementation."
        )
