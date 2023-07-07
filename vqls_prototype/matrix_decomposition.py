"""Methods to decompose a matrix into quantum circuits"""
from collections import namedtuple, OrderedDict
from itertools import product
from typing import Optional, Union, List, Tuple, TypeVar, cast

import numpy as np
from numpy.testing import assert_
import numpy.typing as npt
import scipy.linalg as spla

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Operator, Pauli

import networkx as nx

complex_type = TypeVar("complex_type", float, complex)
complex_array_type = npt.NDArray[np.cdouble]


class MatrixDecomposition:
    """Base class for the decomposition of a matrix in quantum circuits."""

    CircuitElement = namedtuple("CircuitElement", ["coeff", "circuit"])

    @staticmethod
    def _as_complex(
        num_or_arr: Union[float, List[float], complex, List[complex]]
    ) -> npt.NDArray[np.cdouble]:
        """Converts a number or a list of numbers to a complex array.

        Args:
            num_or_arr (Union[complex_type, List[complex_type]]): array of number to convert

        Returns:
            complex_array_type: array of complex numbers
        """
        arr = num_or_arr if isinstance(num_or_arr, List) else [num_or_arr]
        return np.array(arr, dtype=np.cdouble)

    def __init__(
        self,
        matrix: Optional[npt.NDArray] = None,
        circuits: Optional[Union[QuantumCircuit, List[QuantumCircuit]]] = None,
        coefficients: Optional[
            Union[float, complex, List[float], List[complex]]
        ] = None,
    ):
        """Decompose a matrix representing quantum circuits

        Args:
            matrix (Optional[npt.NDArray], optional): Array to decompose;
                only relevant in derived classes where
                `self.decompose_matrix()` has been implemented. Defaults to None.
            circuits (Optional[Union[QuantumCircuit, List[QuantumCircuit]]], optional):
                quantum circuits representing the matrix. Defaults to None.
            coefficients (Optional[ Union[float, complex, List[float], List[complex]] ], optional):
                coefficients associated with the input quantum circuits; `None` is
                valid only for a circuit with 1 element. Defaults to None.
        """

        if matrix is not None:  # ignore circuits & coefficients
            self._matrix, self.num_qubits = self._validate_matrix(matrix)
            self._coefficients, self._matrices, self._circuits = self.decompose_matrix()

        elif circuits is not None:
            self._circuits = circuits if isinstance(circuits, list) else [circuits]

            assert_(
                isinstance(self._circuits[0], QuantumCircuit),
                f"{circuits}: invalid circuit",
            )
            if coefficients is None:
                if len(self._circuits) == 1:
                    self._coefficients = self._as_complex(1.0)
                else:
                    raise ValueError("coefficients mandatory for multiple circuits")
            else:
                self._coefficients = self._as_complex(coefficients)

            if len(self._circuits) != len(self._coefficients):
                raise ValueError("number of coefficients and circuits do not match")

            self.num_qubits = self._circuits[0].num_qubits
            if not all(map(lambda ct: ct.num_qubits == self.num_qubits, self.circuits)):
                _num_qubits = [ct.num_qubits for ct in self.circuits]
                raise ValueError(f"mismatched number of qubits: {_num_qubits}")

            self._matrices = [Operator(qc).data for qc in self.circuits]
            self._matrix = self.recompose()
        else:
            raise ValueError(
                f"inconsistent arguments: matrix={matrix}, \
                    coefficients={coefficients}, circuits={circuits}"
            )

        self.num_circuits = len(self._circuits)
        self.iiter = 0

    @classmethod
    def _compute_circuit_size(cls, matrix: npt.NDArray) -> int:
        """Compute the size of the circuit represented by the matrix

        Args:
            matrix (npt.NDArray): matrix representing the circuit

        Returns:
            int: circuit size
        """
        return int(np.log2(matrix.shape[0]))

    @classmethod
    def _validate_matrix(
        cls, matrix: complex_array_type
    ) -> Tuple[complex_array_type, int]:
        """Check the size of the matrix

        Args:
            matrix (complex_array_type): input matrix

        Raises:
            ValueError: if the matrix is not square
            ValueError: if the matrix size is not a power of 2
            ValueError: if the matrix is not symmetric

        Returns:
            Tuple[complex_array_type, int]: matrix and the number of qubits required
        """
        if len(matrix.shape) == 2 and matrix.shape[0] != matrix.shape[1]:
            raise ValueError(
                f"Input matrix must be square: matrix.shape={matrix.shape}"
            )
        num_qubits = cls._compute_circuit_size(matrix)
        if num_qubits % 1 != 0:
            raise ValueError(
                f"Input matrix dimension is not a power of 2: {num_qubits}"
            )
        if not np.allclose(matrix, matrix.conj().T):
            raise ValueError(f"Input matrix isn't symmetric:\n{matrix}")

        return matrix, num_qubits

    @property
    def matrix(self) -> np.ndarray:
        """matrix of the decomposition"""
        return self._matrix

    @property
    def circuits(self) -> List[QuantumCircuit]:
        """circuits of the decomposition"""
        return self._circuits

    @property
    def coefficients(self) -> complex_array_type:
        """coefficients of the decomposition."""
        return self._coefficients

    @property
    def matrices(self) -> List[complex_array_type]:
        """return the unitary matrices"""
        return self._matrices

    def __iter__(self):
        self.iiter = 0
        return self

    def __next__(self):
        if self.iiter < self.num_circuits:
            out = self.CircuitElement(
                self._coefficients[self.iiter], self._circuits[self.iiter]
            )
            self.iiter += 1
            return out
        raise StopIteration

    def __len__(self):
        return len(self._circuits)

    def __getitem__(self, index):
        return self.CircuitElement(self._coefficients[index], self._circuits[index])

    def recompose(self) -> complex_array_type:
        """Rebuilds the original matrix from the decomposed one.

        Returns:
            complex_array_type: The recomposed matrix.
        """
        coeffs, matrices = self.coefficients, self.matrices
        return (coeffs.reshape(len(coeffs), 1, 1) * matrices).sum(axis=0)

    def decompose_matrix(
        self,
    ) -> Tuple[complex_array_type, List[complex_array_type], List[QuantumCircuit]]:
        raise NotImplementedError(f"can't decompose in {self.__class__.__name__!r}")


class SymmetricDecomposition(MatrixDecomposition):
    """
    A class that represents the symmetric decomposition of a matrix.
    For the mathematical background for the decomposition, see the following
    math.sx answer: https://math.stackexchange.com/a/1710390
    """

    def _create_circuits(
        self, unimatrices: List[np.ndarray], names: List[str]
    ) -> List[QuantumCircuit]:
        """Construct the quantum circuits from unitary matrices

        Args:
            unimatrices (List[np.ndarray]): list of unitary matrices of the decomposition.
            names (List[str]): names of the circuits

        Returns:
            List[QuantumCircuit]: quantum circuits
        """

        def make_qc(mat: complex_array_type, name: str) -> QuantumCircuit:
            circuit = QuantumCircuit(self.num_qubits, name=name)
            circuit.unitary(mat, circuit.qubits)
            return circuit

        return [make_qc(mat, name) for mat, name in zip(unimatrices, names)]

    @staticmethod
    def auxilliary_matrix(
        x: Union[npt.NDArray[np.float_], complex_array_type]
    ) -> complex_array_type:
        """Returns the auxiliary matrix for the decomposition of size n.
           and derfined as defined as : i * sqrt(I - x^2)

        Args:
            x (Union[npt.NDArray[np.float_], complex_array_type]): original matrix.

        Returns:
            complex_array_type: The auxiliary matrix.
        """
        mat = np.eye(len(x)) - x @ x
        mat = cast(npt.NDArray[Union[np.float_, np.cdouble]], spla.sqrtm(mat))
        return 1.0j * mat

    def decompose_matrix(
        self,
    ) -> Tuple[complex_array_type, List[complex_array_type], List[QuantumCircuit]]:
        """Decompose a generic numpy matrix into a sum of unitary matrices.

        Returns:
            Tuple[complex_array_type, List[complex_array_type], List[QuantumCircuit]]:
                A tuple containing the list of coefficients numpy matrices,
                and quantum circuits of the decomposition.
        """

        # Normalize
        norm = np.linalg.norm(self._matrix)
        mat = self._matrix / norm

        mat_real = np.real(mat)
        mat_imag = np.imag(mat)

        coef_real = norm * 0.5
        coef_imag = coef_real * 1j

        # Get the matrices
        unitary_matrices, unitary_coefficients = [], []
        if not np.allclose(mat_real, 0.0):
            aux_mat = self.auxilliary_matrix(mat_real)
            unitary_matrices += [mat_real + aux_mat, mat_real - aux_mat]
            unitary_coefficients += [coef_real] * 2

        if not np.allclose(mat_imag, 0.0):
            aux_mat = self.auxilliary_matrix(mat_imag)
            unitary_matrices += [mat_imag + aux_mat, mat_imag - aux_mat]
            unitary_coefficients += [coef_imag] * 2
        unit_coeffs = np.array(unitary_coefficients, dtype=np.cdouble)

        # create the circuits
        names = ["A+", "A-"]
        circuits = self._create_circuits(unitary_matrices, names)

        return unit_coeffs, unitary_matrices, circuits


class PauliDecomposition(MatrixDecomposition):
    """A class that represents the Pauli decomposition of a matrix."""

    basis = "IXYZ"

    @staticmethod
    def _create_circuit(pauli_string: str) -> QuantumCircuit:
        """creates a quantum circuit for a given pauli string

        Args:
            pauli_string (str): the input pauli string

        Returns:
            QuantumCircuit: quantum circuit for the string
        """
        num_qubit = len(pauli_string)
        circuit = QuantumCircuit(num_qubit, name=pauli_string)
        for iqbit, gate in enumerate(pauli_string[::-1]):
            if (
                gate.upper() != "I"
            ):  # identity gate cannot be controlled by ancillary qubit
                getattr(circuit, gate.lower())(iqbit)
        return circuit

    def decompose_matrix(
        self,
    ) -> Tuple[complex_array_type, List[complex_array_type], List[QuantumCircuit]]:
        """Decompose a generic numpy matrix into a sum of Pauli strings.

        Returns:
            Tuple[complex_array_type, List[complex_array_type]]:
                A tuple containing the list of coefficients and
                the numpy matrix of the decomposition.
        """

        prefactor = 1.0 / (2**self.num_qubits)
        unit_mats, coeffs, circuits = [], [], []
        self.strings = []
        for pauli_gates in product(self.basis, repeat=self.num_qubits):
            pauli_string = "".join(pauli_gates)
            pauli_op = Pauli(pauli_string)
            pauli_matrix = pauli_op.to_matrix()
            coef: complex_array_type = np.trace(pauli_matrix @ self.matrix)

            if coef * np.conj(coef) != 0:
                self.strings.append(pauli_string)
                coeffs.append(prefactor * coef)
                unit_mats.append(pauli_matrix)
                circuits.append(self._create_circuit(pauli_string))

        return np.array(coeffs, dtype=np.cdouble), unit_mats, circuits


class ContractedPauliDecomposition(PauliDecomposition):
    """A class that represents the Pauli decomposition of a matrix with added attributes
    representing the simplification of the Al.T Am terms"""

    contraction_dict = {
        "II": ("I", 1),
        "IX": ("X", 1),
        "IY": ("Y", 1),
        "IZ": ("Z", 1),
        "XI": ("X", 1),
        "YI": ("Y", -1),
        "ZI": ("Z", 1),
        "XX": ("I", 1),
        "YY": ("I", -1),
        "ZZ": ("I", 1),
        "XY": ("Z", 1.0j),
        "YX": ("Z", 1.0j),
        "XZ": ("Y", -1.0j),
        "ZX": ("Y", 1.0j),
        "YZ": ("X", -1.0j),
        "ZY": ("X", -1.0j),
    }

    def __init__(
        self,
        matrix: Optional[npt.NDArray] = None,
        circuits: Optional[Union[QuantumCircuit, List[QuantumCircuit]]] = None,
        coefficients: Optional[
            Union[float, complex, List[float], List[complex]]
        ] = None,
    ):
        super().__init__(matrix, circuits, coefficients)
        self.contract_pauli_terms()
        (
            self.qwc_groups,
            self.qwc_groups_index_mapping,
            self.qwc_groups_eigenvalues,
            self.qwc_groups_shared_basis_transformation,
        ) = self.group_contracted_terms()

    def contract_pauli_terms(
        self,
    ) -> Tuple[complex_array_type, List[complex_array_type], List[QuantumCircuit]]:
        """Compute the contractions of the Pauli Strings.

        Returns:
            Tuple[complex_array_type, List[complex_array_type]]:
                A tuple containing the list of coefficients and
                the numpy matrix of the decomposition.
        """
        self.contraction_map = []
        self.contraction_coefficient = []
        self.contracted_circuits = []
        self.contraction_index_mapping = []

        self.unique_pauli_strings = []
        number_existing_circuits = 0
        nstrings = len(self.strings)

        # loop over combination of gates
        for i1 in range(nstrings):
            for i2 in range(i1 + 1, nstrings):
                # extract pauli strings
                pauli_string_1, pauli_string_2 = self.strings[i1], self.strings[i2]
                contracted_pauli_string, contracted_coefficient = "", 1.0

                # contract pauli gates qubit wise
                for pauli1, pauli2 in zip(pauli_string_1, pauli_string_2):
                    pauli, coefficient = self.contraction_dict[pauli1 + pauli2]
                    contracted_pauli_string += pauli
                    contracted_coefficient *= coefficient

                # contraction mpa -> not  needed
                self.contraction_map.append(
                    [
                        (pauli_string_1, pauli_string_2),
                        (contracted_pauli_string, contracted_coefficient),
                    ]
                )

                # store circuits if we haven't done that yet
                if contracted_pauli_string not in self.unique_pauli_strings:
                    self.unique_pauli_strings.append(contracted_pauli_string)
                    self.contracted_circuits.append(
                        self._create_circuit(contracted_pauli_string)
                    )
                    self.contraction_index_mapping.append(number_existing_circuits)
                    number_existing_circuits += 1
                # otherwise find reference of existing circuit
                else:
                    self.contraction_index_mapping.append(
                        self.unique_pauli_strings.index(contracted_pauli_string)
                    )

                # store the contraction coefficient
                self.contraction_coefficient.append(contracted_coefficient)

    def group_contracted_terms(self):
        """Finds the qubit wise commutating operator to further optimize the number of measurements."""

        def get_eigenvalues(pauli_string):
            """Compute the eigenvalue of the string"""
            ev_dict = {"X": [1, -1], "Y": [1, -1], "Z": [1, -1], "I": [1, 1]}
            pauli_string = pauli_string[::-1]
            evs = ev_dict[pauli_string[0]]
            for p in pauli_string[1:]:
                evs = np.kron(ev_dict[p], evs)
            return evs

        def create_shared_basis_circuit(pauli_string):
            """ "creatae the circuit needed to rotate the qubits in the shared eigenbasis"""

            num_qubits = len(pauli_string)
            circuit = QuantumCircuit(num_qubits)
            for iqbit in range(num_qubits):
                op = pauli_string[iqbit]
                if op == "X":
                    circuit.ry(-np.pi / 2, iqbit)
                if op == "Y":
                    circuit.rx(np.pi / 2, iqbit)

            return circuit

        def string_qw_commutator(pauli1, pauli2):
            """assesses if two pauli string qubit-wise commutes or not

            Args:
                pauli1 (str): first puali string
                pauli2 (str): 2nd pauli string
            """

            return np.all(
                [
                    (p1 == p2) | (p1 == "I") | (p2 == "I")
                    for p1, p2 in zip(pauli1, pauli2)
                ]
            )

        # create qwc edges
        nstrings = len(self.unique_pauli_strings)
        qwc_graph_edges = []
        for i1 in range(nstrings):
            for i2 in range(i1 + 1, nstrings):
                pauli_string_1, pauli_string_2 = (
                    self.unique_pauli_strings[i1],
                    self.unique_pauli_strings[i2],
                )
                if string_qw_commutator(pauli_string_1, pauli_string_2):
                    qwc_graph_edges.append([pauli_string_1, pauli_string_2])

        # create graph
        qwc_graph = nx.Graph()
        qwc_graph.add_nodes_from(self.unique_pauli_strings)
        qwc_graph.add_edges_from(qwc_graph_edges)
        qwc_complement_graph = nx.complement(qwc_graph)

        # greedy clustering
        qwc_groups_flat = nx.coloring.greedy_color(
            qwc_complement_graph, strategy="largest_first"
        )

        # organize the groups
        qwc_groups = OrderedDict()
        qwc_groups_eigenvalues = [None] * nstrings
        qwc_groups_index_mapping = [None] * nstrings

        for pauli, group_id in qwc_groups_flat.items():
            if group_id not in qwc_groups:
                qwc_groups[group_id] = []
            qwc_groups[group_id].append(pauli)

            qwc_groups_eigenvalues[
                self.unique_pauli_strings.index(pauli)
            ] = get_eigenvalues(pauli)
            qwc_groups_index_mapping[self.unique_pauli_strings.index(pauli)] = group_id

        # determine the shared eigenbasis
        qwc_groups_shared_basis_transformation = []
        for group_id, pauli_strings in qwc_groups.items():
            shared_basis = np.array(["I"] * len(pauli_strings[0]))
            for pstr in pauli_strings:
                for ig, pgate in enumerate(list(pstr)):
                    if pgate != "I":
                        shared_basis[ig] = pgate

                if not np.any(shared_basis == "I"):
                    break

            qwc_groups_shared_basis_transformation.append(
                create_shared_basis_circuit(shared_basis[::-1])
            )

        return (
            qwc_groups,
            qwc_groups_index_mapping,
            qwc_groups_eigenvalues,
            qwc_groups_shared_basis_transformation,
        )
