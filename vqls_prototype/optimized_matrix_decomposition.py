"""Methods to decompose a matrix into quantum circuits"""
from collections import namedtuple, OrderedDict
from itertools import product
from typing import Optional, Union, List, Tuple, TypeVar

import numpy as np
import numpy.typing as npt
from qiskit.circuit import QuantumCircuit
import networkx as nx
from .matrix_decomposition import PauliDecomposition

complex_type = TypeVar("complex_type", float, complex)
complex_array_type = npt.NDArray[np.cdouble]


class OptimizedPauliDecomposition(PauliDecomposition):
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

    def post_process_contracted_norm_values(self, hdmr_values_norm):
        """Post process the measurement obtained with the direct

        Args:
            hdmr_values_norm (list): list of measrurement values
        """

        # map the sampled values to all the group members (unique pauli strings)
        hdmr_values_norm = hdmr_values_norm[self.qwc_groups_index_mapping]

        # mulitpliy the sampled values with the eigenvalues
        # of each transformation
        hdmr_values_norm = np.array(
            [
                np.dot(ev, val)
                for ev, val in zip(
                    self.qwc_groups_eigenvalues,
                    hdmr_values_norm,
                )
            ]
        )

        # map the values onto the explicit Al Am terms
        hdmr_values_norm = hdmr_values_norm[self.contraction_index_mapping] * np.array(
            self.contraction_coefficient
        )

        return hdmr_values_norm
