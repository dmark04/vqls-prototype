import qiskit
import numpy as np
from qiskit.quantum_info import Operator
from qiskit_experiments.framework import ParallelExperiment
from qiskit_experiments.library import StateTomography


class FullQST:
    def __init__(self, circuit, backend):
        self.backend = backend
        self.circuit = circuit

    def get_relative_amplitude_sign(self, parameters):
        """_summary_

        Args:
            circuit (_type_): _description_
            parameters (_type_): _description_
            backend (_type_): _description_
        """

        density_matrix = self.get_density_matrix(parameters)
        return self.extract_sign(density_matrix)

    @staticmethod
    def extract_sign(density_matrix):
        """_summary_

        Args:
            density_matrix (_type_): _description_
        """
        return np.sign(density_matrix[0, :].real)

    def get_density_matrix(self, parameters):
        qstexp1 = StateTomography(self.circuit.bind_parameters(parameters))
        qstdata1 = qstexp1.run(self.backend).block_for_results()
        return qstdata1.analysis_results("state").value.data.real
