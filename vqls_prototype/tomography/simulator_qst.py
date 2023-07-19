import qiskit
import numpy as np
from qiskit.quantum_info import Statevector
from qiskit_experiments.framework import ParallelExperiment
from qiskit_experiments.library import StateTomography


class SimulatorQST:
    def __init__(self, circuit):
        self.circuit = circuit

    def get_relative_amplitude_sign(self, parameters):
        """_summary_

        Args:
            circuit (_type_): _description_
            parameters (_type_): _description_
            backend (_type_): _description_
        """
        state_vector = (Statevector(self.circuit.bind_parameters(parameters))).data.real
        return np.sign(state_vector)

    def get_statevector(self, parameters):
        """_summary_

        Args:
            circuit (_type_): _description_
            parameters (_type_): _description_
            backend (_type_): _description_
        """
        return (Statevector(self.circuit.bind_parameters(parameters))).data.real
