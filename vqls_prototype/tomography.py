import qiskit
import numpy as np
from qiskit.quantum_info import Operator
from qiskit_experiments.framework import ParallelExperiment
from qiskit_experiments.library import StateTomography


def get_relative_amplitude_sign(circuit, parameters, backend):
    """_summary_

    Args:
        circuit (_type_): _description_
        parameters (_type_): _description_
        backend (_type_): _description_
    """
    density_matrix = get_density_matrix(circuit, parameters, backend)
    return extract_sign(density_matrix)


def extract_sign(density_matrix):
    """_summary_

    Args:
        density_matrix (_type_): _description_
    """
    return np.sign(density_matrix[0, :].real)


def get_density_matrix(circuit, parameters, backend):
    qstexp1 = StateTomography(circuit.bind_parameters(parameters))
    qstdata1 = qstexp1.run(backend).block_for_results()
    return qstdata1.analysis_results("state").value.data.real
