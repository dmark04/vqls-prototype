# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import numpy as np

from qiskit.test import QiskitTestCase
from qiskit.utils import algorithm_globals

from qiskit.circuit.library import RealAmplitudes
from qiskit.algorithms import optimizers as opt
from qiskit.primitives import Estimator, Sampler
from vqls_prototype import VQLS, VQLSLog, Hybrid_QST_VQLS


from vqls_prototype.hadamard_test.hadamard_test import BatchHadammardTest
from vqls_prototype.hadamard_test.hadamard_overlap_test import BatchHadammardOverlapTest
from vqls_prototype.hadamard_test.direct_hadamard_test import BatchDirectHadammardTest


class TestHadamard(QiskitTestCase):
    def setUp(self):
        super().setUp()
        self.seed = 50
        algorithm_globals.random_seed = self.seed

        # define system
        self.estimator = Estimator()
        self.sampler = Sampler()
        self.log = VQLSLog([], [])
        self.optimizer = opt.COBYLA(maxiter=250)

        # define ansatz
        num_qubits = 2
        size = 2**num_qubits
        self.ansatz = RealAmplitudes(num_qubits=num_qubits, reps=3, entanglement="full")

        # define ref vqls
        self.vqls = VQLS(
            self.estimator,
            self.ansatz,
            self.optimizer,
            sampler=self.sampler,
            options={"matrix_decomposition": "pauli", "shots": None}
        )

        # define matrix/vector of the linear system
        self.matrix = np.random.rand(size, size)
        self.matrix += self.matrix.T
        self.matrix /= np.linalg.norm(self.matrix)

        self.vector = np.random.rand(size)
        self.vector /= np.linalg.norm(self.vector)

        # compute the cricuits
        hdmr_tests_norm, hdmr_tests_overlap = self.vqls.construct_circuit(
            self.matrix, self.vector
        )

        # define random parameters
        self.parameters = np.random.rand(self.ansatz.num_parameters)

        # compute the reference values of the hadamard tests
        self.norm_ref = BatchHadammardTest(hdmr_tests_norm).get_values(
            self.estimator, self.parameters
        )
        self.overlap_ref = BatchHadammardOverlapTest(hdmr_tests_overlap).get_values(
            self.estimator, self.parameters
        )

        # compute the ref of the cost function
        coefficient_matrix = self.vqls.get_coefficient_matrix(
            np.array([mat_i.coeff for mat_i in self.vqls.matrix_circuits])
        )
        cost_evaluation = self.vqls.get_cost_evaluation_function(
            hdmr_tests_norm, hdmr_tests_overlap, coefficient_matrix
        )
        self.cost_ref = cost_evaluation(self.parameters)

    def test_contracted_pauli(self):
        """Test the value obtained with the contracted pauli strings"""

        # set up the system
        vqls = VQLS(
            self.estimator,
            self.ansatz,
            self.optimizer,
            sampler=self.sampler,
            options={"matrix_decomposition": "contracted_pauli", "shots": None}
        )


        # compute the circuits
        hdmr_tests_norm, hdmr_tests_overlap = vqls.construct_circuit(
            self.matrix, self.vector
        )

        # compute the reference values of the hadamard tests
        norm = BatchHadammardTest(hdmr_tests_norm).get_values(
            self.estimator, self.parameters
        )
        overlap = BatchHadammardOverlapTest(hdmr_tests_overlap).get_values(
            self.estimator, self.parameters
        )

        assert np.allclose(norm, self.norm_ref)
        assert np.allclose(overlap, self.overlap_ref)

        # compute the cost function
        coefficient_matrix = vqls.get_coefficient_matrix(
            np.array([mat_i.coeff for mat_i in vqls.matrix_circuits])
        )
        cost_evaluation = vqls.get_cost_evaluation_function(
            hdmr_tests_norm, hdmr_tests_overlap, coefficient_matrix
        )
        cost = cost_evaluation(self.parameters)

        assert np.allclose(cost, self.cost_ref)

    def test_optimized_pauli(self):
        """Test the value obtained with the optimized pauli strings"""

        # set up the system
        vqls = Hybrid_QST_VQLS(
            self.estimator,
            self.ansatz,
            self.optimizer,
            sampler=self.sampler,
            options={"matrix_decomposition": "optimized_pauli", "shots": None}
        )


        # compute the circuits
        hdmr_tests_norm, hdmr_tests_overlap = vqls.construct_circuit(
            self.matrix, self.vector
        )
        num_norm_circuits = len(hdmr_tests_norm)
        circuits = hdmr_tests_norm + hdmr_tests_overlap

        # compute the reference values of the hadamard tests
        samples = BatchDirectHadammardTest(circuits).get_values(
            self.sampler, self.parameters
        )

        # postprocess the values for the norm
        norm = self.vqls.matrix_circuits.get_norm_values(samples[:num_norm_circuits])

        # post process the values for the overlap
        sign_ansatz = self.vqls.get_ansatz_sign_vector(self.parameters)
        overlap = self.vqls.matrix_circuits.get_overlap_values(
            samples[num_norm_circuits:], sign_ansatz
        )

        assert np.allclose(norm, self.norm_ref)
        assert np.allclose(overlap, self.overlap_ref)

        # compute the cost function
        coefficient_matrix = vqls.get_coefficient_matrix(
            np.array([mat_i.coeff for mat_i in vqls.matrix_circuits])
        )
        cost_evaluation = vqls.get_cost_evaluation_function(
            hdmr_tests_norm, hdmr_tests_overlap, coefficient_matrix
        )
        cost = cost_evaluation(self.parameters)

        assert np.allclose(cost, self.cost_ref)
