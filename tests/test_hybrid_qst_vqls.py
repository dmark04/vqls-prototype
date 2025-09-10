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

""" Test VQLS """


import unittest

import numpy as np
from qiskit.circuit.library import RealAmplitudes

from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import StatevectorEstimator, StatevectorSampler
from vqls_prototype import Hybrid_QST_VQLS, VQLSLog


class TestHybridQSTVQLS(unittest.TestCase):
    """Test VQLS"""

    def setUp(self):
        super().setUp()

        self.options = ({"use_local_cost_function": False, "use_overlap_test": False},)

        self.estimators = (
            StatevectorEstimator(),
            # AerEstimator(),
        )

        self.samplers = (
            StatevectorSampler(),
            # AerSampler(),
        )

        self.log = VQLSLog([], [])

    def test_numpy_input(self):
        """Test the VQLS on matrix input using statevector simulator."""

        matrix = np.array(
            [
                [0.50, 0.25, 0.10, 0.00],
                [0.25, 0.50, 0.25, 0.10],
                [0.10, 0.25, 0.50, 0.25],
                [0.00, 0.10, 0.25, 0.50],
            ]
        )

        rhs = np.array([0.1] * 4)
        ansatz = RealAmplitudes(num_qubits=2, reps=3, entanglement="full")

        for estimator, sampler in zip(self.estimators, self.samplers):
            for opt in self.options:
                vqls = Hybrid_QST_VQLS(
                    estimator,
                    ansatz,
                    COBYLA(maxiter=2, disp=True),
                    options=opt,
                    sampler=sampler,
                )
                _ = vqls.solve(matrix, rhs)


if __name__ == "__main__":
    unittest.main()
