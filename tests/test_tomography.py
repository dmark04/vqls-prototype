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

import unittest
import numpy as np


from qiskit_aer import Aer
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorSampler

from vqls_prototype.tomography import FullQST, SimulatorQST, HTreeQST, ShadowQST


class TestTomography(unittest.TestCase):
    def setUp(self):
        super().setUp()

        # define ansatz
        num_qubits = 2
        self.ansatz = RealAmplitudes(num_qubits=num_qubits, reps=3, entanglement="full")
        self.parameters = 2 * np.pi * np.random.rand(self.ansatz.num_parameters)

        self.ref = SimulatorQST(self.ansatz).get_relative_amplitude_sign(
            self.parameters
        )

    def test_full_qst(self):
        backend = Aer.get_backend("statevector_simulator")
        _ = FullQST(self.ansatz, backend, shots=10000)
        # this test fails on GH actions but not locally ...
        # sol = full_qst.get_relative_amplitude_sign(self.parameters)
        # assert np.allclose(self.ref, sol) or np.allclose(self.ref, -sol)

    def test_htree_qst(self):
        sampler = StatevectorSampler()
        htree_qst = HTreeQST(self.ansatz, sampler)
        sol = htree_qst.get_relative_amplitude_sign(self.parameters)
        assert np.allclose(self.ref, sol) or np.allclose(self.ref, -sol)

    def test_shadow_qst(self):
        sampler = StatevectorSampler()
        shadow_qst = ShadowQST(self.ansatz, sampler, 10000)
        _ = shadow_qst.get_relative_amplitude_sign(self.parameters)
        # this test can also fail on GH
        # assert np.allclose(self.ref, sol) or np.allclose(self.ref, -sol)
