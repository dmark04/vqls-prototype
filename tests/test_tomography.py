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
from numpy.testing import assert_allclose
import pytest

import unittest
from qiskit.test import QiskitTestCase
from qiskit.utils import algorithm_globals
import numpy as np

from qiskit import Aer
from qiskit.circuit.library import RealAmplitudes
from qiskit.algorithms import optimizers as opt
from qiskit.algorithms.optimizers import COBYLA
from qiskit.primitives import Estimator, Sampler
from vqls_prototype import VQLS, VQLSLog, EVQLS


from vqls_prototype.hadamard_test.hadamard_test import BatchHadammardTest
from vqls_prototype.hadamard_test.hadamard_overlap_test import BatchHadammardOverlapTest
from vqls_prototype.hadamard_test.direct_hadamard_test import BatchDirectHadammardTest

from vqls_prototype.tomography import FullQST, SimulatorQST, RealQST


class TestTomography(QiskitTestCase):
    def setUp(self):
        super().setUp()
        self.seed = 50
        algorithm_globals.random_seed = self.seed

        # define ansatz
        num_qubits = 2
        size = 2**num_qubits
        self.ansatz = RealAmplitudes(num_qubits=num_qubits, reps=3, entanglement="full")
        self.parameters = np.random.rand(self.ansatz.num_parameters)

        self.ref = SimulatorQST(self.ansatz).get_relative_amplitude_sign(
            self.parameters
        )

    def test_full_qst(self):
        backend = Aer.get_backend("statevector_simulator")
        full_qst = FullQST(self.ansatz, backend)
        sols = full_qst.get_relative_amplitude_sign(self.parameters)
        assert np.allclose(self.ref, sols)

    def test_real_qst(self):
        sampler = Sampler()
        real_qst = RealQST(self.ansatz, sampler)
        sol = real_qst.get_relative_amplitude_signs(self.parameters)
        assert np.allclose(self.ref, sol)
