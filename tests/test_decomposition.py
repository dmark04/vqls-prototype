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

from vqls_prototype.matrix_decomposition.matrix_decomposition import (
    MatrixDecomposition,
    SymmetricDecomposition,
    PauliDecomposition,
)


@pytest.fixture(params=[2, 4, 8, 16])
def symmetric(request):
    """Generate a symmetric matrix"""
    dim = request.param
    mat = np.random.rand(dim, dim)
    mat = mat + mat.T
    return mat


def test_decomposition_raises():
    """Test error raised"""
    mat = np.eye(4)[-1::-1]
    with pytest.raises(NotImplementedError, match="decompose.+MatrixDecomposition"):
        MatrixDecomposition(matrix=mat)


@pytest.mark.parametrize(
    "decomposition_t", [SymmetricDecomposition, PauliDecomposition]
)
def test_decomposition(symmetric, decomposition_t):
    """Test decompositions"""
    decomp = decomposition_t(matrix=symmetric)
    assert_allclose(decomp.recompose(), symmetric)
