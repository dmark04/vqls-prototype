# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
=======================================
Variational Quantum Linear Solver
=======================================
"""
"""Template project."""

from importlib_metadata import version as metadata_version, PackageNotFoundError

from .template_module import TemplateClass
from .solver.vqls import VQLS
from .solver.log import VQLSLog
from .solver.hybrid_qst_vqls import Hybrid_QST_VQLS
from .solver.qst_vqls import QST_VQLS


try:
    __version__ = metadata_version("prototype_template")
except PackageNotFoundError:  # pragma: no cover
    # package is not installed
    pass

__all__ = ["VQLS", "VQLSLog", "Hybrid_QST_VQLS", "QST_VQLS"]
