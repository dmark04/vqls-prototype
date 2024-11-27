![Platform](https://img.shields.io/badge/platform-Linux-blue)
[![Python](https://img.shields.io/badge/Python-3.8-informational)](https://www.python.org/)
[![Qiskit](https://img.shields.io/badge/Qiskit-%E2%89%A5%200.44.2-6133BD)](https://github.com/Qiskit/qiskit)
[![License](https://img.shields.io/github/license/quantumapplicationlab/vqls-prototype?label=License)](https://github.com/quantumapplicationlab/vqls-prototype/blob/main/LICENSE.txt)
[![Code style: Black](https://img.shields.io/badge/Code%20style-Black-000.svg)](https://github.com/psf/black)
[![Tests](https://github.com/quantumapplicationlab/vqls-prototype/actions/workflows/coverage.yml/badge.svg)](https://github.com/quantumapplicationlab/vqls-prototype/actions/workflows/coverage.yml)
[![Coverage Status](https://coveralls.io/repos/github/QuantumApplicationLab/vqls-prototype/badge.svg?branch=main)](https://coveralls.io/github/QuantumApplicationLab/vqls-prototype?branch=main)

# Variational Quantum Linear Solver Prototype

This repository contains a prototype implementation of the variational quantum linear solver [1,2] that allows to solve linear systems, $A\times x = b$, using variational circuits. 

![VQLS Solving a 4x4 linear system](./docs/sol.gif)
### Table of Contents

##### For Users

1.  [About the Project](docs/project_overview.md)
2.  [Beginner's Guide](docs/beginners_guide.md)
3.  [Installation](INSTALL.md)
4.  [Quickstart Guide](docs/quickstart_guide.md)
5.  [Tutorials](docs/tutorials/README.md)
6.  [How-Tos](docs/how_tos/README.md)
7.  [Prototype Template File Glossary](docs/file-map-and-description.md)
8.  [References and Acknowledgements](#references-and-acknowledgements)
9.  [License](#license)

##### For Developers/Contributors

1. [Contribution Guide](CONTRIBUTING.md)
2. [Technical Docs](docs/technical_docs.md)


----------------------------------------------------------------------------------------------------

## References and Acknowledgements
* [1] Variational Quantum Linear Solver, Carlos Bravo-Prieto, Ryan LaRose, M. Cerezo, Yigit Subasi, Lukasz Cincio, Patrick J. Coles, arXiv.1909.05820, 2020
* [2] VQLS Tutorial: https://github.com/qiskit-community/qiskit-textbook/blob/main/content/ch-paper-implementations/vqls.ipynb

----------------------------------------------------------------------------------------------------

### License
[Apache License 2.0](LICENSE.txt)
