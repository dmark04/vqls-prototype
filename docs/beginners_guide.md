# Variational Quantum Linear Solver Beginner's Guide

This document should be a one-stop-shop for new users of the quantum prototype. It should pull together much of the other documentation into a single guide. Prototype users that go through the Beginner's Guide will be given some background on the project and software, walked through the installation process, shown the basics of the API, and given a small example problem to solve.

## About the Project

VQLS allows to solve linear system of the form 

$$ A \cdot x = b $$

with $A$ is a matrix, $b$ a vector and $x$ the solution of the system. For this VQLS use a variational approach with a variational ansatz $V(\theta)$ such as 


$$|x(\theta)\rangle = V(\theta)|0\rangle$$. 


The variational parameters of the ansatz are then optimized by minimizing the cost function 

$$ C_L(\theta) = 1 - \frac{\langle b|\Psi(\theta)\rangle}{\langle \Psi(\theta)|\Psi(\theta)\rangle} $$

with $|\Psi(\theta)\rangle = A|x(\theta)\rangle$. 

## Installation
The code does not rely on external dependencies, except qiskit itself of course. After cloning the repository pip install it.

```
git clone https://github.com/QuantumApplicationLab/vqls-prototype 
cd vqls-prototype
pip install .
```

## Usage
The code is organized very similarly as the VQE class. Hence users may simply instantiate the VQLS class and run the otpimization

```python
from qalcore.qiskit.vqls.vqls import VQLS
from qiskit.primitives import Estimator 

estimator = Estimator()
vqls = VQLS(estimator, ansatz, COBYLA(maxiter=2, disp=True))
res = vqls.solve(A, b)

```

## Example Problem
We show here how to solve a small 4x4 linear system using VQLS

```python
import numpy as np
from qalcore.qiskit.vqls.vqls import VQLS, VQLSLog
from qiskit.primitives import Estimator 
from qiskit.circuit.library.n_local.real_amplitudes import RealAmplitudes
from qiskit.algorithms import optimizers as opt
size = 4

# define the matrix of the problem
A = np.random.rand(size, size)
A = A + A.T

# define the b vector
b = np.random.rand(size)

# define the ansatz
ansatz = RealAmplitudes(2, entanglement="full", reps=3)

# define the VQLS solver
log = VQLSLog([],[])
estimator = Estimator()
vqls = VQLS(
    estimator,
    ansatz,
    opt.CG(maxiter=200),
    callback=log.update 
)

# run the solver
res = vqls.solve(A, b)

```

