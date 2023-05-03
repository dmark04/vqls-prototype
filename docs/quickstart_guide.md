# Quickstart Guide

The quickstart guide is similar to the beginner's guide. The key difference is that the quickstart guide assumes the user already understands what the software intends to do and generally how they would like to use it, and they just need to quickly get spun up on the API. To that end, this document won't provide any scientific background or detailed tutorials/how-tos. It will focus more on installation and usage.

## Installation

```
git clone https://github.com/QuantumApplicationLab/vqls_prototype
cd vqls_prototype
pip install -e .
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
