# VQLS Prototype Installation Guide

This document should walk you through the installation of the vqls prototype.

## Setting up Python Environment

Create a new conda environment. The code has been tested with different python version and should work for version 3.9 onward.

```
conda create -n vqls python==3.9
conda activate vqls
``` 

## Installing Dependencies

The dependencies needed are all included in `pyproject.toml`. Therefore, you do not need to install anything prior to the prototype.

## Installing Quantum Prototype Software

Clone the repository and pip install it.

```
git clone https://github.com/QuantumApplicationLab/vqls-prototype
cd vqls-prototype
pip install .
```


## Testing the Installation

You can test your installation by executing the tests

```
cd  tests
pytest
```
