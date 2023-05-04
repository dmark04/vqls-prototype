# VQLS Prototype Installation Guide

This document should walk you throgh the installation of the vqls prototpe

## Setting up Python Environment

Create a new conda environement. The code has been tested with different python version and should work for version 3.8 onward.

```
conda create -n vqls python==3.9
conda activate vqls
``` 

## Installing Depencencies

The dependencies needed are all included in the `requirements.txt`. Therefore you do not need to install anything prior to the protoype 

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
