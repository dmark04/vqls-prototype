<!-- markdownlint-disable -->

<a href="../vqls_prototype/hadamard_test.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `vqls_prototype.hadamard_test`
Hadammard test. 



---

<a href="../vqls_prototype/hadamard_test.py#L10"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `HadammardTest`
Class to compute the Hadamard Test  



<a href="../vqls_prototype/hadamard_test.py#L13"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    operators: Union[QuantumCircuit, List[QuantumCircuit]],
    use_barrier: Optional[bool] = False,
    apply_control_to_operator: Optional[bool, List[bool]] = True,
    apply_initial_state: Optional[QuantumCircuit] = None,
    apply_measurement: Optional[bool] = False
)
```

Create the quantum circuits required to compute the hadamard test: 

.. math:
``` 

     \\langle \\Psi | U | \\Psi \\rangle 

```


**Args:**
 
 - <b>`operators`</b> (Union[QuantumCircuit, List[QuantumCircuit]]):  quantum circuit or list of quantum circuits representing the U. 
 - <b>`use_barrier`</b> (Optional[bool], optional):  introduce barriers in the description of the circuits.  Defaults to False. 
 - <b>`apply_control_to_operator`</b> (Optional[bool], optional):  Apply control operator to the input quantum circuits. Defaults to True. 
 - <b>`apply_initial_state`</b> (Optional[QuantumCircuit], optional):  Quantum Circuit to create |Psi> from |0>. If None, assume that the qubits are alredy in Psi. 
 - <b>`apply_measurement`</b> (Optional[bool], optional):  apply explicit measurement. Defaults to False. 




---

<a href="../vqls_prototype/hadamard_test.py#L162"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_value`

```python
get_value(estimator, parameter_sets: List) → List
```






---

<a href="../vqls_prototype/hadamard_test.py#L181"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `HadammardOverlapTest`
Class to compute the Hadamard Test  



<a href="../vqls_prototype/hadamard_test.py#L184"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    operators: List[QuantumCircuit],
    use_barrier: Optional[bool] = False,
    apply_initial_state: Optional[QuantumCircuit] = None,
    apply_measurement: Optional[bool] = True
)
```

Create the quantum circuits required to compute the hadamard test: 

.. math:
``` 

     \\langle 0 | U^\dagger A_l V | 0 \\rangle \\langle V^\dagger A_m^\dagger U | 0 \\rangle 

```


**Args:**
 
 - <b>`operators`</b> (List[QuantumCircuit]):  List of quantum circuits representing the operators [U, A_l, A_m]. 
 - <b>`use_barrier`</b> (Optional[bool], optional):  introduce barriers in the description of the circuits.  Defaults to False. 
 - <b>`apply_initial_state`</b> (Optional[QuantumCircuit], optional):  Quantum Circuit to create |Psi> from |0>. If None, assume that the qubits of the firsr register are alredy in Psi. 
 - <b>`apply_measurement`</b> (Optional[bool], optional):  apply explicit measurement. Defaults to False. 



**Returns:**
 
 - <b>`List[QuantumCircuit]`</b>:  List of quamtum circuits required to compute the Hadammard Test. 




---

<a href="../vqls_prototype/hadamard_test.py#L315"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `compute_post_processing_coefficients`

```python
compute_post_processing_coefficients()
```

Compute the coefficients for the postprocessing   



---

<a href="../vqls_prototype/hadamard_test.py#L348"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_value`

```python
get_value(sampler, parameter_sets: List) → float
```

Compute and return the value of Hadmard overlap test 



**Args:**
 
 - <b>`sampler`</b> (Sampler):  a Sampler primitive to extract the output of the circuits 
 - <b>`parameter_sets`</b> (List):  the parameters of the variational circuits 



**Returns:**
 
 - <b>`float`</b>:  value of the overlap hadammard test 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
