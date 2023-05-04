<!-- markdownlint-disable -->

<a href="../vqls_prototype/matrix_decomposition.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `vqls_prototype.matrix_decomposition`






---

<a href="../vqls_prototype/matrix_decomposition.py#L17"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `MatrixDecomposition`
Base class for the decomposition of a matrix in quantum circuits.  



<a href="../vqls_prototype/matrix_decomposition.py#L38"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    matrix: Optional[ndarray[Any, dtype[+ScalarType]]] = None,
    circuits: Optional[QuantumCircuit, List[QuantumCircuit]] = None,
    coefficients: Optional[float, complex, List[float], List[complex]] = None
)
```

Decompose a matrix representing quantum circuits 



**Args:**
 
 - <b>`matrix`</b> (Optional[npt.NDArray], optional):  Array to decompose; only relevant in derived classes where `self.decompose_matrix()` has been implemented. Defaults to None. 
 - <b>`circuits`</b> (Optional[Union[QuantumCircuit, List[QuantumCircuit]]], optional):   quantum circuits representing the matrix. Defaults to None. 
 - <b>`coefficients`</b> (Optional[ Union[float, complex, List[float], List[complex]] ], optional):  coefficients associated with the input quantum circuits; `None` is valid only for a circuit with 1 element. Defaults to None. 


---

#### <kbd>property</kbd> circuits

circuits of the decomposition 

---

#### <kbd>property</kbd> coefficients

coefficients of the decomposition. 

---

#### <kbd>property</kbd> matrices

return the unitary matrices 

---

#### <kbd>property</kbd> matrix

matrix of the decomposition 



---

<a href="../vqls_prototype/matrix_decomposition.py#L185"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `decompose_matrix`

```python
decompose_matrix() → Tuple[ndarray[Any, dtype[complex128]], List[ndarray[Any, dtype[complex128]]], List[QuantumCircuit]]
```





---

<a href="../vqls_prototype/matrix_decomposition.py#L176"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `recompose`

```python
recompose() → ndarray[Any, dtype[complex128]]
```

Rebuilds the original matrix from the decomposed one. 



**Returns:**
 
 - <b>`complex_arr_t`</b>:  The recomposed matrix. 


---

<a href="../vqls_prototype/matrix_decomposition.py#L189"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `SymmetricDecomposition`
A class that represents the symmetric decomposition of a matrix. For the mathematical background for the decomposition, see the following math.sx answer: https://math.stackexchange.com/a/1710390 

<a href="../vqls_prototype/matrix_decomposition.py#L38"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    matrix: Optional[ndarray[Any, dtype[+ScalarType]]] = None,
    circuits: Optional[QuantumCircuit, List[QuantumCircuit]] = None,
    coefficients: Optional[float, complex, List[float], List[complex]] = None
)
```

Decompose a matrix representing quantum circuits 



**Args:**
 
 - <b>`matrix`</b> (Optional[npt.NDArray], optional):  Array to decompose; only relevant in derived classes where `self.decompose_matrix()` has been implemented. Defaults to None. 
 - <b>`circuits`</b> (Optional[Union[QuantumCircuit, List[QuantumCircuit]]], optional):   quantum circuits representing the matrix. Defaults to None. 
 - <b>`coefficients`</b> (Optional[ Union[float, complex, List[float], List[complex]] ], optional):  coefficients associated with the input quantum circuits; `None` is valid only for a circuit with 1 element. Defaults to None. 


---

#### <kbd>property</kbd> circuits

circuits of the decomposition 

---

#### <kbd>property</kbd> coefficients

coefficients of the decomposition. 

---

#### <kbd>property</kbd> matrices

return the unitary matrices 

---

#### <kbd>property</kbd> matrix

matrix of the decomposition 



---

<a href="../vqls_prototype/matrix_decomposition.py#L215"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `auxilliary_matrix`

```python
auxilliary_matrix(
    x: Union[ndarray[Any, dtype[float64]], ndarray[Any, dtype[complex128]]]
) → ndarray[Any, dtype[complex128]]
```

Returns the auxiliary matrix for the decomposition of size n and derfined as defined as : i * sqrt(I - x^2) 



**Args:**
 
 - <b>`x`</b> (Union[npt.NDArray[np.float_], complex_arr_t]):  original matrix. 



**Returns:**
 
 - <b>`complex_arr_t`</b>:  The auxiliary matrix. 

---

<a href="../vqls_prototype/matrix_decomposition.py#L229"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `decompose_matrix`

```python
decompose_matrix() → Tuple[ndarray[Any, dtype[complex128]], List[ndarray[Any, dtype[complex128]]], List[QuantumCircuit]]
```

Decompose a generic numpy matrix into a sum of unitary matrices. 



**Returns:**
 
 - <b>`Tuple[complex_arr_t, List[complex_arr_t], List[QuantumCircuit]]`</b>:   A tuple containing the list of coefficients numpy matrices, and quantum circuits of the decomposition. 

---

<a href="../vqls_prototype/matrix_decomposition.py#L176"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `recompose`

```python
recompose() → ndarray[Any, dtype[complex128]]
```

Rebuilds the original matrix from the decomposed one. 



**Returns:**
 
 - <b>`complex_arr_t`</b>:  The recomposed matrix. 


---

<a href="../vqls_prototype/matrix_decomposition.py#L268"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `PauliDecomposition`
A class that represents the Pauli decomposition of a matrix. 

<a href="../vqls_prototype/matrix_decomposition.py#L38"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    matrix: Optional[ndarray[Any, dtype[+ScalarType]]] = None,
    circuits: Optional[QuantumCircuit, List[QuantumCircuit]] = None,
    coefficients: Optional[float, complex, List[float], List[complex]] = None
)
```

Decompose a matrix representing quantum circuits 



**Args:**
 
 - <b>`matrix`</b> (Optional[npt.NDArray], optional):  Array to decompose; only relevant in derived classes where `self.decompose_matrix()` has been implemented. Defaults to None. 
 - <b>`circuits`</b> (Optional[Union[QuantumCircuit, List[QuantumCircuit]]], optional):   quantum circuits representing the matrix. Defaults to None. 
 - <b>`coefficients`</b> (Optional[ Union[float, complex, List[float], List[complex]] ], optional):  coefficients associated with the input quantum circuits; `None` is valid only for a circuit with 1 element. Defaults to None. 


---

#### <kbd>property</kbd> circuits

circuits of the decomposition 

---

#### <kbd>property</kbd> coefficients

coefficients of the decomposition. 

---

#### <kbd>property</kbd> matrices

return the unitary matrices 

---

#### <kbd>property</kbd> matrix

matrix of the decomposition 



---

<a href="../vqls_prototype/matrix_decomposition.py#L290"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `decompose_matrix`

```python
decompose_matrix() → Tuple[ndarray[Any, dtype[complex128]], List[ndarray[Any, dtype[complex128]]], List[QuantumCircuit]]
```

Decompose a generic numpy matrix into a sum of Pauli strings. 



**Returns:**
  Tuple[complex_arr_t, List[complex_arr_t]]:   A tuple containing the list of coefficients and the numpy matrix of the decomposition. 

---

<a href="../vqls_prototype/matrix_decomposition.py#L176"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `recompose`

```python
recompose() → ndarray[Any, dtype[complex128]]
```

Rebuilds the original matrix from the decomposed one. 



**Returns:**
 
 - <b>`complex_arr_t`</b>:  The recomposed matrix. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
