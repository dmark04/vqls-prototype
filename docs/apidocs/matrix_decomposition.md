<!-- markdownlint-disable -->

<a href="../vqls_prototype/matrix_decomposition.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `vqls_prototype.matrix_decomposition`






---

<a href="../vqls_prototype/matrix_decomposition.py#L17"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `MatrixDecomposition`
Base class for the decomposition of a matrix in quantum circuits.  



<a href="../vqls_prototype/matrix_decomposition.py#L31"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    matrix: Optional[ndarray[Any, dtype[+ScalarType]]] = None,
    circuits: Optional[QuantumCircuit, List[QuantumCircuit]] = None,
    coefficients: Optional[float, complex, List[float], List[complex]] = None
)
```

Decompose a matrix representing quantum circuits 

Parameters 
---------- matrix : npt.NDArray  Array to decompose; only relevant in derived classes where  `self.decompose_matrix()` has been implemented 

circuits : Union[QuantumCircuit, List[QuantumCircuit]]  quantum circuits representing the matrix 

coefficients : Union[float, complex, List[float], List[complex]] (default: None)  coefficients associated with the input quantum circuits; `None` is  valid only for a circuit with 1 element 


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

<a href="../vqls_prototype/matrix_decomposition.py#L171"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `decompose_matrix`

```python
decompose_matrix() → Tuple[ndarray[Any, dtype[complex128]], List[ndarray[Any, dtype[complex128]]], List[QuantumCircuit]]
```





---

<a href="../vqls_prototype/matrix_decomposition.py#L155"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `recompose`

```python
recompose() → ndarray[Any, dtype[complex128]]
```

Rebuilds the original matrix from the decomposed one. 

Returns 
------- np.ndarray  The recomposed matrix. 

See Also 
-------- decompose_matrix : Decompose a generic numpy matrix into a sum of unitary matrices. 


---

<a href="../vqls_prototype/matrix_decomposition.py#L175"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `SymmetricDecomposition`
A class that represents the symmetric decomposition of a matrix. For the mathematical background for the decomposition, see the following math.sx answer: https://math.stackexchange.com/a/1710390 

Methods 
------- decompose_matrix() -> Tuple[complex_arr_t, List[complex_arr_t]]:  Decompose a generic numpy matrix into a sum of unitary matrices. 

See Also 
-------- MatrixDecomposition : A base class for matrix decompositions. recompose : Rebuilds the original matrix from the decomposed one. 

<a href="../vqls_prototype/matrix_decomposition.py#L31"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    matrix: Optional[ndarray[Any, dtype[+ScalarType]]] = None,
    circuits: Optional[QuantumCircuit, List[QuantumCircuit]] = None,
    coefficients: Optional[float, complex, List[float], List[complex]] = None
)
```

Decompose a matrix representing quantum circuits 

Parameters 
---------- matrix : npt.NDArray  Array to decompose; only relevant in derived classes where  `self.decompose_matrix()` has been implemented 

circuits : Union[QuantumCircuit, List[QuantumCircuit]]  quantum circuits representing the matrix 

coefficients : Union[float, complex, List[float], List[complex]] (default: None)  coefficients associated with the input quantum circuits; `None` is  valid only for a circuit with 1 element 


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

<a href="../vqls_prototype/matrix_decomposition.py#L214"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `auxilliary_matrix`

```python
auxilliary_matrix(
    x: Union[ndarray[Any, dtype[float64]], ndarray[Any, dtype[complex128]]]
) → ndarray[Any, dtype[complex128]]
```

Returns the auxiliary matrix for the decomposition of size n. 

Parameters 
---------- x : np.ndarray  original matrix. 

Returns 
------- np.ndarray  The auxiliary matrix. 

Notes 
----- The auxiliary matrix is defined as : i * sqrt(I - x^2) 

---

<a href="../vqls_prototype/matrix_decomposition.py#L238"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `decompose_matrix`

```python
decompose_matrix() → Tuple[ndarray[Any, dtype[complex128]], List[ndarray[Any, dtype[complex128]]], List[QuantumCircuit]]
```

Decompose a generic numpy matrix into a sum of unitary matrices. 

Parameters 
---------- matrix : np.ndarray  The matrix to be decomposed. 

Returns 
------- Tuple[np.ndarray, np.ndarray]  A tuple containing the list of coefficients and the numpy matrix of the decomposition. 

See Also 
-------- recompose : Rebuilds the original matrix from the decomposed one. 

---

<a href="../vqls_prototype/matrix_decomposition.py#L155"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `recompose`

```python
recompose() → ndarray[Any, dtype[complex128]]
```

Rebuilds the original matrix from the decomposed one. 

Returns 
------- np.ndarray  The recomposed matrix. 

See Also 
-------- decompose_matrix : Decompose a generic numpy matrix into a sum of unitary matrices. 


---

<a href="../vqls_prototype/matrix_decomposition.py#L288"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `PauliDecomposition`
A class that represents the Pauli decomposition of a matrix. 

Attributes 
---------- basis : str  The basis of Pauli gates used for the decomposition. 

Methods 
------- decompose_matrix() -> Tuple[complex_arr_t, List[complex_arr_t]]:  Decompose a matrix into a sum of Pauli strings. 

See Also 
-------- MatrixDecomposition : A base class for matrix decompositions. 

<a href="../vqls_prototype/matrix_decomposition.py#L31"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    matrix: Optional[ndarray[Any, dtype[+ScalarType]]] = None,
    circuits: Optional[QuantumCircuit, List[QuantumCircuit]] = None,
    coefficients: Optional[float, complex, List[float], List[complex]] = None
)
```

Decompose a matrix representing quantum circuits 

Parameters 
---------- matrix : npt.NDArray  Array to decompose; only relevant in derived classes where  `self.decompose_matrix()` has been implemented 

circuits : Union[QuantumCircuit, List[QuantumCircuit]]  quantum circuits representing the matrix 

coefficients : Union[float, complex, List[float], List[complex]] (default: None)  coefficients associated with the input quantum circuits; `None` is  valid only for a circuit with 1 element 


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

<a href="../vqls_prototype/matrix_decomposition.py#L326"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `decompose_matrix`

```python
decompose_matrix() → Tuple[ndarray[Any, dtype[complex128]], List[ndarray[Any, dtype[complex128]]], List[QuantumCircuit]]
```

Decompose a generic numpy matrix into a sum of Pauli strings. 



**Returns:**
  Tuple[complex_arr_t, List[complex_arr_t]]:   A tuple containing the list of coefficients and the numpy matrix of the decomposition. 

---

<a href="../vqls_prototype/matrix_decomposition.py#L155"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `recompose`

```python
recompose() → ndarray[Any, dtype[complex128]]
```

Rebuilds the original matrix from the decomposed one. 

Returns 
------- np.ndarray  The recomposed matrix. 

See Also 
-------- decompose_matrix : Decompose a generic numpy matrix into a sum of unitary matrices. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
