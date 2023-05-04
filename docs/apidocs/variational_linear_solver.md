<!-- markdownlint-disable -->

<a href="../vqls_prototype/variational_linear_solver.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `vqls_prototype.variational_linear_solver`
An abstract class for variational linear systems solvers. 



---

<a href="../vqls_prototype/variational_linear_solver.py#L24"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `VariationalLinearSolverResult`
A base class for linear systems results using variational methods 

The  linear systems variational algorithms return an object of the type ``VariationalLinearSystemsResult`` with the information about the solution obtained. 

<a href="../vqls_prototype/variational_linear_solver.py#L31"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__() → None
```






---

#### <kbd>property</kbd> cost_function_evals

Returns number of cost optimizer evaluations 

---

#### <kbd>property</kbd> optimal_circuit

The optimal circuits. Along with the optimal parameters, these can be used to retrieve the minimum eigenstate. 

---

#### <kbd>property</kbd> optimal_parameters

Returns the optimal parameters in a dictionary 

---

#### <kbd>property</kbd> optimal_point

Returns optimal point 

---

#### <kbd>property</kbd> optimal_value

Returns optimal value 

---

#### <kbd>property</kbd> optimizer_evals

Returns number of optimizer evaluations 

---

#### <kbd>property</kbd> optimizer_result

Returns the optimizer result 

---

#### <kbd>property</kbd> optimizer_time

Returns time taken for optimization 

---

#### <kbd>property</kbd> state

return either the circuit that prepares the solution or the solution as a vector 




---

<a href="../vqls_prototype/variational_linear_solver.py#L59"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `VariationalLinearSolver`
An abstract class for linear system solvers in Qiskit. 




---

<a href="../vqls_prototype/variational_linear_solver.py#L62"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `solve`

```python
solve(
    matrix: Union[ndarray, QuantumCircuit],
    vector: Union[ndarray, QuantumCircuit],
    **kwargs
) → VariationalLinearSolverResult
```

Solve the system and compute the observable(s) 



**Args:**
 
 - <b>`matrix`</b>:  The matrix specifying the system, i.e. A in Ax=b. 
 - <b>`vector`</b>:  The vector specifying the right hand side of the equation in Ax=b. 



**Returns:**
 The result of the linear system. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
