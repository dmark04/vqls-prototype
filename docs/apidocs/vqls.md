<!-- markdownlint-disable -->

<a href="../vqls_prototype/vqls.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `vqls_prototype.vqls`
Variational Quantum Linear Solver 

See https://arxiv.org/abs/1909.05820 



---

<a href="../vqls_prototype/vqls.py#L32"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `VQLSLog`
VQLSLog(values: List, parameters: List) 




---

<a href="../vqls_prototype/vqls.py#L35"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `update`

```python
update(count, cost, parameters)
```






---

<a href="../vqls_prototype/vqls.py#L40"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `VQLS`
Systems of linear equations arise naturally in many real-life applications in a wide range of areas, such as in the solution of Partial Differential Equations, the calibration of financial models, fluid simulation or numerical field calculation. The problem can be defined as, given a matrix :math:`A\in\mathbb{C}^{N\times N}` and a vector :math:`\vec{b}\in\mathbb{C}^{N}`, find :math:`\vec{x}\in\mathbb{C}^{N}` satisfying :math:`A\vec{x}=\vec{b}`. 


```python
 from qalcore.qiskit.vqls.vqls import VQLS, VQLSLog  
 from qiskit.circuit.library.n_local.real_amplitudes import RealAmplitudes  
 from qiskit.algorithms import optimizers as opt  
 from qiskit import Aer, BasicAer  
 import numpy as np 

 from qiskit.quantum_info import Statevector  
 import matplotlib.pyplot as plt  f
 rom qiskit.primitives import Estimator, Sampler, BackendEstimator 

 # create random symmetric matrix  
 A = np.random.rand(4, 4)  A = A + A.T 

 # create rhight hand side  
 b = np.random.rand(4) 

 # solve using numpy  
 classical_solution = np.linalg.solve(A, b / np.linalg.norm(b))  
 ref_solution = classical_solution / np.linalg.norm(classical_solution) 

 # define the wave function ansatz  
 ansatz = RealAmplitudes(2, entanglement="full", reps=3, insert_barriers=False) 

 # define an estimator primitive  
 estimator = Estimator() 

 # define the logger  
 log = VQLSLog([],[]) 

 # create the solver  
 vqls = VQLS(  estimator,  ansatz,  opt.CG(maxiter=200),  callback=log.update  ) 

 # solve   
 res = vqls.solve(A, b, opt)  vqls_solution = np.real(Statevector(res.state).data) 

 # plot solution  
 plt.scatter(ref_solution, vqls_solution)  plt.plot([-1, 1], [-1, 1], "--")  plt.show() 

 # plot cost function   
 plt.plot(log.values)  plt.ylabel('Cost Function')  plt.xlabel('Iterations')  plt.show() 

 ```

References: 

 [1] Carlos Bravo-Prieto, Ryan LaRose, M. Cerezo, Yigit Subasi, Lukasz Cincio, Patrick J. Coles  Variational Quantum Linear Solver  `arXiv:1909.05820 <https://arxiv.org/abs/1909.05820>` 

<a href="../vqls_prototype/vqls.py#L115"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    estimator: BaseEstimator,
    ansatz: QuantumCircuit,
    optimizer: Union[Optimizer, Minimizer],
    sampler: Optional[BaseSampler] = None,
    initial_point: Optional[ndarray] = None,
    gradient: Optional[GradientBase, Callable] = None,
    max_evals_grouped: Optional[int] = 1,
    callback: Optional[Callable[[int, ndarray, float, float]], NoneType] = None
) → None
```



**Args:**
 
 - <b>`estimator`</b>:  an Estimator primitive to compute the expected values of the   quantum circuits needed for the cost function 
 - <b>`ansatz`</b>:  A parameterized circuit used as Ansatz for the wave function. 
 - <b>`optimizer`</b>:  A classical optimizer. Can either be a Qiskit optimizer or a callable  that takes an array as input and returns a Qiskit or SciPy optimization result. 
 - <b>`sampler`</b>:  a Sampler primitive to sample the output of some quantum circuits needed to  compute the cost function. This is only needed if overal Hadammard tests are used. 
 - <b>`initial_point`</b>:  An optional initial point (i.e. initial parameter values)  for the optimizer. If ``None`` then VQLS will look to the ansatz for a preferred  point and if not will simply compute a random one. 
 - <b>`gradient`</b>:  An optional gradient function or operator for optimizer. 
 - <b>`max_evals_grouped`</b>:  Max number of evaluations performed simultaneously. Signals the  given optimizer that more than one set of parameters can be supplied so that  potentially the expectation values can be computed in parallel. Typically this is  possible when a finite difference gradient is used by the optimizer such that  multiple points to compute the gradient can be passed and if computed in parallel  improve overall execution time. Deprecated if a gradient operator or function is  given. 
 - <b>`callback`</b>:  a callback that can access the intermediate data during the optimization.  Three parameter values are passed to the callback as follows during each evaluation  by the optimizer for its current set of parameters as it works towards the minimum. 
 - <b>`These are`</b>:  the evaluation count, the cost and the optimizer parameters for the ansatz 


---

#### <kbd>property</kbd> ansatz

Returns the ansatz. 

---

#### <kbd>property</kbd> callback

Returns callback 

---

#### <kbd>property</kbd> initial_point

Returns initial point 

---

#### <kbd>property</kbd> max_evals_grouped

Returns max_evals_grouped 

---

#### <kbd>property</kbd> num_clbits

return the numner of classical bits 

---

#### <kbd>property</kbd> num_qubits

return the numner of qubits 

---

#### <kbd>property</kbd> optimizer

Returns optimizer 



---

<a href="../vqls_prototype/vqls.py#L274"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `construct_circuit`

```python
construct_circuit(
    matrix: Union[ndarray, QuantumCircuit, List],
    vector: Union[ndarray, QuantumCircuit],
    options: Dict
) → List[QuantumCircuit]
```

Returns the a list of circuits required to compute the expectation value 



**Args:**
 
 - <b>`matrix`</b> (Union[np.ndarray, QuantumCircuit, List]):  matrix of the linear system 
 - <b>`vector`</b> (Union[np.ndarray, QuantumCircuit]):  rhs of thge linear system 
 - <b>`options`</b> (Dict):  Options to compute define the quantum circuits that compute the cost function  



**Raises:**
 
 - <b>`ValueError`</b>:  if vector and matrix have different size 
 - <b>`ValueError`</b>:  if vector and matrix have different numner of qubits 
 - <b>`ValueError`</b>:  the input matrix is not a numoy array nor a quantum circuit 



**Returns:**
 
 - <b>`List[QuantumCircuit]`</b>:  Quantum Circuits required to compute the cost function 

---

<a href="../vqls_prototype/vqls.py#L465"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_coefficient_matrix`

```python
get_coefficient_matrix(coeffs) → ndarray
```

Compute all the vi* vj terms 



**Args:**
 
 - <b>`coeffs`</b> (np.ndarray):  list of complex coefficients 

---

<a href="../vqls_prototype/vqls.py#L648"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_cost_evaluation_function`

```python
get_cost_evaluation_function(
    hdmr_tests_norm: List,
    hdmr_tests_overlap: List,
    coefficient_matrix: ndarray,
    options: Dict
) → Callable[[ndarray], Union[float, List[float]]]
```

Generate the cost function of the minimazation process 



**Args:**
 
 - <b>`hdmr_tests_norm`</b> (List):  list of quantum circuits needed to compute the norm 
 - <b>`hdmr_tests_overlap`</b> (List):  list of quantum circuits needed to compute the norm 
 - <b>`coefficient_matrix`</b> (np.ndarray):  the matrix values of the c_n^* c_m coefficients 
 - <b>`options`</b> (Dict):  Option to compute the cost function 



**Raises:**
 
 - <b>`RuntimeError`</b>:  If the ansatz is not parametrizable 



**Returns:**
 
 - <b>`Callable[[np.ndarray], Union[float, List[float]]]`</b>:  the cost function 

---

<a href="../vqls_prototype/vqls.py#L738"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `solve`

```python
solve(
    matrix: Union[ndarray, QuantumCircuit, List[QuantumCircuit]],
    vector: Union[ndarray, QuantumCircuit],
    options: Optional[Dict] = None
) → VariationalLinearSolverResult
```

Solve the linear system 



**Args:**
 
 - <b>`matrix`</b> (Union[List, np.ndarray, QuantumCircuit]):  matrix of the linear system 
 - <b>`vector`</b> (Union[np.ndarray, QuantumCircuit]):  rhs of the linear system 
 - <b>`options`</b> (Union[Dict, None]):  options for the calculation of the cost function   



**Returns:**
 
 - <b>`VariationalLinearSolverResult`</b>:  Result of the optimization and solution vector of the linear system 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
