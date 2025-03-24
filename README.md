# SI-NNLS+ implementation using Julia

All core code is copyrighted by the authors of the paper. The purpose of this module is to improve the interface and make it easier to use regardless of the type of type matrix.

```julia
using SINNLS, SparseArrays

A = [1.0 3.0;3.0 8.0]
B = [3.0,8.0]
A = sparse(A)

w = SI_NNLS_simple(A, B)

isapprox(A * w, b, atol=1e-3)
```

[1] A Fast Scale-Invariant Algorithm for Non-negative Least Squares with Non-negative Data (2022)