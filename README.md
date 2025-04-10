# SI-NNLS+ implementation using Julia

All core code is copyrighted by the authors of the paper. The purpose of this module is to improve the interface and make it easier to use regardless of the type of type sparse matrix or matrix.

```julia
using SINNLS, SparseArrays

A = [2.0 2.0;2.0 8.0]
B = [2.0,6.0]
As = sparse(A)

ws = SI_NNLS(As, B) # Sparse matrix input
w = SI_NNLS(A, B) # Normal matrix input
```

[1] A Fast Scale-Invariant Algorithm for Non-negative Least Squares with Non-negative Data (2022)