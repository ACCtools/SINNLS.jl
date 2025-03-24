module Tests

using SINNLS
using LinearAlgebra, SparseArrays
using Test

for i = 5:10
    a = reshape(collect(0.0:i * 5 - 1), :, 5)
    x_true = collect(0.0:4.0)
    y = a * x_true

    a = sparse(a)
    x = SI_NNLS_simple(a, y)

    @test norm(a * x - y) < 1e-3
end

end