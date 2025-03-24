module Tests

using SINNLS
using LinearAlgebra, SparseArrays
using Test

for i = 5:5
    a::Matrix{Float64} = reshape(collect(0.0:i * 5 - 1), :, 5)
    x_true = collect(0.0:4.0)
    y::Vector{Float64} = a * x_true

    as = sparse(a)
    x = SI_NNLS_simple(as, y)

    @test norm(a * x - y) < 1e-3
end

for i = 5:5
    a::Matrix{Float32} = reshape(collect(0.0:i * 5 - 1), :, 5)
    x_true = collect(0.0:4.0)
    y::Vector{Float32} = a * x_true

    as = sparse(a)
    x = SI_NNLS_simple(as, y)

    @test norm(a * x - y) < 1e-3
end
end