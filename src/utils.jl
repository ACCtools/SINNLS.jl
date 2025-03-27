using LinearAlgebra
using SparseArrays
using IterativeSolvers
using KrylovKit
using Octavian

function reformulation_sparse!(A::SparseMatrixCSC{T, Int}, b::Vector{T}) where T <: AbstractFloat
    col_norm = norm.(eachcol(A))
    non_zero_col_idx = Int[]
    non_zero_col_norm = T[]
    for i in 1:length(col_norm)
        if abs(col_norm[i]) > T(1e-10)
            push!(non_zero_col_idx, i)
            push!(non_zero_col_norm, col_norm[i])
        end
    end
    Â = A[:, non_zero_col_idx]
    A_T_b = (b' * Â)[:]

    final_non_zero_col_idx = Int[]
    for i in 1:length(A_T_b)
        if A_T_b[i] > zero(T)
            push!(final_non_zero_col_idx, i)
        end
    end
    Â = Â[:, final_non_zero_col_idx]
    A_T_b = A_T_b[final_non_zero_col_idx]
    non_zero_col_norm = non_zero_col_norm[final_non_zero_col_idx]

    return Â, A_T_b, non_zero_col_norm, non_zero_col_idx[final_non_zero_col_idx]
end

function reformulation_sparse!(A::AbstractMatrix{T}, b::Vector{T}) where T <: AbstractFloat
    col_norm = norm.(eachcol(A))
    non_zero_col_idx = Int[]
    non_zero_col_norm = T[]
    for i in 1:length(col_norm)
        if abs(col_norm[i]) > T(1e-10)
            push!(non_zero_col_idx, i)
            push!(non_zero_col_norm, col_norm[i])
        end
    end
    @views Â = A[:, non_zero_col_idx]
    A_T_b = matmul(b', Â)

    final_non_zero_col_idx = Int[]
    for i in 1:length(A_T_b)
        if A_T_b[i] > zero(T)
            push!(final_non_zero_col_idx, i)
        end
    end
    @views Â = Â[:, final_non_zero_col_idx]
    A_T_b = A_T_b[final_non_zero_col_idx]
    non_zero_col_norm = non_zero_col_norm[final_non_zero_col_idx]

    return Â, A_T_b, non_zero_col_norm, non_zero_col_idx[final_non_zero_col_idx]
end

function compute_blocks_rows_slice(C::SparseMatrixCSC{T, Int}, blocksize::Int) where T <: AbstractFloat
    d, n = size(C)

    blocks = UnitRange{Int}[]
    row_idxs = Vector{Int}[]
    len_b = n % blocksize == 0 ? n ÷ blocksize : n ÷ blocksize + 1
    for i in 1:len_b
        if i == len_b
            push!(blocks, (1+(i-1)*blocksize):n)
        else
            push!(blocks, (1+(i-1)*blocksize):(i*blocksize))
        end

        row_set = Set{Int}()
        for j in blocks[i]
            loc = C.colptr[j]:(C.colptr[j+1]-1)
            union!(row_set, C.rowval[loc])
        end
        push!(row_idxs, collect(row_set))
    end

    sliced_Cs = Vector{SparseMatrixCSC{T, Int}}()
    for j in 1:length(row_idxs)
        push!(sliced_Cs, C[row_idxs[j], blocks[j]])
    end

    return blocks, row_idxs, sliced_Cs
end

function compute_blocks_rows_slice(C::AbstractMatrix{T}, blocksize::Int) where T <: AbstractFloat
    d, n = size(C)
    blocks = UnitRange{Int}[]
    row_idxs = Vector{Int}[]
    
    len_b = (n % blocksize == 0) ? (n ÷ blocksize) : (n ÷ blocksize + 1)
    for i in 1:len_b
        block = (i == len_b) ? ((1+(i-1)*blocksize):n) : ((1+(i-1)*blocksize):(i*blocksize))
        push!(blocks, block)
        
        # @view를 사용해 메모리 복사 없이 서브행렬 참조
        subC = @view C[:, block]
        # 각 행에 대해 0이 아닌 원소가 존재하면 해당 행 인덱스를 저장
        active_rows = findall(r -> any(!iszero, r), eachrow(subC))
        push!(row_idxs, active_rows)
    end

    sliced_Cs = Vector{SubArray}()
    for i in 1:length(row_idxs)
        push!(sliced_Cs, @view C[row_idxs[i], blocks[i]])
    end

    return blocks, row_idxs, sliced_Cs
end

function compute_Lips(C::SparseMatrixCSC{T, Int}, blocks, row_idxs) where T <: AbstractFloat
    etas = zeros(T, length(blocks))
    blocksize = length(blocks[1])
    if blocksize >= 5
        for i in 1:length(blocks)
            sub_C = C[row_idxs[i], blocks[i]]
            c, _, _, _ = svdsolve(sub_C)
            etas[i] = 1 / (c[1]^2)
        end
    elseif blocksize == 1
        for i in 1:length(blocks)
            etas[i] = 1 / (norm(C[row_idxs[i], blocks[i]])^2)
        end
    else
        @info "not supported for block size for 2,3,4"
    end
    return etas
end

function compute_Lips(C::AbstractMatrix{T}, blocks, row_idxs) where T <: AbstractFloat
    etas = zeros(T, length(blocks))
    blocksize = length(blocks[1])
    if blocksize >= 5
        for i in 1:length(blocks)
            sub_C = C[row_idxs[i], blocks[i]]
            c, _, _, _ = svdsolve(sub_C)
            etas[i] = 1 / (c[1]^2)
        end
    elseif blocksize == 1
        for i in 1:length(blocks)
            etas[i] = 1 / (norm(C[row_idxs[i], blocks[i]])^2)
        end
    else
        @info "not supported for block size for 2,3,4"
    end
    return etas
end
