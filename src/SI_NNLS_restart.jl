using IterativeSolvers

function SI_NNLS_simple(C::SparseMatrixCSC{T, Int}, b::Vector{T};
                         blocksize::Int = 1, 
                         K::Int = 100000, 
                         total_time::Float64 = 3600.0,
                         num_restart::Int = 200, 
                         freq::Int = 20, 
                         restart_ratio::Float64 = 0.7,  
                         ϵ::Float64 = 1e-7, 
                         time_limit::Bool = true) where T <: AbstractFloat

    C, C_b, non_zero_col_norm = reformulation_sparsev4!(C, b)
    m, n = size(C)
    x0_ = zeros(T, n)
    C_x0_ = zeros(T, m)

    return SI_NNLS_restart!(C, b, C_b, x0_, C_x0_, blocksize, K, total_time,
                            num_restart, freq, restart_ratio, ϵ, time_limit)
end

function SI_NNLS_restart!(C::SparseMatrixCSC{T, Int}, b::Vector{T}, C_b::Vector{T}, 
                          x0::Vector{T}, C_x0::Vector{T}, blocksize::Int, K::Int, total_time::Float64,
                          num_restart::Int, freq::Int, restart_ratio::Float64, ϵ::Float64, time_limit::Bool = true) where T <: AbstractFloat
    col_norm_square = norm.(eachcol(C)).^2

    init_metric = first_order_opt(C, b, x0, C_x0, C_b, col_norm_square)
    init_epoch = 0
    init_time = 0.0

    blocks, row_idxs, sliced_Cs = compute_blocks_rows_slice(C, blocksize)
    ηs = compute_Lips(C, blocks, row_idxs)
    
    for i in 1:num_restart
        x0, C_x0, init_metric, init_epoch, init_time = SI_NNLS!(C, b, C_b, x0, C_x0,
                                                               blocks, row_idxs, sliced_Cs, ηs, K,
                                                               total_time, freq, init_metric,
                                                               init_epoch, init_time, restart_ratio, ϵ, col_norm_square)
        if (init_time >= total_time || !time_limit) || init_metric < ϵ
            break
        end
    end
    return x0
end

function SI_NNLS!(C::SparseMatrixCSC{T, Int}, b::Vector{T}, C_b::Vector{T}, 
                  x0::Vector{T}, C_x0::Vector{T}, blocks::Array{UnitRange{Int}}, 
                  row_idxs::Array{Vector{Int}}, sliced_Cs, ηs, K::Int, total_time::Float64, 
                  freq::Int, init_metric::T, init_epoch::Int, 
                  init_time::Float64, restart_ratio::Float64, ϵ::Float64, col_norm_square::Vector{T}) where T <: AbstractFloat
    t0 = time()
    m, n = size(C)

    num_blks = length(blocks)
    K *= num_blks
    prev_a, prev_A = zero(T), zero(T)
    C_b = (b' * C)[:]
    upper_bounds = C_b ./ col_norm_square

    a = 1 / (sqrt(2) * (num_blks)^(1.5))
    A = a
    later_a = a / T(num_blks - 1)
    later_A = A + later_a

    p = zeros(T, n)
    r = zeros(T, n)
    s = zeros(T, m)
    t = zeros(T, m)
    q = copy(C_x0)
    x = copy(x0)

    idx_seq = 1:num_blks
    prev_jk, jk = 0, 0
    for k in 1:K
        prev_jk = jk
        jk = rand(idx_seq)
        j = blocks[jk]
        row_j = row_idxs[jk]
        sliced_C = sliced_Cs[jk]
        if k == 1
            product = (q[row_j]' * sliced_C)'
        elseif k == 2
            product = ((q[row_j] + (prev_a/a)*t[row_j])' * sliced_C)'
            t[row_idxs[prev_jk]] .= zero(T)
        else
            ratio = (prev_a^2) / (a * (prev_A - prev_a))
            product = ((q[row_j] + (1-ratio)/prev_A*s[row_j] + (num_blks - 1)*ratio*t[row_j])' * sliced_C)'
            t[row_idxs[prev_jk]] .= zero(T)
        end
        p[j] = p[j] .+ num_blks * a * (product .- C_b[j])
        prev_xj = x[j]
        x[j] = max.(zero(T), min.(x0[j] - (ηs[jk]) * p[j], upper_bounds[j]))
        t[row_j] = sliced_C * (x[j] - prev_xj)
        q[row_j] = q[row_j] + t[row_j]
        if k >= 2
            r[j] = r[j] + ((num_blks-1)*a - prev_A) * (x[j]-prev_xj)
            s[row_j] = s[row_j] + ((num_blks-1)*a - prev_A) * t[row_j]
        end
        prev_a, prev_A = a, A
        a, A = later_a, later_A
        later_a = min(num_blks/(num_blks-1) * later_a, sqrt(later_A)/(2*num_blks))
        later_A = later_A + later_a
        if k % (freq * num_blks) == 0
            x_tilde = x + 1/prev_A * r
            C_x_tilde = C * x_tilde
            metric = first_order_opt(C, b, x_tilde, C_x_tilde, C_b, col_norm_square)
            td = time() - t0
            if metric <= restart_ratio * init_metric || td + init_time > total_time || metric < ϵ || k == K
                return x_tilde, C_x_tilde, metric, k + init_epoch, td + init_time
            end
        end
    end
end

function first_order_opt(C::SparseMatrixCSC{T, Int}, b::Vector{T}, x::Vector{T}, 
                         C_x::Vector{T}, C_b::Vector{T}, col_norm_square::Vector{T}) where T <: AbstractFloat
    tmp = (C_x' * C)[:]
    val = norm(x - max.(x - (tmp - C_b) ./ col_norm_square, zero(T)))
    return val
end

