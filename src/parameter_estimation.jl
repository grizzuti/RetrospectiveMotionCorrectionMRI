# Parameter estimation utilities

export OptionsParameterEstimation, parameter_estimation_options, parameter_estimation


## Parameter-estimation options

mutable struct HessianRegularizationParameters
    scaling_diagonal::Real
    scaling_mean::Real
    scaling_id::Real
end

mutable struct OptionsParameterEstimation
    niter::Integer
    steplength::Real
    λ::Real
    reg_matrix::Union{Nothing,AbstractMatrix}
    interp_matrix::Union{Nothing,AbstractMatrix}
    reg_Hessian::HessianRegularizationParameters
    verbose::Bool
    fun_history::Union{Nothing,AbstractVector}
end

function parameter_estimation_options(; niter::Integer=10,
                                        steplength::Real=1f0,
                                        λ::Real=0f0,
                                        scaling_diagonal::Real=0f0, scaling_mean::Real=0f0, scaling_id::Real=0f0,
                                        reg_matrix::Union{Nothing,AbstractMatrix{<:Real}}=nothing,
                                        interp_matrix::Union{Nothing,AbstractMatrix{<:Real}}=nothing,
                                        verbose::Bool=false,
                                        fun_history::Bool=false)
    fun_history ? (fval = Array{typeof(steplength),1}(undef,niter)) : (fval = nothing)
    return OptionsParameterEstimation(niter, steplength, λ, isnothing(reg_matrix) ? nothing : reg_matrix, isnothing(interp_matrix) ? nothing : interp_matrix, HessianRegularizationParameters(scaling_diagonal, scaling_mean, scaling_id), verbose, fval)
end

AbstractProximableFunctions.fun_history(options::OptionsParameterEstimation) = options.fun_history


## Parameter-estimation algorithms

function parameter_estimation(F::StructuredNFFTtype2LinOp{T}, u::AbstractArray{CT,3}, d::AbstractArray{CT,2}, initial_estimate::AbstractArray{T}; options::Union{Nothing,OptionsParameterEstimation}=nothing) where {T<:Real,CT<:RealOrComplex{T}}

    # Initialize variables
    θ = deepcopy(initial_estimate)
    interp_flag = ~isnothing(options.interp_matrix); interp_flag && (Ip = options.interp_matrix)
    reg_flag    = ~isnothing(options.reg_matrix) && (options.λ != T(0)); reg_flag && (D = options.reg_matrix)

    # Iterative solution
    @inbounds for n = 1:options.niter

        # Evaluate forward operator
        interp_flag ? (Iθ = reshape(Ip*vec(θ), :, 6)) : (Iθ = θ)
        Fθu, _, Jθ = ∂(F()*u, Iθ)

        # Data misfit
        r = Fθu-d
        (~isnothing(options.fun_history) || options.verbose) && (fval_n = T(0.5)*norm(r)^2)

        # Regularization term
        if reg_flag
            Dθ = reshape(D*vec(θ), :, 6)
            (~isnothing(options.fun_history) || options.verbose) && (fval_n += T(0.5)*options.λ^2*norm(Dθ)^2)
        end

        # Print message
        ~isnothing(options.fun_history) && (options.fun_history[n] = fval_n)
        options.verbose && (@info string("Iter [", n, "/", options.niter, "], fval = ", fval_n))

        # Compute gradient
        g = Jθ'*r
        interp_flag && (g = reshape(Ip'*vec(g), :, 6))
        reg_flag && (g .+= options.λ^2*reshape(D'*vec(Dθ), :, 6))

        # Hessian
        H = sparse_matrix_GaussNewton(Jθ)
        interp_flag && (H = Ip'*H*Ip)
        reg_flag && (H .+= options.λ^2*(D'*D))

        # Marquardt-Levenberg regularization
        if ~isnothing(options.reg_Hessian)
            H, ΔH = regularize_Hessian!(H; regularization_options=options.reg_Hessian)
            g .+= reshape(ΔH*vec(θ), :, 6) # consistency correction
        end

        # Preconditioning
        g = reshape(lu(H)\vec(g), :, 6)

        # Update
        θ .-= T(options.steplength)*g

    end

    return θ

end

function regularize_Hessian!(H::AbstractMatrix{T}; regularization_options::Union{Nothing,HessianRegularizationParameters}=nothing) where {T<:Real}
    isnothing(regularization_options) && (return (H, nothing))
    diagH = reshape(diag(H), :, 6)
    mean_diag = sum(diagH; dims=1)/size(diagH, 1)
    ΔH = spdiagm(vec(T(regularization_options.scaling_diagonal)*diagH.+T(regularization_options.scaling_mean)*mean_diag.+T(regularization_options.scaling_id)))
    H .+= ΔH
    return H, ΔH
end