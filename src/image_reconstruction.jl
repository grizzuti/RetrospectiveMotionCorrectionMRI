# Image reconstruction utilities

export image_reconstruction_options, image_reconstruction


struct ImageReconstructionOptionsFISTA<:AbstractImageReconstructionOptions
    prox::AbstractProximableFunction
    options::ArgminFISTA
    niter_estimate_Lipschitz::Union{Nothing,Integer}
end

"""
    image_reconstruction_options(; prox::AbstractProximableFunction,
                                   Lipschitz_constant=nothing,
                                   niter_estimate_Lipschitz=nothing
                                   Nesterov=true,
                                   reset_counter=nothing,
                                   niter=nothing,
                                   verbose=false,
                                   fun_history=false)

Returns image reconstruction options for the routine [`image_reconstruction`](@ref):
- `prox`: regularization function (for which a proximal operator is implemented)
- `Lipschitz_constant`: Lipschitz constant of a smooth objective
- `niter_estimate_Lipschitz`: when set to an `Integer`, the iterative power method is invoked in order to estimate the Lipschitz constant with the specified number of iteration
- `Nesterov`: allows Nesterov acceleration (default)
- `reset_counter`: number of iterations after which the Nesterov acceleration is reset
- `niter`: number of iterations
- `verbose`, `fun_history`: for debugging purposes

Note: for more details on each of these parameters, consult [this section](@ref imrecon).
"""
function image_reconstruction_options(; prox::AbstractProximableFunction,
                                        Lipschitz_constant::Union{Nothing,Real}=nothing,
                                        niter_estimate_Lipschitz::Union{Nothing,Integer}=nothing,
                                        Nesterov::Bool=true,
                                        reset_counter::Union{Nothing,Integer}=nothing,
                                        niter::Union{Nothing,Integer}=nothing,
                                        verbose::Bool=false,
                                        fun_history::Bool=false)
    return ImageReconstructionOptionsFISTA(prox, FISTA_options(Lipschitz_constant; Nesterov=Nesterov, reset_counter=reset_counter, niter=niter, verbose=verbose, fun_history=fun_history), niter_estimate_Lipschitz)
end

AbstractProximableFunctions.fun_history(options::ImageReconstructionOptionsFISTA) = fun_history(options.options)

AbstractProximableFunctions.set_Lipschitz_constant(options::ImageReconstructionOptionsFISTA, L::Real) = ImageReconstructionOptionsFISTA(options.prox, set_Lipschitz_constant(options.options, L), options.niter_estimate_Lipschitz)

"""
    image_reconstruction(F, d, initial_estimate, options)

Performs image reconstruction by fitting the data `d` through a linear operator `F` (e.g. the Fourier transform). `initial_estimate` sets the initial guess. The minimization `options` are passed via the routine [`image_reconstruction_options`](@ref).

See this [section](@ref imrecon) for more details on the solution algorithm.
"""
function image_reconstruction(F::AbstractLinearOperator{CT,3,2}, d::AbstractArray{CT,2}, initial_estimate::AbstractArray{CT,3}, options::ImageReconstructionOptionsFISTA) where {T<:Real,CT<:RealOrComplex{T}}
    if ~isnothing(options.niter_estimate_Lipschitz)
        L = T(1.1)*spectral_radius(F*F'; niter=options.niter_estimate_Lipschitz)
        opt_FISTA = set_Lipschitz_constant(options.options, L)
    end
    return leastsquares_solve(F, d, options.prox, initial_estimate, opt_FISTA)
end