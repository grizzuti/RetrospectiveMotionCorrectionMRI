# Motion-corrected reconstruction utilities

export motion_correction_options, motion_corrected_reconstruction


## Motion-corrected reconstruction options

struct MotionCorrectionOptionsAlternatingFISTADiff<:AbstractMotionCorrectionOptions
    options_imrecon::ImageReconstructionOptionsFISTA
    options_parest::ParameterEstimationOptionsDiff
    niter::Integer
    verbose::Bool
    fun_history::Union{Nothing,AbstractArray}
end

"""
    motion_correction_options(; image_reconstruction_options,
                                parameter_estimation_options,
                                niter,
                                verbose, fun_history)

Return motion correction options for alternating rigid motion estimation and image reconstruction:
- [`image_reconstruction_options`](@ref): set via the corresponding option routine
- [`parameter_estimation_options`](@ref): set via the corresponding option routine
- `niter`: number of outer two-step loops
- `verbose`, `fun_history`: for debugging purposes

Note: for more details consult [this section](@ref theory).
"""
motion_correction_options(; image_reconstruction_options::ImageReconstructionOptionsFISTA, parameter_estimation_options::ParameterEstimationOptionsDiff, niter::Integer, verbose::Bool=false, fun_history::Bool=false) = MotionCorrectionOptionsAlternatingFISTADiff(image_reconstruction_options, parameter_estimation_options, niter, verbose, fun_history ? Vector{NTuple{2,Any}}(undef,niter) : nothing)

AbstractProximableFunctions.fun_history(options::MotionCorrectionOptionsAlternatingFISTADiff) = options.fun_history


## Motion-corrected reconstruction algorithms

"""
    motion_corrected_reconstruction(F, d, u, θ, options)

Performs retrospective motion correction of some data `d`. `F` is the Fourier operator, initialized via the package `UtilitiesForMRI`. The initial estimates for image and motion parameters are `u` and `θ`. The minimization `options` are passed via the routine [`motion_correction_options`](@ref).

See this [section](@ref retromoco) for more details on the solution algorithm.
"""
function motion_corrected_reconstruction(F::StructuredNFFTtype2LinOp{T}, d::AbstractArray{CT,2}, u::AbstractArray{CT,3}, θ::AbstractArray{T,2}, options::MotionCorrectionOptionsAlternatingFISTADiff) where {T<:Real,CT<:RealOrComplex{T}}

    flag_interp = ~isnothing(options.options_parest.interp_matrix)
    flag_interp && (Ip = options.options_parest.interp_matrix)

    # Iterative solution
    for n = 1:options.niter

        # Print message
        options.verbose && (@info string("*** Outer loop: Iter {", n, "/", options.niter, "}"))

        # Image reconstruction
        options.verbose && (@info string("--- Image reconstruction..."))
        flag_interp ? (θ_ = reshape(Ip*vec(θ), :, 6)) : (θ_ = θ); Fθ = F(θ_)
        u = image_reconstruction(Fθ, d, u, options.options_imrecon)

        # Motion-parameter estimation
        (n == options.niter) && break
        options.verbose && (@info string("--- Motion-parameter estimation..."))
        θ = parameter_estimation(F, u, d, θ, options.options_parest)

        # Saving history
        ~isnothing(options.fun_history) && (options.fun_history[n] = (deepcopy(fun_history(options.options_imrecon)), deepcopy(fun_history(options.options_parest))))

    end

    return u, θ

end