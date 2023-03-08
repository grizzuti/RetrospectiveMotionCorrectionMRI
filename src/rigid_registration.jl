# Rigid registration utilities

export rigid_registration_options, rigid_registration

"""
    rigid_registration_options(; niter=10, verbose=false, fun_history=false)

Returns options for the rigid registration routine. `niter` sets the number of iterations, `verbose=true` can be used for debugging purposes, and `fun_history=true` allows the storage of the objective values at each iteration.
"""
rigid_registration_options(; niter::Integer=10, verbose::Bool=false, fun_history::Bool=false) = parameter_estimation_options(; niter=niter, steplength=1.0, λ=0.0,
scaling_diagonal=0.0, scaling_mean=0.0, verbose=verbose, fun_history=fun_history)

"""
    rigid_registration(u_moving, u_fixed, θ, options;
                       spatial_geometry=nothing,
                       nscales=1)

Performs the rigid registration of the 3D image `u_moving` with the fixed 3D image `u_fixed`. The user can input a rigid motion parameter `θ` (`nothing` in case no prior knowledge is available), and `options` via [`rigid_registration_options`](@ref). In case the 3D image are associated to a specific spatial discretization, it can be specified with the keyword argument `spatial_geometry`. Multiscale acceleration can be specified with by setting the number of levels with the keyword `nscales` (in this case, each scaled subproblem solver runs `options.niter` iterations).
"""
function rigid_registration(u_moving::AbstractArray{CT,3}, u_fixed::AbstractArray{CT,3}, θ::Union{Nothing,AbstractArray{T}}, options::ParameterEstimationOptionsDiff; spatial_geometry::Union{Nothing,CartesianSpatialGeometry{T}}=nothing, nscales::Integer=1) where {T<:Real,CT<:RealOrComplex{T}}

    # Initialize variables
    isnothing(spatial_geometry) ? (X = UtilitiesForMRI.spatial_geometry((T(1),T(1),T(1)), size(u_moving))) : (X = spatial_geometry)
    isnothing(θ) && (θ = zeros(T, 1, 6))
    n = X.nsamples

    # Rigid registration (multi-scale)
    for scale = nscales-1:-1:0

        # Down-scaling the problem...
        options.verbose && (@info string("@@@ Scale = ", scale))
        n_h = div.(n, 2^scale)
        X_h = resample(X, n_h)
        kx_h, ky_h, kz_h = k_coord(X_h; mesh=true)
        K_h = cat(reshape(kx_h, 1, :, 1), reshape(ky_h, 1, :, 1), reshape(kz_h, 1, :, 1); dims=3)
        F_h = nfft_linop(X_h, K_h)
        u_fixed_h  = resample(u_fixed, n_h)
        u_moving_h = resample(u_moving, n_h)

        # Parameter estimation
        θ = parameter_estimation(F_h, u_moving_h, F_h*u_fixed_h, θ, options)
        (scale == 0) && (return (F_h'*(F_h(θ)*u_moving_h), θ))

    end

end