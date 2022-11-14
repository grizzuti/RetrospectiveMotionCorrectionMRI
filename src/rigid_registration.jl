# Rigid registration utilities

export rigid_registration_options, rigid_registration


rigid_registration_options(; niter::Integer=10, verbose::Bool=false, fun_history::Bool=false) = parameter_estimation_options(; niter=niter, steplength=1.0, λ=0.0,
scaling_diagonal=0.0, scaling_mean=0.0, verbose=verbose, fun_history=fun_history)

function rigid_registration(u_moving::AbstractArray{CT,3}, u_fixed::AbstractArray{CT,3}, θ::Union{Nothing,AbstractArray{T}}; options::Union{Nothing,OptionsParameterEstimation}=nothing, spatial_geometry::Union{Nothing,CartesianSpatialGeometry{T}}=nothing, nscales::Integer=1) where {T<:Real,CT<:RealOrComplex{T}}

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
        θ = parameter_estimation(F_h, u_moving_h, F_h*u_fixed_h, θ; options=options)
        (scale == 0) && (return (F_h'*(F_h(θ)*u_moving_h), θ))

    end

end