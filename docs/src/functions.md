# Main functions

## Motion correction

```@docs
motion_corrected_reconstruction(F::RetrospectiveMotionCorrectionMRI.StructuredNFFTtype2LinOp{T}, d::AbstractArray{CT,2}, u::AbstractArray{CT,3}, θ::AbstractArray{T,2}, options::RetrospectiveMotionCorrectionMRI.MotionCorrectionOptionsAlternatingFISTADiff) where {T<:Real,CT<:RealOrComplex{T}}
```

```@docs
motion_correction_options(; image_reconstruction_options::ImageReconstructionOptionsFISTA, parameter_estimation_options::ParameterEstimationOptionsDiff, niter::Integer, niter_estimate_Lipschitz::Union{Nothing,Integer}=nothing, verbose::Bool=false, fun_history::Bool=false)
```

## Image reconstruction

```@docs
image_reconstruction(F::AbstractLinearOperator{CT,3,2}, d::AbstractArray{CT,2}, initial_estimate::AbstractArray{CT,3}, options::RetrospectiveMotionCorrectionMRI.ImageReconstructionOptionsFISTA) where {T<:Real,CT<:RealOrComplex{T}}
```

```@docs
image_reconstruction_options(; prox::AbstractProximableFunction,
                               Lipschitz_constant::Union{Nothing,Real}=nothing,
                               Nesterov::Bool=true,
                               reset_counter::Union{Nothing,Integer}=nothing,
                               niter::Union{Nothing,Integer}=nothing,
                               verbose::Bool=false,
                               fun_history::Bool=false)
```

## Rigid motion parameter estimation

```@docs
parameter_estimation(F::RetrospectiveMotionCorrectionMRI.StructuredNFFTtype2LinOp{T}, u::AbstractArray{CT,3}, d::AbstractArray{CT,2}, initial_estimate::AbstractArray{T}, options::RetrospectiveMotionCorrectionMRI.ParameterEstimationOptionsDiff) where {T<:Real,CT<:RealOrComplex{T}}
```

```@docs
parameter_estimation_options(; niter::Integer=10,
                                        steplength::Real=1f0,
                                        λ::Real=0f0,
                                        scaling_diagonal::Real=0f0, scaling_mean::Real=0f0, scaling_id::Real=0f0,
                                        reg_matrix::Union{Nothing,AbstractMatrix{<:Real}}=nothing,
                                        interp_matrix::Union{Nothing,AbstractMatrix{<:Real}}=nothing,
                                        calibration::Bool=false,
                                        verbose::Bool=false,
                                        fun_history::Bool=false)
```

## Rigid registration utilities

```@docs
rigid_registration(u_moving::AbstractArray{CT,3}, u_fixed::AbstractArray{CT,3}, θ::Union{Nothing,AbstractArray{T}}, options::RetrospectiveMotionCorrectionMRI.ParameterEstimationOptionsDiff; spatial_geometry::Union{Nothing,CartesianSpatialGeometry{T}}=nothing, nscales::Integer=1) where {T<:Real,CT<:RealOrComplex{T}}
```

```@docs
rigid_registration_options(; niter::Integer=10, verbose::Bool=false, fun_history::Bool=false)
```