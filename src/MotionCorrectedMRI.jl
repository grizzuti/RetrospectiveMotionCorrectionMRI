module MotionCorrectedMRI

using LinearAlgebra, AbstractLinearOperators, FastSolversForWeightedTV, UtilitiesForMRI, Flux, ImageQualityIndexes, SparseArrays, Statistics
import Flux.Optimise: update!

const RealOrComplex{T<:Real} = Union{T,Complex{T}}

include("./abstract_types.jl")
include("./data_residual_loss.jl")
include("./calibration_utils.jl")
include("./optimization_utils.jl")
include("./imagequality_utils.jl")
include("./image_reconstruction.jl")
# include("./parameter_estimation.jl")
# include("./motion_corrected_reconstruction.jl")

end
