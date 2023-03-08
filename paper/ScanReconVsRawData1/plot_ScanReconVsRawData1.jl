using LinearAlgebra, RetrospectiveMotionCorrectionMRI, AbstractProximableFunctions, FastSolversForWeightedTV, UtilitiesForMRI, PyPlot, JLD

# Load data
results_folder = string(pwd(), "/paper/ScanReconVsRawData1/")
results_file = string(results_folder, "ScanReconVsRawData1.jld")
corrupted, ground_truth, fov, u_reg, u, θ, prior = load(results_file, "u_conventional", "ground_truth", "fov", "u_reg", "u", "θ", "prior")
orientation = Orientation((1,2,3), (false,false,false))
X = spatial_geometry(Float32.(fov), size(prior))

# Denoising ground-truth/corrupted volumes
# @info "@@@ Post-processing figures"
# C = zero_set(ComplexF32, (!).(mask))
# corrupted_reg = proj(corrupted_reg, C)
# u_reg = proj(u_reg, C)
opt_reg = rigid_registration_options(; niter=10, verbose=true)
u_reg, _ = rigid_registration(u, ground_truth, nothing, opt_reg; spatial_geometry=X, nscales=5)
corrupted_reg, _ = rigid_registration(corrupted, ground_truth, nothing, opt_reg; spatial_geometry=X, nscales=5)
@save string(results_folder, "registered_scans.jld") u_reg corrupted_reg

# Reconstruction quality
nx, ny, nz = size(ground_truth)[[invperm(orientation.perm)...]]
slices = (VolumeSlice(1, div(nx,2)+1, nothing),
          VolumeSlice(2, div(ny,2)+1, nothing),
          VolumeSlice(3, div(nz,2)+1, nothing))
fact = norm(ground_truth, Inf)
psnr_corrected = psnr(abs.(u_reg)/fact, abs.(ground_truth)/fact; slices=slices, orientation=orientation)
psnr_corrupted = psnr(abs.(corrupted_reg)/fact, abs.(ground_truth)/fact; slices=slices, orientation=orientation)
ssim_corrected = ssim(abs.(u_reg)/fact, abs.(ground_truth)/fact; slices=slices, orientation=orientation)
ssim_corrupted = ssim(abs.(corrupted_reg)/fact, abs.(ground_truth)/fact; slices=slices, orientation=orientation)
@save string(results_folder, "indexes.jld") psnr_corrected psnr_corrupted ssim_corrected ssim_corrupted

# Plot results
plot_volume_slices(abs.(ground_truth); slices=slices, spatial_geometry=X, vmin=0f0, vmax=norm(ground_truth,Inf), savefile=string(results_folder, "ground_truth.eps"), orientation=orientation)
plot_volume_slices(abs.(prior); slices=slices, spatial_geometry=X, vmin=0f0, vmax=norm(prior,Inf), savefile=string(results_folder, "reference.eps"), orientation=orientation)
plot_volume_slices(abs.(u_reg); slices=slices, spatial_geometry=X, vmin=0f0, vmax=norm(ground_truth,Inf), savefile=string(results_folder, "corrected.eps"), orientation=orientation)
plot_volume_slices(abs.(corrupted_reg); slices=slices, spatial_geometry=X, vmin=0f0, vmax=norm(ground_truth,Inf), savefile=string(results_folder, "corrupted.eps"), orientation=orientation)

# Plot results (detail)
c = (div(nx,2)+21, div(ny,2)+55)
l = 50
slices = (VolumeSlice(3, div(nz,2)+1, (c[1]-l:c[1]+l, c[2]-l:c[2]+l)), )

plot_volume_slices(abs.(ground_truth); slices=slices, spatial_geometry=X, vmin=0f0, vmax=norm(ground_truth,Inf), savefile=string(results_folder, "ground_truth_detail.eps"), orientation=orientation)
plot_volume_slices(abs.(prior); slices=slices, spatial_geometry=X, vmin=0f0, vmax=norm(prior,Inf), savefile=string(results_folder, "reference_detail.eps"), orientation=orientation)
plot_volume_slices(abs.(u_reg); slices=slices, spatial_geometry=X, vmin=0f0, vmax=norm(ground_truth,Inf), savefile=string(results_folder, "corrected_detail.eps"), orientation=orientation)
plot_volume_slices(abs.(corrupted_reg); slices=slices, spatial_geometry=X, vmin=0f0, vmax=norm(ground_truth,Inf), savefile=string(results_folder, "corrupted_detail.eps"), orientation=orientation)

# Plot motion parameters
θ[:, 1:3] .*= 1e3
θ_min =  minimum(θ; dims=1); θ_max = maximum(θ; dims=1); Δθ = θ_max-θ_min; θ_middle = (θ_min+θ_max)/2
Δθ = [ones(1,3)*max(Δθ[1:3]...)/2 ones(1,3)*max(Δθ[4:end]...)/2]
# Δθ[1:3] .= 1
plot_parameters(1:size(θ,1), θ, nothing; xlabel="t = phase encoding index", vmin=vec(θ_middle-1.1f0*Δθ), vmax=vec(θ_middle+1.1f0*Δθ), fmt1="b", linewidth1=2, savefile=string(results_folder, "parameters_motion.eps"))
close("all")