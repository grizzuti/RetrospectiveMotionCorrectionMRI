using LinearAlgebra, RetrospectiveMotionCorrectionMRI, AbstractProximableFunctions, UtilitiesForMRI,  FastSolversForWeightedTV, PyPlot, JLD

# Folders & files
experiment_name = "prior_smoothness_study"; @info experiment_name
exp_folder = string(pwd(), "/data/", experiment_name, "/")
figures_folder = string(exp_folder, "figures/")
~isdir(figures_folder) && mkdir(figures_folder)
data_file = "data.jld"
unprocessed_scans_file = "unprocessed_scans.jld"

# Loading unprocessed data
prior, mask_prior, prior_reg, ground_truth, mask_ground_truth, corrupted, fov, permutation_dims, idx_phase_encoding, idx_readout = load(string(exp_folder, unprocessed_scans_file), "prior", "mask_prior", "prior_reg", "ground_truth", "mask_ground_truth", "corrupted", "fov", "permutation_dims", "idx_phase_encoding", "idx_readout")

# Denoise prior
X = spatial_geometry(fov, size(prior)); h = spacing(X)
opt = FISTA_options(4f0*sum(1f0./h.^2); Nesterov=true, niter=20)
g = gradient_norm(2, 1, size(prior), h; complex=true, options=opt)
prior = proj(prior, 0.8f0*g(prior), g)
z = (sin.(range(-Float32(pi)/2, Float32(pi)/2; length=size(prior,3)-130)).+1)./2
prior[:,:,end-length(z)+1:end] .*= reshape(z[end:-1:1], 1, 1, :)
z_reg = (sin.(range(-Float32(pi)/2, Float32(pi)/2; length=size(prior,3)-180)).+1)./2
prior_reg[:,:,end-length(z_reg)+1:end] .*= reshape(z_reg[end:-1:1], 1, 1, :)

# Generating synthetic data
K = kspace_sampling(X, permutation_dims[1:2]; phase_encode_sampling=idx_phase_encoding, readout_sampling=idx_readout)
F = nfft_linop(X, K)
data = F*corrupted

# Rigid registration
opt_reg = rigid_registration_options(; niter=20, verbose=true)
corrupted_reg, _ = rigid_registration(corrupted, ground_truth, nothing, opt_reg; spatial_geometry=X, nscales=4)
C = zero_set(ComplexF32, (!).(mask_ground_truth))
corrupted_reg = proj(corrupted_reg, C)

# Plotting
orientation = Orientation((2,1,3), (true,false,true))
nx, ny, nz = size(ground_truth)[[invperm(orientation.perm)...]]
slices = (VolumeSlice(1, div(nx,2)+1, nothing),
          VolumeSlice(2, div(ny,2)+1, nothing),
          VolumeSlice(3, div(nz,2)+1, nothing))
plot_volume_slices(abs.(ground_truth); spatial_geometry=X, vmin=0, vmax=norm(ground_truth, Inf), savefile=string(figures_folder, "ground_truth.png"), slices=slices, orientation=orientation)
plot_volume_slices(abs.(prior); spatial_geometry=X, vmin=0, vmax=norm(prior, Inf), savefile=string(figures_folder, "prior.png"), slices=slices, orientation=orientation)
plot_volume_slices(abs.(prior_reg); spatial_geometry=X, vmin=0, vmax=norm(prior, Inf), savefile=string(figures_folder, "prior_reg.png"), slices=slices, orientation=orientation)
plot_volume_slices(abs.(corrupted); spatial_geometry=X, vmin=0, vmax=norm(ground_truth, Inf), savefile=string(figures_folder, "corrupted.png"), slices=slices, orientation=orientation)
plot_volume_slices(abs.(corrupted_reg); spatial_geometry=X, vmin=0, vmax=norm(ground_truth, Inf), savefile=string(figures_folder, "corrupted_reg.png"), slices=slices, orientation=orientation)
close("all")

# Saving data
save(string(exp_folder, data_file), "data", data, "X", X, "K", K, "ground_truth", ground_truth, "prior", prior, "prior_reg", prior_reg, "corrupted", corrupted, "corrupted_reg", corrupted_reg, "orientation", orientation, "mask_prior", mask_prior, "mask_ground_truth", mask_ground_truth)