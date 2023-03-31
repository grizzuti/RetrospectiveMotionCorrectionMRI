using LinearAlgebra, RetrospectiveMotionCorrectionMRI, AbstractProximableFunctions, FastSolversForWeightedTV, UtilitiesForMRI, PyPlot, JLD

# Folders
experiment_name = "17-06-2022-SENSE"; @info string(experiment_name, "\n")
exp_folder = string(pwd(), "/data/", experiment_name, "/")
data_folder = string(exp_folder, "data/")
~isdir(data_folder) && mkdir(data_folder)
figures_folder = string(exp_folder, "figures/")
~isdir(figures_folder) && mkdir(figures_folder)
results_folder = string(exp_folder, "results/")
~isdir(results_folder) && mkdir(results_folder)

# Loading data
data_file = "data.jld"
X = load(string(data_folder, data_file))["X"]
K = load(string(data_folder, data_file))["K"]
data = load(string(data_folder, data_file))["data"]
prior = load(string(data_folder, data_file))["prior"]
ground_truth = load(string(data_folder, data_file))["ground_truth"]
corrupted = load(string(data_folder, data_file))["corrupted"]
vmin = 0f0; vmax = maximum(abs.(ground_truth))
orientation = load(string(data_folder, data_file))["orientation"]

# Setting Fourier operator
F = nfft_linop(X, K)
nt, nk = size(K)

# Multi-scale inversion schedule
n_scales = 3
niter_imrecon = ones(Integer, n_scales)
niter_parest  = ones(Integer, n_scales)
niter_outloop = 100*ones(Integer, n_scales); niter_outloop[end] = 10;
# ε_schedule = range(0.1f0, 0.8f0; length=4)
# ε_schedule = [0.01f0, 0.1f0, 0.2f0]
# ε_schedule = [0.01f0, 0.1f0, 0.5f0]
ε_schedule = [0.01f0, 0.1f0, 0.5f0, 0.8f0]
niter_registration = 10
nt, _ = size(K)

# Setting starting values
u = deepcopy(corrupted)
θ = zeros(Float32, nt, 6)

# Loop over scales
damping_factor = nothing
for (i, scale) in enumerate(n_scales-1:-1:0)
    global u

    # Down-scaling the problem (spatially)...
    n_h = div.(X.nsamples, 2^scale)
    X_h = resample(X, n_h)
    K_h = subsample(K, X_h; radial=false)
    F_h = nfft_linop(X_h, K_h)
    nt_h, _ = size(K_h)
    data_h = subsample(K, data, K_h; norm_constant=F.norm_constant/F_h.norm_constant, damping_factor=damping_factor)
    prior_h = resample(prior, n_h; damping_factor=damping_factor)
    ground_truth_h = resample(ground_truth, n_h; damping_factor=damping_factor)
    u = resample(u, n_h)

    # Down-scaling the problem (temporally)...
    nt_h = 50
    t_coarse = Float32.(range(1, nt; length=nt_h))
    t_fine = Float32.(1:nt)
    # interp = :spline
    interp = :linear
    Ip_c2f = interpolation1d_motionpars_linop(t_coarse, t_fine; interp=interp)
    Ip_f2c = interpolation1d_motionpars_linop(t_fine, t_coarse; interp=interp)
    t_fine_h = K_h.subindex_phase_encoding

    ### Optimization options: ###

    ## Parameter estimation
    scaling_diagonal = 1f-3
    scaling_mean     = 1f-4
    scaling_id       = 0f0
    Ip_c2fh = interpolation1d_motionpars_linop(t_coarse, Float32.(t_fine_h); interp=interp)
    D = derivative1d_motionpars_linop(t_coarse, 1; pars=(true, true, true, true, true, true))
    λ = 1f-3*sqrt(norm(data_h)^2/spectral_radius(D'*D; niter=3))
    opt_parest = parameter_estimation_options(; niter=niter_parest[i], steplength=1f0, λ=λ, scaling_diagonal=scaling_diagonal, scaling_mean=scaling_mean, scaling_id=scaling_id, reg_matrix=D, interp_matrix=Ip_c2fh)

    ## Image reconstruction
    h = spacing(X_h)
    η = 1f-2*structural_maximum(prior_h; h=h)
    P = structural_weight(prior_h; h=h, η=η, γ=1f0)
    # P = nothing
    opt_inner = FISTA_optimizer(4f0*sum(1 ./h.^2); Nesterov=true, niter=10)
    g = gradient_norm(2, 1, n_h, h, opt_inner; weight=P, complex=true)
    opt_imrecon(ε) = image_reconstruction_options(; prox=indicator(g ≤ ε), Lipschitz_constant=1f0, Nesterov=true, niter=niter_imrecon[i])

    ## Global
    opt(ε) = motion_correction_options(; image_reconstruction_options=opt_imrecon(ε), parameter_estimation_options=opt_parest, niter=niter_outloop[i], niter_estimate_Lipschitz=3)

    ### End optimization options ###

    # Loop over smoothing factor
    for (j, ε_rel) in enumerate(ε_schedule)

        # Selecting motion parameters on low-dimensional space
        θ_coarse = reshape(Ip_f2c*vec(θ), :, 6)
        # θ_coarse = reshape(Float32.(Ip_c2f\vec(θ)), :, 6)

        # Joint reconstruction
        @info string("@@@ Scale = ", scale, ", regularization = ", ε_rel)
        θ_h = reshape(Ip_c2fh*vec(θ_coarse), :, 6)
        corrupted_h = F_h(θ_h)'*data_h
        # ε = ε_rel*g(corrupted_h)
        ε = ε_rel*g(ground_truth_h)
        u, θ_coarse = motion_corrected_reconstruction(F_h, data_h, u, θ_coarse, opt(ε))

        # Up-scaling motion parameters
        θ .= reshape(Ip_c2f*vec(θ_coarse), :, 6)
        n_avg = div(size(θ, 1), 10)
        θ[1:t_fine_h[1]+n_avg-1,:] .= sum(θ[t_fine_h[1]:t_fine_h[1]+n_avg-1,:]; dims=1)/n_avg
        θ[t_fine_h[end]-n_avg+1:end,:] .= sum(θ[t_fine_h[end]-n_avg+1:t_fine_h[end],:]; dims=1)/n_avg

        # Plot
        plot_volume_slices(abs.(u); spatial_geometry=X_h, vmin=vmin, vmax=vmax, savefile=string(figures_folder, "temp.png"), orientation=orientation)
        close("all")
        plot_parameters(1:nt, θ, nothing; xlabel="t = phase encoding", vmin=[-10, -10, -10, -10, -10, -10], vmax=[10, 10, 10, 10, 10, 10], fmt1="b", linewidth1=2, savefile=string(figures_folder, "temp_motion_pars.png"))
        close("all")

    end

    # Up-scaling reconstruction
    u = resample(u, X.nsamples)

end

# Denoising ground-truth/corrupted volumes
@info "@@@ Post-processing figures"
opt_reg = rigid_registration_options(; T=Float32, niter=10, verbose=false)
u_reg, _ = rigid_registration(u, ground_truth, nothing, opt_reg; spatial_geometry=X, nscales=3)
corrupted_reg, _ = rigid_registration(corrupted, ground_truth, nothing, opt_reg; spatial_geometry=X, nscales=3)

# Reconstruction quality
psnr_recon = psnr(abs.(u_reg), abs.(ground_truth))
psnr_conv = psnr(abs.(corrupted_reg), abs.(ground_truth))
ssim_recon = ssim(abs.(u_reg), abs.(ground_truth))
ssim_conv = ssim(abs.(corrupted_reg), abs.(ground_truth))
@info string("@@@ Conventional reconstruction: psnr = ", psnr_conv)
@info string("@@@ Conventional reconstruction: ssim = ", ssim_conv)
@info string("@@@ Joint reconstruction: psnr = ", psnr_recon)
@info string("@@@ Joint reconstruction: ssim = ", ssim_recon)

# Save and plot results
@save string(results_folder, "results.jld") u u_reg θ psnr_recon psnr_conv ssim_recon ssim_conv
plot_volume_slices(abs.(u_reg); spatial_geometry=X, vmin=vmin, vmax=vmax, savefile=string(figures_folder, "joint_sTV.png"), orientation=orientation)
plot_volume_slices(abs.(corrupted_reg); spatial_geometry=X, vmin=vmin, vmax=vmax, savefile=string(figures_folder, "corrupted_reg.png"), orientation=orientation)
plot_parameters(1:size(θ,1), θ, nothing; xlabel="t = phase encoding", vmin=[-10, -10, -10, -10, -10, -10], vmax=[10, 10, 10, 10, 10, 10], fmt1="b", linewidth1=2, savefile=string(figures_folder, "motion_pars.png"))
close("all")