using LinearAlgebra, RetrospectiveMotionCorrectionMRI, AbstractProximableFunctions, FastSolversForWeightedTV, UtilitiesForMRI, PyPlot, JLD

# Load data
results_folder = string(pwd(), "/paper/robustness/vol2/")
results_file = string(results_folder, "data_vol2_priorT1.jld")
corrupted1, corrupted2, corrupted3, ground_truth, X, prior, mask = load(results_file, "corrupted_motion1", "corrupted_motion2", "corrupted_motion3", "ground_truth", "X", "prior", "mask")
results_file = string(results_folder, "results_52782_motion1_priorT1.jld")
u1, θ1 = load(results_file, "u", "θ")
results_file = string(results_folder, "results_52782_motion2_priorT1.jld")
u2, θ2 = load(results_file, "u", "θ")
results_file = string(results_folder, "results_52782_motion3_priorT1.jld")
u3, θ3 = load(results_file, "u", "θ")
orientation = Orientation((2,1,3), (true,false,true))

# # Registration
# opt_reg = rigid_registration_options(; niter=20, verbose=true)
# u_reg, _ = rigid_registration(u, ground_truth, nothing, opt_reg; spatial_geometry=X, nscales=5)
# corrupted_reg, _ = rigid_registration(corrupted, ground_truth, nothing, opt_reg; spatial_geometry=X, nscales=5)

# # Denoising
# h = spacing(X)
# opt = FISTA_options(4f0*sum(1f0./h.^2); Nesterov=true, niter=20)
# g = gradient_norm(2, 1, size(prior), h; complex=true, options=opt)
# ε = g(u)
# corrupted_reg = proj(corrupted_reg, ε, g)
# u_reg = proj(u_reg, ε, g)
# ground_truth_reg = proj(ground_truth, ε, g)

# Masking
C = zero_set(ComplexF32, (!).(mask))
u1 = proj(u1, C)
u2 = proj(u2, C)
u3 = proj(u3, C)
corrupted1 = proj(corrupted1, C)
corrupted2 = proj(corrupted2, C)
corrupted3 = proj(corrupted3, C)
ground_truth = proj(ground_truth, C)
@save string(results_folder, "post_processed_scans_vol2_robustness.jld") u1 u2 u3 corrupted1 corrupted2 corrupted3 ground_truth

# Reconstruction quality
fact = norm(ground_truth, Inf)
psnr_corrected1 = psnr(abs.(u1)/fact, abs.(ground_truth)/fact; orientation=orientation)
psnr_corrupted1 = psnr(abs.(corrupted1)/fact, abs.(ground_truth)/fact; orientation=orientation)
ssim_corrected1 = ssim(abs.(u1)/fact, abs.(ground_truth)/fact; orientation=orientation)
ssim_corrupted1 = ssim(abs.(corrupted1)/fact, abs.(ground_truth)/fact; orientation=orientation)
psnr_corrected2 = psnr(abs.(u2)/fact, abs.(ground_truth)/fact; orientation=orientation)
psnr_corrupted2 = psnr(abs.(corrupted2)/fact, abs.(ground_truth)/fact; orientation=orientation)
ssim_corrected2 = ssim(abs.(u2)/fact, abs.(ground_truth)/fact; orientation=orientation)
ssim_corrupted2 = ssim(abs.(corrupted2)/fact, abs.(ground_truth)/fact; orientation=orientation)
psnr_corrected3 = psnr(abs.(u3)/fact, abs.(ground_truth)/fact; orientation=orientation)
psnr_corrupted3 = psnr(abs.(corrupted3)/fact, abs.(ground_truth)/fact; orientation=orientation)
ssim_corrected3 = ssim(abs.(u3)/fact, abs.(ground_truth)/fact; orientation=orientation)
ssim_corrupted3 = ssim(abs.(corrupted3)/fact, abs.(ground_truth)/fact; orientation=orientation)
@save string(results_folder, "indexes_vol2_robustness.jld") psnr_corrected1 psnr_corrupted1 ssim_corrected1 ssim_corrupted1 psnr_corrected2 psnr_corrupted2 ssim_corrected2 ssim_corrupted2 psnr_corrected3 psnr_corrupted3 ssim_corrected3 ssim_corrupted3

# Plot results
plot_volume_slices(abs.(ground_truth); spatial_geometry=X, vmin=0f0, vmax=norm(ground_truth,Inf), savefile=string(results_folder, "ground_truth_vol2_robustness.eps"), orientation=orientation)
plot_volume_slices(abs.(prior); spatial_geometry=X, vmin=0f0, vmax=norm(prior,Inf), savefile=string(results_folder, "reference_vol2_robustness.eps"), orientation=orientation)

plot_volume_slices(abs.(u1); spatial_geometry=X, vmin=0f0, vmax=norm(ground_truth,Inf), savefile=string(results_folder, "corrected1_vol2_robustness.eps"), orientation=orientation)
plot_volume_slices(abs.(corrupted1); spatial_geometry=X, vmin=0f0, vmax=norm(ground_truth,Inf), savefile=string(results_folder, "corrupted1_vol2_robustness.eps"), orientation=orientation)

plot_volume_slices(abs.(u2); spatial_geometry=X, vmin=0f0, vmax=norm(ground_truth,Inf), savefile=string(results_folder, "corrected2_vol2_robustness.eps"), orientation=orientation)
plot_volume_slices(abs.(corrupted2); spatial_geometry=X, vmin=0f0, vmax=norm(ground_truth,Inf), savefile=string(results_folder, "corrupted2_vol2_robustness.eps"), orientation=orientation)

plot_volume_slices(abs.(u3); spatial_geometry=X, vmin=0f0, vmax=norm(ground_truth,Inf), savefile=string(results_folder, "corrected3_vol2_robustness.eps"), orientation=orientation)
plot_volume_slices(abs.(corrupted3); spatial_geometry=X, vmin=0f0, vmax=norm(ground_truth,Inf), savefile=string(results_folder, "corrupted3_vol2_robustness.eps"), orientation=orientation)

# Plot results (detail)
h = spacing(X)
c = div.(size(prior), 2).+(-41, 31, 11)
l = 50
detail = (c[1]-l:c[1]+l, c[2]-l:c[2]+l, c[3]-l:c[3]+l)
ground_truth_detail = ground_truth[detail...]
corrupted1_detail = corrupted1[detail...]
corrupted2_detail = corrupted2[detail...]
corrupted3_detail = corrupted3[detail...]
u1_detail = u1[detail...]
u2_detail = u2[detail...]
u3_detail = u3[detail...]
prior_detail = prior[detail...]
X_detail = spatial_geometry(spacing(X).*size(prior_detail), size(prior_detail))
plot_volume_slices(abs.(ground_truth_detail); spatial_geometry=X_detail, vmin=0f0, vmax=norm(ground_truth,Inf), savefile=string(results_folder, "ground_truth_vol2_robustness_detail.eps"), orientation=orientation)
plot_volume_slices(abs.(prior_detail); spatial_geometry=X_detail, vmin=0f0, vmax=norm(prior,Inf), savefile=string(results_folder, "reference_vol2_robustness_detail.eps"), orientation=orientation)

plot_volume_slices(abs.(u1_detail); spatial_geometry=X_detail, vmin=0f0, vmax=norm(ground_truth,Inf), savefile=string(results_folder, "corrected1_vol2_robustness_detail.eps"), orientation=orientation)
plot_volume_slices(abs.(corrupted1_detail); spatial_geometry=X_detail, vmin=0f0, vmax=norm(ground_truth,Inf), savefile=string(results_folder, "corrupted1_vol2_robustness_detail.eps"), orientation=orientation)

plot_volume_slices(abs.(u2_detail); spatial_geometry=X_detail, vmin=0f0, vmax=norm(ground_truth,Inf), savefile=string(results_folder, "corrected2_vol2_robustness_detail.eps"), orientation=orientation)
plot_volume_slices(abs.(corrupted2_detail); spatial_geometry=X_detail, vmin=0f0, vmax=norm(ground_truth,Inf), savefile=string(results_folder, "corrupted2_vol2_robustness_detail.eps"), orientation=orientation)

plot_volume_slices(abs.(u3_detail); spatial_geometry=X_detail, vmin=0f0, vmax=norm(ground_truth,Inf), savefile=string(results_folder, "corrected3_vol2_robustness_detail.eps"), orientation=orientation)
plot_volume_slices(abs.(corrupted3_detail); spatial_geometry=X_detail, vmin=0f0, vmax=norm(ground_truth,Inf), savefile=string(results_folder, "corrupted3_vol2_robustness_detail.eps"), orientation=orientation)

close("all")

# Plot motion parameters
θ1_min =  minimum(θ1; dims=1); θ1_max = maximum(θ1; dims=1); Δθ1 = θ1_max-θ1_min; θ1_middle = (θ1_min+θ1_max)/2
Δθ1 = [ones(1,3)*max(Δθ1[1:3]...)/2 ones(1,3)*max(Δθ1[4:end]...)/2]
plot_parameters(1:size(θ1,1), θ1, nothing; xlabel="t = phase encoding index", vmin=vec(θ1_middle-1.1f0*Δθ1), vmax=vec(θ1_middle+1.1f0*Δθ1), fmt1="b", linewidth1=2, savefile=string(results_folder, "parameters_motion1_vol2_robustness.eps"))

θ2_min =  minimum(θ2; dims=1); θ2_max = maximum(θ2; dims=1); Δθ2 = θ2_max-θ2_min; θ2_middle = (θ2_min+θ2_max)/2
Δθ2 = [ones(1,3)*max(Δθ2[1:3]...)/2 ones(1,3)*max(Δθ2[4:end]...)/2]
plot_parameters(1:size(θ2,1), θ2, nothing; xlabel="t = phase encoding index", vmin=vec(θ2_middle-1.1f0*Δθ2), vmax=vec(θ2_middle+1.1f0*Δθ2), fmt1="b", linewidth1=2, savefile=string(results_folder, "parameters_motion2_vol2_robustness.eps"))

θ3_min =  minimum(θ3; dims=1); θ3_max = maximum(θ3; dims=1); Δθ3 = θ3_max-θ3_min; θ3_middle = (θ3_min+θ3_max)/2
Δθ3 = [ones(1,3)*max(Δθ3[1:3]...)/2 ones(1,3)*max(Δθ3[4:end]...)/2]
plot_parameters(1:size(θ3,1), θ3, nothing; xlabel="t = phase encoding index", vmin=vec(θ3_middle-1.1f0*Δθ3), vmax=vec(θ3_middle+1.1f0*Δθ3), fmt1="b", linewidth1=2, savefile=string(results_folder, "parameters_motion3_vol2_robustness.eps"))
close("all")