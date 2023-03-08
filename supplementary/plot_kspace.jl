using UtilitiesForMRI, PyPlot

# Cartesian domain
n = (128,128,128)
h = [1f0, 1f0, 1f0]
X = spatial_sampling(n; h=h)

# Cartesian sampling in k-space
readout = :z
phase_encode = :xy
K = kspace_sampling(X; readout=readout, phase_encode=phase_encode)

# Plot
k = reshape(K.K, 128, 128, 128, 3)
figure()
idx = [5, 25, 45, 65, 85, 105, 125]
for i = idx, j = idx
    plot3D(k[i,j,:,1],  k[i,j,:,2],  k[i,j,:,3], ".")
end
xlabel(L"\kappa_x")
ylabel(L"\kappa_y")
zlabel(L"\kappa_z\ \mathrm{(readout)}")
savefig("./kspace3D.png", dpi=300, transparent=false, bbox_inches="tight")

# Perturb
φ = randn(Float32, nt, 3)
nt = size(K)[1]
# for i = 1:3
#     φ[:, i] .*= 2*pi*30f0/(180f0*norm(φ[:, i], Inf))
# end
φ[:, 1] .= 5f0*pi/180f0
φ[:, 2] .= 0
φ[:, 3] .= 0
Rφ = rotation()(φ)
Kφ = reshape(Rφ*K.K, 128, 128, 128, 3)

# Plot
figure()
for i = idx, j = idx
    plot3D(Kφ[i,j,:,1],  Kφ[i,j,:,2],  Kφ[i,j,:,3], ".")
end
xlabel(L"\kappa_x")
ylabel(L"\kappa_y")
zlabel(L"\kappa_z\ \mathrm{(readout)}")
savefig("./kspace3D_perturbed.png", dpi=300, transparent=false, bbox_inches="tight")