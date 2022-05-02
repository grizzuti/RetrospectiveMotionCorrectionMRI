using LinearAlgebra, BlindMotionCorrectionMRI, Test, AbstractLinearOperators

# Calibration operator
d1 = randn(ComplexF64, 10, 5)
d2 = randn(ComplexF64, 10, 5)
λ = 0.1
loss = data_residual_loss(ComplexF64, 2, 2)
C = calibration(loss, λ; type=:readout)
A = C(d1, d2)

# Random input
d = randn(ComplexF64, 10, 5)
e = randn(ComplexF64, 10, 5)

# Adjoint test
@test dot(A*d, e) ≈ dot(d, A'*e) rtol=1f-6

# Gradient test
function funeval(C, d̄::AbstractArray{CT,2}, d::AbstractArray{CT,2}) where {T<:Real,CT<:Complex{T}}
    A = C(d̄, d)
    r = A*d̄-d
    normd = sqrt.(sum(abs.(d).^2; dims=2))
    f = T(0.5)*norm(r)^2+T(0.5)*C.λ^2*norm(normd.*(A.α.-T(1)))^2
    g = A'*r
    return f, g
end
d̄0 = randn(ComplexF64, 10, 5)
p = randn(ComplexF64, 10, 5); p .*= norm(d̄0)/norm(p)
d = randn(ComplexF64, 10, 5)
_, g0 = funeval(C, d̄0, d)
t = 1e-6
d̄p1 = d̄0+t/2*p
fp1, _ = funeval(C, d̄p1, d)
d̄m1 = d̄0-t/2*p
fm1, _ = funeval(C, d̄m1, d)
@test (fp1-fm1)/t ≈ real(dot(g0, p)) rtol=1e-5