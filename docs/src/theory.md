# [Motion correction for MRI](@id theory)

In this section, we briefly summarize the methodology that underpins reference-guided motion correction for MRI. The method is introduced and discussed in detail in [[2]](@ref references).


## [Motion-perturbed Fourier transform](@id Fourier)

### The non-uniform Fourier transform

For a discretized (complex-valued) image ``\mathbf{u}`` on some computational grid ``\{\mathbf{x}_j\}_j``, that is
```math
\mathbf{u}(\mathbf{x})=\sum_{j=1}^n\mathbf{u}_j\delta(\mathbf{x}-\mathbf{x}_j),
```
the (non-uniform) Fourier transform is defined as:
```math
F\mathbf{u}\,(\mathbf{k})=\sum_j\mathbf{u}_j\exp{(2\pi\mathrm{i}\,\mathbf{k}\cdot\mathbf{x}_j)},\qquad\forall\,\mathbf{k}\in K
```
for some ``k``-space subset ``K``. For general ``k``-space trajectories, this can be evaluated by non-uniform Fourier transform libraries such as [`FINUFFT.jl`](https://github.com/ludvigak/FINUFFT.jl) or [`NFFT.jl`](https://github.com/JuliaMath/NFFT.jl).

### Global rigid motion under the Fourier transform

Let's now assume that an image undergoes a rigid motion ``T(\pmb{\theta})`` parameterized by ``\pmb{\theta}\in\mathbb{R}^d`` (``d=3`` in 2D, and ``d=6`` in 3D):
```math
T(\pmb{\theta})(\mathbf{x})=R(\pmb{\varphi})\mathbf{x}+\pmb{\tau},\qquad\mathrm{where\ }\pmb{\theta}=(\pmb{\varphi},\pmb{\tau}).
```
The parameter ``\pmb{\theta}=(\pmb{\varphi},\pmb{\tau})`` represents translation distances and rotation angles, respectively. The operator ``R(\pmb{\varphi})`` performs the coordinate rotations associated to the angles ``\pmb{\varphi}``.

The transformed image is ``\mathbf{u}_{T(\pmb{\theta})}=\mathbf{u}\circ T(\pmb{\theta})^{-1}``, and the *perturbed* Fourier transform of ``\mathbf{u}`` can be defined as ``F(\pmb{\theta})\mathbf{u}:=F\mathbf{u}_{T(\pmb{\theta})}``. Thanks to elementary Fourier identities, this gives:
```math
F(\pmb{\theta})\mathbf{u}\,(\mathbf{k})=\exp{(-2\pi\mathrm{i}\,\mathbf{k}\cdot\pmb{\tau})}F\mathbf{u}\,(R(\pmb{\varphi})^{\mathrm{T}}\mathbf{k}).
```

### Time-dependent rigid motion under the Fourier transform

In MRI, the Fourier transform is evaluated in ``k``-space according to a certain temporal sequence of wavenumbers ``K=\{\mathbf{k}_t\}_{t=1}^{n_t}``. For motion correction purposes, we must then postulate (in principle) as many rigid motion parameters ``\pmb{\theta}_t`` as "time" steps.

With a slight abuse of notation, we will now denote with ``\pmb{\theta}`` the actual motion evolution ``\{\pmb{\theta}_t\}_{t=1}^{n_t}``, and consider as perturbed Fourier transform the following:
```math
F(\pmb{\theta})\mathbf{u}:=(F(\pmb{\theta}_1)\mathbf{u}(\mathbf{k}_1),\ldots,F(\pmb{\theta}_{n_t})\mathbf{u}(\mathbf{k}_{n_t})).
```
Note that, in practice, we can assume that ``\pmb{\theta}_t`` remains constant for time indexes associated to wavenumbers ``\mathbf{k}_t`` that belongs to the same readout. In general, we can also assume that the rigid motion parameters vary smoothly in time.


## [Alternating motion correction and reconstruction](@id retromoco)

A notable problem in MRI is motion corruption. We can assume that some data ``\mathbf{d}`` has been acquired in ``k``-space while the patient was moving inside the scanner. The conventional Fourier inverse transform of ``\mathbf{d}`` will produce a corrupted image ``\mathbf{u}``, and won't be suitable for radiological assessment.

Retrospective rigid motion correction tries to estimate the motion parameter ``\pmb{\theta}`` together with a motion-clean image ``\mathbf{u}`` such that the motion artifacts are no longer present. Mathematically, it is equivalent to the bi-level minimization problem:
```math
\min_{\mathbf{u},\pmb{\theta}}J(\mathbf{u},\pmb{\theta})=\dfrac{1}{2}||F(\pmb{\theta})\mathbf{u}-\mathbf{d}||^2+g_u(\mathbf{u})+g_{\theta}(\pmb{\theta}),
```
for some regularization terms ``g_u`` and ``g_{\theta}``.

### Solution method

The optimization problem is solved by alternating update of the unknowns ``\mathbf{u}`` and ``\pmb{\theta}``. The two subproblems are associated to:
- [image reconstruction](@ref imrecon): ``\min_{\mathbf{u}}J(\mathbf{u},\pmb{\theta})``
- [motion parameter estimation](@ref parest): ``\min_{\pmb{\theta}}J(\mathbf{u},\pmb{\theta})``.
More details are expounded in the respective sections.

We offer the convenience function [`motion_correction_options`](@ref) to set the overall options of this alternating scheme. The suboptions [`image_reconstruction_options`](@ref) and [`parameter_estimation_options`](@ref) for each specific step are set by dedicated option routines described in their respective section. The alternating solver is called with the function [`motion_corrected_reconstruction`](@ref).


## [Image reconstruction](@id imrecon)

In this section, we describe the optimization problem related to image reconstruction, given a *known* rigid motion parameter.

The minimization problem is:
```math
\min_{\mathbf{u}}J(\mathbf{u})=\dfrac{1}{2}||F(\pmb{\theta})\mathbf{u}-\mathbf{d}||^2+g(\mathbf{u}).
```
The linear operator ``F(\pmb{\theta})`` is the motion-perturbed Fourier transform (described [here](@ref Fourier)), ``\pmb{\theta}`` a fixed rigid motion state, and ``\mathbf{d}`` some given data.

The regularization term ``g`` is typically a variant of total variation, and is handled through the package `FastSolversForWeightedTV`. For more details on how to set this regularization term consult the documentation in `FastSolversForWeightedTV` and the convenience function [`image_reconstruction_options`](@ref).

### Solution method

In order to solve the non-smooth convex optimization problem, we focus on the FISTA method (see [3](@ref references)). For a generic optimization problem of the form ``\min_{\mathbf{x}}f(\mathbf{x})+g(\mathbf{x})``, it results in the iterative scheme:
```math
\begin{aligned}
\tilde{\mathbf{x}}_{k+1}&=\mathrm{prox}_{1/L,g}(\mathbf{x}_k-\nabla_{\mathbf{x}_k}f/L),\\
t_{k+1}&=\dfrac{1+\sqrt{1+4t_k^2}}{2},\\
\mathbf{x}_{k+1}&=\tilde{\mathbf{x}}_{k+1}+\dfrac{t_k-1}{t_{k+1}}(\tilde{\mathbf{x}}_{k+1}-\tilde{\mathbf{x}}_k).
\end{aligned}
```
Here, ``\mathrm{prox}`` represents the proximal operator, ``L`` is the Lipschitz constant of ``\nabla f``, and ``t_k`` are the momentum factors associated to Nesterov acceleration (``t_0=1``). Despite the theoretical analysis, it is reported in the literature that it may be beneficial to reset the Nesterov momentum to ``1`` after a certain number of iterations to avoid convergence stagnation.

A special note must be dedicated to the choice of ``L``. For image reconstruction ``f`` corresponds to the function ``J``. It can be easily seen that the Lipschitz constant for ``\nabla J`` is the spectral radius of ``F(\pmb{\theta})``. Due to the fact that the unknowns ``\pmb{\theta}`` will be updated frequently, this constant must be computed every time such an update occurs and cannot be optimally set for any choice of ``\pmb{\theta}``. The option routine [`image_reconstruction_options`](@ref) additionally offers the option to estimate the Lipschitz constant via the power method with the keyword `niter_estimate_Lipschitz`, which must be set to a certain integer value equal to the number of iterations the power method is run.


## [Rigid motion parameter estimation](@id parest)

In this section, we lay out a detailed formulation of the optimization problem related to motion parameter estimation, given a *known* image. This is, essentially, a generalized rigid registration routine.

The mathematical problem considered is:
```math
\min_{\pmb{\theta}}J(\pmb{\theta})=\dfrac{1}{2}||F(\pmb{\theta})\mathbf{u}-\mathbf{d}||^2+\dfrac{\lambda^2}{2}||D\pmb{\theta}||^2.
```
The linear operator ``F(\pmb{\theta})`` is the motion-perturbed Fourier transform (described [here](@ref Fourier)), ``\mathbf{u}`` a fixed image, and ``\mathbf{d}`` some given data.

The regularization term is typically aimed at enforcing temporal smoothness. Its strength is regulated by the weight ``\lambda`` and regularization operator ``D`` (e.g. a time-derivative operator).

Temporal smoothness regularization can be integrated or replaced by hard constraints. It is convenient, for example, to define an time-wise interpolation operator ``I`` that transforms coarse-grid arrays ``\tilde{\pmb{\theta}}`` to fine-grid arrays ``\tilde{\pmb{\theta}}``. In this case, we solve the problem
```math
\min_{\tilde{\pmb{\theta}}}J(\tilde{\pmb{\theta}})=\dfrac{1}{2}||F(I\tilde{\pmb{\theta}})\mathbf{u}-\mathbf{d}||^2+\dfrac{\lambda^2}{2}||D\tilde{\pmb{\theta}}||^2.
```

### Solution method

We implemented a rather conventional pseudo-Hessian method for the optimization problem:
```math
\pmb{\theta}\leftarrow\pmb{\theta}-\mathrm{steplength}*B^{-1}\nabla_{\pmb{\theta}}J.
```
All the parameters described in this section can be set via the routine [`parameter_estimation_options`](@ref). Typically ``\mathrm{steplength}=1``.

The pseudo-Hessian ``B`` is based on the Gauss-Newton approximation of the analytical Hessian ``\nabla_{\pmb{\theta}}^2J``. Note, however that ``\nabla_{\pmb{\theta}}^2J`` is ill-conditioned for the wavenumber ``\mathbf{k}=0`` and high-frequency wavenumbers. Therefore, we use the Levenberg-Marquardt regularization:
```math
\tilde{B}=B+\mathrm{scaling}_{\mathrm{diag}}*\mathrm{diag}(B)+\mathrm{scaling}_{\mathrm{mean}}*\mathrm{mean}(\mathrm{diag}(B))\otimes\mathrm{Id}+\mathrm{scaling}_{\mathrm{id}}*\mathrm{Id}.
```
Here, the ``\mathrm{mean}`` is performed for the diagonal elements of each rigid-motion parameter.
Typically, ``\mathrm{scaling}_{\mathrm{id}}=0``, ``\mathrm{scaling}_{\mathrm{mean}}=\mathrm{1e-1}*\mathrm{scaling}_{\mathrm{diag}}`` and ``\mathrm{scaling}_{\mathrm{diag}}`` is a relatively small number.