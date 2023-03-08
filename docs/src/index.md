# [Introduction](@id intro)

This package implements the algorithmic tools to perform retrospective motion correction for MRI based on a reference-guided TV regularization (see [[2]](@ref references)). It combines several custom packages, the most important building blocks are:
- `FastSolversForWeightedTV`: for the computation of proximal/projection operators 
- `UtilitiesForMRI`: for the computation and differentiation of the non-uniform Fourier transform with respect to rigid-body motion perturbation.

## [Related publications](@id references)

1. Ehrhardt, M. J., and Betcke, M. M., (2015). Multi-Contrast MRI Reconstruction with Structure-Guided Total Variation (https://arxiv.org/abs/1511.06631), _SIAM J. IMAGING SCIENCES_, **9(3)**, 1084-1106, doi:[10.1137/15M1047325](https://doi.org/10.1137/15M1047325)
2. Rizzuti, G., Sbrizzi, A., and van Leeuwen, T., (2022). Joint Retrospective Motion Correction and Reconstruction for Brain MRI With a Reference Contrast, _IEEE Transaction on Computational Imaging_, **8**, 490-504, doi:[10.1109/TCI.2022.3183383](hhtps://doi.org/10.1109/TCI.2022.3183383)
3. Beck, A., and Teboulle, M., (2009). A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems, _SIAM Journal on Imaging Sciences_, **2(1)**, 183-202, doi:[10.1137/080716542](https://doi.org/10.1137/080716542)