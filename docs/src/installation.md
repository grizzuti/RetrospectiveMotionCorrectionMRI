# [Installation instructions](@id install)

In the Julia REPL, simply type `]` and
```julia
(@v1.8) pkg> add https://github.com/grizzuti/AbstractLinearOperators.git
(@v1.8) pkg> add https://github.com/grizzuti/AbstractProximableFunctions.git
(@v1.8) pkg> add https://github.com/grizzuti/FastSolversForWeightedTV.git
(@v1.8) pkg> add https://github.com/grizzuti/UtilitiesForMRI.git
(@v1.8) pkg> add https://github.com/grizzuti/RetrospectiveMotionCorrectionMRI.git
```
The packages `AbstractLinearOperators`, `AbstractProximableFunctions`, `FastSolversForWeightedTV`, and `UtilitiesForMRI` have to be explicitly installed since they are unregistered at the moment.