# diffable_transforms_for_autograd
Implements common transforms that map constrained parameter spaces to unconstrained real values. Useful for gradient-based optimization.

All functions are differentiable using the autograd Python package.


# Included transforms

* positive reals
* unit interval
* unit simplex
* * using log transform
* * using centered stick-breaking transform

# TODO
* cholesky

