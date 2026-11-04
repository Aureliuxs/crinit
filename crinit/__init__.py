"""
crinit: critical initialisation of deep networks at finite width.

    import crinit
    r = crinit.crinit("tanh", depth=32, width=64)      # sigma_w^2, sigma_b^2 and q* of the finite-width edge of chaos
    print(r)

With PyTorch:

    import crinit.torch_init as ct
    ct.crinit_(model, "tanh")                         # initialise every nn.Linear of a feed-forward model in place
    model, r = ct.mlp(784, 64, 32, 10, "leaky_relu")  # a ready-made network with its shaped activation

The theory and the measurements behind the rule are in A. Caesar, "Finite-width shift of
the edge of chaos in deep neural networks" (2026).
"""
from .activations import Activation, custom, get, leaky_relu
from .core import Crinit, crinit
from .finite_width import a1_b1, shift, shift_at_bias, sigma_c2_leaky, sigma_c2_relu_exact, sigma_c2_smooth
from .meanfield import (chi1, compression, eoc_at_bias, eoc_point, fixed_point, kappa, operating_point,
                        shaped_leaky_alpha, slope)
from .sample import sample

__version__ = "0.3.1"
__all__ = ["crinit", "Crinit", "Activation", "custom", "get", "leaky_relu", "a1_b1", "shift", "shift_at_bias",
           "sigma_c2_leaky", "sigma_c2_relu_exact", "sigma_c2_smooth", "chi1", "compression", "eoc_at_bias",
           "eoc_point", "fixed_point", "kappa", "operating_point", "shaped_leaky_alpha", "slope", "sample"]
