"""NumPy sampler: draw the weights and biases of a fully connected network from a Crinit result."""
import math

import numpy as np

from .core import Crinit


def _orthogonal(rng, n):
    q, r = np.linalg.qr(rng.standard_normal((n, n)))
    return q * np.sign(np.diag(r))


def sample(r: Crinit, d_in, d_out, rng=None, first_layer="steady"):
    """Weights W_l (fan_in x fan_out) and biases b_l for `r.depth` hidden layers of width r.width and a
    LeCun-initialised output layer.  first_layer: 'steady' (default; the first hidden layer starts at q*),
    'standard' (like the hidden layers) or a number q1.  Returns (weights, biases) as lists; the forward pass is
    h = phi(h @ W + b)."""
    rng = np.random.default_rng(rng)
    N, L = r.width, r.depth
    Ws, bs = [], []
    for i in range(L):
        fan_in = d_in if i == 0 else N
        if i == 0 and first_layer not in ("standard", False):
            W = rng.standard_normal((fan_in, N)) * r.weight_std(fan_in, first=first_layer)
        elif r.ensemble == "orthogonal" and fan_in == N:
            W = _orthogonal(rng, N) * math.sqrt(r.sigma_w2)
        else:
            W = rng.standard_normal((fan_in, N)) * r.weight_std(fan_in)
        Ws.append(W)
        bs.append(rng.standard_normal(N) * r.bias_std() if r.sigma_b2 > 0 else np.zeros(N))
    Ws.append(rng.standard_normal((N, d_out)) / math.sqrt(N))
    bs.append(np.zeros(d_out))
    return Ws, bs
