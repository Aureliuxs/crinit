"""
All orders in 1/N.

The closed-form correction of `crinit` is the first order in 1/N.  Two routes give all orders.

Homogeneous activations (linear, ReLU, leaky ReLU).  The signal size of one input is multiplied at every layer by
sigma_w^2 X/N, with X = chi^2_K + alpha^2 chi^2_{N-K} and K ~ Bin(N, 1/2) the number of active neurons,
independently from layer to layer.  The typical signal is preserved exactly when ln sigma_w^2 = ln N - E ln X, and
by Frullani's identity ln x = int_0^inf (e^{-t} - e^{-tx}) dt/t,

    E ln X = int_0^inf [ e^{-t} - ( (1+2t)^{-1/2}/2 + (1+2 alpha^2 t)^{-1/2}/2 )^N ] dt/t .

For alpha = 1 (linear) this is psi(N/2) + ln 2, the digamma formula; for alpha = 0 (ReLU) the all-dead event
K = 0 is excluded.

Smooth activations.  The drift of ln d^2 for two nearby inputs is measured in the exact finite-width chain
(Gaussian weights: given layer l, the pre-activations of layer l+1 are exactly Gaussian with the empirical
variances of layer l), and its zero is found by a few secant steps with common random numbers.
"""
import math

import numpy as np
from scipy.integrate import quad

from .activations import get
from .finite_width import a1_b1, sigma_c2_relu_exact


# ----------------------------------------------------------------------------------------- homogeneous ----------
def sigma_c2_homogeneous_exact(N, alpha):
    """Typical-preserving sigma_w^2 of a linear (alpha = 1), ReLU (alpha = 0) or leaky ReLU network, all orders."""
    if alpha == 0.0:
        return sigma_c2_relu_exact(N)
    a2 = alpha * alpha

    def integrand(t):
        m = 0.5 * (1 + 2 * t) ** -0.5 + 0.5 * (1 + 2 * a2 * t) ** -0.5
        return (math.exp(-t) - m ** N) / t if t > 0 else 0.0

    e_ln_x = quad(integrand, 0, 1, limit=200)[0] + quad(integrand, 1, np.inf, limit=200)[0]
    return math.exp(math.log(N) - e_ln_x)


# ----------------------------------------------------------------------------------------- smooth --------------
def chain_drift(act, sw2, sb2, N, B=None, L=1200, burn=None, seed=0, q0=None):
    """Drift (mean of ln D_{l+1}/D_l) of the distance between two nearby inputs in the exact finite-width chain
    with Gaussian weights, and its standard error from block sums.  Returns (mu, se)."""
    act = get(act)
    if B is None:
        B = max(64, 51200 // N)
    if q0 is None:
        q0 = max(sb2, 0.0) + sw2 * act.V(1.0) * 0.5
    if burn is None:
        burn = 300
    rng = np.random.default_rng(seed)
    z = math.sqrt(q0) * rng.standard_normal((B, N))
    dl = rng.standard_normal((B, N))
    d_prev = np.mean(dl * dl, 1)
    steps = np.empty((L, B))
    for ell in range(burn + L):
        y, dy = act.phi(z), act.dphi(z) * dl
        Q = sw2 * np.mean(y * y, 1) + sb2
        D = sw2 * np.mean(dy * dy, 1)
        K = sw2 * np.mean(y * dy, 1)
        g1 = rng.standard_normal((B, N))
        g2 = rng.standard_normal((B, N))
        z = np.sqrt(Q)[:, None] * g1
        dl = (K / Q)[:, None] * z + np.sqrt(np.maximum(D - K * K / Q, 0.0))[:, None] * g2
        if ell >= burn:
            steps[ell - burn] = np.log(D / d_prev)
        s = np.sqrt(np.mean(dl * dl, 1))
        dl = dl / s[:, None]
        d_prev = D / (s * s)
    block = 100
    nb = L // block
    sums = steps[: nb * block].reshape(nb, block, B).sum(1)
    return float(steps.mean()), float(sums.std() / math.sqrt(sums.size) / block)


def exact_point(act, q, N, sw2_first, sb2_first, zero_bias=False, seed=0, B=None, L=1200, iters=3):
    """Zero of the chain drift starting from the first-order point.  Along the path of fixed steady signal q
    (sigma_b^2 = q - sigma_w^2 V(q)), or at zero bias if `zero_bias`.  Returns (sigma_w^2, sigma_b^2, se) where se is
    the statistical uncertainty of sigma_w^2."""
    act = get(act)
    Vq = act.V(q)
    bias = (lambda s: 0.0) if zero_bias else (lambda s: q - s * Vq)
    Fp = abs(a1_b1(act, q)["Fp"])
    burn = int(max(300, 8 / max(1 - Fp, 1e-3)))

    def mu(s):
        return chain_drift(act, s, max(bias(s), 0.0), N, B=B, L=L, burn=burn, seed=seed, q0=q)

    s0 = sw2_first
    h = max(0.5 * abs(sw2_first * (1 - math.exp(a1_b1(act, q)["a1"] / N))), 2e-3)
    m0, se0 = mu(s0)
    s1 = s0 + (h if m0 < 0 else -h)
    m1, _ = mu(s1)
    for _ in range(iters - 1):
        if m1 == m0:
            break
        slope = (m1 - m0) / (s1 - s0)
        s2 = s1 - m1 / slope
        s0, m0 = s1, m1
        s1 = s2
        m1, se0 = mu(s1)
    slope = (m1 - m0) / (s1 - s0) if s1 != s0 else 1.0
    s = s1 - m1 / slope
    return s, max(bias(s), 0.0), abs(se0 / slope)
