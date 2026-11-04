"""
Finite-width corrections.

Distance between two close inputs.  At width N the logarithm of the distance performs a random walk with
drift mu_d = ln chi_1 + a_1/N.  The finite-width edge of chaos is mu_d = 0, which moves the weight variance
by Delta sigma_w^2 = -a_1 / (N d ln chi_1 / d sigma_w^2) at fixed bias variance.  a_1 is an explicit
Gaussian average (a1_b1), valid when N (1 - F') >> 1.

Signal size of one input.  For homogeneous activations (ReLU, leaky ReLU, linear) a common rescaling of all
weights does not change the geometry, and the only finite-width effect is that the typical signal shrinks as
exp(-C/(2N)) per layer.  The typical-preserving variance is
    sigma_c^2(N) = (N/2) exp(-psi(N/2))                      (linear; smooth phi with phi'(0) = 1 at q* -> 0)
    sigma_c^2(N) = 2/(1+alpha^2) exp(C_alpha/(2N)),  C_alpha = (5 - 2 alpha^2 + 5 alpha^4)/(1 + alpha^2)^2
                                                             (leaky ReLU, leading order)
    sigma_c^2(N) = (N/2) exp(-E_K psi(K/2)),  K ~ Bin(N, 1/2), K >= 1      (ReLU, exact)
"""
import math

import numpy as np
from scipy.special import comb, digamma

from .activations import get
from .meanfield import eoc_at_bias


def a1_b1(act, q, sw2=None, orthogonal=False):
    """Drift coefficient a_1 and long-run variance coefficient b_1 of ln D at steady signal q.

    sw2 = None puts the point on the mean-field edge of chaos.  orthogonal=True uses the orthogonal-ensemble
    variances (the parts of the per-neuron fluctuations fixed by norm preservation are removed)."""
    act = get(act)
    m = act.moments(q)
    if sw2 is None:
        sw2 = 1.0 / m["P"]
    P, dP, d2P, V, dV = m["P"], m["dP"], m["d2P"], m["V"], m["dV"]
    Fp, Fpp = sw2 * dV, sw2 * m["d2V"]
    chi = sw2 * P
    var_w = 3 * m["P4"] / P ** 2 - 1
    var_Q = sw2 ** 2 * (m["V4"] - V ** 2)
    var_r = sw2 * m["EphiDphi2"] / (q * P)
    cov_wQ = sw2 * (m["EphiDphi2"] - P * V) / P
    if orthogonal:
        var_w -= (2 * P ** 2 + 2 * q ** 2 * dP ** 2) / P ** 2
        var_Q -= sw2 ** 2 * 2 * q ** 2 * dV ** 2
        var_r -= sw2 * q * dV ** 2 / (q * P)
        cov_wQ -= sw2 * 2 * q ** 2 * dP * dV / P
    NEx2 = var_Q / (1 - Fp ** 2)
    NEx = 0.5 * Fpp * NEx2 / (1 - Fp)
    NEr2 = var_r / (1 - Fp ** 2 / chi)
    a1 = -0.5 * var_w + 2 * q * (dP / P) * NEr2 + (dP / P) * NEx + 0.5 * (d2P / P - (dP / P) ** 2) * NEx2
    b1 = var_w + (dP / P) ** 2 * var_Q / (1 - Fp) ** 2 + 2 * (dP / P) * cov_wQ / (1 - Fp)
    return dict(q=q, sw2=sw2, chi1=chi, Fp=Fp, a1=a1, b1=b1)


def dlnchi_dsw2(act, q, sw2):
    """d ln chi_1 / d sigma_w^2 at fixed sigma_b^2 (the steady signal moves with sigma_w^2)."""
    act = get(act)
    m = act.moments(q)
    Fp = sw2 * m["dV"]
    return 1 / sw2 + (m["dP"] / m["P"]) * m["V"] / (1 - Fp)


def shift(act, q, N, sw2=None, orthogonal=False):
    """Delta sigma_w^2 that moves the edge-of-chaos point with steady signal q (bias fixed) to the
    finite-width edge of chaos at width N.  Returns (Delta sigma_w^2, a_1, validity) where validity =
    N (1 - F') must be large compared with 1."""
    act = get(act)
    if sw2 is None:
        sw2 = 1.0 / act.P(q)
    r = a1_b1(act, q, sw2, orthogonal=orthogonal)
    d = -r["a1"] / (N * dlnchi_dsw2(act, q, sw2))
    return d, r["a1"], N * (1 - r["Fp"])


def shift_at_bias(act, sb2, N, orthogonal=False):
    """Finite-width edge of chaos at bias variance sb2: (sigma_w^2 mean field, q*, Delta sigma_w^2)."""
    sw2, q = eoc_at_bias(act, sb2)
    d, _, _ = shift(act, q, N, sw2, orthogonal)
    return sw2, q, d


def sigma_c2_smooth(N, a2=1.0):
    return (N / 2) * math.exp(-digamma(N / 2)) / a2


def leaky_C(alpha):
    return (5 - 2 * alpha ** 2 + 5 * alpha ** 4) / (1 + alpha ** 2) ** 2


def sigma_c2_leaky(N, alpha=0.0):
    return 2 / (1 + alpha ** 2) * math.exp(leaky_C(alpha) / (2 * N))


def sigma_c2_relu_exact(N):
    K = np.arange(1, N + 1)
    p = comb(N, K) / (2.0 ** N - 1)
    return (N / 2) * math.exp(-float(np.sum(p * digamma(K / 2))))
