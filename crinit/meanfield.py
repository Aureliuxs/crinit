"""
Infinite-width (mean-field) theory of a deep random network.

    q_{l+1} = sigma_w^2 V(q_l) + sigma_b^2        signal size, with steady value q* and slope F' = sigma_w^2 V'(q*)
    chi_1   = sigma_w^2 P(q*)                     growth factor per layer of the distance between two close inputs

The edge of chaos is chi_1 = 1.  Written in terms of the steady signal q* it is the line

    sigma_w^2 = 1/P(q*),      sigma_b^2 = q* - V(q*)/P(q*),

which exists for every q* because q P(q) >= V(q) for any activation (a Hermite-series identity).

The correlation C between two inputs obeys 1 - C' = chi_1 (1 - C) - kappa (1 - C)^2 + ... for smooth
activations, with kappa = sigma_w^2 q* E[phi''^2]/2.  Over L layers on the edge of chaos the range of input
similarities is compressed by the factor exp(comp), comp = ln(eps_0/eps_L) with eps = 1 - C.
"""
import math

import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from scipy.optimize import brentq

from .activations import get

_X2, _W2 = hermegauss(80)
_W2 = _W2 / _W2.sum()
_ZA, _ZB = np.meshgrid(_X2, _X2, indexing="ij")
_WAB = np.outer(_W2, _W2)


def fixed_point(act, sw2, sb2, q0=1.0, iters=5000, tol=1e-14):
    """Steady signal q* of the variance map for the point (sigma_w^2, sigma_b^2)."""
    act = get(act)
    q = q0
    for _ in range(iters):
        qn = sw2 * act.V(q) + sb2
        if not np.isfinite(qn) or abs(qn - q) < tol:
            break
        q = qn
    return q


def eoc_point(act, q):
    """(sigma_w^2, sigma_b^2) of the edge of chaos with steady signal q."""
    act = get(act)
    P, V = act.P(q), act.V(q)
    return 1.0 / P, q - V / P


def eoc_at_bias(act, sb2, lo=0.05, hi=20.0):
    """(sigma_w^2, q*) of the edge of chaos at a given bias variance."""
    act = get(act)

    def chi_minus_1(sw2):
        q = fixed_point(act, sw2, sb2)
        return sw2 * act.P(q) - 1.0

    sw2 = brentq(chi_minus_1, lo, hi, xtol=1e-12)
    return sw2, fixed_point(act, sw2, sb2)


def chi1(act, q, sw2=None):
    act = get(act)
    return (1.0 / act.P(q) if sw2 is None else sw2) * act.P(q)


def slope(act, q, sw2=None):
    """F' = sigma_w^2 V'(q): stability of the steady signal (|F'| < 1 is stable)."""
    act = get(act)
    if sw2 is None:
        sw2 = 1.0 / act.P(q)
    h = 1e-4 * max(q, 1e-3)
    return sw2 * (act.V(q + h) - act.V(q - h)) / (2 * h)


def kappa(act, q, sw2=None):
    """Compression coefficient of the correlation map on the edge of chaos (smooth activations)."""
    act = get(act)
    if sw2 is None:
        sw2 = 1.0 / act.P(q)
    return sw2 * q / 2 * act.avg(lambda u: act.d2phi(u) ** 2, q)


def dist_map(act, eps, q, sw2):
    """One layer of the map for eps = 1 - C at steady signal q (midpoint form, stable at small eps)."""
    act = get(act)
    d2 = 2 * q * eps
    m = math.sqrt(max(q - d2 / 4, 0.0)) * _ZA
    dl = math.sqrt(d2) * _ZB
    return sw2 * float(np.sum(_WAB * (act.phi(m + dl / 2) - act.phi(m - dl / 2)) ** 2)) / (2 * q)


def compression(act, q, L, sw2=None, eps0=1.0):
    """comp = ln(eps_0/eps_L): how much the range of input similarities shrinks over L layers (eps_0 = 1 means
    orthogonal inputs)."""
    act = get(act)
    if sw2 is None:
        sw2 = 1.0 / act.P(q)
    eps = eps0
    for _ in range(L):
        eps = dist_map(act, min(eps, 1.999), q, sw2)
    return math.log(eps0) - math.log(max(eps, 1e-300))


def operating_point(act, depth, gamma=1.0, q_lo=1e-5, q_hi=20.0):
    """Steady signal q* on the edge of chaos at which the compression over `depth` layers equals gamma
    (gamma = 1 is the operating point recommended by the paper).  Returns (q*, sigma_w^2, sigma_b^2)."""
    act = get(act)

    def f(lq):
        return compression(act, math.exp(lq), depth) - gamma

    lo, hi = math.log(q_lo), math.log(q_hi)
    flo, fhi = f(lo), f(hi)
    if flo * fhi > 0:
        raise ValueError(f"compression {gamma} is not reachable for {act.name} at depth {depth}: the range over "
                         f"q* in [{q_lo:g}, {q_hi:g}] is [{min(flo, fhi) + gamma:.3g}, {max(flo, fhi) + gamma:.3g}]")
    q = math.exp(brentq(f, lo, hi, xtol=1e-10))
    sw2, sb2 = eoc_point(act, q)
    return q, sw2, sb2


# ----------------------------------------------------------------- homogeneous activations -------------------
def leaky_cmap(c, alpha):
    """Closed-form correlation map of the leaky ReLU on its edge of chaos (sigma_w^2 = 2/(1+alpha^2), bias 0)."""
    a, b = (1 + alpha) / 2, (1 - alpha) / 2
    c = min(c, 1.0)
    return (a * a * c + 2 * b * b / math.pi * (math.sqrt(max(1 - c * c, 0.0)) + c * math.asin(c))) / (a * a + b * b)


def leaky_compression(alpha, depth, c0=0.0):
    c = c0
    for _ in range(depth):
        c = leaky_cmap(c, alpha)
    return -math.log(max(1 - c, 1e-300))


def shaped_leaky_alpha(depth, gamma=1.0):
    """Negative slope alpha of the leaky ReLU whose compression of orthogonal inputs over `depth` layers is
    gamma (the depth-shaped leaky ReLU)."""
    if leaky_compression(0.0, depth) < gamma:
        raise ValueError(f"compression {gamma} is not reachable with a leaky ReLU at depth {depth}; "
                         f"ReLU itself gives {leaky_compression(0.0, depth):.3g}")
    return brentq(lambda a: leaky_compression(a, depth) - gamma, 0.0, 0.9999, xtol=1e-10)
