"""
Activation functions and their Gaussian averages.

Every quantity of the theory is an average of the activation phi, or of its derivatives, over a Gaussian
pre-activation of variance q:

    V(q) = E[phi(sqrt(q) z)^2],    P(q) = E[phi'(sqrt(q) z)^2],    z ~ N(0, 1).

Averages are computed with Gauss-Hermite quadrature (160 nodes): exact to machine precision for smooth
activations, accurate to about 1e-8 for activations with a kink.

An activation is described by an `Activation` object holding phi, phi' and phi'' as NumPy callables and two
flags: `homogeneous` (phi(cx) = c phi(x) for c > 0, as for ReLU and leaky ReLU, which makes the edge of chaos
a single point) and `smooth` (phi'' exists, so the compression coefficient kappa is finite).
"""
import math
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from scipy.special import erf

_X, _W = hermegauss(160)
_W = _W / _W.sum()
_SQ2PI = math.sqrt(2.0 / math.pi)


@dataclass(frozen=True)
class Activation:
    name: str
    phi: Callable
    dphi: Callable
    d2phi: Callable
    homogeneous: bool = False
    smooth: bool = True
    alpha: Optional[float] = None       # negative slope of a leaky ReLU

    def avg(self, f, q):
        """E[f(sqrt(q) z)] for z ~ N(0, 1)."""
        return float(np.sum(_W * f(math.sqrt(q) * _X)))

    def V(self, q):
        return self.avg(lambda u: self.phi(u) ** 2, q)

    def P(self, q):
        return self.avg(lambda u: self.dphi(u) ** 2, q)

    def moments(self, q, h=None):
        """All averages used by the finite-width formulas at variance q: P, V, their first two q-derivatives,
        P4 = E[phi'^4], V4 = E[phi^4] and E[(phi phi')^2]."""
        if h is None:
            h = 1e-4 * max(q, 1e-3)
        Pq = lambda s: self.P(s)
        Vq = lambda s: self.V(s)
        return dict(
            q=q, P=Pq(q), V=Vq(q),
            dP=(Pq(q + h) - Pq(q - h)) / (2 * h), d2P=(Pq(q + h) - 2 * Pq(q) + Pq(q - h)) / h ** 2,
            dV=(Vq(q + h) - Vq(q - h)) / (2 * h), d2V=(Vq(q + h) - 2 * Vq(q) + Vq(q - h)) / h ** 2,
            P4=self.avg(lambda u: self.dphi(u) ** 4, q), V4=self.avg(lambda u: self.phi(u) ** 4, q),
            EphiDphi2=self.avg(lambda u: (self.phi(u) * self.dphi(u)) ** 2, q),
        )


def _gelu(u):
    return 0.5 * u * (1 + erf(u / math.sqrt(2)))


def _dgelu(u):
    return 0.5 * (1 + erf(u / math.sqrt(2))) + u * np.exp(-u * u / 2) / math.sqrt(2 * math.pi)


def _d2gelu(u):
    return (2 - u * u) * np.exp(-u * u / 2) / math.sqrt(2 * math.pi)


def _softplus(u):
    return np.logaddexp(0.0, u)


def _sigmoid(u):
    return 1 / (1 + np.exp(-u))


TANH = Activation("tanh", np.tanh, lambda u: 1 / np.cosh(u) ** 2, lambda u: -2 * np.tanh(u) / np.cosh(u) ** 2)
ERF = Activation("erf", lambda u: erf(u / math.sqrt(2)), lambda u: _SQ2PI * np.exp(-u * u / 2),
                 lambda u: -_SQ2PI * u * np.exp(-u * u / 2))
GELU = Activation("gelu", _gelu, _dgelu, _d2gelu)
SOFTPLUS = Activation("softplus", _softplus, _sigmoid, lambda u: _sigmoid(u) * (1 - _sigmoid(u)))
LINEAR = Activation("linear", lambda u: u, lambda u: np.ones_like(u), lambda u: np.zeros_like(u), homogeneous=True)
RELU = Activation("relu", lambda u: np.maximum(u, 0.0), lambda u: (u > 0).astype(float), lambda u: np.zeros_like(u),
                  homogeneous=True, smooth=False, alpha=0.0)


def leaky_relu(alpha):
    """Leaky ReLU with negative slope alpha."""
    return Activation(f"leaky_relu({alpha:g})", lambda u: np.where(u > 0, u, alpha * u),
                      lambda u: np.where(u > 0, 1.0, alpha), lambda u: np.zeros_like(u),
                      homogeneous=True, smooth=False, alpha=float(alpha))


def custom(phi, dphi, d2phi=None, name="custom"):
    """A user-defined smooth activation.  phi, dphi (and d2phi, optional) are NumPy callables; if d2phi is
    omitted it is obtained by central differences of dphi."""
    if d2phi is None:
        d2phi = lambda u, h=1e-4: (dphi(u + h) - dphi(u - h)) / (2 * h)
    return Activation(name, phi, dphi, d2phi)


REGISTRY = {"tanh": TANH, "erf": ERF, "gelu": GELU, "softplus": SOFTPLUS, "linear": LINEAR, "relu": RELU,
            "identity": LINEAR}


def get(activation, alpha=None):
    """Resolve an activation given as an Activation, a name ('tanh', 'erf', 'gelu', 'softplus', 'relu',
    'leaky_relu', 'linear') or a torch module/function name such as 'ReLU' or 'Tanh'."""
    if isinstance(activation, Activation):
        return activation
    key = str(activation).lower().replace("torch.", "").replace("nn.", "").replace("f.", "")
    if key in ("leaky_relu", "leakyrelu", "leaky"):
        return leaky_relu(0.01 if alpha is None else alpha)
    if key in REGISTRY:
        return REGISTRY[key]
    raise ValueError(f"unknown activation {activation!r}; use one of {sorted(REGISTRY)} or crinit.custom(...)")
