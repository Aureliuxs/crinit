"""
Crinit: critical initialisation of deep networks at finite width.

    >>> import crinit
    >>> r = crinit.crinit("tanh", depth=32, width=64)
    >>> r.sigma_w2, r.sigma_b2, r.q_star
    (1.498..., 0.0065..., 0.306...)

The rule has two parts.

1.  Where on the edge of chaos to sit.  The mean-field edge of chaos is a line; its points differ in how
    much the network compresses the range of input similarities over its depth.  Crinit picks the steady
    signal q* at which this compression over `depth` layers equals `gamma` (default 1).

2.  The finite-width correction.  At width N the distance between two close inputs drifts as
    ln chi_1 + a_1/N per layer; the true edge of chaos is where this drift vanishes, ln chi_1 = -a_1/N.
    Crinit keeps the steady signal q* chosen in step 1 (it sets the compression) and moves the point across the
    line: sigma_w^2 = exp(-a_1/N)/P(q*), sigma_b^2 = q* - sigma_w^2 V(q*).  (Moving at fixed bias instead, by
    Delta sigma_w^2 = -a_1/(N d ln chi_1/d sigma_w^2), lands on the same line but at a larger q*, i.e. at a point
    that compresses more than the budget; `shift_at_bias` gives that displacement.)

For homogeneous activations (ReLU, leaky ReLU, linear) the edge of chaos is the single point
(2/(1+alpha^2), 0) and a common rescaling of the weights does not change the geometry; the compression over
the depth is then set by the negative slope alpha instead (`shape=True` picks alpha for the requested gamma),
and the finite-width correction keeps the typical signal size constant.

Orders in 1/N.  `order=1` (default) uses the closed-form first-order correction.  `order="exact"` includes all
orders: for homogeneous activations through the exact typical-preserving variance (digamma formula and its
generalisation, see crinit.exact), for smooth activations by locating the zero of the drift numerically in the exact
finite-width chain of two inputs (Gaussian weights; about 15 seconds).  The difference matters only for narrow
networks: for tanh at depth 32 it is 2 to 6% of the (already small) correction for N >= 64, but up to 30% at N = 32
and up to 50% on the zero-bias branch (depth 96, N = 64), where q* moves with sigma_w^2.
"""
import math
from dataclasses import dataclass, field, asdict
from typing import Optional

from .activations import Activation, get, leaky_relu
from .finite_width import a1_b1, shift, sigma_c2_leaky, sigma_c2_relu_exact, sigma_c2_smooth
from .meanfield import eoc_point, leaky_compression, operating_point, shaped_leaky_alpha, slope, compression


@dataclass
class Crinit:
    """Result of crinit(): the variances to initialise with and how they were obtained."""
    activation: str
    depth: int
    width: int
    ensemble: str
    sigma_w2: float                  # variance of the hidden weights times fan-in
    sigma_b2: float                  # variance of the hidden biases
    q_star: float                    # steady signal size (pre-activation variance) of the hidden layers
    gamma: float                     # compression of input similarities over the depth
    sigma_w2_meanfield: float        # the mean-field edge of chaos at the same bias
    shift: float                     # finite-width correction Delta sigma_w^2 = sigma_w2 - sigma_w2_meanfield
    alpha: Optional[float] = None    # negative slope (leaky ReLU)
    validity: Optional[float] = None  # N (1 - F'), must be large compared with 1
    notes: list = field(default_factory=list)

    @property
    def first_layer_var(self):
        """Variance (times fan-in) for the first layer so that the first hidden layer already has signal
        q*, assuming inputs standardised to unit variance per component."""
        return max(self.q_star - self.sigma_b2, 1e-12)

    def first_layer_variance(self, first_layer="steady"):
        """Variance (times fan-in) of the first-layer weights for the chosen first-layer rule:
        'steady' (or True): (q* - sigma_b^2)/fan_in, the first hidden layer starts at the steady signal (default);
        'standard' (or False): sigma_w^2/fan_in, the same rule as the hidden layers;
        a number q1: (q1 - sigma_b^2)/fan_in, the first hidden layer starts at the signal q1."""
        if first_layer in ("steady", True):
            return self.first_layer_var
        if first_layer in ("standard", False):
            return self.sigma_w2
        return max(float(first_layer) - self.sigma_b2, 1e-12)

    def weight_std(self, fan_in, first=False):
        """Standard deviation of a weight: hidden layers (first=False), or the first layer for the rule
        `first` ('steady', 'standard', a number q1, or True for 'steady')."""
        var = self.sigma_w2 if first is False else self.first_layer_variance(first)
        return math.sqrt(var / fan_in)

    def bias_std(self):
        return math.sqrt(max(self.sigma_b2, 0.0))

    def as_dict(self):
        return asdict(self)

    def __str__(self):
        s = (f"Crinit({self.activation}, depth={self.depth}, width={self.width}, {self.ensemble}): "
             f"sigma_w^2 = {self.sigma_w2:.4f} (mean field {self.sigma_w2_meanfield:.4f}, shift {self.shift:+.4f}), "
             f"sigma_b^2 = {self.sigma_b2:.4g}, q* = {self.q_star:.4g}, gamma = {self.gamma:.3g}")
        if self.alpha is not None:
            s += f", alpha = {self.alpha:.4g}"
        for n in self.notes:
            s += "\n  note: " + n
        return s


def crinit(activation="tanh", depth=32, width=64, gamma=1.0, finite_width=True, ensemble="gaussian",
           alpha=None, shape=None, q_star=None, bias=None, order=1, seed=0):
    """Critical initialisation at finite width.

    activation : 'tanh', 'erf', 'gelu', 'softplus', 'relu', 'leaky_relu', 'linear', or an Activation
    depth      : number of hidden layers L
    width      : hidden width N (used for the 1/N correction; with unequal widths pass the typical one)
    gamma      : target compression (nats of log-range of input similarities lost over the depth); 1 by default
    finite_width : apply the 1/N correction (True) or return the mean-field point (False)
    ensemble   : 'gaussian' or 'orthogonal' hidden weights
    alpha      : negative slope for 'leaky_relu'; None lets the slope be chosen from gamma
    shape      : for ReLU-type activations, choose the slope alpha so that the compression equals gamma;
                 defaults to True for 'leaky_relu' without a given alpha and False otherwise (a plain ReLU
                 is then kept as ReLU and the result notes its fixed compression)
    q_star     : fix the steady signal directly instead of choosing it from gamma
    bias       : fix the bias variance directly (q_star is then found from it) instead of choosing it from gamma
    order      : 1 for the closed-form first-order correction in 1/N, or "exact" for all orders (homogeneous
                 activations: exact formula; smooth activations: numerical, from the exact finite-width chain,
                 Gaussian weights only; `seed` fixes its random numbers)
    """
    if order not in (1, "exact"):
        raise ValueError("order must be 1 or 'exact'")
    if ensemble not in ("gaussian", "orthogonal"):
        raise ValueError("ensemble must be 'gaussian' or 'orthogonal'")
    name = activation.name if isinstance(activation, Activation) else str(activation).lower()
    is_leaky = name.replace("torch.", "").replace("nn.", "") in ("leaky_relu", "leakyrelu", "leaky") or name.startswith("leaky_relu(")
    if isinstance(activation, Activation) and alpha is None and activation.alpha not in (None, 0.0):
        alpha = activation.alpha          # a slope given through crinit.leaky_relu(alpha) is kept, not re-chosen
    act = get(activation, alpha)
    notes = []
    N, L = int(width), int(depth)

    if act.homogeneous:
        if shape is None:
            shape = is_leaky and alpha is None
        return _homogeneous(act, L, N, gamma, finite_width, ensemble, alpha, shape, notes, order)

    if bias is not None:
        from .meanfield import eoc_at_bias
        sw2_mf, q = eoc_at_bias(act, bias)
        sb2 = bias
        g = compression(act, q, L)
    elif q_star is not None:
        q = float(q_star)
        sw2_mf, sb2 = eoc_point(act, q)
        g = compression(act, q, L)
    else:
        q, sw2_mf, sb2 = operating_point(act, L, gamma)
        g = gamma
    if sb2 < -1e-12:
        raise ValueError(f"{act.name} has a non-zero mean E[phi] at the origin, so no point of its edge of chaos has a "
                         "non-negative bias variance; centre the activation (subtract its mean) first")
    Fp = slope(act, q, sw2_mf)
    if Fp >= 1:
        q_st = _stable_branch(act, q)
        if q_st is None:
            raise ValueError(f"the steady signal is unstable (F' >= 1) everywhere on the edge of chaos of {act.name}")
        notes.append(f"the steady signal q* = {q:.3g} is unstable on the edge of chaos of {act.name} (F' = {Fp:.3f} >= 1) "
                     f"because phi has an even part at the origin; moved to the smallest stable point q* = {q_st:.3g}")
        q = q_st
        sw2_mf, sb2 = eoc_point(act, q)
        g = compression(act, q, L)
        Fp = slope(act, q, sw2_mf)
        if abs(g - gamma) > 1e-6 and q_star is None and bias is None:
            notes.append(f"the compression over the depth is then {g:.3g} instead of the requested {gamma:g}")
    validity = N * (1 - Fp)
    sw2, keep_bias = sw2_mf, True
    if finite_width:
        # The finite-width edge of chaos is ln chi_1 = -a_1/N.  Crinit keeps the steady signal q* (which sets the
        # compression) and moves the point across the line: sigma_w^2 = exp(-a_1/N)/P(q*), sigma_b^2 = q* - sigma_w^2 V(q*).
        # If that bias would be negative, the point closest to the budget is at zero bias with a slightly larger q*.
        orth = ensemble == "orthogonal"
        q, sw2, sb2, moved = _finite_width_point(act, q, N, orth)
        if moved:
            g = compression(act, q, L)
            notes.append(f"the finite-width point with the requested compression would need a negative bias variance; "
                         f"moved along the finite-width line to zero bias, q* = {q:.3g}, compression {g:.3g}")
        if order == "exact":
            if orth:
                notes.append("order='exact' is available for Gaussian weights only; the first-order correction is used")
            else:
                from .exact import exact_point
                sw2_e, sb2_e, se = exact_point(act, q, N, sw2, sb2, zero_bias=moved, seed=seed)
                if not moved and sb2_e <= 0:
                    sw2_e, sb2_e, se = exact_point(act, q, N, sw2, 0.0, zero_bias=True, seed=seed)
                    moved = True
                if moved:
                    from .meanfield import fixed_point
                    q = fixed_point(act, sw2_e, 0.0)
                    g = compression(act, q, L)
                notes.append(f"all orders in 1/N (exact chain): sigma_w^2 = {sw2_e:.4f} +- {se:.4f} "
                             f"(first order {sw2:.4f})")
                sw2, sb2 = sw2_e, sb2_e
        sw2_mf = 1.0 / act.P(q)
        Fp = slope(act, q, sw2_mf)
        validity = N * (1 - Fp)
        if validity < 5 and order == 1:
            notes.append(f"N(1 - F') = {validity:.2f}: the 1/N correction is not reliable this close to the "
                         "linear point; use a larger q* (smaller gamma) or a wider network")
    return Crinit(act.name, L, N, ensemble, sw2, sb2, q, g, sw2_mf, sw2 - sw2_mf, None, validity, notes)


def _finite_width_point(act, q, N, orthogonal):
    """(q*, sigma_w^2, sigma_b^2, moved) on the finite-width edge of chaos with steady signal q, or, if that needs a
    negative bias, at zero bias with the smallest q* >= q that lies on the line."""
    from scipy.optimize import brentq

    def point(qq):
        a1 = a1_b1(act, qq, 1.0 / act.P(qq), orthogonal=orthogonal)["a1"]
        sw2 = math.exp(-a1 / N) / act.P(qq)
        return sw2, qq - sw2 * act.V(qq)

    sw2, sb2 = point(q)
    if sb2 >= 0:
        return q, sw2, sb2, False
    # the bias on the finite-width line grows with q*: find where it crosses zero
    hi = q
    for _ in range(60):
        hi *= 1.15
        if point(hi)[1] > 0:
            break
    q0 = brentq(lambda qq: point(qq)[1], q, hi, xtol=1e-10)
    sw2, _ = point(q0)
    return q0, sw2, 0.0, True


def _stable_branch(act, q0, q_max=200.0, margin=1e-3):
    """Smallest q >= q0 on the edge of chaos with F' < 1 - margin (None if there is none up to q_max)."""
    import numpy as np
    from scipy.optimize import brentq
    grid = np.geomspace(q0, q_max, 60)
    f = lambda q: slope(act, q, 1.0 / act.P(q)) - (1 - margin)
    prev = f(grid[0])
    for q_prev, q in zip(grid[:-1], grid[1:]):
        cur = f(q)
        if prev > 0 and cur < 0:
            return brentq(f, q_prev, q, xtol=1e-8)
        prev = cur
    return None


def _homogeneous(act, L, N, gamma, finite_width, ensemble, alpha, shape, notes, order=1):
    from .exact import sigma_c2_homogeneous_exact
    a = act.alpha
    if act.name == "linear":
        sw2_mf = 1.0
        sw2 = (sigma_c2_smooth(N) if order == "exact" else math.exp(1.0 / N)) if finite_width else 1.0
        if ensemble == "orthogonal":
            sw2 = 1.0
        notes.append("linear network: every input similarity is preserved; the only finite-width effect is on the "
                     "typical signal size")
        return Crinit(act.name, L, N, ensemble, sw2, 0.0, 1.0, 0.0, sw2_mf, sw2 - sw2_mf, None, None, notes)
    # ReLU family
    if shape and (alpha is None):
        try:
            a = shaped_leaky_alpha(L, gamma)
        except ValueError as e:
            a = 0.0 if a is None else a
            notes.append(str(e))
        act = leaky_relu(a) if a > 0 else act
    g = leaky_compression(a, L)
    sw2_mf = 2.0 / (1 + a * a)
    if ensemble == "orthogonal" or not finite_width:
        sw2 = sw2_mf
    elif order == "exact":
        sw2 = sigma_c2_homogeneous_exact(N, a)
    else:
        sw2 = sigma_c2_leaky(N, a)          # 2/(1+a^2) exp(C_a/(2N)); for ReLU 2 exp(5/(2N))
    if a == 0.0:
        notes.append("ReLU: the edge of chaos is the single point (2, 0) and the compression over the depth is fixed "
                     f"at {g:.3g}; pass activation='leaky_relu' (or shape=True) to set it to gamma")
    return Crinit(act.name, L, N, ensemble, sw2, 0.0, 1.0, g, sw2_mf, sw2 - sw2_mf, a, None, notes)
