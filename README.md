# crinit: critical initialisation at finite width

`crinit` initialises a deep network on the **finite-width edge of chaos**: the weight and bias variances at which
the distance between two nearby inputs neither grows nor shrinks as it passes through the layers, including the
`1/N` correction that moves this point away from its infinite-width (mean-field) position. It plays the same role
as LeCun, Glorot or He initialisation, with two differences: the point is chosen on the edge of chaos of the
activation actually used, and it is corrected for the width of the network.

The rule comes from A. Caesar, *Finite-width shift of the edge of chaos in deep neural
networks* (2026). In short:

1. **Where to sit on the edge of chaos.** For a smooth activation the edge of chaos is a line of points, labelled
   by the steady signal size `q*` (the pre-activation variance deep in the network). Points differ in how much
   they compress the range of input similarities over the depth. `crinit` picks the point whose compression over
   `depth` layers equals `gamma` (default 1 nat, the operating point found best in training).
2. **The finite-width correction.** At width `N` the logarithm of the distance between two inputs drifts by
   `ln chi_1 + a_1/N` per layer, with `a_1 < 0` an explicit Gaussian average of the activation. The true edge of
   chaos is where this drift vanishes, `ln chi_1 = -a_1/N`. `crinit` keeps the steady signal of step 1 (it sets
   the compression) and moves the point across the line: `sigma_w^2 = exp(-a_1/N)/P(q*)`,
   `sigma_b^2 = q* - sigma_w^2 V(q*)`. If that bias would be negative (deep, narrow networks), the nearest point
   of the line at zero bias is taken and the result says so. (Moving at fixed bias instead, by
   `Delta sigma_w^2 = -a_1/(N d ln chi_1/d sigma_w^2)`, about `8/N` for tanh, lands on the same line at a larger
   signal; `crinit.shift` gives that displacement.)

For ReLU-type activations the edge of chaos is a single point, `sigma_w^2 = 2/(1 + alpha^2)`, `sigma_b^2 = 0`;
there the compression is set by the negative slope `alpha` of a leaky ReLU, which `crinit` chooses from `gamma`,
and the finite-width correction keeps the typical signal size constant (`sigma_w^2 = 2 e^{5/(2N)}` for ReLU at
first order).

### Orders in 1/N

By default (`order=1`) the correction is the closed-form first order in `1/N`. With `order="exact"` all orders are
included:

* homogeneous activations (linear, ReLU, leaky ReLU): the exact typical-preserving variance, from the digamma
  formula `(N/2) exp(-psi(N/2))` (linear), its binomial average (ReLU), and a one-dimensional integral (leaky ReLU);
  instantaneous;
* smooth activations (tanh, erf, ...): the zero of the drift of the distance between two inputs is found
  numerically in the exact finite-width chain (Gaussian weights only), which takes about 15 seconds.

For tanh at depth 32 the two agree to within 2 to 6% of the correction for `N >= 64` (the correction itself is
about `2/N` in `sigma_w^2` there). The exact option matters for narrow networks (`N <~ 32`, where the difference
reaches 30% of the correction) and on the zero-bias branch of deep networks (depth 96 at `N = 64`: 1.298 instead
of 1.311), where `q*` moves with `sigma_w^2` (paper, Supplemental Material Table S2).

## Install

```
pip install .            # numpy and scipy
pip install .[torch]     # also PyTorch, for crinit.torch_init
```

## Use

```python
import crinit

r = crinit.crinit("tanh", depth=32, width=64)     # 32 hidden layers of width 64
print(r)
# Crinit(tanh, depth=32, width=64, gaussian): sigma_w^2 = 1.4980 (mean field 1.4655, shift +0.0324),
#   sigma_b^2 = 0.006478, q* = 0.306, gamma = 1
```

`r.sigma_w2` is the variance of the hidden weights **times the fan-in** (draw `W ~ N(0, r.sigma_w2 / fan_in)`)
and `r.sigma_b2` the bias variance.

**The first layer.** Crinit draws the first-layer weights with variance `r.first_layer_var / fan_in`, where
`r.first_layer_var = q* - r.sigma_b2`, so that the first hidden layer already carries the steady signal `q*`
(inputs standardised to unit variance per component). This is the `first_layer="steady"` default of `crinit_`,
`apply_`, `mlp` and `sample`. Because `q*` is smaller than `sigma_w2`, the first layer is smaller than under the
standard rules, and with plain SGD it then takes larger steps relative to its size; in the paper's training runs
this is worth up to two per cent of test accuracy in deep networks (Supplemental Material Sec. S8). The other
choices are `first_layer="standard"` (variance `r.sigma_w2 / fan_in`, like the hidden layers and like the
standard rules) and `first_layer=q1` with a number `q1` (variance `(q1 - r.sigma_b2) / fan_in`). Do not rescale
the input instead of the weights: that gives the same forward pass but a smaller gradient for the first layer,
and trains worse.

With PyTorch:

```python
import torch.nn as nn
import crinit.torch_init as ct

model = nn.Sequential(nn.Linear(784, 64), nn.Tanh(), *[m for _ in range(31) for m in (nn.Linear(64, 64), nn.Tanh())],
                      nn.Linear(64, 10))
r = ct.crinit_(model, "tanh")                      # initialises every nn.Linear in place; depth and width are read off the model

model, r = ct.mlp(784, 64, 96, 10, "leaky_relu")   # a ready-made network; the leaky ReLU gets the slope chosen by crinit
```

Options of `crinit.crinit(activation, depth, width, ...)`:

| argument | meaning | default |
|---|---|---|
| `activation` | `'tanh'`, `'erf'`, `'gelu'`, `'relu'`, `'leaky_relu'`, `'linear'`, or `crinit.custom(phi, dphi)` | `'tanh'` |
| `gamma` | compression of input similarities over the depth, in nats | `1.0` |
| `finite_width` | apply the `1/N` correction | `True` |
| `ensemble` | `'gaussian'` or `'orthogonal'` hidden weights (the correction is smaller for orthogonal weights) | `'gaussian'` |
| `alpha` | negative slope of a leaky ReLU; `None` lets `gamma` choose it | `None` |
| `q_star`, `bias` | fix the steady signal or the bias variance directly instead of choosing them from `gamma` | `None` |
| `order` | `1` (closed-form first order in `1/N`) or `"exact"` (all orders; numerical for smooth activations) | `1` |

The result carries `notes` when something should be known: the ReLU point has a fixed compression; the
requested compression needed a negative bias variance and the zero-bias point was taken instead; and the
correction is flagged when `N(1 - F')` is not large compared with one (too close to the linear point). Softplus,
whose mean at the origin is not zero, has no edge of chaos with a non-negative bias variance, and `crinit` raises
an error for it.

**GELU is supported but the rule works poorly for it**, and the result says so in its notes. GELU has an even part
at the origin, so its steady signal is unstable on the small-`q*` part of the edge of chaos; `crinit` moves to the
smallest stable point, and the finite-width correction then pushes the bias variance negative, so the result is
the zero-bias point at a large `q*` (where GELU is close to ReLU), with a compression well above the requested
`gamma` and `N(1 - F')` of order one, where the `1/N` correction is not reliable. The paper's tests cover tanh,
erf-type, ReLU and leaky ReLU; for a ReLU-like network use `'leaky_relu'`, whose compression `crinit` can set.

A slope given explicitly, either as `alpha=0.2` or as `crinit.leaky_relu(0.2)`, is kept; only `'leaky_relu'`
without a slope lets `gamma` choose it.

NumPy users can draw the weights directly:

```python
Ws, bs = crinit.sample(r, d_in=784, d_out=10, rng=0)     # lists of (fan_in, fan_out) weights and biases
```

## Lower-level functions

`eoc_point(act, q)` gives `(sigma_w^2, sigma_b^2)` of the mean-field edge of chaos at steady signal `q`;
`operating_point(act, depth, gamma)` the point with the requested compression; `compression(act, q, depth)` the
compression itself; `shift(act, q, N)` the correction `Delta sigma_w^2`, the drift coefficient `a_1` and the
validity measure `N(1 - F')`; `a1_b1(act, q)` the drift and diffusion coefficients of the distance walk;
`sigma_c2_relu_exact(N)`, `sigma_c2_leaky(N, alpha)` and `sigma_c2_smooth(N)` the typical-preserving variances;
`crinit.exact.sigma_c2_homogeneous_exact(N, alpha)` the all-orders typical-preserving variance of a leaky ReLU, and
`crinit.exact.chain_drift(act, sw2, sb2, N)` the drift of the distance measured in the exact finite-width chain.

## Tests

```
python -m pytest tests
```

## Citation

If you use `crinit`, please cite the paper:

```
@article{caesar2026finitewidth,
  title   = {Finite-width shift of the edge of chaos in deep neural networks},
  author  = {Caesar, Arn},
  year    = {2026},
  journal = {arXiv preprint},
  note    = {arXiv ID to be added}
}
```

## Licence

MIT.
