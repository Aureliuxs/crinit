"""
PyTorch helpers.

    import crinit, crinit.torch_init as ct
    model = nn.Sequential(nn.Linear(784, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), ..., nn.Linear(64, 10))
    r = ct.crinit_(model, "tanh")              # in place; returns the Crinit result that was applied

or, layer by layer, like torch.nn.init:

    r = crinit.crinit("tanh", depth=L, width=N)
    ct.normal_(layer.weight, r)                # hidden weight, N(0, sigma_w^2 / fan_in)
    ct.normal_(first.weight, r, first=True)    # first layer, N(0, (q* - sigma_b^2) / fan_in); first="standard" for sigma_w^2/fan_in
    ct.bias_(layer.bias, r)                    # N(0, sigma_b^2)
"""
import math

import torch
import torch.nn as nn

from .core import Crinit, crinit


@torch.no_grad()
def normal_(weight, r: Crinit, first=False):
    """Gaussian weights with variance sigma_w^2/fan_in.  For the first layer pass first='steady' (or True):
    variance (q* - sigma_b^2)/fan_in, so that the first hidden layer starts at the steady signal (inputs
    standardised to unit variance per component); first='standard': sigma_w^2/fan_in like the hidden layers;
    first=q1 (a number): (q1 - sigma_b^2)/fan_in."""
    fan_in = weight.shape[1]
    return weight.normal_(0.0, r.weight_std(fan_in, first=first))


@torch.no_grad()
def orthogonal_(weight, r: Crinit):
    """Orthogonal weights scaled so that every layer multiplies the signal norm by sigma_w^2 exactly."""
    return nn.init.orthogonal_(weight, gain=math.sqrt(r.sigma_w2))


@torch.no_grad()
def bias_(bias, r: Crinit):
    if bias is None:
        return None
    return bias.normal_(0.0, r.bias_std()) if r.sigma_b2 > 0 else bias.zero_()


def activation_module(r: Crinit):
    """The torch activation that goes with the result (a leaky ReLU carries the chosen slope)."""
    name = r.activation.split("(")[0]
    return {"tanh": nn.Tanh, "erf": _Erf, "gelu": nn.GELU, "softplus": nn.Softplus, "relu": nn.ReLU,
            "linear": nn.Identity, "leaky_relu": lambda: nn.LeakyReLU(r.alpha)}[name]()


class _Erf(nn.Module):
    def forward(self, x):
        return torch.erf(x / math.sqrt(2.0))


def linear_layers(model):
    """All nn.Linear modules of a model in registration order."""
    return [m for m in model.modules() if isinstance(m, nn.Linear)]


@torch.no_grad()
def apply_(model, r: Crinit, first_layer="steady", head=True):
    """Initialise every nn.Linear of `model` from the Crinit result: the first layer by the rule `first_layer`
    ('steady', the default: the first hidden layer starts at the steady signal q*; 'standard': like the hidden
    layers; or a number q1), the hidden layers with (sigma_w^2, sigma_b^2), and the last layer (the head) with
    LeCun's 1/fan_in."""
    layers = linear_layers(model)
    if len(layers) < 2:
        raise ValueError("expected at least two nn.Linear layers (hidden layers and a head)")
    hidden, out = layers[:-1], layers[-1]
    for i, lin in enumerate(hidden):
        if i == 0 and first_layer not in ("standard", False):
            normal_(lin.weight, r, first=first_layer)
        elif r.ensemble == "orthogonal":
            orthogonal_(lin.weight, r)
        else:
            normal_(lin.weight, r)
        bias_(lin.bias, r)
    if head:
        out.weight.normal_(0.0, math.sqrt(1.0 / out.weight.shape[1]))
        if out.bias is not None:
            out.bias.zero_()
    return r


def crinit_(model, activation="tanh", gamma=1.0, finite_width=True, ensemble="gaussian", depth=None, width=None,
            first_layer="steady", head=True, **kw):
    """Initialise a feed-forward model in place.  depth and width are read off the model when not given:
    depth = number of nn.Linear layers minus one (the head), width = the fan-in of the last hidden layer."""
    layers = linear_layers(model)
    if depth is None:
        depth = len(layers) - 1
    if width is None:
        width = layers[-1].weight.shape[1]
    r = crinit(activation, depth=depth, width=width, gamma=gamma, finite_width=finite_width, ensemble=ensemble, **kw)
    return apply_(model, r, first_layer=first_layer, head=head)


def mlp(d_in, width, depth, d_out, activation="tanh", first_layer="steady", **kw):
    """A fully connected network (depth hidden layers of the given width) initialised with Crinit; the
    activation module is the one chosen by Crinit (a leaky ReLU gets its shaped slope).  first_layer as in
    apply_ ('steady', 'standard' or a number q1).  Returns (model, r)."""
    r = crinit(activation, depth=depth, width=width, **kw)
    mods = []
    for i in range(depth):
        mods += [nn.Linear(d_in if i == 0 else width, width), activation_module(r)]
    mods.append(nn.Linear(width, d_out))
    model = nn.Sequential(*mods)
    apply_(model, r, first_layer=first_layer)
    return model, r
