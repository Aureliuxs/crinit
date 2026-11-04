import math

import numpy as np
import pytest

import crinit


def test_edge_of_chaos_is_chi1_equal_one():
    for act in ["tanh", "erf"]:
        for q in [0.05, 0.3, 1.0]:
            sw2, sb2 = crinit.eoc_point(act, q)
            assert abs(crinit.chi1(act, q, sw2) - 1) < 1e-10
            assert sb2 >= 0
            assert abs(crinit.fixed_point(act, sw2, sb2) - q) < 1e-8


def test_linear_point():
    r = crinit.crinit("linear", depth=10, width=64, order="exact")
    assert abs(r.sigma_w2 - crinit.sigma_c2_smooth(64)) < 1e-12
    r = crinit.crinit("linear", depth=10, width=64)
    assert abs(r.sigma_w2 - math.exp(1 / 64)) < 1e-12
    assert abs(crinit.sigma_c2_smooth(64) - 1.0158) < 1e-3


def test_tanh_default_point():
    r = crinit.crinit("tanh", depth=32, width=64)
    assert abs(r.q_star - 0.306) < 2e-3
    assert abs(r.sigma_w2_meanfield - 1.4655) < 2e-3
    assert abs(r.sigma_b2 - 0.0065) < 3e-4                # the mean-field point has 0.0130
    assert 0.03 < r.shift < 0.04
    assert abs(crinit.chi1("tanh", r.q_star, r.sigma_w2) - math.exp(-crinit.a1_b1("tanh", r.q_star)["a1"] / 64)) < 1e-10
    assert abs(crinit.compression("tanh", r.q_star, 32) - 1.0) < 1e-6
    assert abs(crinit.shift("tanh", r.q_star, 64)[0] - 0.1147) < 2e-3        # the fixed-bias displacement of Eq. (10)


def test_shift_scales_as_one_over_N():
    d64 = crinit.crinit("tanh", depth=32, width=64).shift
    d256 = crinit.crinit("tanh", depth=32, width=256).shift
    assert abs(d64 / d256 - 4) < 0.1                     # exp(-a_1/N) - 1 is 1/N up to the next order
    s64 = crinit.shift("tanh", 0.3, 64)[0]
    s256 = crinit.shift("tanh", 0.3, 256)[0]
    assert abs(s64 / s256 - 4) < 1e-6


def test_zero_bias_branch():
    r = crinit.crinit("tanh", depth=96, width=64)
    assert r.sigma_b2 == 0.0 and r.q_star > 0.143 and any("zero bias" in n for n in r.notes)
    assert abs(crinit.fixed_point("tanh", r.sigma_w2, 0.0) - r.q_star) < 1e-8


def test_linear_network_a1_b1():
    r = crinit.a1_b1("linear", 1.0)
    assert abs(r["a1"] + 1) < 1e-8 and abs(r["b1"] - 2) < 1e-8


def test_relu_family():
    r = crinit.crinit("relu", depth=20, width=64)
    assert r.alpha == 0.0 and r.sigma_b2 == 0 and abs(r.sigma_w2_meanfield - 2) < 1e-12
    assert abs(r.sigma_w2 - 2 * math.exp(5 / 128)) < 1e-12
    assert abs(crinit.sigma_c2_relu_exact(64) / (2 * math.exp(5 / 128)) - 1) < 2e-3
    r = crinit.crinit("leaky_relu", depth=20, width=64)
    assert 0 < r.alpha < 1 and abs(r.gamma - 1) < 1e-8
    assert abs(crinit.shaped_leaky_alpha(20, 1.0) - r.alpha) < 1e-12
    r = crinit.crinit("leaky_relu", depth=20, width=64, alpha=0.2)
    assert r.alpha == 0.2 and abs(r.sigma_w2_meanfield - 2 / 1.04) < 1e-12


def test_orthogonal_shift_smaller():
    g = crinit.crinit("tanh", depth=32, width=64, ensemble="gaussian").shift
    o = crinit.crinit("tanh", depth=32, width=64, ensemble="orthogonal").shift
    assert 0 < o < g


def test_gelu_moves_to_stable_branch():
    r = crinit.crinit("gelu", depth=32, width=64)
    assert crinit.slope("gelu", r.q_star, r.sigma_w2_meanfield) < 1
    assert any("unstable" in n for n in r.notes)


def test_softplus_has_no_edge():
    with pytest.raises(ValueError):
        crinit.crinit("softplus", depth=32, width=64)


def test_custom_activation_matches_builtin():
    act = crinit.custom(np.tanh, lambda u: 1 / np.cosh(u) ** 2, name="mytanh")
    a, b = crinit.crinit(act, depth=32, width=64), crinit.crinit("tanh", depth=32, width=64)
    assert abs(a.sigma_w2 - b.sigma_w2) < 1e-6 and abs(a.sigma_b2 - b.sigma_b2) < 1e-8


def test_sampler_shapes_and_variances():
    r = crinit.crinit("tanh", depth=5, width=200)
    Ws, bs = crinit.sample(r, 300, 10, rng=1)
    assert [W.shape for W in Ws] == [(300, 200)] + [(200, 200)] * 4 + [(200, 10)]
    assert abs(Ws[1].var() * 200 / r.sigma_w2 - 1) < 0.05
    assert abs(Ws[0].var() * 300 / r.first_layer_var - 1) < 0.05


def test_torch_helpers():
    torch = pytest.importorskip("torch")
    import crinit.torch_init as ct
    model, r = ct.mlp(100, 128, 8, 10, "tanh")
    layers = ct.linear_layers(model)
    assert len(layers) == 9
    assert abs(layers[3].weight.var().item() * 128 / r.sigma_w2 - 1) < 0.1
    assert abs(layers[0].weight.var().item() * 100 / r.first_layer_var - 1) < 0.1
    model, r = ct.mlp(100, 128, 8, 10, "leaky_relu")
    assert isinstance(model[1], torch.nn.LeakyReLU) and abs(model[1].negative_slope - r.alpha) < 1e-12
    r2 = ct.crinit_(model, "tanh")
    assert r2.depth == 8 and r2.width == 128


def test_exact_homogeneous_formulas():
    from crinit.exact import sigma_c2_homogeneous_exact
    for N in (16, 64, 256):
        assert abs(sigma_c2_homogeneous_exact(N, 1.0) - crinit.sigma_c2_smooth(N)) < 1e-9       # digamma
        assert abs(sigma_c2_homogeneous_exact(N, 0.0) - crinit.sigma_c2_relu_exact(N)) < 1e-12
        # leaky ReLU: the first-order formula is the leading term of the exact one
        e, f = sigma_c2_homogeneous_exact(N, 0.5), crinit.sigma_c2_leaky(N, 0.5)
        assert abs(e / f - 1) < 2.0 / N ** 2
    r = crinit.crinit("relu", depth=20, width=64, order="exact")
    assert abs(r.sigma_w2 - crinit.sigma_c2_relu_exact(64)) < 1e-12


def test_exact_smooth_close_to_first_order():
    r1 = crinit.crinit("tanh", depth=32, width=128)
    re = crinit.crinit("tanh", depth=32, width=128, order="exact")
    assert abs(re.q_star - r1.q_star) < 1e-12
    assert abs(re.shift - r1.shift) < 0.15 * r1.shift          # second order is a few per cent here
    assert any("all orders" in n for n in re.notes)


def test_first_layer_option():
    torch = pytest.importorskip("torch")
    import crinit.torch_init as ct
    r0 = crinit.crinit("tanh", depth=6, width=128)
    assert r0.first_layer_variance("steady") == r0.first_layer_var
    assert r0.first_layer_variance("standard") == r0.sigma_w2
    assert abs(r0.first_layer_variance(0.5) - (0.5 - r0.sigma_b2)) < 1e-12
    for fl, var in (("standard", r0.sigma_w2), ("steady", r0.first_layer_var), (0.5, 0.5 - r0.sigma_b2), (False, r0.sigma_w2)):
        model, r = ct.mlp(400, 128, 6, 10, "tanh", first_layer=fl)
        first = ct.linear_layers(model)[0]
        assert abs(first.weight.var().item() * 400 / var - 1) < 0.05
    Ws, _ = crinit.sample(r0, 300, 10, rng=0, first_layer="standard")
    assert abs(Ws[0].var() * 300 / r0.sigma_w2 - 1) < 0.05


def test_explicit_slope_is_kept():
    r = crinit.crinit(crinit.leaky_relu(0.2), depth=32, width=64)
    assert r.alpha == 0.2
    r2 = crinit.crinit("leaky_relu", depth=32, width=64, alpha=0.2)
    assert r2.alpha == 0.2 and abs(r2.sigma_w2 - r.sigma_w2) < 1e-12
    r3 = crinit.crinit("leaky_relu", depth=32, width=64)
    assert abs(r3.alpha - 0.586) < 2e-3                   # chosen from gamma = 1 (paper, Table S5)
