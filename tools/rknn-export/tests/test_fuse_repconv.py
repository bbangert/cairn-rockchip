import numpy as np
import pytest

import fuse_repconv as fr


def conv2d(x, w, b=None, pad=0):
    """x [C, H, W], w [O, C, k, k] -> [O, H', W'] (cross-correlation, as torch)."""
    x = np.pad(x, ((0, 0), (pad, pad), (pad, pad)))
    win = np.lib.stride_tricks.sliding_window_view(x, w.shape[-2:], axis=(1, 2))  # [C, H', W', k, k]
    out = np.einsum("chwij,ocij->ohw", win, w)
    return out if b is None else out + b[:, None, None]


def bn(y, gamma, beta, mean, var, eps):
    s = lambda v: v[:, None, None]
    return (y - s(mean)) / np.sqrt(s(var) + eps) * s(gamma) + s(beta)


def test_fold_and_merge_reproduce_the_two_branch_sum():
    rng = np.random.default_rng(1)
    cin, cout, eps = 3, 4, 1e-3
    x = rng.normal(size=(cin, 9, 7))
    w3, w1 = rng.normal(size=(cout, cin, 3, 3)), rng.normal(size=(cout, cin, 1, 1))
    p3 = [rng.uniform(0.5, 1.5, cout), rng.normal(size=cout), rng.normal(size=cout), rng.uniform(0.2, 2, cout)]
    p1 = [rng.uniform(0.5, 1.5, cout), rng.normal(size=cout), rng.normal(size=cout), rng.uniform(0.2, 2, cout)]
    want = bn(conv2d(x, w3, pad=1), *p3, eps) + bn(conv2d(x, w1), *p1, eps)
    w, b = fr.merge(*fr.fold_bn(w3, *p3, eps), *fr.fold_bn(w1, *p1, eps))
    np.testing.assert_allclose(conv2d(x, w, b, pad=1), want, atol=1e-12)


def test_eps_matters():
    # MIT's BN eps is 1e-3, not torch's default 1e-5: folding with the wrong one is a visible error
    w = np.ones((2, 1, 1, 1))
    args = (np.ones(2), np.zeros(2), np.zeros(2), np.full(2, 0.01))
    assert abs(fr.fold_bn(w, *args, 1e-3)[0][0, 0, 0, 0] - fr.fold_bn(w, *args, 1e-5)[0][0, 0, 0, 0]) > 0.04


def test_pad_kernel_centres_and_rejects_bad_sizes():
    k = fr.pad_kernel(np.full((1, 1, 1, 1), 7.0), 3)
    assert k[0, 0].tolist() == [[0, 0, 0], [0, 7, 0], [0, 0, 0]]
    assert fr.pad_kernel(np.ones((1, 1, 3, 3)), 3).shape == (1, 1, 3, 3)
    with pytest.raises(ValueError):
        fr.pad_kernel(np.ones((1, 1, 2, 2)), 3)
    with pytest.raises(ValueError):
        fr.pad_kernel(np.ones((1, 1, 3, 3)), 1)


def test_fuse_matches_a_synthetic_repconv_in_torch():
    torch = pytest.importorskip("torch")
    nn = torch.nn

    class Conv(nn.Module):
        def __init__(self, cin, cout, k):
            super().__init__()
            self.conv = nn.Conv2d(cin, cout, k, padding=k // 2, bias=False)
            self.bn = nn.BatchNorm2d(cout, eps=1e-3)

        def forward(self, x):
            return self.bn(self.conv(x))

    class RepConv(nn.Module):
        def __init__(self, cin, cout):
            super().__init__()
            self.act = nn.SiLU()
            self.conv1, self.conv2 = Conv(cin, cout, 3), Conv(cin, cout, 1)

        def forward(self, x):
            return self.act(self.conv1(x) + self.conv2(x))

    torch.manual_seed(0)
    net = nn.Sequential(RepConv(3, 5), nn.Sequential(RepConv(5, 4)), nn.Conv2d(4, 2, 1)).double()
    for m in net.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.running_mean.normal_()
            m.running_var.uniform_(0.2, 2)
            m.weight.data.uniform_(0.5, 1.5)
            m.bias.data.normal_()
    net.eval()
    x = torch.randn(1, 3, 11, 9, dtype=torch.float64)
    want = net(x)
    assert fr.fuse(net) == 2
    assert not any(type(m).__name__ == "RepConv" for m in net.modules())
    assert (net(x) - want).abs().max() < 1e-10
    assert sum(isinstance(m, nn.BatchNorm2d) for m in net.modules()) == 0


def test_tolerance_gate_flags_nan_inf_leftovers_and_real_differences():
    ok = [(1e-4, 30.0)] * 6
    assert fr.failures(ok, 0) == []
    assert fr.failures([(3e-2, 30.0)] + ok[1:], 0) == []  # inside 1e-3 x max |plain|
    assert fr.failures([(3.5e-2, 30.0)] + ok[1:], 0)  # just outside
    assert fr.failures([(float("nan"), 30.0)] + ok[1:], 0)
    assert fr.failures([(float("inf"), 1.0)] + ok[1:], 0)
    assert "2 RepConv left" in fr.failures(ok, 2)[0]


def _torch_conv_bn(torch, cin, cout, k, eps=1e-3, stride=1, padding=None, **conv_kw):
    nn = torch.nn

    class Conv(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(cin, cout, k, stride, k // 2 if padding is None else padding,
                                  **{"bias": False, **conv_kw})
            self.bn = nn.BatchNorm2d(cout, eps=eps)

    return Conv()


def _rep(torch, big, small):
    class RepConv(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.act = torch.nn.SiLU()
            self.conv1, self.conv2 = big, small

    return RepConv()


def test_fold_conv_uses_the_bns_own_eps():
    torch = pytest.importorskip("torch")
    block = _torch_conv_bn(torch, 2, 3, 1, eps=0.5)
    block.bn.running_var.fill_(1.0)
    w, _ = fr._fold_conv(block)
    expect = block.conv.weight.detach().double().numpy() / np.sqrt(1.0 + 0.5)
    np.testing.assert_allclose(w, expect, rtol=1e-12)
    block.bn.eps = 1e-5
    assert abs(fr._fold_conv(block)[0] - w).max() > 1e-2  # so the eps read is the BN's, not a constant


@pytest.mark.parametrize("conv_kw", [{"bias": True}, {"groups": 2}, {"dilation": 2}])
def test_fold_conv_refuses_bias_groups_and_dilation(conv_kw):
    torch = pytest.importorskip("torch")
    block = _torch_conv_bn(torch, 4, 4, 3, **conv_kw)
    with pytest.raises(ValueError, match="bias-free ungrouped undilated"):
        fr._fold_conv(block)


@pytest.mark.parametrize("big_kw,small_kw", [
    ({"stride": 2}, {}),  # strides differ
    ({}, {"padding": 1}),  # the 1x1 branch pads
    ({"padding": 0}, {}),  # the 3x3 branch does not read the same window
])
def test_fuse_one_refuses_branches_that_do_not_line_up(big_kw, small_kw):
    torch = pytest.importorskip("torch")
    rep = _rep(torch, _torch_conv_bn(torch, 2, 3, 3, **big_kw), _torch_conv_bn(torch, 2, 3, 1, **small_kw))
    with pytest.raises(ValueError, match="do not line up"):
        fr.fuse_one(rep)


def test_negative_image_count_is_rejected_and_zero_kept(capsys):
    with pytest.raises(SystemExit):
        fr.main(["v9-t", "--images", "-1"])
    assert "is below 0" in capsys.readouterr().err
    assert fr.count("0") == 0 and fr.count("5") == 5
