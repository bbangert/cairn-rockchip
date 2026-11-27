#!/usr/bin/env python3
"""Fold MIT's two-branch RepConv (3x3 Conv+BN plus 1x1 Conv+BN, summed, then the activation) into one conv, in float.

Done on the loaded module so the MIT clone stays untouched. The fold maths is numpy so the
tests run without torch; the CLI runs both models in float32 on a random and K real inputs and
exits non-zero when a RepConv is left or an output differs by more than TOLERANCE x max(1, max |plain|).
usage: fuse_repconv.py {v9-t|v9-s|v9-m} [--images K]
"""
import argparse, collections, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# Float32 fold error measured at 7e-4 on logits of magnitude ~30 (v9-s); the float64 fold is exact to ~3e-12,
# so a wrong weight or eps is orders above this.
TOLERANCE = 1e-3


def fold_bn(weight, gamma, beta, mean, var, eps):
    """Conv (no bias) followed by BatchNorm in eval mode -> (weight, bias) of the equivalent conv."""
    scale = np.asarray(gamma, np.float64) / np.sqrt(np.asarray(var, np.float64) + eps)
    w = np.asarray(weight, np.float64) * scale[:, None, None, None]
    return w, np.asarray(beta, np.float64) - np.asarray(mean, np.float64) * scale


def pad_kernel(weight, size):
    """Centre a k x k kernel in a size x size one: with 'same' padding the two convs read the same pixels."""
    k = weight.shape[-1]
    if weight.shape[-2] != k or size < k or (size - k) % 2:
        raise ValueError(f"cannot centre a {weight.shape[-2]}x{k} kernel in {size}x{size}")
    p = (size - k) // 2
    return np.pad(weight, ((0, 0), (0, 0), (p, p), (p, p)))


def merge(w_big, b_big, w_small, b_small):
    return w_big + pad_kernel(w_small, w_big.shape[-1]), b_big + b_small


def _fold_conv(block):
    """MIT Conv (conv, bn) -> (weight, bias) float64 numpy, using that BN's own eps and running stats."""
    c, bn = block.conv, block.bn
    if c.bias is not None or c.groups != 1 or c.dilation != (1, 1) or not bn.track_running_stats:
        raise ValueError("fold expects a bias-free ungrouped undilated conv before a BN with running stats")
    n = lambda t: t.detach().double().numpy()
    return fold_bn(n(c.weight), n(bn.weight), n(bn.bias), n(bn.running_mean), n(bn.running_var), bn.eps)


def fuse_one(rep):
    import torch

    big, small = rep.conv1, rep.conv2
    k = big.conv.kernel_size[0]
    # the 1x1 branch is only a centred 3x3 tap if both read the same window at the same stride
    if (big.conv.kernel_size != (k, k) or small.conv.kernel_size != (1, 1) or big.conv.stride != small.conv.stride
            or big.conv.padding != (k // 2, k // 2) or small.conv.padding != (0, 0)):
        raise ValueError(f"RepConv branches do not line up: {big.conv} vs {small.conv}")
    w, b = merge(*_fold_conv(big), *_fold_conv(small))
    src = big.conv
    conv = torch.nn.Conv2d(src.in_channels, src.out_channels, src.kernel_size, src.stride, src.padding, bias=True,
                            dtype=src.weight.dtype)
    with torch.no_grad():
        conv.weight.copy_(torch.from_numpy(w))
        conv.bias.copy_(torch.from_numpy(b))

    class Fused(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv, self.act = conv, rep.act

        def forward(self, x):
            return self.act(self.conv(x))

    return Fused().eval()


def fuse(module):
    """Replace every RepConv under `module` in place; -> how many."""
    n = 0
    for parent in list(module.modules()):
        for name, child in list(parent.named_children()):
            if type(child).__name__ == "RepConv":
                setattr(parent, name, fuse_one(child))
                n += 1
    return n


def batchnorm_parents(module):
    """Counts of the module types that directly hold a BatchNorm."""
    import torch

    return collections.Counter(type(m).__name__ for m in module.modules()
                               if any(isinstance(c, torch.nn.BatchNorm2d) for c in m.children()))


def failures(worst, left):
    """worst: [(max abs difference, max |plain|)] per output -> why the check fails (empty: it passes)."""
    bad = [f"{left} RepConv left"] if left else []
    for i, (d, m) in enumerate(worst):
        if not (np.isfinite(d) and np.isfinite(m) and d <= TOLERANCE * max(1.0, m)):
            bad.append(f"output {i}: max |diff| {d} (max |plain| {m}) above {TOLERANCE} x max(1, max |plain|)")
    return bad


def count(text):
    # a negative K would reach a slice and select all but the last K images
    n = int(text)
    if n < 0:
        raise argparse.ArgumentTypeError(f"{text} is below 0")
    return n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("usage:")[0].strip())
    ap.add_argument("name", choices=("v9-t", "v9-s", "v9-m"))
    ap.add_argument("--images", type=count, default=5, help="real val2017 images besides the random input")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args(argv)
    sys.path.insert(0, HERE)  # python -I leaves the script directory off sys.path

    import mit_model

    undo = mit_model.block_network()
    try:
        return _check(a)
    finally:
        undo()


def _check(a):
    import coco_run, mit_model, sources
    import torch

    torch.set_num_threads(a.threads)
    weights, _, _ = mit_model.checkpoint(a.name)
    plain = mit_model.load(a.name, weights, mit_model.MIT_DIR)
    fused = mit_model.load(a.name, weights, mit_model.MIT_DIR, fuse=True)
    n = fused.load_report["repconv_fused"]
    left = sum(type(m).__name__ == "RepConv" for m in fused.modules())
    ann = os.path.join(sources.COCO, "annotations", "instances_val2017.json")
    xs = [torch.randn(1, 3, 640, 640).clamp(0, 1)]
    for im in coco_run._images(ann, a.images):
        canvas, _ = coco_run.prepare(os.path.join(sources.COCO, "val2017", im["file_name"]), 640, "origin", "bilinear-cv2")
        xs.append(torch.from_numpy(canvas).permute(2, 0, 1).unsqueeze(0).float().div(255))
    worst = [(0.0, 0.0)] * 6
    with torch.inference_mode():
        for x in xs:
            for i, (p, f) in enumerate(zip(plain(x), fused(x))):
                # max() drops a NaN, so a non-finite difference is carried as inf
                d = float((p - f).abs().max())
                m = float(p.abs().max())
                worst[i] = (max(worst[i][0], d if np.isfinite(d) else float("inf")), max(worst[i][1], m))
    print(f"{a.name}: {n} RepConv fused, {left} left; max |fused - plain| over 1 random + {len(xs) - 1} real inputs")
    for name, (d, m) in zip(mit_model.OUTPUT_NAMES, worst):
        print(f"  {name}: {d:.3g} (max |plain| {m:.3g})")
    print(f"  BatchNorm holders after fusing: {dict(batchnorm_parents(fused))}")
    bad = failures(worst, left)
    if bad:
        sys.exit("fusion check failed: " + "; ".join(bad))


if __name__ == "__main__":
    main()
