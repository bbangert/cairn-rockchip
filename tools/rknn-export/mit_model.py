#!/usr/bin/env python3
"""MIT YOLOv9 (MultimediaTechLab/YOLO) loaded without its DFL decode: the one loader every stage shares.

The model ends at the head's last anchor conv (`box`, [B, 64, h, w], channel = side * 16 + bin)
and last class conv (`cls`), so the graph an exporter traces from here has no softmax,
Conv3d or rearrange in it; decode_split.py does that on the host. torch is imported inside
the functions so the module imports in venvs without it.
"""
import os
import socket
import sys

import sources

MIT_DIR = os.path.join(sources.MIT, "YOLO")
WEIGHTS = os.path.join(sources.MIT, "weights")
OUTPUT_NAMES = ("box0", "cls0", "box1", "cls1", "box2", "cls2")  # stride 8, 16, 32


class NetworkBlocked(OSError):
    pass


def block_network():
    """Returns the function that undoes it."""
    # The MIT repo downloads weights when a path is missing and imports wandb; a silent fetch
    # would make a baseline that is not the pinned checkpoint. This guards against accidental
    # downloads by Python code and is not a sandbox: the venv having no network path after install
    # and the pinned commit are the boundary.
    wandb = os.environ.get("WANDB_MODE")
    os.environ["WANDB_MODE"] = "disabled"
    real = socket.socket.connect, socket.socket.connect_ex

    def refuse(self, address, *a):
        raise NetworkBlocked(f"network blocked: connect to {address!r}")

    socket.socket.connect = socket.socket.connect_ex = refuse

    def undo():
        socket.socket.connect, socket.socket.connect_ex = real
        if wandb is None:
            os.environ.pop("WANDB_MODE", None)
        else:
            os.environ["WANDB_MODE"] = wandb

    return undo


def checkpoint(name):
    """-> (path, sha256, MIT commit) of a checkpoint whose hash matches sources.lock."""
    path = os.path.join(WEIGHTS, f"{name}.pt")
    lock = sources.load_lock()["mit"]
    want = next((c["sha256"] for c in lock["checkpoints"] if c["name"] == f"{name}.pt"), None)
    if want is None:
        raise ValueError(f"{name}.pt is not in sources.lock")
    if not os.path.isfile(path):
        raise ValueError(f"{path}: checkpoint missing")
    sha = sources.sha256_file(path)
    if sha != want:
        raise ValueError(f"{path}: sha256 {sha}, lock says {want}")
    return path, sha, lock["commit"]


def _build(name, repo_dir):
    from omegaconf import OmegaConf

    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    from yolo.model.yolo import create_model

    cfg = OmegaConf.load(os.path.join(repo_dir, "yolo", "config", "model", f"{name}.yaml"))
    cfg.model.auxiliary = {}  # deploy_utils.FastModelLoader "deploy": the aux branch is training-only
    # weight_path=False skips both the load and the download branch of create_model
    return create_model(cfg, weight_path=False, class_num=80)


def _fit_class_neck(net, ckpt):
    # The released v9-t checkpoint has class_conv width 80 where the pinned code builds
    # max(first_neck, min(2 * nc, 128)) = 128, so MIT's own load would warn and leave the class
    # heads random. Only this width follows the checkpoint; every other shape is still checked.
    import torch

    fitted = {}
    for idx, layer in enumerate(net.model):
        if getattr(layer, "layer_type", None) != "MultiheadDetection":
            continue
        for j, h in enumerate(layer.heads):
            key = f"{idx}.heads.{j}.class_conv.0.conv.weight"
            if key not in ckpt:
                continue
            neck, cin = int(ckpt[key].shape[0]), h.class_conv[0].conv.in_channels
            if neck == h.class_conv[0].conv.out_channels:
                continue
            nc, conv = h.class_conv[2].out_channels, type(h.class_conv[0])
            h.class_conv = torch.nn.Sequential(conv(cin, neck, 3), conv(neck, neck, 3), torch.nn.Conv2d(neck, nc, 1))
            fitted[f"{idx}.heads.{j}"] = neck
    return fitted


def _load_strict(net, weights_path):
    import torch

    ckpt = torch.load(weights_path, map_location="cpu", weights_only=True)
    fitted = _fit_class_neck(net, ckpt)
    # save_load_weights compares against net.model.state_dict(), keys "<layer>.<...>"; it only
    # warns on a mismatch and leaves random weights behind.
    want = net.model.state_dict()
    missing = sorted(k for k in want if k not in ckpt)
    bad = sorted(k for k in want if k in ckpt and want[k].shape != ckpt[k].shape)
    if missing or bad:
        raise ValueError(f"{weights_path}: {len(missing)} keys missing from the checkpoint, e.g. {missing[:5]}; "
                         f"{len(bad)} with a different shape, e.g. {bad[:5]}")
    unused = sorted(k for k in ckpt if k not in want)
    n_layers = len(net.model)
    # aux layers are appended after the main head, so kept layers keep their indices
    stray = [k for k in unused if not (k.split(".")[0].isdigit() and int(k.split(".")[0]) >= n_layers)]
    if stray:
        raise ValueError(f"{weights_path}: {len(stray)} checkpoint keys belong to layers the model has, "
                         f"e.g. {stray[:5]}")
    net.model.load_state_dict({k: ckpt[k] for k in want}, strict=True)
    return {"loaded": len(want), "unused": len(unused), "unused_layers": sorted({int(k.split(".")[0]) for k in unused}),
            "class_neck_fitted": fitted}


def _cut(net, sigmoid):
    import torch

    class Cut(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = net
            self.sigmoid = sigmoid
            last = net.model[-1]
            if last.layer_type != "MultiheadDetection":
                raise ValueError(f"expected the main detection head last, found {last.layer_type}")
            if any(layer.external for layer in net.model):
                raise ValueError("a layer takes external inputs; the cut only follows the main chain")

        def forward(self, x):
            # YOLO.forward's loop up to the head; the head's own forward would add the Anchor2Vec decode
            y = {0: x}
            for index, layer in enumerate(self.net.model[:-1], start=1):
                inp = [y[i] for i in layer.source] if isinstance(layer.source, list) else y[layer.source]
                y[-1] = out = layer(inp)
                if layer.usable:
                    y[index] = out
            outs = []
            head = self.net.model[-1]
            for i, h in zip(head.source, head.heads):
                feat = y[i]
                cls = h.class_conv(feat)
                outs += [h.anchor_conv(feat), cls.sigmoid() if self.sigmoid else cls]
            return tuple(outs)

    return Cut()


def load(name, weights_path, repo_dir, sigmoid=True, fuse=False):
    if name not in ("v9-t", "v9-s", "v9-m"):
        raise ValueError(f"unknown model {name!r}")
    net = _build(name, repo_dir)
    report = _load_strict(net, weights_path)
    net.eval()
    if fuse:
        import fuse_repconv

        report["repconv_fused"] = fuse_repconv.fuse(net)
    cut = _cut(net, sigmoid).eval()
    cut.load_report = report
    return cut
