#!/usr/bin/env python3
"""Host-side decode, NMS and box un-mapping for a YOLOv9 exported with the DFL decode cut out.

One implementation shared by the scoring runners (pytorch and onnx so far; the quantised
stages are to use it too), so a difference between stages is the model and never the decode. numpy + stdlib only: it
runs in venvs that have torch, onnxruntime or the RKNN toolkit, and in none of them
together. Maths follows MultimediaTechLab/YOLO (Vec2Box, Anchor2Vec, PadAndResize);
NMS follows the reference YOLOv9 validation protocol.

Per level: box [1, 64, h, w] raw logits, channel = side * 16 + bin, sides left, top,
right, bottom; cls [1, 80, h, w] already sigmoided.
"""
import numbers
from typing import NamedTuple

import numpy as np

from prep_calib import scaled_dim  # cv2 is imported there only under __main__

STRIDES = (8, 16, 32)
BINS = 16


def _wh(size):
    w, h = (size, size) if isinstance(size, numbers.Integral) else size
    if w <= 0 or h <= 0:
        raise ValueError(f"input size must be positive, got {size!r}")
    return int(w), int(h)


def parse_size(text):
    """'640' or 'WxH' -> (w, h), each a positive multiple of the coarsest stride."""
    parts = text.split("x")
    if len(parts) > 2 or not all(p.isdigit() for p in parts):
        raise ValueError(f"size {text!r}: want 640 or WxH, e.g. 640x384")
    w, h = int(parts[0]), int(parts[-1])
    if w == 0 or h == 0 or w % STRIDES[-1] or h % STRIDES[-1]:
        raise ValueError(f"size {text!r}: width and height must be positive multiples of {STRIDES[-1]}")
    return w, h


def size_suffix(size):
    """Stem part that keeps a non-default input size from overwriting the 640x640 result."""
    w, h = _wh(size)
    return "" if (w, h) == (640, 640) else f"-{w}x{h}"


def decode(boxes, clss, size, strides=STRIDES):
    """-> xyxy [A, 4] float32 in input pixels, scores [A, nc]; anchors by level, then y * w + x."""
    w, h = _wh(size)
    if not (len(boxes) == len(clss) == len(strides)):
        raise ValueError(f"need one box and one cls tensor per stride {tuple(strides)}, "
                         f"got {len(boxes)} and {len(clss)}")
    xyxy, scores = [], []
    for i, (stride, b, c) in enumerate(zip(strides, boxes, clss)):
        gh, gw = h // stride, w // stride
        if b.shape != (1, 4 * BINS, gh, gw):
            raise ValueError(f"box[{i}] shape {tuple(b.shape)}, want (1, {4 * BINS}, {gh}, {gw}) "
                             f"for stride {stride} at {w}x{h}")
        if c.ndim != 4 or c.shape[0] != 1 or c.shape[2:] != (gh, gw):
            raise ValueError(f"cls[{i}] shape {tuple(c.shape)}, want (1, nc, {gh}, {gw}) "
                             f"for stride {stride} at {w}x{h}")
        # float64 leaves the final float32 cast as the only rounding; the float32 reference
        # softmax differs from this by about 1e-4 px at most. np.array, not asarray: the
        # in-place maths below must not write through to a float64 caller's tensor.
        logits = np.array(b, dtype=np.float64).reshape(4, BINS, gh, gw)
        logits -= logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        p /= p.sum(axis=1, keepdims=True)
        ltrb = np.tensordot(np.arange(BINS, dtype=np.float64), p, axes=(0, 1)) * stride  # [4, gh, gw]
        ys, xs = np.mgrid[0:gh, 0:gw]
        ax, ay = (xs + 0.5) * stride, (ys + 0.5) * stride
        xyxy.append(np.stack([ax - ltrb[0], ay - ltrb[1], ax + ltrb[2], ay + ltrb[3]], axis=-1)
                    .reshape(-1, 4).astype(np.float32))
        scores.append(np.asarray(c, dtype=np.float32)[0].transpose(1, 2, 0).reshape(gh * gw, -1))
    return np.concatenate(xyxy), np.concatenate(scores)


def _greedy(box, iou, limit=None):
    """Keep indices of score-descending `box` under strict `> iou` suppression, as torchvision.ops.nms.

    With `limit`, stops after that many keeps: a prefix of the unlimited result, since a later keep never undoes an earlier one.
    """
    area = (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
    alive = np.arange(len(box))
    keep = []
    while alive.size and (limit is None or len(keep) < limit):
        i = alive[0]
        keep.append(i)
        rest = alive[1:]
        if not rest.size:
            break
        lt = np.maximum(box[i, :2], box[rest, :2])
        rb = np.minimum(box[i, 2:], box[rest, 2:])
        inter = np.clip(rb - lt, 0, None).prod(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            ov = inter / (area[i] + area[rest] - inter)
        alive = rest[~(ov > iou)]
    return np.asarray(keep, dtype=np.int64)


def nms(xyxy, scores, conf=0.001, iou=0.7, max_nms=30000, max_det=300):
    """Multi-label, class-aware NMS -> boxes [N, 4] float32, scores [N] float32, class ids [N] int64, by score descending."""
    a, c = np.nonzero(scores > conf)
    s = scores[a, c]
    order = np.argsort(-s, kind="stable")[:max_nms]
    a, c, s = a[order], c[order], s[order]
    b = xyxy[a]
    kept = []
    for cls in np.unique(c):
        idx = np.nonzero(c == cls)[0]  # still score-descending
        # a class never contributes more than max_det to the final top max_det
        kept.append(idx[_greedy(b[idx], iou, max_det)])
    kept = np.concatenate(kept) if kept else np.empty(0, np.int64)
    kept = kept[np.argsort(-s[kept], kind="stable")][:max_det]
    return b[kept].astype(np.float32), s[kept].astype(np.float32), c[kept].astype(np.int64)


class Placement(NamedTuple):
    src_w: int
    src_h: int
    new_w: int  # resized content, what the preprocessor resizes the image to
    new_h: int
    pad_left: int
    pad_top: int
    scale_x: float  # input pixels per source pixel; what unmap divides by
    scale_y: float


def placement(src_w, src_h, size, mode):
    w, h = _wh(size)
    if src_w <= 0 or src_h <= 0:
        raise ValueError(f"source size must be positive, got {src_w}x{src_h}")
    if mode == "centre":
        # MIT PadAndResize: one scale, un-mapped by that scale rather than new / src.
        scale = min(w / src_w, h / src_h)
        new_w, new_h = int(src_w * scale), int(src_h * scale)
        if new_w < 1 or new_h < 1:
            raise ValueError(f"{src_w}x{src_h} collapses to {new_w}x{new_h} at {w}x{h}")
        return Placement(src_w, src_h, new_w, new_h, (w - new_w) // 2, (h - new_h) // 2, scale, scale)
    if mode == "origin":
        # Core's Fit::fit builds the content and Fit::projection / Projection::unproject
        # un-map it by the per-axis new / src (even-floored), not by the single ratio.
        ratio = min(w / src_w, h / src_h)
        new_w, new_h = scaled_dim(src_w, ratio, w), scaled_dim(src_h, ratio, h)
        return Placement(src_w, src_h, new_w, new_h, 0, 0, new_w / src_w, new_h / src_h)
    if mode == "stretch":
        return Placement(src_w, src_h, w, h, 0, 0, w / src_w, h / src_h)
    raise ValueError(f"unknown placement {mode!r}; want centre, origin or stretch")


def unmap(boxes, p):
    """Input-pixel xyxy [N, 4] -> original-image pixels clipped to the source frame, float64."""
    b = np.array(boxes, dtype=np.float64).reshape(-1, 4)
    b[:, [0, 2]] = (b[:, [0, 2]] - p.pad_left) / p.scale_x
    b[:, [1, 3]] = (b[:, [1, 3]] - p.pad_top) / p.scale_y
    b[:, [0, 2]] = b[:, [0, 2]].clip(0, p.src_w)
    b[:, [1, 3]] = b[:, [1, 3]].clip(0, p.src_h)
    return b
