#!/usr/bin/env python3
"""Host-side decode, NMS and box un-mapping for a YOLOv9 exported with the DFL decode cut out.

One implementation shared by the scoring runners (pytorch, onnx and the toolkit's simulator),
so a difference between stages is the model and never the decode. numpy + stdlib only: it
runs in venvs that have torch, onnxruntime or the RKNN toolkit, and in none of them
together. Maths follows MultimediaTechLab/YOLO (Vec2Box, Anchor2Vec, PadAndResize);
NMS follows the reference YOLOv9 validation protocol; `core_nms` follows instead what core's
`SplitDfl` decode puts on the wire (plugins/cairn-detect/src/infer/heads.rs). Scored at a 0.001
floor that is core's decode, not what a camera emits: that is cut at its `min_score` (0.5 by default).

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


def decode(boxes, clss, size, strides=STRIDES, dtype=np.float32):
    """-> xyxy [A, 4] `dtype` in input pixels, scores [A, nc]; anchors by level, then y * w + x."""
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
                    .reshape(-1, 4).astype(dtype))
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


RULES = ("reference", "core")
CORE_MAX_EXTENT = 4.0  # heads.rs MAX_EXTENT: a box wider or taller than 4x the input is refused


def round_half_away(x, places):
    """Rust's f64::round at `places` decimals; numpy rounds half to even. trunc keeps the fraction exact."""
    v = np.asarray(x, np.float64) * 10.0 ** places
    t = np.trunc(v)
    return (t + np.where(np.abs(v - t) >= 0.5, np.sign(v), 0)) / 10.0 ** places


def _wire(box, p):
    """heads.rs wire_bbox over un-projected boxes: normalised, clamped, 4 dp -> (x0, y0, w, h), alive."""
    n = (np.array(box, np.float64).reshape(-1, 4) - [p.pad_left, p.pad_top, p.pad_left, p.pad_top])
    n /= [p.scale_x * p.src_w, p.scale_y * p.src_h, p.scale_x * p.src_w, p.scale_y * p.src_h]
    n = n.clip(0, 1)
    w = round_half_away(np.maximum(n[:, 2] - n[:, 0], 0), 4)
    h = round_half_away(np.maximum(n[:, 3] - n[:, 1], 0), 4)
    return round_half_away(n[:, 0], 4), round_half_away(n[:, 1], 4), w, h, (w != 0) & (h != 0)


def _iou_ge(box, i, rest, iou):
    # heads.rs iou: areas floored at 0, an empty union is IoU 0; suppression at >=, not >
    a, b = box[i], box[rest]
    inter = np.prod(np.clip(np.minimum(a[2:], b[:, 2:]) - np.maximum(a[:2], b[:, :2]), 0, None), axis=1)
    area = lambda r: np.clip(r[..., 2] - r[..., 0], 0, None) * np.clip(r[..., 3] - r[..., 1], 0, None)
    union = area(a) + area(b) - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        ov = np.where(union > 0, inter / union, 0.0)
    return ov >= iou


def core_nms(xyxy, scores, p, size, floor=0.001, iou=0.45, max_cand=300, max_det=32):
    """-> (boxes [N, 4] float64 in source pixels, scores [N] rounded to 3 dp, class ids [N]), as core emits them.

    One class per anchor (the last of tied maxima, heads.rs argmax), score >= floor, boxes past
    CORE_MAX_EXTENT or that the wire would refuse (wholly in the pad) dropped before the top
    `max_cand` by score, class-aware NMS in input pixels, then the top `max_det`. Sorts are stable,
    as Rust's sort_by: with 8-bit scores, ties are common and anchor order breaks them.
    """
    w, h = _wh(size)
    scores = np.asarray(scores, np.float64)
    nc = scores.shape[1]
    c = nc - 1 - scores[:, ::-1].argmax(axis=1)
    s = scores[np.arange(len(c)), c]
    box = np.asarray(xyxy, np.float64)
    ok = (s >= floor) & np.isfinite(box).all(axis=1)
    ok &= (box[:, 2] - box[:, 0] <= CORE_MAX_EXTENT * w) & (box[:, 3] - box[:, 1] <= CORE_MAX_EXTENT * h)
    box, c, s = box[ok], c[ok], s[ok]
    box, c, s = (t[_wire(box, p)[4]] for t in (box, c, s))
    order = np.argsort(-s, kind="stable")[:max_cand]
    box, c, s = box[order], c[order], s[order]
    keep = np.ones(len(s), bool)
    for i in range(len(s)):
        if keep[i]:
            rest = np.nonzero(keep[i + 1:] & (c[i + 1:] == c[i]))[0] + i + 1
            keep[rest[_iou_ge(box, i, rest, iou)]] = False
    box, c, s = box[keep][:max_det], c[keep][:max_det], s[keep][:max_det]  # already score-descending
    x0, y0, bw, bh, _ = _wire(box, p)
    px = np.stack([x0 * p.src_w, y0 * p.src_h, (x0 + bw) * p.src_w, (y0 + bh) * p.src_h], axis=1)
    return px.reshape(-1, 4), round_half_away(s.clip(0, 1), 3), c.astype(np.int64)


def detections(xyxy, scores, p, size, rules="reference"):
    """Decoded anchors (float64 boxes, as core keeps them) -> source-pixel boxes, scores, class ids under one of RULES.
    The reference rules see float32 boxes, the precision their baselines were scored at."""
    if rules == "reference":
        box, score, cls = nms(np.asarray(xyxy, np.float32), scores)
        return unmap(box, p), score, cls
    if rules == "core":
        return core_nms(xyxy, scores, p, size)
    raise ValueError(f"unknown rules {rules!r}; want one of {RULES}")
