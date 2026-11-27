#!/usr/bin/env python3
"""COCO bbox scorer shared by every runner, so every stage is scored by the same code.

Detections interchange as an .npz (see save_detections): original-image pixels, dense
COCO-80 class ids, plus the ids of every image that was run -- an image with no
detections still counts, and a subset run scores only the images it ran.
usage: score_coco.py ANN_JSON DETS_NPZ [--out RESULT_JSON] [--label NAME]
"""
import argparse, contextlib, hashlib, io, json
from typing import NamedTuple

import numpy as np

# dense 0..79 -> COCO category id (1..90, ten gaps)
COCO80_TO_91 = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
    46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
    67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90,
]
STAT_NAMES = ("AP", "AP50", "AP75", "APs", "APm", "APl")


class Detections(NamedTuple):
    image_id: np.ndarray  # int64 [N]
    class_id: np.ndarray  # int64 [N]
    score: np.ndarray     # float32 [N]
    box: np.ndarray       # float32 [N, 4] xyxy
    images: np.ndarray    # int64 [M]


def _check_columns(image_id, class_id, score, box):
    # zip over mismatched columns truncates silently and would drop detections from the AP
    n = np.shape(image_id)[0] if np.ndim(image_id) == 1 else None
    if n is None or any(np.shape(a) != (n,) for a in (class_id, score)) or np.shape(box) != (n, 4):
        raise ValueError(
            "image_id, class_id and score must be 1-D of one length N and box [N, 4]; got shapes "
            f"{np.shape(image_id)}, {np.shape(class_id)}, {np.shape(score)}, {np.shape(box)}")


def save_detections(path, image_id, class_id, score, box, images):
    box = np.asarray(box, np.float32)
    if box.size == 0:
        box = box.reshape(0, 4)
    _check_columns(image_id, class_id, score, box)
    np.savez_compressed(path, image_id=np.asarray(image_id, np.int64), class_id=np.asarray(class_id, np.int64),
                        score=np.asarray(score, np.float32), box=box, images=np.asarray(images, np.int64))


def load_detections(path):
    with np.load(path) as z:
        d = Detections(*(z[k] for k in Detections._fields))
    _check_columns(*d[:4])
    return d


def _quiet(fn, *a):
    # pycocotools narrates on stdout; the caller prints the one line that matters
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a)


def evaluate(ann_path, dets):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    _check_columns(*dets[:4])
    gt = _quiet(COCO, ann_path)
    images = np.unique(dets.images)
    if not len(images):
        raise ValueError("no images were run; there is nothing to score")
    unknown = set(images.tolist()) - set(gt.getImgIds())
    if unknown:
        raise ValueError(f"{len(unknown)} run images are not in the annotation file, e.g. {sorted(unknown)[:3]}")
    stray = set(np.unique(dets.image_id).tolist()) - set(images.tolist())
    if stray:
        raise ValueError(f"detections for images not listed in `images`, e.g. {sorted(stray)[:3]}")
    if len(dets.class_id) and not (0 <= dets.class_id.min() and dets.class_id.max() < len(COCO80_TO_91)):
        raise ValueError("class_id outside 0..79")

    out = {"n_images": int(len(images)), "n_detections": int(len(dets.score)),
           "full_set": set(gt.getImgIds()) <= set(images.tolist())}
    if len(dets.score):
        box = dets.box.astype(np.float64)
        wh = box[:, 2:] - box[:, :2]
        xywh = np.round(np.concatenate([box[:, :2], wh], axis=1), 3)  # val.py rounding
        cat = np.asarray(COCO80_TO_91)[dets.class_id]
        score = np.round(dets.score.astype(np.float64), 5)
        res = [{"image_id": int(i), "category_id": int(c), "bbox": b, "score": float(s)}
               for i, c, b, s in zip(dets.image_id, cat, xywh.tolist(), score)]
        dt = _quiet(gt.loadRes, res)
    else:
        # loadRes([]) raises IndexError; an empty result set keeps COCOeval deciding which slices have no ground truth
        dt = COCO()
        dt.dataset = {"images": gt.dataset["images"], "categories": gt.dataset["categories"], "annotations": []}
        _quiet(dt.createIndex)
    ev = COCOeval(gt, dt, "bbox")
    ev.params.imgIds = images.tolist()
    _quiet(ev.evaluate)
    _quiet(ev.accumulate)
    _quiet(ev.summarize)
    stats = [float(x) for x in ev.stats[:6]]
    # pycocotools reports -1 for "no ground truth in this slice"; null keeps that from reading as a score
    out.update({k: (round(v * 100, 2) if v >= 0 else None) for k, v in zip(STAT_NAMES, stats)})
    out["raw"] = {k: (v if v >= 0 else None) for k, v in zip(STAT_NAMES, stats)}
    return out


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fmt(v):
    return "n/a" if v is None else f"{v:.2f}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("usage:")[0].strip())
    ap.add_argument("ann")
    ap.add_argument("dets")
    ap.add_argument("--out")
    ap.add_argument("--label", default="")
    a = ap.parse_args(argv)
    r = evaluate(a.ann, load_detections(a.dets))
    tail = "" if r["full_set"] else " [subset of the annotation images]"
    print(f"{a.label + ': ' if a.label else ''}AP {_fmt(r['AP'])} AP50 {_fmt(r['AP50'])} AP75 {_fmt(r['AP75'])} "
          f"({r['n_images']} images, {r['n_detections']} dets){tail}")
    if a.out:
        r.update({"label": a.label, "dets_sha256": _sha256(a.dets), "ann_sha256": _sha256(a.ann)})
        with open(a.out, "w") as f:
            json.dump(r, f, indent=1)
            f.write("\n")
    return r


if __name__ == "__main__":
    main()
