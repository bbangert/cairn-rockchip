#!/usr/bin/env python3
"""COCO image loop shared by every stage's runner: preprocess, infer, decode, NMS, un-map, save.

The runner supplies `infer(canvas) -> (box0, cls0, box1, cls1, box2, cls2)` for one uint8
HWC RGB canvas and does its own normalising, since later stages feed uint8. Decode + NMS is
slow at conf 0.001, so it runs in a process pool while inference stays in the calling process.
cv2 and PIL are imported only by `prepare`.
"""
import collections, concurrent.futures, contextlib, json, multiprocessing, os, sys, time

import numpy as np

from decode_split import decode, nms, placement, unmap
from score_coco import save_detections

PAD = 114
MODES = ("centre", "origin", "stretch")
RESAMPLES = ("lanczos-pil", "bilinear-cv2")


def _wh(size):
    return (size, size) if isinstance(size, int) else (int(size[0]), int(size[1]))


def prepare(path, size, mode, resample):
    """-> uint8 HWC RGB canvas of the input size, and its Placement."""
    w, h = _wh(size)
    if resample == "lanczos-pil":
        from PIL import Image

        with Image.open(path) as im:
            im = im.convert("RGB")
        p = placement(im.width, im.height, size, mode)
        # what MIT's PadAndResize does to a PIL RGB image
        content = np.asarray(im.resize((p.new_w, p.new_h), Image.Resampling.LANCZOS))
    elif resample == "bilinear-cv2":
        import cv2

        # IGNORE_ORIENTATION: annotation sizes are the stored pixels, not the EXIF-rotated ones
        bgr = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if bgr is None:
            raise FileNotFoundError(f"cannot decode {path}")
        p = placement(bgr.shape[1], bgr.shape[0], size, mode)
        content = cv2.resize(bgr, (p.new_w, p.new_h), interpolation=cv2.INTER_LINEAR)[:, :, ::-1]
    else:
        raise ValueError(f"unknown resample {resample!r}; want one of {RESAMPLES}")
    canvas = np.full((h, w, 3), PAD, np.uint8)
    canvas[p.pad_top:p.pad_top + p.new_h, p.pad_left:p.pad_left + p.new_w] = content
    return canvas, p


def _post(job):
    image_id, p, size, t = job
    t0 = time.perf_counter()
    xyxy, scores = decode(t[0::2], t[1::2], size)
    box, score, cls = nms(xyxy, scores)
    return image_id, unmap(box, p), score, cls, time.perf_counter() - t0


@contextlib.contextmanager
def _worker_pool(workers):
    # ProcessPoolExecutor spawns its workers lazily inside submit(), so the variables have to stay
    # set for the pool's whole life, not just its construction. Without them 32 workers each open a
    # 32-thread BLAS pool and thrash.
    keys = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
    old = {k: os.environ.get(k) for k in keys}
    os.environ.update({k: "1" for k in keys})
    pool = concurrent.futures.ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn"))
    try:
        yield pool
    finally:
        pool.shutdown(cancel_futures=True)
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _images(ann_path, limit):
    with open(ann_path) as f:
        images = sorted(json.load(f)["images"], key=lambda i: i["id"])
    return images if limit is None else images[:limit]


def run(infer, images_dir, ann_path, size, mode, resample, out_npz, limit=None,
        workers=os.cpu_count() or 1, prepare=prepare):
    """Returns timings in seconds: wall, prepare, infer, and post (summed over workers)."""
    images = _images(ann_path, limit)
    if not images:
        raise ValueError("no images to run")
    size = _wh(size)
    done = {"ids": [], "box": [], "score": [], "cls": [], "post": 0.0}

    def collect(res):
        image_id, box, score, cls, dt = res
        done["ids"].append(np.full(len(score), image_id, np.int64))
        done["box"].append(box)
        done["score"].append(score)
        done["cls"].append(cls)
        done["post"] += dt

    t_prep = t_infer = 0.0
    start = time.perf_counter()
    pending = collections.deque()
    with _worker_pool(workers) if workers > 1 else contextlib.nullcontext() as pool:
        for n, im in enumerate(images, 1):
            t0 = time.perf_counter()
            canvas, p = prepare(os.path.join(images_dir, im["file_name"]), size, mode, resample)
            t1 = time.perf_counter()
            out = infer(canvas)
            t2 = time.perf_counter()
            t_prep, t_infer = t_prep + t1 - t0, t_infer + t2 - t1
            # a copy: the pool pickles a job on its feeder thread, after a runtime that reuses its
            # output buffers may already have written the next image into them
            job = (im["id"], p, size, [np.array(a, np.float32) for a in out])
            if pool is None:
                collect(_post(job))
            else:
                pending.append(pool.submit(_post, job))
                while len(pending) > 2 * workers:  # bounds the arrays held in flight
                    collect(pending.popleft().result())
            if n % 500 == 0:
                print(f"{n}/{len(images)} images, {n / (time.perf_counter() - start):.1f}/s", file=sys.stderr, flush=True)
        while pending:
            collect(pending.popleft().result())

    cat = lambda k, shape, dt: np.concatenate(done[k]) if done[k] else np.empty(shape, dt)
    save_detections(out_npz, cat("ids", (0,), np.int64), cat("cls", (0,), np.int64), cat("score", (0,), np.float32),
                    cat("box", (0, 4), np.float32), [im["id"] for im in images])
    return {"images": len(images), "wall": time.perf_counter() - start, "prepare": t_prep, "infer": t_infer,
            "post": done["post"]}
