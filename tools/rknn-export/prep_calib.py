#!/usr/bin/env python3
"""Build INT8 calibration sets.

Policies letterbox / stretch write NCHW uint8 .npy (the toolkit rejects NHWC .npy) and mirror
core's plugins/cairn-detect/src/infer/catalog.rs and the letterbox size rule
src/infer/geometry.rs (scaled_dim): yolox = BGR letterbox (pad 114, content at the origin, pad
right/bottom); yolov8/yolo26 = RGB stretch; rfdetr = RGB stretch. Stored as .npy so channel order
is explicit rather than depending on how rknn reads image files.
Policy png is the YOLOv9 scoring pipeline itself (coco_run.prepare, top-left letterbox, cv2
bilinear, RGB), written as PNG at the exact input size so the toolkit resizes nothing. The
toolkit reads a PNG as RGB: a cv2-written file with R=200, B=20 calibrated model channel 0 to
max 200 (2.3.2, rk3566). Order must be rgb.
Also writes set.json (per-frame source geometry and sha256, and an id over the frame bytes that
manifests record as the calibration/parity set id; subset_id gives the id of the first N).
usage: prep_calib.py FRAMES_DIR OUT_DIR SIZE {letterbox|stretch|png} {rgb|bgr} [--list FILE]
SIZE is 640 or WxH (png only). --list names the frames in order, COCO image ids or file names separated
by white space (`#` lines skipped); without it every .jpg/.png in FRAMES_DIR, sorted.
"""
import argparse, glob, hashlib, json, os, sys
import numpy as np


def scaled_dim(side, ratio, limit):
    # geometry.rs scaled_dim: floor, even, clamp. Rounding instead calibrates on content the detector never produces.
    even = (min(int(side * ratio), limit) // 2) * 2
    return max(min(max(even, 2), limit), 1)


def subset_id(frames, n=None):
    """The set id over the first n frames (all by default): the id a calibration subset is recorded under."""
    frames = frames if n is None else frames[:n]
    digest = hashlib.sha256()
    for f in frames:
        digest.update(f"{f['sha256']}  {f.get('file') or f['npy']}\n".encode())
    return digest.hexdigest()[:16]


def listed(frames_dir, list_file):
    names = []
    for ln in open(list_file):
        if not ln.startswith("#"):
            names += [f"{int(t):012d}.jpg" if t.isdigit() else t for t in ln.split()]
    if len(set(names)) != len(names):
        raise ValueError(f"{list_file}: a frame is listed twice")
    return [os.path.join(frames_dir, n) for n in names]


def canvas_npy(path, size, policy, order):
    import cv2

    im = cv2.imread(path)  # BGR
    h, w = im.shape[:2]
    if policy == "letterbox":
        r = min(size / w, size / h)
        nw, nh = scaled_dim(w, r, size), scaled_dim(h, r, size)
        canvas = np.full((size, size, 3), 114, np.uint8)
        canvas[:nh, :nw] = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    else:
        canvas = cv2.resize(im, (size, size), interpolation=cv2.INTER_LINEAR)
    if order == "rgb":
        canvas = canvas[:, :, ::-1]
    return np.ascontiguousarray(canvas.transpose(2, 0, 1)[None]), w, h  # NCHW: what the toolkit demands of .npy


def write_set(paths, out, size, policy, order, source):
    import cv2
    import coco_run

    os.makedirs(out, exist_ok=True)
    lines, meta = [], []
    for i, p in enumerate(paths):
        if policy == "png":
            rgb, pl = coco_run.prepare(p, size, "origin", "bilinear-cv2")
            f = os.path.join(out, f"{i:03d}.png")
            if not cv2.imwrite(f, np.ascontiguousarray(rgb[:, :, ::-1])):  # cv2 writes BGR; the toolkit reads RGB
                raise OSError(f"cannot write {f}")
            w, h, key = pl.src_w, pl.src_h, "file"
        else:
            arr, w, h = canvas_npy(p, size, policy, order)
            f = os.path.join(out, f"{i:03d}.npy")
            np.save(f, arr)
            key = "npy"
        lines.append(os.path.abspath(f))
        meta.append({key: os.path.basename(f), "src": os.path.basename(p), "w": w, "h": h,
                     "sha256": hashlib.sha256(open(f, "rb").read()).hexdigest()})
    open(os.path.join(out, "dataset.txt"), "w").write("\n".join(lines) + "\n")
    size_rec = size if isinstance(size, int) else list(size)
    # set.json is the set's done marker (build_rungs.sh, quant_matrix.sh): never half-written
    with open(os.path.join(out, "set.json.tmp"), "w") as f:
        json.dump({"source": os.path.abspath(source), "policy": policy, "order": order, "size": size_rec, "n": len(lines),
                   "id": subset_id(meta), "frames": meta}, f, indent=1)
    os.replace(os.path.join(out, "set.json.tmp"), os.path.join(out, "set.json"))
    return len(lines)


def main(argv=None):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # python -I leaves the script directory off sys.path
    from decode_split import parse_size

    ap = argparse.ArgumentParser(description=__doc__.split("usage:")[0].strip())
    ap.add_argument("frames")
    ap.add_argument("out")
    ap.add_argument("size")
    ap.add_argument("policy", choices=("letterbox", "stretch", "png"))
    ap.add_argument("order", choices=("rgb", "bgr"))
    ap.add_argument("--list")
    a = ap.parse_args(argv)
    if a.policy == "png":
        if a.order != "rgb":
            ap.error("png sets are RGB: the toolkit converts a PNG to RGB as it reads it")
        size = parse_size(a.size)
    elif a.size.isdigit():
        size = int(a.size)
    else:
        ap.error("letterbox and stretch take one square size")
    paths = (listed(a.frames, a.list) if a.list else
             sorted(glob.glob(os.path.join(a.frames, "*.jpg")) + glob.glob(os.path.join(a.frames, "*.png"))))
    if not paths: sys.exit(f"no .jpg/.png frames in {a.frames}")
    missing = [p for p in paths if not os.path.isfile(p)]
    if missing: sys.exit(f"{len(missing)} listed frames are missing, e.g. {missing[0]}")
    print(write_set(paths, a.out, size, a.policy, a.order, a.frames), "frames ->", a.out)


if __name__ == "__main__":
    main()
