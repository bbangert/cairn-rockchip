#!/usr/bin/env python3
"""Float baseline: an MIT YOLOv9 checkpoint through the six-tensor cut and decode_split, scored on COCO val2017.

Every later build stage (ONNX, fp16, int8) is compared with this number. Run with the venv
that has torch and the MIT repo's dependencies, e.g. work/mit/venv/bin/python -I.
usage: baseline_mit.py {v9-t|v9-s|v9-m} --mode centre --resample lanczos-pil [--size 640|WxH] [--limit N]
                       [--out-dir DIR] [--check-mit-decode K] [--snap-scores] [--print-stem]
"""
import argparse, json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # python -I leaves the script directory off sys.path

import coco_run, mit_model, score_coco, sources
from decode_split import decode, parse_size, size_suffix

MIT_DIR = mit_model.MIT_DIR
COCO = sources.COCO


def variant_stem(name, mode, resample, limit, size=(640, 640), snap=False):
    return (f"{name}-pytorch-{mode}-{resample}{size_suffix(size)}" + ("-snap256" if snap else "")
            + (f"-limit{limit}" if limit else ""))


def snap_scores(p, steps=256):
    """What an 8-bit sigmoid output with scale 1/steps can hold: k / steps, k = 0..steps-1."""
    # floor(x + 0.5), not np.round: p just under 1/(2 * steps) must go to 0 and a tie must go up
    return np.clip(np.floor(p * steps + 0.5) / steps, 0, (steps - 1) / steps).astype(np.float32)


def check_mit_decode(net, canvases, size):
    """Max abs difference of (boxes px, class probabilities) between MIT's Vec2Box path and ours."""
    import torch
    from types import SimpleNamespace
    from yolo.utils.bounding_box_utils import Vec2Box

    w, h = size
    v2b = Vec2Box(net.net, SimpleNamespace(strides=[8, 16, 32]), (w, h), "cpu")
    db = ds = 0.0
    with torch.inference_mode():
        for canvas in canvases:
            x = to_tensor(canvas)
            ours = [t.numpy() for t in net(x)]
            cls_logits, _, mit_box = v2b(net.net(x)["Main"])
            box, score = decode(ours[0::2], ours[1::2], size)
            db = max(db, float(abs(mit_box[0].numpy() - box).max()))
            ds = max(ds, float(abs(torch.sigmoid(cls_logits[0]).numpy() - score).max()))
    return db, ds


def to_tensor(canvas):
    import torch

    return torch.from_numpy(canvas).permute(2, 0, 1).unsqueeze(0).float().div(255)


def positive_int(text):
    n = int(text)
    if n < 1:
        raise argparse.ArgumentTypeError(f"{text} is below 1")
    return n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("usage:")[0].strip())
    ap.add_argument("name", choices=("v9-t", "v9-s", "v9-m"))
    ap.add_argument("--mode", choices=coco_run.MODES, required=True)
    ap.add_argument("--resample", choices=coco_run.RESAMPLES, required=True)
    ap.add_argument("--size", type=parse_size, default=(640, 640), help="640 or WxH (width x height)")
    ap.add_argument("--limit", type=positive_int)
    ap.add_argument("--out-dir", default=os.path.join(sources.MIT, "baseline"))
    ap.add_argument("--snap-scores", action="store_true",
                    help="round the class probabilities to k/256 before decode: the 8-bit sigmoid output cost, alone")
    ap.add_argument("--check-mit-decode", type=int, default=0, metavar="K")
    ap.add_argument("--print-stem", action="store_true", help="print the result file's basename and exit")
    # batch-1 CPU inference stops scaling early: on 32 cores v9-t ran 13.9 img/s at 6 threads and
    # 10.2 at 20; the default was not itself measured
    ap.add_argument("--threads", type=int, default=8, help="torch intra-op threads")
    ap.add_argument("--workers", type=int, default=6, help="decode + NMS processes")
    a = ap.parse_args(argv)

    if a.print_stem:
        print(variant_stem(a.name, a.mode, a.resample, a.limit, a.size, a.snap_scores))
        return None

    undo = mit_model.block_network()
    try:
        return _run(a)
    finally:
        undo()


def _run(a):
    import torch

    torch.set_num_threads(a.threads)
    try:
        weights, sha, commit = mit_model.checkpoint(a.name)
    except ValueError as e:
        sys.exit(str(e))

    net = mit_model.load(a.name, weights, MIT_DIR)
    print(f"{a.name}: strict load {json.dumps(net.load_report)}", file=sys.stderr)
    size = a.size

    mit_check = None
    if a.check_mit_decode:
        ann = coco_run._images(os.path.join(COCO, "annotations", "instances_val2017.json"), a.check_mit_decode)
        canvases = [coco_run.prepare(os.path.join(COCO, "val2017", im["file_name"]), size, a.mode, a.resample)[0]
                    for im in ann]
        db, ds = check_mit_decode(net, canvases, size)
        print(f"MIT decode check over {len(canvases)} images: max |box| diff {db:.3g} px, max |score| diff {ds:.3g}")
        mit_check = {"images": len(canvases), "max_box_px": db, "max_score": ds}
        if db > 1e-2 or ds > 1e-4:
            sys.exit("our cut + decode differs from the MIT model's own decode")

    def infer(canvas):
        with torch.inference_mode():
            outs = [t.numpy() for t in net(to_tensor(canvas))]
        if a.snap_scores:
            outs[1::2] = [snap_scores(c) for c in outs[1::2]]
        return outs

    os.makedirs(a.out_dir, exist_ok=True)
    stem = os.path.join(a.out_dir, variant_stem(a.name, a.mode, a.resample, a.limit, size, a.snap_scores))
    ann_path = os.path.join(COCO, "annotations", "instances_val2017.json")
    t = coco_run.run(infer, os.path.join(COCO, "val2017"), ann_path, size, a.mode, a.resample, stem + ".npz",
                     limit=a.limit, workers=a.workers)
    res = score_coco.evaluate(ann_path, score_coco.load_detections(stem + ".npz"))
    import PIL

    versions = {"PIL": PIL.__version__}
    if a.resample == "bilinear-cv2":
        import cv2

        versions["cv2"] = cv2.__version__
    res.update({"name": a.name, "stage": "pytorch", "mode": a.mode, "resample": a.resample,
                "size": size[0] if size[0] == size[1] else list(size), "snap_scores": a.snap_scores,
                "limit": a.limit, "torch": torch.__version__, "checkpoint_sha256": sha,
                "mit_commit": commit, "load": net.load_report, "mit_decode_check": mit_check, "versions": versions,
                "seconds": {k: round(v, 2) for k, v in t.items() if k != "images"},
                "images_per_second": round(t["images"] / t["wall"], 2)})
    # the .json is the resume marker, so it must never exist half-written
    with open(stem + ".json.tmp", "w") as f:
        json.dump(res, f, indent=1)
        f.write("\n")
    os.replace(stem + ".json.tmp", stem + ".json")
    n = t["images"]
    print(f"{a.name} {a.mode} {a.resample}: AP {res['AP']} AP50 {res['AP50']} AP75 {res['AP75']} "
          f"({n} images, {res['n_detections']} dets) wall {t['wall']:.1f}s {n / t['wall']:.2f} img/s; "
          f"s/img prepare {t['prepare'] / n:.4f} infer {t['infer'] / n:.4f} post {t['post'] / n:.4f} (CPU-s, all workers)")
    return res


if __name__ == "__main__":
    main()
