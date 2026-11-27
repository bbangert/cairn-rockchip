#!/usr/bin/env python3
"""Step-1 proof: an exported MIT YOLOv9 ONNX under onnxruntime (CPU), against PyTorch and scored on COCO val2017.

--compare K first checks the graph against the PyTorch module it was exported from (flags read
from FILE.onnx.json) on K images; a --logits model then gets its sigmoid on the host before decode. A
--class-clamp export is refused there: the torch module has no clamp.
--baseline FILE.json (a baseline_mit.py result for the same name, mode, resample, size and limit)
fails the run when the AP differs by more than --ap-tol; it must also record this export's checkpoint
hash and MIT commit. --is-current exits 0 when the model still matches its sidecar and the pinned
checkpoint and commit, and its result exists with the same onnx_sha256, passes the AP gate and (given --compare K)
holds a passing tensor comparison over at least K images; 1 when it must be (re)run, 2 on a refusal
(a model that does not match its sidecar included) or a gate failure; --print-stem prints the result's basename.
Run with the MIT venv, e.g. work/mit/venv/bin/python -B -I.
usage: onnx_proof.py {v9-t|v9-s|v9-m} FILE.onnx --mode origin --resample bilinear-cv2 [--limit N]
                     [--out-dir DIR] [--compare K] [--baseline FILE.json] [--ap-tol AP]
                     [--is-current] [--print-stem]
"""
import argparse, json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # python -I leaves the script directory off sys.path

import coco_run, mit_model, score_coco, sources
from decode_split import size_suffix

COCO = sources.COCO
ANN = os.path.join(COCO, "annotations", "instances_val2017.json")
# Relative to the reference tensor's largest magnitude (at least 1): float32 conv reordering alone
# moves a logit of magnitude 30 by ~3e-3 (torch float32 vs float64 differs by as much as ORT does),
# i.e. ~1e-4 relative. The largest box0 difference in work/mit/onnx/run.log is 1.85e-3 at a max of 31.1.
TOLERANCE = 1e-3
AP_TOLERANCE = 0.05


def variant_stem(name, meta, mode, resample, limit):
    # a model without sum tensors is a different export of the same flags, so it gets its own result
    nosum = "-nosum" if not meta.get("with_sum", True) and not meta["logits"] else ""
    k = meta.get("class_clamp")
    # repr round-trips, so two clamps never share a stem; an integral K keeps the short spelling
    clamp = "" if k is None else f"-clamp{int(k) if float(k).is_integer() else repr(float(k))}"
    return (f"{name}-onnx{'-fused' if meta['fuse'] else ''}{'-logits' if meta['logits'] else ''}{nosum}{clamp}"
            f"-{mode}-{resample}{size_suffix(meta['size'])}" + (f"-limit{limit}" if limit else ""))


def sigmoid(x):
    with np.errstate(over="ignore"):  # exp overflows to inf for very negative logits, which 1 / (1 + inf) takes to 0
        return (1 / (1 + np.exp(-x))).astype(np.float32)


def session(path, threads):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads, opts.inter_op_num_threads = threads, 1
    return ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])


def raw_outputs(sess, canvas):
    x = np.ascontiguousarray(canvas.transpose(2, 0, 1)[None], dtype=np.float32) / 255
    return dict(zip((o.name for o in sess.get_outputs()), sess.run(None, {"images": x})))


def accumulate(worst, theirs, ours):
    """Fold one image into per-tensor (max abs difference, max |reference|); non-finite on either side or in the
    difference makes the difference inf, because max() would silently drop a NaN."""
    for name, t in theirs.items():
        ref = ours[name]
        with np.errstate(invalid="ignore"):
            diff = np.abs(t.astype(np.float64) - ref)
        finite = bool(np.isfinite(t).all() and np.isfinite(ref).all() and np.isfinite(diff).all())
        d, m = worst.get(name, (0.0, 0.0))
        worst[name] = (max(d, float(diff.max()) if finite else float("inf")),
                       max(m, float(np.abs(ref).max()) if np.isfinite(ref).all() else float("inf")))
    return worst


def tensor_failures(cmp, tol=TOLERANCE):
    """-> {tensor: why} for every tensor whose difference is non-finite or above tol x max(1, max |reference|)."""
    bad = {}
    for name, (d, m) in cmp.items():
        if not (np.isfinite(d) and np.isfinite(m)):
            bad[name] = f"{name}: non-finite (max |diff| {d}, max |reference| {m})"
        elif d > tol * max(1.0, m):
            bad[name] = f"{name}: max |diff| {d:.3g} above {tol} x max(1, {m:.3g})"
    return bad


def compare(net, sess, canvases, logits):
    import torch

    worst = {}
    with torch.inference_mode():
        for canvas in canvases:
            theirs = raw_outputs(sess, canvas)
            x = torch.from_numpy(canvas).permute(2, 0, 1).unsqueeze(0).float().div(255)
            ours = dict(zip(mit_model.OUTPUT_NAMES, (t.numpy() for t in net(x))))
            if not logits:
                for i in range(3):
                    ours[f"sum{i}"] = ours[f"cls{i}"].sum(axis=1, keepdims=True)
            accumulate(worst, theirs, ours)
    return worst


def result_size(size):
    return size[0] if size[0] == size[1] else list(size)


def check_baseline(base, run):
    """Refuses a baseline scored under other settings than this run: its AP would not be comparable. A field the
    baseline lacks reads as null and so differs."""
    got = {k: base.get(k) for k in ("name", "mode", "resample", "size", "limit", "checkpoint_sha256", "mit_commit")}
    got["snap_scores"] = base.get("snap_scores", False)  # absent in baselines written before the flag existed
    got["stage"] = base.get("stage")
    # an ONNX result handed in as the baseline would gate the model against itself
    want = {**run, "snap_scores": False, "stage": "pytorch"}
    bad = {k: (got[k], want[k]) for k in want if got[k] != want[k]}
    if bad:
        raise ValueError("baseline differs from this run (baseline, run): " + json.dumps(bad))


def load_baseline(path, run):
    with open(path) as f:
        base = json.load(f)
    check_baseline(base, run)
    return base


def ap_failure(ap, base_ap, tol):
    # 1e-9: 38.16 - 38.21 is a hair over 0.05 in binary and must pass an inclusive 0.05; `not <=` is also true for NaN
    if not abs(ap - base_ap) <= tol + 1e-9:
        return f"AP {ap} differs from the baseline AP {base_ap} by {ap - base_ap:+.4g}, above {tol}"
    return None


class NonFinite(ValueError):
    pass


def scoring_outputs(out, logits):
    """The six decode inputs. Raw values are checked first: decode drops a NaN score and the host sigmoid turns
    an infinite logit into a probability, so the AP gate alone would pass them."""
    names = [n for i in range(3) for n in (f"box{i}", f"cls{i}")]
    bad = [n for n in names if not np.isfinite(out[n]).all()]
    if bad:
        raise NonFinite(f"non-finite values in {', '.join(bad)}")
    return [sigmoid(out[n]) if logits and n.startswith("cls") else out[n] for n in names]


def positive_int(text):
    n = int(text)
    if n < 1:
        raise argparse.ArgumentTypeError(f"{text} is below 1")
    return n


def verify_model(model, name, meta):
    """-> (checkpoint path, sha256, MIT commit); ValueError unless the file and the pinned sources match the sidecar."""
    weights, sha, commit = mit_model.checkpoint(name)
    # the commit too: the same checkpoint exported from another revision of the code is another model
    if (meta["checkpoint_sha256"] != sha or meta.get("mit_commit") != commit
            or sources.sha256_file(model) != meta["onnx_sha256"]):
        raise ValueError(f"{model} does not match its .json record, the pinned checkpoint or the pinned MIT commit")
    return weights, sha, commit


def compared_images(res):
    """Images behind a result's passing tensor comparison; 0 when none, and for a result written before the
    count was recorded (bool `compared`)."""
    n = res.get("compared")
    try:
        cmp = {k: (v["max_abs_diff"], v["max_abs_value"]) for k, v in res["compare"].items()}
    except (KeyError, TypeError, AttributeError):
        return 0
    if isinstance(n, bool) or not isinstance(n, int) or not cmp or tensor_failures(cmp):
        return 0
    return n


def is_current(result_path, meta, base, tol, compare=None):
    """Exit code: 0 skip, 1 run, 2 the existing result fails the AP gate. The sidecar sha is enough here: main
    verifies the file against the sidecar first, so a rebuilt model changes both."""
    try:
        with open(result_path) as f:
            res = json.load(f)
    except (OSError, ValueError):
        return 1
    if res.get("onnx_sha256") != meta["onnx_sha256"]:
        return 1
    if compare and compared_images(res) < compare:
        return 1
    why = base and ap_failure(res["AP"], base["AP"], tol)
    if why:
        print(f"{result_path}: {why}", file=sys.stderr)
        return 2
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("usage:")[0].strip())
    ap.add_argument("name", choices=("v9-t", "v9-s", "v9-m"))
    ap.add_argument("model")
    ap.add_argument("--mode", choices=coco_run.MODES, required=True)
    ap.add_argument("--resample", choices=coco_run.RESAMPLES, required=True)
    ap.add_argument("--limit", type=positive_int)
    ap.add_argument("--out-dir", default=os.path.join(sources.MIT, "onnx"))
    ap.add_argument("--compare", type=positive_int, metavar="K")
    ap.add_argument("--baseline", metavar="FILE.json")
    ap.add_argument("--ap-tol", type=float, default=AP_TOLERANCE)
    ap.add_argument("--is-current", action="store_true")
    ap.add_argument("--print-stem", action="store_true", help="print the result file's basename and exit")
    ap.add_argument("--threads", type=int, default=8, help="intra-op threads, as baseline_mit.py")
    ap.add_argument("--workers", type=int, default=6, help="decode + NMS processes")
    a = ap.parse_args(argv)

    with open(a.model + ".json") as f:
        meta = json.load(f)
    if meta["name"] != a.name:
        sys.exit(f"{a.model} was exported from {meta['name']}, not {a.name}")
    if a.compare and meta.get("class_clamp") is not None:
        sys.exit(f"{a.model} is clamped (--class-clamp {meta['class_clamp']:g}); the torch module is not, so they "
                 "cannot be compared: its proof is the float AP")
    size = tuple(meta["size"])
    meta["size"] = size
    stem = os.path.join(a.out_dir, variant_stem(a.name, meta, a.mode, a.resample, a.limit))
    if a.print_stem:
        print(os.path.basename(stem))
        return None

    base = None
    if a.baseline:
        run = {"name": a.name, "mode": a.mode, "resample": a.resample, "size": result_size(size), "limit": a.limit,
               "checkpoint_sha256": meta["checkpoint_sha256"], "mit_commit": meta["mit_commit"]}
        try:
            base = load_baseline(a.baseline, run)
        except ValueError as e:
            print(f"{a.baseline}: {e}", file=sys.stderr)
            sys.exit(2)
    if a.is_current:
        try:
            verify_model(a.model, a.name, meta)
        except ValueError as e:
            print(e, file=sys.stderr)
            sys.exit(2)
        sys.exit(is_current(stem + ".json", meta, base, a.ap_tol, a.compare))

    undo = mit_model.block_network()
    try:
        return _run(a, meta, size, stem, base)
    finally:
        undo()


def _run(a, meta, size, stem, base):
    try:
        weights, sha, commit = verify_model(a.model, a.name, meta)
    except ValueError as e:
        sys.exit(str(e))

    sess = session(a.model, a.threads)
    names = {o.name for o in sess.get_outputs()}
    want = {f"{k}{i}" for i in range(3) for k in ("box", "cls")}
    if not want <= names:
        sys.exit(f"outputs {sorted(names)} lack {sorted(want - names)}")

    cmp, n_compared = None, 0
    if a.compare:
        import torch

        torch.set_num_threads(a.threads)
        net = mit_model.load(a.name, weights, mit_model.MIT_DIR, sigmoid=not meta["logits"], fuse=meta["fuse"])
        canvases = [coco_run.prepare(os.path.join(COCO, "val2017", im["file_name"]), size, a.mode, a.resample)[0]
                    for im in coco_run._images(ANN, a.compare)]
        cmp, n_compared = compare(net, sess, canvases, meta["logits"]), len(canvases)
        print(f"pytorch vs onnxruntime over {len(canvases)} images, max |diff| (of max |value|) per output: "
              + " ".join(f"{k} {d:.3g} ({m:.3g})" for k, (d, m) in sorted(cmp.items())))
        bad = tensor_failures(cmp)
        if bad:
            sys.exit("tensor gate failed: " + "; ".join(bad.values()))

    os.makedirs(a.out_dir, exist_ok=True)
    try:
        t = coco_run.run(lambda canvas: scoring_outputs(raw_outputs(sess, canvas), meta["logits"]),
                         os.path.join(COCO, "val2017"), ANN, size, a.mode, a.resample, stem + ".npz",
                         limit=a.limit, workers=a.workers)
    except NonFinite as e:
        sys.exit(f"scoring refused: {e}")
    res = score_coco.evaluate(ANN, score_coco.load_detections(stem + ".npz"))
    import onnxruntime

    res.update({"name": a.name, "stage": "onnx", "fused": meta["fuse"], "logits": meta["logits"],
                "class_clamp": meta.get("class_clamp"),
                "mode": a.mode, "resample": a.resample, "size": result_size(size),
                "limit": a.limit, "onnxruntime": onnxruntime.__version__, "model": os.path.basename(a.model),
                "onnx_sha256": meta["onnx_sha256"], "checkpoint_sha256": sha, "mit_commit": commit,
                "compared": n_compared,
                "compare": cmp and {k: {"max_abs_diff": d, "max_abs_value": m} for k, (d, m) in cmp.items()},
                "seconds": {k: round(v, 2) for k, v in t.items() if k != "images"},
                "images_per_second": round(t["images"] / t["wall"], 2)})
    why = None
    if base:
        why = ap_failure(res["AP"], base["AP"], a.ap_tol)
        res.update({"baseline": os.path.abspath(a.baseline), "baseline_ap": base["AP"],
                    "ap_delta": round(res["AP"] - base["AP"], 6), "ap_tol": a.ap_tol})
    # written even when the AP gate fails: the result is the evidence, and is_current re-applies the gate
    with open(stem + ".json.tmp", "w") as f:
        json.dump(res, f, indent=1)
        f.write("\n")
    os.replace(stem + ".json.tmp", stem + ".json")
    n = t["images"]
    print(f"{os.path.basename(stem)}: AP {res['AP']} AP50 {res['AP50']} AP75 {res['AP75']} "
          f"({n} images, {res['n_detections']} dets) wall {t['wall']:.1f}s {n / t['wall']:.2f} img/s; "
          f"s/img prepare {t['prepare'] / n:.4f} infer {t['infer'] / n:.4f} post {t['post'] / n:.4f} (per worker)")
    if why:
        sys.exit(f"baseline gate failed: {why}")
    return res


if __name__ == "__main__":
    main()
