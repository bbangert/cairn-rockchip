#!/usr/bin/env python3
"""COCO val2017 detections from a YOLOv9 build in the toolkit's host simulator, or from its float ONNX
under onnxruntime, decoded under every rule set from one pass (decode_split.RULES).

The simulator only runs a graph built in the same process, so `rknn` mode builds it here with
convert.py's own build (same arguments, written to that --out) and scores the live graph; a .rknn
file cannot be scored. `onnx` mode is the float reference on the same images through the same
prepare and decode. The toolkit's venv has no pycocotools, so this writes, per rule set,
STEM.npz and STEM.meta.json; `score_coco.py ANN STEM.npz --meta STEM.meta.json --out STEM.json`
then gives a result of the float runners' shape. --limit N takes the first N of the sorted id
list, as the float runners do. Resumable: chunks of --chunk images land in STEM.parts/ and a
rerun with the same arguments skips them. A resumed rknn run rebuilds the graph, and is refused
unless the rebuild's outputs on a fixed input (convert.graph_id) equal the first build's: the
.rknn hash differs between identical builds. STEM-core.meta.json, written last, is the done marker.
For a logits export the -core result scores a decode core does not have (SplitDfl applies no
sigmoid). Run with venv/bin/python -I -B.
usage: sim_coco.py [--limit N] [--out-dir DIR] [--chunk 250] [--workers 6] onnx ONNX
       sim_coco.py [--limit N] [--out-dir DIR] [--chunk 250] [--workers 6] rknn NAME ONNX {i8|fp16} [convert.py options]
       sim_coco.py ... --print-stem   prints the reference-rules stem and exits
"""
import argparse, json, os, sys, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # python -I leaves the script directory off sys.path

import coco_run, convert, onnx_proof, prep_calib, score_coco, sources
from decode_split import RULES, size_suffix

COCO = sources.COCO
ANN = os.path.join(COCO, "annotations", "instances_val2017.json")
MODE, RESAMPLE = "origin", "bilinear-cv2"  # core's letterbox, the pipeline the float baselines compare under


def stems(base, limit):
    """-> {rules: stem}; the float runners' tail (-origin-bilinear-cv2[-limitN]) after base, -core for core rules."""
    tail = f"-{MODE}-{RESAMPLE}" + (f"-limit{limit}" if limit else "")
    return {r: base + tail + ("" if r == "reference" else f"-{r}") for r in RULES}


def onnx_meta(path):
    with open(path + ".json") as f:
        return json.load(f)


def chunks(images, size):
    return [images[i:i + size] for i in range(0, len(images), size)]


def chunk_done(parts, k):
    return all(os.path.exists(os.path.join(parts, f"{k}.{r}.npz")) for r in RULES)


def merge(parts, n_chunks, rule, out_npz):
    ds = [score_coco.load_detections(os.path.join(parts, f"{k}.{rule}.npz")) for k in range(n_chunks)]
    score_coco.save_detections(out_npz, *(np.concatenate([getattr(d, f) for d in ds])
                                          for f in ("image_id", "class_id", "score", "box", "images")))


def write_json(path, obj):
    # these files are resume markers: a kill mid-write must not leave a truncated one behind
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(path + ".tmp", path)


def fingerprint(a):
    # what the detections depend on; a resume under anything else would mix two models in one result
    keep = {k: v for k, v in sorted(vars(a).items()) if k not in ("workers", "threads", "print_stem")}
    calib = getattr(a, "calib", None)
    if calib and os.path.exists(os.path.join(calib, "set.json")):
        frames = json.load(open(os.path.join(calib, "set.json")))["frames"]
        keep["calib_set_id"] = prep_calib.subset_id(frames, a.calib_count)
    return {**keep, "source_sha256": sources.sha256_file(a.onnx)}


def check_graph(fp_path, fp, gid):
    """Record the first build's graph id; refuse a rebuild whose outputs differ from it."""
    have = fp.get("graph_id")
    if have is None:
        write_json(fp_path, {**fp, "graph_id": gid})
    elif have != gid:
        sys.exit(f"{os.path.dirname(fp_path)}: this build's outputs differ from those of the build that scored the "
                 f"chunks already there (graph id {gid[:12]} vs {have[:12]}); remove the directory to rescore")


def build_quietly(a):
    """convert.build points fds 1 and 2 at its NAME.out for the operator table; this run's log needs them back.
    convert.py's --parity, --ranges and --analysis run here too, on the same graph: an mmse calibration is
    slow enough that building once per variant matters."""
    sys.stdout.flush(); sys.stderr.flush()
    saved, cwd = (os.dup(1), os.dup(2)), os.getcwd()
    try:
        r, rec = convert.build(a)
        ok = r is not None and rec["_step"]("sim_init", lambda: r.init_runtime())
        if ok: convert.post_steps(r, a, rec)
        rec = convert.save(a, rec)
        ok = ok and rec["steps"].get("graph_unchanged") == "ok"
    finally:
        sys.stdout.flush(); sys.stderr.flush()
        os.dup2(saved[0], 1); os.dup2(saved[1], 2)
        os.close(saved[0]); os.close(saved[1])
        os.chdir(cwd)
    if not ok:
        sys.exit(f"build failed: {rec['steps']}")
    return r, rec


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    ap = argparse.ArgumentParser(description=__doc__.split("usage:")[0].strip())
    ap.add_argument("--limit", type=onnx_proof.positive_int)
    ap.add_argument("--out-dir", default=os.path.join(sources.MIT, "quant", "sim"))
    ap.add_argument("--chunk", type=onnx_proof.positive_int, default=250)
    ap.add_argument("--workers", type=int, default=6, help="decode + NMS processes")
    ap.add_argument("--threads", type=int, default=8, help="onnxruntime intra-op threads (onnx mode)")
    ap.add_argument("--print-stem", action="store_true")
    sub = ap.add_subparsers(dest="stage", required=True)
    so = sub.add_parser("onnx")
    so.add_argument("onnx")
    sub.add_parser("rknn", parents=[convert.parser(add_help=False)])
    a = ap.parse_args(argv)
    a.onnx = os.path.abspath(a.onnx)
    a.out_dir = os.path.abspath(a.out_dir)
    meta = onnx_meta(a.onnx)
    size = tuple(meta["size"])
    base = (f"{a.name}-sim" if a.stage == "rknn" else
            f"{os.path.splitext(os.path.basename(a.onnx))[0]}-ort") + size_suffix(size)
    st = stems(os.path.join(a.out_dir, base), a.limit)
    if a.print_stem:
        print(os.path.basename(st["reference"]))
        return None
    if a.stage == "rknn":
        convert.check_args(a)
    os.makedirs(a.out_dir, exist_ok=True)
    images = coco_run._images(ANN, a.limit)
    parts = st["reference"] + ".parts"
    os.makedirs(parts, exist_ok=True)
    fp = fingerprint(a)
    fp_path = os.path.join(parts, "config.json")
    if os.path.exists(fp_path):
        with open(fp_path) as f:
            old = json.load(f)
        if {k: v for k, v in old.items() if k != "graph_id"} != fp:
            sys.exit(f"{parts} was written under other arguments or another model; remove it to start over")
        fp = old
    else:
        write_json(fp_path, fp)
    todo = [k for k in range(len(chunks(images, a.chunk))) if not chunk_done(parts, k)]

    build_rec, infer = None, None
    if todo:
        if a.stage == "rknn":
            r, build_rec = build_quietly(a)
            names = convert.output_names(a)

            def infer(canvas):
                return onnx_proof.scoring_outputs(dict(zip(names, r.inference(inputs=[canvas[None]], data_format="nhwc"))), meta["logits"])
        else:
            sess = onnx_proof.session(a.onnx, a.threads)

            def infer(canvas):
                return onnx_proof.scoring_outputs(onnx_proof.raw_outputs(sess, canvas), meta["logits"])
        check_graph(fp_path, fp, convert.graph_id(infer, size))

    runs_path = os.path.join(parts, "runs.json")
    runs = json.load(open(runs_path)) if os.path.exists(runs_path) else []
    all_chunks = chunks(images, a.chunk)
    for k in todo:
        tmp = {rule: os.path.join(parts, f"{k}.{rule}.tmp.npz") for rule in RULES}
        t = coco_run.run(infer, os.path.join(COCO, "val2017"), ANN, size, MODE, RESAMPLE, tmp,
                         workers=a.workers, images=all_chunks[k])
        for rule in RULES:  # not `r`: that is the live graph the next chunk runs on
            os.replace(tmp[rule], os.path.join(parts, f"{k}.{rule}.npz"))
        runs.append({"chunk": k, "images": t["images"], "seconds": {key: round(v, 2) for key, v in t.items() if key != "images"},
                     "rknn_sha256": build_rec and build_rec.get("rknn_sha256")})
        write_json(runs_path, runs)
        done = sum(x["images"] for x in runs)
        print(f"{time.strftime('%H:%M:%S')} chunk {k}: {done}/{len(images)} images, "
              f"{t['images'] / t['wall']:.2f} img/s", file=sys.stderr, flush=True)

    wall = sum(x["seconds"]["wall"] for x in runs)
    build_json = os.path.join(a.out, a.name + ".json") if a.stage == "rknn" else None
    build = json.load(open(build_json)) if build_json and os.path.exists(build_json) else None
    for rule, stem in st.items():
        merge(parts, len(all_chunks), rule, stem + ".npz")
        rec = {"name": meta["name"], "stage": "rknn-sim" if a.stage == "rknn" else "onnx", "rules": rule,
               "fused": meta["fuse"], "logits": meta["logits"], "mode": MODE, "resample": RESAMPLE,
               "size": onnx_proof.result_size(size), "limit": a.limit, "model": os.path.basename(a.onnx),
               "onnx_sha256": meta["onnx_sha256"], "checkpoint_sha256": meta["checkpoint_sha256"],
               "mit_commit": meta["mit_commit"],
               "build": build and {k: build.get(k) for k in (
                   "name", "mode", "quantized_algorithm", "quantized_method", "optimization_level", "auto_hybrid",
                   "auto_hybrid_cos_thresh", "calib_policy", "calib_set_id", "calib_count", "toolkit",
                   "rknn_sha256", "float16_layers", "off_npu")},
               # a resumed run rebuilt the graph: one graph id, a file hash per build
               "graph_id": json.load(open(fp_path)).get("graph_id"),
               "rknn_sha256s": sorted({x["rknn_sha256"] for x in runs if x.get("rknn_sha256")}),
               "note": ("core rules on logits outputs: core's SplitDfl applies no sigmoid, so no shipping path decodes "
                        "this build" if rule == "core" and meta["logits"] else None),
               "seconds": {"wall": round(wall, 2)},
               "images_per_second": round(sum(x["images"] for x in runs) / wall, 2) if wall else None}
        write_json(stem + ".meta.json", rec)  # RULES puts core last, so its meta is the done marker
        print(f"{os.path.basename(stem)}: {len(images)} images; score_coco.py ANN {os.path.basename(stem)}.npz "
              f"--meta {os.path.basename(stem)}.meta.json --out {os.path.basename(stem)}.json")
    return st


if __name__ == "__main__":
    main()
