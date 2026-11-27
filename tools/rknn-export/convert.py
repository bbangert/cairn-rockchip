#!/usr/bin/env python3
"""One rknn-toolkit2 conversion (x86 host, no board).

usage: convert.py NAME ONNX {i8|fp16} [--mean 0,0,0] [--std 1,1,1] [--calib DIR [--calib-count N]]
         [--size WxH] [--input-name images] [--outputs a,b] [--analysis N]
         [--algo normal|mmse|kl_divergence] [--hybrid [--hybrid-cos 0.999]] [--opt-level 3]
         [--ranges] [--float16 PATTERNS] [--refuse-cpu] [--disable-rules r1,r2]
         [--out DIR] [--kind yolox|split|mit --parity DIR [--dump N]]
Writes OUT/NAME.rknn and OUT/NAME.json (steps ok/err, timings, toolkit and
quantization settings, calibration/parity set ids, sha256 of the source ONNX
and of the .rknn, output tensors seen on the simulator, op targets from the
operator table, build warnings, ops left in float16). With --parity (a
prep_calib.py set of held-out frames) also OUT/NAME.parity.json: pooled
simulator-vs-FP32 metrics per source geometry; no tensors kept beyond the
first --dump frames. Without it, 5 calibration frames are dumped as
NAME.parity.npz for parity.py.
--calib-count N calibrates on the first N frames of the set, recorded under
that subset's id. --refuse-cpu fails the build (step `cpu`) if any op other
than the graph's input/output lands off the NPU; that is the compiler's
placement, which is all the host can see: an op the board's librknnrt then
cannot run shows only there. --ranges builds through
hybrid_quantization_step1 (proposal=False) + step2 instead of build, for the
per-tensor ranges build does not report (with --hybrid, which only build
does, step 1 runs again beside it): NAME.quantization.cfg and
NAME.ranges.json (each output's scale, zero point and representable range,
clipped box tensors, score ceilings, narrow Concat inputs, widest tensors).
With --hybrid those describe the graph before auto_hybrid's float16 moves
(`plan` says so) and NAME.json takes no `outputs` or `flags` from them.
--float16 PATTERNS (comma-separated fnmatch patterns over the cfg's tensor
names; `*` crosses `/`) builds through step 1 + 2 too, and pins every tensor
they match to float16 in the cfg's custom_quantize_layers before step 2; a
pattern matching nothing fails the build. Not with --hybrid. NAME.json
records `float16_patterns` and `float16_tensors`; `float16_layers` is still
what the operator table shows.
--analysis N runs accuracy_analysis on N of the frames calibration used (the
--calib-count prefix), one call each (a call reads one frame per model
input), into NAME.analysis.json; the ~0.75 GB of tensor dumps per frame are
deleted.
The simulator only runs in the session that built the graph, so parity
cannot be a separate step. Run under `venv/bin/python -I`; stdout carries the
toolkit's operator table and is redirected to OUT/NAME.out unless it is that
file already.
"""
import argparse, collections, hashlib, json, os, shutil, sys, time, traceback
from importlib.metadata import version
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import parity, prep_calib, quant_report, summarize

HERE = os.path.dirname(os.path.abspath(__file__))
ALGOS = ("normal", "mmse", "kl_divergence")


def parser(add_help=True):
    ap = argparse.ArgumentParser(add_help=add_help)
    ap.add_argument("name"); ap.add_argument("onnx"); ap.add_argument("mode", choices=["i8", "fp16"])
    ap.add_argument("--mean", default="0,0,0"); ap.add_argument("--std", default="1,1,1")
    ap.add_argument("--calib"); ap.add_argument("--calib-count", type=int); ap.add_argument("--size")
    ap.add_argument("--outputs"); ap.add_argument("--analysis", type=int, default=0)
    ap.add_argument("--algo", default="normal", choices=ALGOS); ap.add_argument("--disable-rules"); ap.add_argument("--input-name", default="images")
    ap.add_argument("--hybrid", action="store_true"); ap.add_argument("--hybrid-cos", type=float, default=0.999)
    ap.add_argument("--opt-level", type=int, default=3, choices=range(4)); ap.add_argument("--ranges", action="store_true")
    ap.add_argument("--refuse-cpu", action="store_true"); ap.add_argument("--float16")
    ap.add_argument("--out", default=os.path.join(HERE, "work", "rknn")); ap.add_argument("--kind", choices=["yolox", "split", "mit"])
    ap.add_argument("--parity"); ap.add_argument("--dump", type=int, default=0)
    return ap


def check_args(a):
    for k in ("onnx", "calib", "parity", "out"):
        if getattr(a, k): setattr(a, k, os.path.abspath(getattr(a, k)))
    if a.mode == "i8" and not a.calib: sys.exit("i8 needs --calib")
    if (a.hybrid or a.ranges or a.float16) and a.mode != "i8": sys.exit("--hybrid, --ranges and --float16 are INT8 options")
    if a.float16 and a.hybrid: sys.exit("--float16 and --hybrid exclude each other: auto_hybrid picks its own float16 layers")
    if a.parity:
        assert a.kind and a.calib, "--parity needs --kind and --calib"
        cf, pf = (json.load(open(d + "/set.json"))["frames"] for d in (a.calib, a.parity))
        if not (cf and pf): sys.exit("calibration and parity sets must both have frames")
        # only the --calib-count prefix trains the quantizer; calib_dataset refuses a count out of range
        if a.calib_count and a.calib_count > 0: cf = cf[:a.calib_count]
        # frames of a calibration source would score the quantizer on its own training data
        shared = parity.shared_sources(cf, pf)
        if shared: sys.exit(f"calibration and parity share sources: {shared}")


def own_stdout(path):
    """Point fds 1 and 2 at path unless stdout is that file already: the operator table is only on stdout."""
    try:
        if os.path.exists(path) and os.path.samefile("/proc/self/fd/1", path): return
    except OSError:
        pass
    sys.stdout.flush(); sys.stderr.flush()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    os.dup2(fd, 1); os.dup2(fd, 2); os.close(fd)


sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
def vec(s): return [[float(x) for x in s.split(",")]]


def onnx_logits(a):
    """The exporter's sidecar says whether cls* are logits rather than probabilities."""
    return os.path.exists(a.onnx + ".json") and json.load(open(a.onnx + ".json")).get("logits", False)


def graph_id(infer, size):
    """sha256 over the outputs for one fixed random input: what identifies a live graph. The .rknn file does
    not: two builds of v9-t normal and kl_divergence (2026-11-27) gave equal cfgs and parity but files
    differing in 200-240 bytes at offsets 4912-7040 that read as heap addresses."""
    w, h = size
    x = np.random.default_rng(0).integers(0, 256, (h, w, 3), np.uint8)
    d = hashlib.sha256()
    for t in infer(x):
        d.update(np.ascontiguousarray(t, np.float32).tobytes())
    return d.hexdigest()


def hashed(frames, files):
    """Frames with the per-frame hash a set written before set.json carried one lacks, read off its files."""
    return [f if "sha256" in f else {**f, "sha256": hashlib.sha256(open(p, "rb").read()).hexdigest()}
            for f, p in zip(frames, files)]


def calib_dataset(a, out):
    """-> (dataset file the toolkit reads, set record): a --calib-count subset gets its own list and id."""
    if not a.calib: return None, {}
    meta = json.load(open(a.calib + "/set.json")) if os.path.exists(a.calib + "/set.json") else None
    files = open(a.calib + "/dataset.txt").read().split()
    rec = {"calib_set_id": meta and meta["id"], "calib_policy": os.path.basename(a.calib), "calib_count": len(files)}
    if a.calib_count is None: return a.calib + "/dataset.txt", rec
    if not 0 < a.calib_count <= len(files): sys.exit(f"--calib-count {a.calib_count}: the set has {len(files)} frames")
    ds = os.path.join(out, a.name + ".dataset.txt")
    open(ds, "w").write("\n".join(files[:a.calib_count]) + "\n")
    rec.update(calib_count=a.calib_count, calib_full_set_id=rec["calib_set_id"],
               calib_set_id=meta and prep_calib.subset_id(hashed(meta["frames"][:a.calib_count], files)))
    return ds, rec


def configure(r, a, algo=None):
    r.config(mean_values=vec(a.mean), std_values=vec(a.std), target_platform="rk3566",
             quantized_algorithm=algo or a.algo, quantized_method="channel", float_dtype="float16",
             optimization_level=a.opt_level,
             **({"auto_hybrid_cos_thresh": a.hybrid_cos} if a.hybrid else {}),
             **({"disable_rules": a.disable_rules.split(",")} if a.disable_rules else {}))


def load_kw(a):
    kw = {}
    if a.size:
        w, h = map(int, a.size.split("x")); kw = dict(inputs=[a.input_name], input_size_list=[[1, 3, h, w]])
    if a.outputs: kw["outputs"] = a.outputs.split(",")
    return kw


def output_names(a):
    if a.outputs: return a.outputs.split(",")
    import onnx
    return [o.name for o in onnx.load(a.onnx, load_external_data=False).graph.output]


def build(a):
    """Load, build, export and re-load; -> (RKNN with the live graph, or None, result dict). The sidecar is
    written by the caller via save(res)."""
    from rknn.api import RKNN

    out = a.out; os.makedirs(out, exist_ok=True)
    os.chdir(out)  # the toolkit drops check*_*.onnx (20-40 MB each) into the cwd on every build
    for ext in ("json", "rknn", "parity.json", "parity.npz", "ranges.json", "analysis.json", "quantization.cfg"):
        if os.path.exists(f"{a.name}.{ext}"): os.remove(f"{a.name}.{ext}")
    own_stdout(os.path.join(out, a.name + ".out"))
    dataset, crec = calib_dataset(a, out)
    setid = lambda d: json.load(open(d + "/set.json"))["id"] if d and os.path.exists(d + "/set.json") else None
    res = {"name": a.name, "mode": a.mode, "kind": a.kind, "steps": {}, "secs": {},
           "toolkit": version("rknn-toolkit2"), "target_platform": "rk3566", "quantized_algorithm": a.algo,
           "quantized_method": "channel", "optimization_level": a.opt_level, "mean": a.mean, "std": a.std,
           "auto_hybrid": a.hybrid, "auto_hybrid_cos_thresh": a.hybrid_cos if a.hybrid else None,
           "float16_patterns": a.float16 and a.float16.split(","), "float16_tensors": None,
           "calib_set_id": None, "calib_policy": None, **crec, "parity_set_id": setid(a.parity),
           "source": os.path.basename(a.onnx), "source_sha256": sha(a.onnx)}
    if a.mode == "fp16":  # a dataset only feeds quantization; the record must not claim one
        for k in list(crec): res[k] = None
    r = RKNN(verbose=True, verbose_file=os.path.join(out, a.name + ".build.log"))
    configure(r, a)

    def step(k, fn):
        t = time.time()
        try:
            v = fn(); res["steps"][k] = "ok" if v in (0, None) else f"ret={v}"
            return v in (0, None)
        except Exception as e:
            res["steps"][k] = "ERR " + repr(e)[:300]; traceback.print_exc(); return False
        finally:
            res["secs"][k] = round(time.time() - t, 1)
    res["_step"] = step
    ok = step("load", lambda: r.load_onnx(model=a.onnx, **load_kw(a)))
    if a.float16 or (a.ranges and not a.hybrid):
        # step 1 + step 2 is a build whose per-tensor record is the cfg it was built from, at the cost of one
        # calibration (v9-t mmse, 20 frames: 400 s); v9-t normal gave build's parity counts exactly (2026-11-27)
        res["build_path"] = "hybrid_quantization_step1+2"
        ok = ok and step("build", lambda: step1(r, a, dataset))
        if a.float16: ok = ok and step("float16", lambda: pin_float16(a, res))
        if ok:
            r = RKNN(verbose=True, verbose_file=os.path.join(out, a.name + ".step2.log"))
            configure(r, a)
            ok = step("build_step2", lambda: r.hybrid_quantization_step2(
                f"{a.name}.model", f"{a.name}.data", f"{a.name}.quantization.cfg"))
    else:
        res["build_path"] = "build"
        ok = ok and step("build", lambda: r.build(do_quantization=(a.mode == "i8"), dataset=dataset, auto_hybrid=a.hybrid))
    sys.stdout.flush()
    lines = open(os.path.join(out, a.name + ".out"), errors="replace").read().splitlines()
    try:
        rows = summarize.optable(lines)
    except ValueError as e:
        rows = []; res["steps"]["optable"] = "ERR " + str(e)[:300]; ok = False
    res["op_targets"] = dict(sorted(collections.Counter(x[3] for x in rows).items()))
    res["off_npu"] = dict(summarize.off_npu(rows))
    res["warnings"] = dict(summarize.warnings(lines).most_common())
    if a.mode == "i8":
        f16 = quant_report.float16_layers(rows)
        res["float16_layers"] = len(f16); res["float16_layer_list"] = f16
    if ok and a.refuse_cpu:
        if not rows: res["steps"]["cpu"] = "ERR no operator table in stdout"; ok = False
        elif res["off_npu"]: res["steps"]["cpu"] = f"ERR off the NPU: {res['off_npu']}"; ok = False
        else: res["steps"]["cpu"] = "ok"
    ok = ok and step("export", lambda: r.export_rknn(os.path.join(out, a.name + ".rknn")))
    if ok: res["rknn_sha256"] = sha(os.path.join(out, a.name + ".rknn"))
    ok = ok and step("load_rknn", lambda: RKNN().load_rknn(os.path.join(out, a.name + ".rknn")))  # the file itself, not the live graph
    if ok: res["size_mb"] = round(os.path.getsize(os.path.join(out, a.name + ".rknn")) / 1e6, 2)
    for ext in ("model", "data"):
        if os.path.exists(f"{a.name}.{ext}"): os.remove(f"{a.name}.{ext}")  # step 2's inputs, 10-80 MB
    return (r if ok else None), res


def save(a, res):
    rec = {k: v for k, v in res.items() if not k.startswith("_")}
    json.dump(rec, open(os.path.join(a.out, a.name + ".json.tmp"), "w"), indent=1, default=str)
    os.replace(os.path.join(a.out, a.name + ".json.tmp"), os.path.join(a.out, a.name + ".json"))
    return rec


def read_frame(f):
    """A calibration or parity frame as the toolkit's inference takes it, and NCHW float for onnxruntime."""
    if f.endswith(".npy"):
        x = np.load(f)
        return x, "nchw", x.astype(np.float32)
    import cv2
    rgb = np.ascontiguousarray(cv2.imread(f)[:, :, ::-1][None])  # what the toolkit made of the PNG at calibration
    return rgb, "nhwc", rgb.transpose(0, 3, 1, 2).astype(np.float32)


def sim_parity(r, a, res):
    import onnxruntime as ort
    mean = np.array(vec(a.mean)[0], np.float32).reshape(1, 3, 1, 1)
    std = np.array(vec(a.std)[0], np.float32).reshape(1, 3, 1, 1)
    sess = ort.InferenceSession(a.onnx, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    names = [o.name for o in sess.get_outputs()]
    if a.parity:
        meta = json.load(open(a.parity + "/set.json"))
        frames = [(f"{a.parity}/{f.get('file') or f['npy']}", f"{f['w']}x{f['h']}") for f in meta["frames"]]
        dump = a.dump
    else:
        files = open(a.calib + "/dataset.txt").read().split()
        files = files[:: max(1, len(files) // 5)][:5]
        frames, dump = [(f, "all") for f in files], 5
    # the exporter's sidecar says whether cls* are logits; parity thresholds probabilities
    logits = onnx_logits(a)
    acc = parity.Acc(a.kind, names, logits) if a.parity else None
    sim, ref = [], []
    t0 = time.time()
    try:
        for n, (f, grp) in enumerate(frames):
            x, fmt, xf = read_frame(f)
            o = r.inference(inputs=[x], data_format=fmt)
            rf = sess.run(None, {iname: (xf - mean) / std})
            if n == 0: res["sim_outputs"] = [[list(t.shape), str(t.dtype)] for t in o]
            if n < dump: sim.append(o); ref.append(rf)
            if acc: acc.add(grp, o, rf)
        if sim:
            np.savez(os.path.join(a.out, a.name + ".parity.npz"),
                     **{f"sim{i}_{j}": t for i, o in enumerate(sim) for j, t in enumerate(o)},
                     **{f"ref{i}_{j}": t for i, o in enumerate(ref) for j, t in enumerate(o)},
                     names=np.array(names))
        if acc:
            groups = acc.result()
            rec = {"set_id": meta["id"], "n_frames": len(frames), "groups": groups}
            if a.kind == "mit":
                rec["verdict"] = parity.mit_verdict(groups["all"])
                res["parity_pass"] = rec["verdict"]["pass"]
            json.dump(rec, open(os.path.join(a.out, a.name + ".parity.json"), "w"), indent=1)
    except Exception as e:
        res["sim_outputs"] = "ERR " + repr(e)[:300]; traceback.print_exc()
    res["secs"]["parity"] = round(time.time() - t0, 1)


def step1(r, a, dataset):
    """hybrid_quantization_step1 (proposal=False) writes ONNXSTEM.{model,data,quantization.cfg} to the cwd;
    -> 0 once they are renamed NAME.*, so a stale file of another build cannot be read."""
    stem = os.path.splitext(os.path.basename(a.onnx))[0]
    v = r.hybrid_quantization_step1(dataset=dataset, proposal=False)
    if v != 0: return v
    for ext in ("model", "data", "quantization.cfg"):
        if not os.path.exists(f"{stem}.{ext}"): raise RuntimeError(f"step 1 wrote no {stem}.{ext}")
        os.replace(f"{stem}.{ext}", f"{a.name}.{ext}")
    return 0


def pin_float16(a, res):
    p = f"{a.name}.quantization.cfg"
    text, res["float16_tensors"] = quant_report.pin_float16(open(p).read(), a.float16.split(","))
    open(p, "w").write(text)


def ranges(a, res, dataset):
    """Without NAME.quantization.cfg from the build (an --hybrid build is made by build()), calibrate again
    through step 1 in its own instance; without auto_hybrid its scales equal the build's (v9-t normal, 2026-11-27:
    cls0 0.0037029 both, read off the simulator's output steps)."""
    if not os.path.exists(f"{a.name}.quantization.cfg"):
        from rknn.api import RKNN
        r2 = RKNN(verbose=False)
        configure(r2, a)
        ok = res["_step"]("ranges", lambda: r2.load_onnx(model=a.onnx, **load_kw(a)) or step1(r2, a, dataset))
        r2.release()
        if not ok: return
    else:
        res["steps"]["ranges"] = "ok"
    for ext in ("model", "data"):
        if os.path.exists(f"{a.name}.{ext}"): os.remove(f"{a.name}.{ext}")  # step 2's inputs, 10-80 MB
    cfg = quant_report.parse_cfg(open(f"{a.name}.quantization.cfg").read())
    # a pinned tensor's int8 parameters stay in the cfg but describe nothing the exported model runs, so the
    # reports leave it out (a Concat over one counts as unmatched)
    f16 = set(res.get("float16_tensors") or ())
    cfg = {k: v for k, v in cfg.items() if k not in f16 and k.removesuffix("_int8") not in f16}
    names = output_names(a)
    outs = quant_report.qparams(cfg, names)
    for o in outs:
        if o["name"].startswith("cls") and o.get("quantized"):
            # a logits output's representable maximum is a logit; the ceiling that matters is its probability
            o["max_score"] = float(1 / (1 + np.exp(-o["max_representable"]))) if onnx_logits(a) else o["max_representable"]
    box = [n for n in names if n.startswith("box")]
    narrow, unmatched = quant_report.narrow_concat(cfg, quant_report.onnx_concats(a.onnx))
    rep = {"outputs": outs, "clipped_box": quant_report.clipped(cfg, box),
           "clipped_any_output": quant_report.clipped(cfg, names),
           "score_ceiling": [o for o in outs if "max_score" in o and o["max_score"] < 1.0],
           "narrow_concat": narrow, "unmatched_concats": unmatched, "widest": quant_report.widest(cfg)}
    # step 1 knows nothing of the layers auto_hybrid moved to float16, so under --hybrid this is the plan
    # before those moves and none of it is recorded as the exported model's
    if a.hybrid: rep = {"plan": "before auto_hybrid", **rep}
    json.dump(rep, open(f"{a.name}.ranges.json", "w"), indent=1)
    if a.hybrid: return
    res["outputs"] = outs
    res["flags"] = {k: len(rep[k]) for k in ("clipped_box", "score_ceiling", "narrow_concat")}


def analysis(r, a, res):
    files = open(calib_dataset(a, a.out)[0]).read().split()
    files = files[:: max(1, len(files) // a.analysis)][: a.analysis]
    res["analysis_inputs"] = files
    runs = []
    t0 = time.time()
    try:
        for i, f in enumerate(files):
            snap = os.path.join(a.out, f"{a.name}.snap{i}")
            r.accuracy_analysis(inputs=[f], output_dir=snap, target=None)
            text = open(os.path.join(snap, "error_analysis.txt")).read()
            open(os.path.join(a.out, f"{a.name}.error_analysis{i}.txt"), "w").write(text)
            shutil.rmtree(snap)
            runs.append(quant_report.parse_error_analysis(text))
        layers = quant_report.merge_analysis(runs)
        json.dump({"inputs": files, "layers": layers,
                   "worst_single": quant_report.worst(layers, "single_cos"),
                   "worst_entire": quant_report.worst(layers, "entire_cos")},
                  open(os.path.join(a.out, a.name + ".analysis.json"), "w"), indent=1)
        res["steps"]["analysis"] = "ok"
        res["worst_single"] = [(l["name"], round(l["single_cos"], 5)) for l in quant_report.worst(layers, "single_cos", 5)]
    except Exception as e:
        res["steps"]["analysis"] = "ERR " + repr(e)[:300]; traceback.print_exc()
    res["secs"]["analysis"] = round(time.time() - t0, 1)


def input_size(a):
    """(w, h) the graph takes."""
    if a.size: return tuple(map(int, a.size.split("x")))
    import onnx
    dims = onnx.load(a.onnx, load_external_data=False).graph.input[0].type.tensor_type.shape.dim
    return dims[3].dim_value, dims[2].dim_value


def post_steps(r, a, res):
    """Parity, ranges and analysis on the live graph. accuracy_analysis runs inference itself, so the graph's
    outputs are fingerprinted before and after: a scorer that runs after these must see the graph that was built."""
    infer = lambda x: r.inference(inputs=[x[None]], data_format="nhwc")
    size = input_size(a)
    res["graph_id"] = before = graph_id(infer, size)
    if a.calib: sim_parity(r, a, res)
    if a.ranges: ranges(a, res, calib_dataset(a, a.out)[0])
    if a.analysis and a.mode == "i8": analysis(r, a, res)
    res["steps"]["graph_unchanged"] = "ok" if graph_id(infer, size) == before else "ERR outputs moved during parity/ranges/analysis"


def main(argv=None):
    a = parser().parse_args(argv)
    check_args(a)
    r, res = build(a)
    if r is not None:
        # simulator run: shapes of what the model emits
        if res["_step"]("sim_init", lambda: r.init_runtime()):
            try:
                post_steps(r, a, res)
            except Exception as e:  # the .rknn is exported by now; its record must still be written
                res["steps"]["post"] = "ERR " + repr(e)[:300]; traceback.print_exc()
    rec = save(a, res)
    print(json.dumps(rec, indent=1, default=str))
    sys.exit(int(any(v != "ok" for v in rec["steps"].values()) or isinstance(rec.get("sim_outputs"), str)
                 or rec.get("parity_pass") is False))


if __name__ == "__main__":
    main()
