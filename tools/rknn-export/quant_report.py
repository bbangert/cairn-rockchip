#!/usr/bin/env python3
"""What convert.py reads out of a quantised build, kept free of the toolkit so it can be tested.

- `parse_cfg`: the NAME.quantization.cfg `hybrid_quantization_step1` writes. Per activation
  tensor `min`/`max` is the range quantised to, `ori_min`/`ori_max` the float range calibration
  saw; `NAME_int8` carries scale and zero point. (No YAML module in the toolkit's venv; the file's
  shape is fixed.)
- `parse_error_analysis`: `accuracy_analysis`'s error_analysis.txt, one row per layer.
usage: quant_report.py CFG [ONNX]   prints output qparams and the range flags
"""
import json, re, sys

import numpy as np

CONCAT_FRACTION = 0.25  # a Concat input using less than this share of its output's range


def _scalar(v):
    v = v.strip().strip("'\"")
    try:
        return float(v)
    except ValueError:
        return v


def parse_cfg(text):
    """-> {tensor: {key: scalar | [scalars]}} from the quantize_parameters block."""
    out, name, key, inside = {}, None, None, False
    for ln in text.splitlines():
        if not ln.strip():
            continue
        if not ln.startswith(" "):
            inside = ln.startswith("quantize_parameters:")
            continue
        if not inside:
            continue
        indent = len(ln) - len(ln.lstrip(" "))
        body = ln.strip()
        if body.startswith("- ") or body == "-":
            if name is not None and key is not None:
                out[name][key].append(_scalar(body[1:]))
        elif indent == 4 and body.endswith(":"):
            name, key = body[:-1].strip().strip("'\""), None
            out[name] = {}
        elif indent == 8 and name is not None:
            key, _, val = body.partition(":")
            val = val.strip()
            out[name][key] = [] if val in ("", "[]") else ([_scalar(x) for x in val[1:-1].split(",")] if val.startswith("[")
                                                          else _scalar(val))
    return out


def _one(v):
    return v[0] if isinstance(v, list) and len(v) == 1 else v


def qparams(cfg, names):
    """Per output tensor: scale, zero point and the float values int8 can represent."""
    rows = []
    for n in names:
        q = cfg.get(n + "_int8") or cfg.get(n)
        if not q or q.get("dtype") != "int8":
            rows.append({"name": n, "quantized": False})
            continue
        scale, zp = float(_one(q["scale"])), int(_one(q["zero_point"]))
        rows.append({"name": n, "quantized": True, "scale": scale, "zero_point": zp,
                     "min_representable": (-128 - zp) * scale, "max_representable": (127 - zp) * scale,
                     "ori_min": float(_one(q["ori_min"])), "ori_max": float(_one(q["ori_max"]))})
    return rows


def clipped(cfg, names, rel=1e-6):
    """Tensors quantised to a narrower range than calibration saw (kl_divergence and mmse clip by design)."""
    bad = []
    for n in names:
        q = cfg.get(n + "_int8") or cfg.get(n)
        if not q:
            continue
        lo, hi, olo, ohi = (float(_one(q[k])) for k in ("min", "max", "ori_min", "ori_max"))
        tol = rel * max(1.0, ohi - olo)
        if hi < ohi - tol or lo > olo + tol:
            bad.append({"name": n, "min": lo, "max": hi, "ori_min": olo, "ori_max": ohi})
    return bad


def _range(cfg, n):
    q = cfg.get(n)
    if not q or "max" not in q or not q["max"]:
        return None
    return float(_one(q["max"])) - float(_one(q["min"]))


def narrow_concat(cfg, concats, fraction=CONCAT_FRACTION):
    """concats: [(input names, output name)]. Activations are per-tensor, so a Concat's inputs share its
    output's scale: an input using a small part of that range keeps few of the 256 levels.
    -> (flagged, concats whose tensors the toolkit renamed or fused away, so they could not be checked)."""
    flagged, unmatched = [], 0
    for ins, out in concats:
        r_out = _range(cfg, out)
        r_in = [_range(cfg, i) for i in ins]
        if r_out is None or any(r is None for r in r_in):
            unmatched += 1
            continue
        for i, r in zip(ins, r_in):
            if r_out > 0 and r < fraction * r_out:
                flagged.append({"concat": out, "input": i, "input_range": r, "output_range": r_out,
                                "share": r / r_out})
    return flagged, unmatched


def onnx_concats(path):
    import onnx

    g = onnx.load(path, load_external_data=False).graph
    return [(list(n.input), n.output[0]) for n in g.node if n.op_type == "Concat"]


ROW = re.compile(r"^\[(\w+)\]\s+(\S+)\s+([\d.]+)\s*\|\s*(\S+)\s+([\d.]+)\s*\|\s*(\S+)\s*$")


def parse_error_analysis(text):
    """-> [{op, name, entire_cos, entire_euc, single_cos, single_euc}]; rows without numbers (a layer the
    toolkit fused into the next) are skipped."""
    rows = []
    for ln in text.splitlines():
        m = ROW.match(ln.strip())
        if m:
            op, name, ec, ee, sc, se = m.groups()
            rows.append({"op": op, "name": name, "entire_cos": float(ec), "entire_euc": float(ee),
                         "single_cos": float(sc), "single_euc": float(se)})
    return rows


def merge_analysis(runs):
    """Per layer over several images (each its own accuracy_analysis): mean and min of both cosines, in the
    first run's layer order."""
    by = {}
    for rows in runs:
        for r in rows:
            by.setdefault((r["op"], r["name"]), []).append(r)
    out = []
    for (op, name), rs in by.items():
        d = {"op": op, "name": name, "images": len(rs)}
        for k in ("entire_cos", "single_cos"):
            v = np.array([r[k] for r in rs])
            d[k] = float(v.mean())
            d[k + "_min"] = float(v.min())
        out.append(d)
    return out


def worst(layers, key, n=15):
    # exDataConvert rows repeat the layer before them (the int8 -> float output cast)
    real = [l for l in layers if l["op"] not in ("Input", "exDataConvert")]
    return sorted(real, key=lambda l: l[key])[:n]


def widest(cfg, n=10):
    """The int8 activations with the largest range: per-tensor, so one outlier channel sets the step for all."""
    rows = [(k, float(_one(v["max"])) - float(_one(v["min"])), float(_one(v["scale"])))
            for k, v in cfg.items() if v.get("dtype") == "int8" and v.get("scale")]
    return [{"name": k, "range": r, "scale": sc} for k, r, sc in sorted(rows, key=lambda x: -x[1])[:n]]


def float16_layers(rows):
    """Ops left in float16 in an INT8 build (auto_hybrid's moves, and ops it could not quantise)."""
    return [f"{r[1]}#{r[0]}" for r in rows if r[2].upper().startswith("FLOAT16") and r[1] not in ("InputOperator", "OutputOperator")]


if __name__ == "__main__":
    cfg = parse_cfg(open(sys.argv[1]).read())
    outs = [n[:-5] for n in cfg if n.endswith("_int8") and re.match(r"(box|cls|sum)\d_int8$", n)]
    print(json.dumps({"outputs": qparams(cfg, outs), "clipped": clipped(cfg, outs)}, indent=1))
    if len(sys.argv) > 2:
        flagged, unmatched = narrow_concat(cfg, onnx_concats(sys.argv[2]))
        print(json.dumps({"narrow_concat": flagged, "unmatched_concats": unmatched}, indent=1))
