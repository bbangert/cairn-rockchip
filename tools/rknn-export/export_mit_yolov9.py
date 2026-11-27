#!/usr/bin/env python3
"""ONNX export of an MIT YOLOv9 (v9-t, v9-s, v9-m) cut at the head's last convs, for the RKNN conversion.

Outputs are the split layout core's SplitDfl reads, by name (box0..2 [1,64,h,w] DFL logits,
cls0..2 [1,80,h,w] sigmoided) plus, as the ultralytics split rungs carry them, sum0..2
[1,1,h,w] = cls summed over classes, interleaved box0 cls0 sum0 box1 ... Core binds by name and
never reads sum*, so --no-sum is still accepted by it. --logits leaves cls raw and emits no
sum: a sum over logits means nothing. --class-clamp K clips the class neck's first conv output
(class_conv.1, before its SiLU) to [-K, K] at each stride: on v9-t one or two of its channels reach
85-627 against a median channel maximum of 6-7, and per-tensor INT8 gives them the step of all. That
changes the model, so the sidecar records it and onnx_proof.py refuses to compare it with the torch
module. check_graph asserts the saved graph with onnx alone.
usage: export_mit_yolov9.py NAME --out FILE.onnx [--size 640|WxH] [--logits] [--fuse] [--no-sum] [--class-clamp K]
"""
import argparse, collections, json, os, sys

import numpy as np
import onnx
from onnx import numpy_helper

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # python -I leaves the script directory off sys.path

from decode_split import parse_size

OPSET = 12  # as export_ultralytics_rknn.py
STRIDES = (8, 16, 32)
NC = 80
BOX_CH = 64  # 4 sides x 16 DFL bins
CLASS_CLAMP_AT = [f"/net/class_conv/class_conv.1/conv{s}/Conv_output_0" for s in ("", "_1", "_2")]


def expected_outputs(size, logits, with_sum):
    w, h = size
    out = []
    for i, s in enumerate(STRIDES):
        grid = [h // s, w // s]
        out.append((f"box{i}", [1, BOX_CH, *grid]))
        out.append((f"cls{i}", [1, NC, *grid]))
        if with_sum and not logits:
            out.append((f"sum{i}", [1, 1, *grid]))
    return out


def _dims(value_info):
    return [d.dim_value if d.HasField("dim_value") else None for d in value_info.type.tensor_type.shape.dim]


def _attr(node, name, default=None):
    for a in node.attribute:
        if a.name == name:
            return onnx.helper.get_attribute_value(a)
    return default


def check_graph(model, size, logits, with_sum, class_clamp=None):
    """Raises ValueError naming the first violated property; -> the record written beside the model."""
    w, h = size
    if logits and with_sum:
        raise ValueError("logits and sum tensors cannot be combined")
    g = model.graph
    nodes = list(g.node)
    consts = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    for n in nodes:
        if n.op_type == "Constant" and _attr(n, "value") is not None:
            consts[n.output[0]] = numpy_helper.to_array(_attr(n, "value"))
    producer = {o: n for n in nodes for o in n.output}
    consumers = collections.defaultdict(list)
    for n in nodes:
        for i in n.input:
            consumers[i].append(n)

    if [i.name for i in g.input if i.name not in consts] != ["images"]:
        raise ValueError(f"inputs {[i.name for i in g.input]}, want ['images']")
    img = next(i for i in g.input if i.name == "images")
    if img.type.tensor_type.elem_type != onnx.TensorProto.FLOAT or _dims(img) != [1, 3, h, w]:
        raise ValueError(f"input images is {_dims(img)}, want float32 [1, 3, {h}, {w}]")

    clips = [n for n in nodes if n.op_type == "Clip"]
    if class_clamp is None and clips:
        raise ValueError(f"{len(clips)} Clip nodes in an export without --class-clamp")
    if class_clamp is not None:
        if sorted(n.input[0] for n in clips) != sorted(CLASS_CLAMP_AT):
            raise ValueError(f"Clip inputs {sorted(n.input[0] for n in clips)}, want {sorted(CLASS_CLAMP_AT)}")
        for n in clips:
            # the graph holds float32 bounds: 32.1 is stored as 32.099998
            if [consts[t] for t in n.input[1:]] != [np.float32(-class_clamp), np.float32(class_clamp)]:
                raise ValueError(f"Clip {n.name!r} is not [-{class_clamp}, {class_clamp}]")
            # the SiLU's Sigmoid and Mul must both read the clipped tensor
            if [c.op_type for c in consumers[n.input[0]]] != ["Clip"]:
                raise ValueError(f"{n.input[0]!r} also feeds {[c.op_type for c in consumers[n.input[0]]]}")

    resize = []
    for n in nodes:
        t = n.op_type
        if t in ("Softmax", "LogSoftmax"):
            raise ValueError(f"{t} node {n.name!r} in the graph: the DFL softmax belongs to the host decode")
        if t == "BatchNormalization":
            raise ValueError(f"BatchNormalization node {n.name!r}: Conv+BN was not folded")
        if t == "Conv":
            weight = consts.get(n.input[1])
            if weight is None:
                raise ValueError(f"Conv {n.name!r} has a weight that is not a constant")
            if weight.ndim != 4 or len(_attr(n, "kernel_shape", [0, 0])) != 2:
                raise ValueError(f"3-D Conv {n.name!r}: weight shape {list(weight.shape)}")
        if t == "Reshape" and n.input[1] in consts and len(consts[n.input[1]]) > 4:
            raise ValueError(f"Reshape {n.name!r} to {len(consts[n.input[1]])}-D shape {consts[n.input[1]].tolist()}")
        if t == "Transpose" and len(_attr(n, "perm", [])) > 4:
            raise ValueError(f"Transpose {n.name!r} of a {len(_attr(n, 'perm'))}-D tensor")
        if t in ("MaxPool", "AveragePool") and _attr(n, "ceil_mode", 0) == 1:
            raise ValueError(f"{t} {n.name!r} has ceil_mode=1")
        if t == "Resize":
            scales = consts.get(n.input[2]) if len(n.input) > 2 and n.input[2] else None
            sizes = consts.get(n.input[3]) if len(n.input) > 3 and n.input[3] else None
            dec = lambda k, d: (_attr(n, k, d.encode()) or d.encode()).decode()
            resize.append({"name": n.name, "mode": dec("mode", "nearest"),
                           "coordinate_transformation_mode": dec("coordinate_transformation_mode", "half_pixel"),
                           "nearest_mode": dec("nearest_mode", "round_prefer_floor"),
                           "scales": None if scales is None else scales.tolist(),
                           "sizes": None if sizes is None else sizes.tolist()})

    inferred = onnx.shape_inference.infer_shapes(model).graph
    for v in [*inferred.value_info, *inferred.output]:
        if len(_dims(v)) > 4:
            src = producer.get(v.name)
            raise ValueError(f"{len(_dims(v))}-D tensor {v.name!r} produced by {src.op_type if src else '?'}")

    want = expected_outputs((w, h), logits, with_sum)
    got = [(o.name, _dims(o)) for o in inferred.output]
    if got != want:
        raise ValueError(f"outputs {got}, want {want}")

    out_names = {o.name for o in g.output}
    out_sigmoids = [n for n in nodes if n.op_type == "Sigmoid" and n.output[0] in out_names]
    if len(out_sigmoids) != (0 if logits else 3):
        raise ValueError(f"{len(out_sigmoids)} Sigmoid nodes produce graph outputs, want {0 if logits else 3}")

    def head_conv(tensor, channels, what):
        n = producer.get(tensor)
        if n is None or n.op_type != "Conv" or consts[n.input[1]].shape[0] != channels:
            raise ValueError(f"{what} {tensor!r} is not produced by a {channels}-channel Conv "
                             f"(found {n.op_type if n else 'nothing'})")

    for i in range(3):
        head_conv(f"box{i}", BOX_CH, "box output")
        if logits:
            head_conv(f"cls{i}", NC, "class logits output")
            continue
        sig = producer[f"cls{i}"]
        head_conv(sig.input[0], NC, f"input of the cls{i} Sigmoid")
        # a SiLU is Sigmoid + Mul on the same tensor; the output Sigmoid is a bare squash
        extra = {c.op_type for c in consumers[sig.output[0]]} - ({"ReduceSum"} if with_sum else set())
        if extra:
            raise ValueError(f"cls{i} Sigmoid also feeds {sorted(extra)}")
        if with_sum:
            red = producer[f"sum{i}"]
            if red.op_type != "ReduceSum" or red.input[0] != f"cls{i}" or list(_attr(red, "axes", [])) != [1] \
                    or _attr(red, "keepdims", 1) != 1:
                raise ValueError(f"sum{i} is not ReduceSum(cls{i}, axes=[1], keepdims=1)")

    live, todo = set(), [o.name for o in g.output]
    while todo:
        t = todo.pop()
        n = producer.get(t)
        if n is not None and id(n) not in live:
            live.add(id(n))
            todo += list(n.input)
    dead = [n.name or n.op_type for n in nodes if id(n) not in live]
    if dead:
        raise ValueError(f"{len(dead)} nodes feed no output, e.g. {dead[:3]} (an auxiliary branch?)")

    return {"outputs": [[n, s] for n, s in got],
            "opset": next(o.version for o in model.opset_import if o.domain in ("", "ai.onnx")),
            "nodes": len(nodes),
            "op_histogram": dict(sorted(collections.Counter(n.op_type for n in nodes).items())),
            "output_sigmoids": len(out_sigmoids),
            "resize": resize,
            "initializers": len(g.initializer),
            "parameters": int(sum(int(np.prod(i.dims)) for i in g.initializer))}


def resize_note(check):
    seen = sorted({(r["mode"], r["coordinate_transformation_mode"], r["nearest_mode"],
                    tuple(r["scales"][2:]) if r["scales"] else None) for r in check["resize"]}, key=str)
    if not seen:
        return "Resize: none"
    desc = ", ".join(f"{m}/{c}/{nm} scales(h,w) {s}" for m, c, nm, s in seen)
    return (f"Resize x{len(check['resize'])}: {desc}; the RKNN op list names half_pixel, "
            f"pytorch_half_pixel and align_corners")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("usage:")[0].strip())
    ap.add_argument("name", choices=("v9-t", "v9-s", "v9-m"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=parse_size, default=(640, 640), help="640 or WxH (width x height), multiples of 32")
    ap.add_argument("--logits", action="store_true", help="raw class logits, no Sigmoid; implies no sum tensors")
    ap.add_argument("--fuse", action="store_true", help="fold each RepConv into one conv before export")
    ap.add_argument("--no-sum", action="store_true", help="omit sum0..2")
    ap.add_argument("--class-clamp", type=float, metavar="K", help="clip class_conv.1's conv outputs to [-K, K]")
    a = ap.parse_args(argv)
    if a.class_clamp is not None and not a.class_clamp > 0:
        ap.error("--class-clamp must be positive")

    import mit_model, sources

    with_sum = not (a.logits or a.no_sum)
    w, h = a.size
    try:
        sources.check_clone(sources.load_lock()["mit"], mit_model.MIT_DIR)
    except sources.SourceError as e:
        sys.exit(f"refusing to export from an MIT clone that is not the pinned one: {e}")
    undo = mit_model.block_network()
    try:
        return _export(a, with_sum, w, h)
    finally:
        undo()


def forget(out):
    # the sidecar is the resume marker: it goes before the ONNX changes, so a failed or interrupted re-export
    # is retried instead of leaving a marker for a file that is gone
    try:
        os.remove(out + ".json")
    except FileNotFoundError:
        pass


def _export(a, with_sum, w, h):
    import mit_model, sources
    import torch

    weights, sha, commit = mit_model.checkpoint(a.name)
    net = mit_model.load(a.name, weights, mit_model.MIT_DIR, sigmoid=not a.logits, fuse=a.fuse)

    class Split(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = net

        def forward(self, x):
            outs = self.net(x)
            if not with_sum:
                return outs
            res = []
            for i in range(3):
                box, cls = outs[2 * i:2 * i + 2]
                res += [box, cls, cls.sum(1, keepdim=True)]
            return tuple(res)

    names = [n for n, _ in expected_outputs((w, h), a.logits, with_sum)]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    forget(a.out)
    with torch.no_grad():
        torch.onnx.export(Split().eval(), torch.zeros(1, 3, h, w), a.out, input_names=["images"],
                          output_names=names, opset_version=OPSET, do_constant_folding=True)
    try:
        model = onnx.load(a.out)
        if a.class_clamp is not None:
            import onnx_edit
            onnx.save(onnx_edit.clamp(model, CLASS_CLAMP_AT, a.class_clamp, "class_clamp"), a.out)
        check = check_graph(model, (w, h), a.logits, with_sum, a.class_clamp)
    except BaseException:
        os.remove(a.out)
        raise
    record = {"name": a.name, "size": [w, h], "logits": a.logits, "fuse": a.fuse, "with_sum": with_sum,
              "class_clamp": a.class_clamp,
              "checkpoint_sha256": sha, "mit_commit": commit, "load": net.load_report,
              "torch": torch.__version__, "onnx": onnx.__version__, "numpy": np.__version__,
              "python": sys.version.split()[0], "opset": check["opset"],
              "tool_sha256": {f: sources.sha256_file(os.path.join(HERE, f))
                              for f in ("export_mit_yolov9.py", "mit_model.py", "fuse_repconv.py", "onnx_edit.py")},
              "onnx_sha256": sources.sha256_file(a.out), "check": check}
    # the sidecar is the resume marker, so it must never exist half-written
    with open(a.out + ".json.tmp", "w") as f:
        json.dump(record, f, indent=1)
        f.write("\n")
    os.replace(a.out + ".json.tmp", a.out + ".json")
    print(f"wrote {a.out}: {check['nodes']} nodes, {check['parameters']} parameters, "
          f"{os.path.getsize(a.out)} bytes")
    print(resize_note(check))
    return record


if __name__ == "__main__":
    main()
