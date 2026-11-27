#!/usr/bin/env python3
"""ONNX graph edits shared by the export scripts. usage: onnx_edit.py resize IN.onnx OUT.onnx HxW"""
import sys
import numpy as np
import onnx
from onnx import helper, numpy_helper


def rename_outputs(model, names):
    g = model.graph
    if len(names) != len(g.output):
        raise ValueError(f"{len(names)} names for {len(g.output)} outputs")
    taken = {t for n in g.node for t in (*n.input, *n.output)}
    taken |= {i.name for i in g.initializer} | {i.name for i in g.input}
    taken |= {v.name for v in g.value_info}
    # a rename onto a name the graph already uses would merge two tensors
    clash = [n for o, n in zip(g.output, names) if n != o.name and n in taken]
    if clash:
        raise ValueError(f"names already in the graph: {clash}")
    mapping = {o.name: n for o, n in zip(g.output, names)}
    # consumers too: in a split head the cls tensor also feeds its ReduceSum
    for n in g.node:
        n.input[:] = [mapping.get(t, t) for t in n.input]
        n.output[:] = [mapping.get(t, t) for t in n.output]
    for v in g.value_info:
        v.name = mapping.get(v.name, v.name)
    for o, n in zip(g.output, names):
        o.name = n
    return model


def clamp(model, tensors, k, prefix="clamp"):
    """Clip(-k, k) on each named tensor, every consumer reading the clipped one."""
    g = model.graph
    produced = {o for n in g.node for o in n.output}
    if [t for t in tensors if t not in produced] or {o.name for o in g.output} & set(tensors):
        raise ValueError(f"not all of {tensors} are internal tensors of the graph")
    lo, hi = f"{prefix}_min", f"{prefix}_max"
    g.initializer.extend([numpy_helper.from_array(np.array(-k, np.float32), lo),
                          numpy_helper.from_array(np.array(k, np.float32), hi)])
    nodes = []
    for n in g.node:
        n.input[:] = [t + "_clamped" if t in tensors else t for t in n.input]
        nodes.append(n)
        # right after its producer, so the node list stays topologically sorted
        nodes += [helper.make_node("Clip", [o, lo, hi], [o + "_clamped"], name=o + "_clamp") for o in n.output if o in tensors]
    del g.node[:]
    g.node.extend(nodes)
    return model


def set_input_size(model, h, w):
    g = model.graph
    if len(g.input) != 1:
        raise ValueError("expected a single graph input")
    dims = g.input[0].type.tensor_type.shape.dim
    dims[2].dim_value, dims[3].dim_value = h, w
    # stale value_info and output dims pin the old size and make inference
    # keep them instead of recomputing
    del g.value_info[:]
    for o in g.output:
        for d in o.type.tensor_type.shape.dim:
            d.Clear()
    return onnx.shape_inference.infer_shapes(model)


if __name__ == "__main__":
    cmd, src, dst, size = sys.argv[1:5]
    assert cmd == "resize"
    h, w = map(int, size.split("x"))
    onnx.save(set_input_size(onnx.load(src), h, w), dst)
