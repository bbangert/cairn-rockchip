import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

import export_mit_yolov9 as ex

W, H = 64, 32  # small, but every stride divides it: grids 8x4, 4x2, 2x1


def build(size=(W, H), logits=False, with_sum=True, extra_nodes=(), extra_inits=(), box0_sigmoid=False):
    w, h = size
    rng = np.random.default_rng(0)
    nodes, inits = [], []

    def init(name, arr):
        inits.append(numpy_helper.from_array(np.asarray(arr, np.float32), name))
        return name

    # a SiLU in the stem, as the real graph has everywhere: Sigmoid + Mul that must not count as output sigmoids
    nodes += [helper.make_node("Conv", ["images", init("stem_w", rng.normal(size=(3, 3, 1, 1))), init("stem_b", np.zeros(3))],
                               ["stem"], name="stem", kernel_shape=[1, 1]),
              helper.make_node("Sigmoid", ["stem"], ["stem_sig"], name="stem_sig"),
              helper.make_node("Mul", ["stem", "stem_sig"], ["feat"], name="stem_silu")]
    outputs = []
    for i, s in enumerate(ex.STRIDES):
        grid = [h // s, w // s]
        nodes.append(helper.make_node("Conv", ["feat", init(f"bw{i}", np.zeros((ex.BOX_CH, 3, s, s)))], [f"box{i}_raw" if box0_sigmoid and i == 0 else f"box{i}"],
                                      name=f"box_conv{i}", kernel_shape=[s, s], strides=[s, s]))
        if box0_sigmoid and i == 0:
            nodes.append(helper.make_node("Sigmoid", ["box0_raw"], ["box0"], name="box0_sig"))
        cls_in = f"cls{i}" if logits else f"cls{i}_raw"
        nodes.append(helper.make_node("Conv", ["feat", init(f"cw{i}", np.zeros((ex.NC, 3, s, s)))], [cls_in],
                                      name=f"cls_conv{i}", kernel_shape=[s, s], strides=[s, s]))
        if not logits:
            nodes.append(helper.make_node("Sigmoid", [cls_in], [f"cls{i}"], name=f"cls_sig{i}"))
        outputs += [(f"box{i}", [1, ex.BOX_CH, *grid]), (f"cls{i}", [1, ex.NC, *grid])]
        if with_sum and not logits:
            nodes.append(helper.make_node("ReduceSum", [f"cls{i}"], [f"sum{i}"], name=f"sum{i}", axes=[1], keepdims=1))
            outputs.append((f"sum{i}", [1, 1, *grid]))
    graph = helper.make_graph(
        [*nodes, *extra_nodes], "g",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, h, w])],
        [helper.make_tensor_value_info(n, TensorProto.FLOAT, s) for n, s in outputs], [*inits, *extra_inits])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 12)])


def init(name, shape):
    return numpy_helper.from_array(np.zeros(shape, np.float32), name)


def test_good_graph_passes_and_reports():
    rec = ex.check_graph(build(), (W, H), logits=False, with_sum=True)
    assert [o[0] for o in rec["outputs"]] == ["box0", "cls0", "sum0", "box1", "cls1", "sum1", "box2", "cls2", "sum2"]
    assert rec["output_sigmoids"] == 3
    # the stem's SiLU sigmoid is in the graph and is not one of the output sigmoids
    assert rec["op_histogram"]["Sigmoid"] == 4
    assert rec["opset"] == 12 and rec["resize"] == []


@pytest.mark.parametrize("logits,with_sum", [(False, False), (True, False)])
def test_other_output_sets_pass(logits, with_sum):
    rec = ex.check_graph(build(logits=logits, with_sum=with_sum), (W, H), logits, with_sum)
    assert [o[0] for o in rec["outputs"]] == ["box0", "cls0", "box1", "cls1", "box2", "cls2"]
    assert rec["output_sigmoids"] == (0 if logits else 3)


def test_rectangular_size_shapes():
    rec = ex.check_graph(build(size=(32, 64)), (32, 64), False, True)
    assert dict(rec["outputs"])["box0"] == [1, 64, 8, 4] and dict(rec["outputs"])["sum2"] == [1, 1, 2, 1]
    with pytest.raises(ValueError, match="images is"):
        ex.check_graph(build(size=(W, H)), (H, W), False, True)


def test_logits_with_sum_is_refused():
    with pytest.raises(ValueError, match="cannot be combined"):
        ex.check_graph(build(logits=True, with_sum=False), (W, H), True, True)


def fails(model, match, logits=False, with_sum=True, size=(W, H)):
    with pytest.raises(ValueError, match=match):
        ex.check_graph(model, size, logits, with_sum)


def test_softmax_is_refused():
    fails(build(extra_nodes=[helper.make_node("Softmax", ["feat"], ["sm"], name="dfl")]), "Softmax node 'dfl'")


def test_batchnorm_is_refused():
    node = helper.make_node("BatchNormalization", ["feat", "g", "b", "m", "v"], ["bn"], name="bn0")
    fails(build(extra_nodes=[node], extra_inits=[init(n, [3]) for n in "gbmv"]), "BatchNormalization node 'bn0'")


def test_3d_conv_is_refused():
    node = helper.make_node("Conv", ["feat", "w5"], ["c3"], name="dfl_conv", kernel_shape=[1, 1, 1])
    fails(build(extra_nodes=[node], extra_inits=[init("w5", [4, 3, 1, 1, 1])]), "3-D Conv 'dfl_conv'")


def test_5d_reshape_and_transpose_are_refused():
    shape = numpy_helper.from_array(np.array([1, 4, 16, 8, 4], np.int64), "shape5")
    fails(build(extra_nodes=[helper.make_node("Reshape", ["feat", "shape5"], ["r"], name="rs")], extra_inits=[shape]),
          "Reshape 'rs' to 5-D")
    fails(build(extra_nodes=[helper.make_node("Transpose", ["feat"], ["t"], name="tp", perm=[0, 1, 2, 3, 4])]),
          "Transpose 'tp' of a 5-D")


def test_5d_tensor_anywhere_is_refused():
    model = build()
    model.graph.value_info.append(helper.make_tensor_value_info("stem", TensorProto.FLOAT, [1, 4, 16, 8, 4]))
    fails(model, "5-D tensor 'stem'")


@pytest.mark.parametrize("op", ["MaxPool", "AveragePool"])
def test_ceil_mode_is_refused(op):
    node = helper.make_node(op, ["feat"], ["p"], name="pool", kernel_shape=[2, 2], ceil_mode=1)
    fails(build(extra_nodes=[node]), f"{op} 'pool' has ceil_mode=1")


def test_wrong_output_name_order_and_shape_are_refused():
    model = build()
    model.graph.output[0].name, model.graph.output[1].name = "cls0", "box0"
    fails(model, "outputs")
    model = build()
    out = list(model.graph.output)
    del model.graph.output[:]
    model.graph.output.extend([out[1], out[0], *out[2:]])
    fails(model, "outputs")
    fails(build(), "outputs", with_sum=False)  # an extra sum tensor
    fails(build(with_sum=False), "outputs", with_sum=True)  # a missing one
    model = build()
    model.graph.output[0].type.tensor_type.shape.dim[1].dim_value = 63
    fails(model, "outputs")


def test_missing_and_extra_output_sigmoids_are_refused():
    fails(build(logits=True, with_sum=False), "0 Sigmoid nodes produce graph outputs, want 3", logits=False, with_sum=False)
    fails(build(with_sum=False), "3 Sigmoid nodes produce graph outputs, want 0", logits=True, with_sum=False)
    fails(build(with_sum=False, box0_sigmoid=True), "4 Sigmoid nodes produce graph outputs, want 3", with_sum=False)


def test_cls_sigmoid_must_be_a_bare_squash_on_the_80_channel_conv():
    # a SiLU-shaped consumer on the cls Sigmoid's output
    node = helper.make_node("Mul", ["cls0", "cls0_raw"], ["silu"], name="silu_like")
    fails(build(extra_nodes=[node]), r"cls0 Sigmoid also feeds \['Mul'\]")
    model = build()
    model.graph.initializer[[i.name for i in model.graph.initializer].index("cw1")].CopyFrom(init("cw1", [79, 3, 16, 16]))
    fails(model, "80-channel Conv")


def test_box_output_must_come_from_a_64_channel_conv():
    model = build()
    model.graph.initializer[[i.name for i in model.graph.initializer].index("bw2")].CopyFrom(init("bw2", [63, 3, 32, 32]))
    fails(model, "64-channel Conv")


def test_sum_must_be_a_reducesum_of_cls():
    model = build()
    next(n for n in model.graph.node if n.name == "sum1").input[0] = "cls1_raw"
    fails(model, "sum1 is not ReduceSum")


def test_dead_branch_is_refused():
    node = helper.make_node("Conv", ["feat", "aux_w"], ["aux"], name="aux_conv", kernel_shape=[8, 8], strides=[8, 8])
    fails(build(extra_nodes=[node], extra_inits=[init("aux_w", [64, 3, 8, 8])]), "1 nodes feed no output")


def test_resize_attributes_and_scales_are_recorded():
    model = build()
    model.graph.initializer.append(numpy_helper.from_array(np.array([1, 1, 1, 1], np.float32), "scales"))
    model.graph.initializer.append(numpy_helper.from_array(np.array([], np.float32), "roi"))
    rs = helper.make_node("Resize", ["stem", "roi", "scales"], ["stem_rs"], name="up", mode="nearest",
                          coordinate_transformation_mode="asymmetric", nearest_mode="floor")
    model.graph.node.insert(1, rs)
    next(n for n in model.graph.node if n.name == "stem_sig").input[0] = "stem_rs"
    rec = ex.check_graph(model, (W, H), False, True)
    assert rec["resize"] == [{"name": "up", "mode": "nearest", "coordinate_transformation_mode": "asymmetric",
                              "nearest_mode": "floor", "scales": [1.0, 1.0, 1.0, 1.0], "sizes": None}]
    assert "asymmetric" in ex.resize_note(rec) and "half_pixel" in ex.resize_note(rec)


def test_export_refuses_a_clone_that_is_not_the_pinned_one(monkeypatch, tmp_path):
    import sources

    def dirty(mit, clone):
        raise sources.SourceError("YOLO has local changes")

    monkeypatch.setattr(sources, "check_clone", dirty)
    with pytest.raises(SystemExit, match="not the pinned one: YOLO has local changes"):
        ex.main(["v9-t", "--out", str(tmp_path / "m.onnx")])
    assert not (tmp_path / "m.onnx").exists()


def test_forget_drops_the_resume_marker_and_tolerates_none(tmp_path):
    out = tmp_path / "m.onnx"
    (tmp_path / "m.onnx.json").write_text("{}")
    ex.forget(str(out))
    assert not (tmp_path / "m.onnx.json").exists()
    ex.forget(str(out))


def test_class_clamp_is_checked_for_count_bounds_and_sole_consumer(monkeypatch):
    import onnx_edit

    assert ex.CLASS_CLAMP_AT == [f"/net/class_conv/class_conv.1/conv{s}/Conv_output_0" for s in ("", "_1", "_2")]
    monkeypatch.setattr(ex, "CLASS_CLAMP_AT", ["stem"])  # the toy graph's one SiLU input
    clamped = onnx_edit.clamp(build(), ["stem"], 32.0)
    assert ex.check_graph(clamped, (W, H), False, True, 32.0)["op_histogram"]["Clip"] == 1
    fails(clamped, "Clip nodes in an export without")
    with pytest.raises(ValueError, match="is not"):
        ex.check_graph(clamped, (W, H), False, True, 16.0)
    ex.check_graph(onnx_edit.clamp(build(), ["stem"], 32.1), (W, H), False, True, 32.1)  # not a float32 value
    with pytest.raises(ValueError, match="Clip inputs"):
        ex.check_graph(build(), (W, H), False, True, 32.0)
    side = onnx_edit.clamp(build(), ["stem"], 32.0)
    side.graph.node.append(helper.make_node("Mul", ["stem", "stem"], ["dead"], name="raw_reader"))
    with pytest.raises(ValueError, match="also feeds"):
        ex.check_graph(side, (W, H), False, True, 32.0)


def test_class_clamp_must_be_positive(tmp_path):
    with pytest.raises(SystemExit):
        ex.main(["v9-t", "--out", str(tmp_path / "m.onnx"), "--class-clamp", "0"])
