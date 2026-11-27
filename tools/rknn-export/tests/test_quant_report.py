import pytest
from pytest import approx

import quant_report as q
import summarize

# the shape hybrid_quantization_step1 writes (2.3.2), cut down
CFG = """custom_quantize_layers: {}
quantize_parameters:
    box0:
        qtype: asym
        dtype: float32
        min:
        -   -31.3
        max:
        -   8.4
        scale: []
        zero_point: []
        ori_min:
        -   -31.3
        ori_max:
        -   8.4
    box0_int8:
        qtype: asym
        dtype: int8
        min:
        -   -20.0
        max:
        -   8.4
        scale:
        -   0.1
        zero_point:
        -   72
        ori_min:
        -   -31.3
        ori_max:
        -   8.4
    cls0_int8:
        dtype: int8
        min:
        -   0.0
        max:
        -   0.9442
        scale:
        -   0.0037029
        zero_point:
        -   -128
        ori_min:
        -   0.0
        ori_max:
        -   0.9442
    '/a/out':
        dtype: int8
        min:
        -   0.0
        max:
        -   16.0
    /a/x:
        dtype: int8
        min:
        -   0.0
        max:
        -   3.0
    /a/y:
        dtype: int8
        min:
        -   0.0
        max:
        -   12.0
"""


def test_parse_cfg_lists_scalars_and_quoted_names():
    c = q.parse_cfg(CFG)
    assert c["box0"]["scale"] == [] and c["box0"]["min"] == [-31.3]
    assert c["box0_int8"]["zero_point"] == [72] and c["box0_int8"]["dtype"] == "int8"
    assert "/a/out" in c and c["/a/y"]["max"] == [12.0]


def test_qparams_give_the_representable_range():
    o = {r["name"]: r for r in q.qparams(q.parse_cfg(CFG), ["box0", "cls0", "sum0"])}
    assert o["cls0"]["max_representable"] == approx(255 * 0.0037029)
    assert o["box0"]["min_representable"] == approx(-200 * 0.1) and o["box0"]["zero_point"] == 72
    assert o["sum0"] == {"name": "sum0", "quantized": False}


def test_clipped_reads_the_int8_record():
    bad = q.clipped(q.parse_cfg(CFG), ["box0", "cls0"])
    assert [b["name"] for b in bad] == ["box0"] and bad[0]["min"] == -20.0 and bad[0]["ori_min"] == -31.3


def test_narrow_concat_flags_inputs_under_a_quarter():
    flagged, unmatched = q.narrow_concat(q.parse_cfg(CFG), [(["/a/x", "/a/y"], "/a/out"), (["/gone"], "/a/out")])
    assert [(f["input"], f["share"]) for f in flagged] == [("/a/x", approx(3 / 16))]
    assert unmatched == 1


ANALYSIS = """layer_name   simulator_error
---------------------------------------------
[Input] images                                     1.00000 | 0.0       1.00000 | 0.0
[Conv] /net/a/Conv_output_0                        0.99000 | 34.98     0.99900 | 34.98
[Conv] /net/class_conv.2/Conv_output_0
[Sigmoid] cls0_int8                                0.79333 | 0.0261    0.86670 | 0.0262
[exDataConvert] cls0                               0.79333 | 0.0261    0.88209 | 0.0203
"""


def test_error_analysis_rows_skip_fused_layers():
    rows = q.parse_error_analysis(ANALYSIS)
    assert [r["name"] for r in rows] == ["images", "/net/a/Conv_output_0", "cls0_int8", "cls0"]
    assert rows[2] == {"op": "Sigmoid", "name": "cls0_int8", "entire_cos": 0.79333, "entire_euc": 0.0261,
                       "single_cos": 0.8667, "single_euc": 0.0262}


def test_merge_and_worst_skip_input_and_output_casts():
    a = q.parse_error_analysis(ANALYSIS)
    b = [dict(r, single_cos=r["single_cos"] - 0.1) for r in a]
    m = q.merge_analysis([a, b])
    s = {r["name"]: r for r in m}["cls0_int8"]
    assert s["single_cos"] == approx(0.8167) and s["single_cos_min"] == approx(0.7667) and s["images"] == 2
    assert [r["name"] for r in q.worst(m, "single_cos")] == ["cls0_int8", "/net/a/Conv_output_0"]


TABLE = """D RKNN: [04:12:03.429] ID   OpType             DataType Target InputShape    OutputShape   Cycles  RW  FullName
D RKNN: [04:12:03.429] ----------------------------------------------------------------
D RKNN: [04:12:03.429] 0    InputOperator      INT8     CPU    \\           (1,3,640,640)  0/0/0  0  InputOperator:images
D RKNN: [04:12:03.429] 1    ConvExSwish        INT8     NPU    (1,3,640,640) (1,16,320,320) 1/2/3 2401 Conv:/a
D RKNN: [04:12:03.429] 2    Conv               FLOAT16  NPU    (1,16,320,320) (1,16,320,320) 1/2/3 2401 Conv:/b
D RKNN: [04:12:03.429] 3    AveragePool        INT8     CPU    (1,16,320,320) (1,16,320,320) 1/2/3 2401 Pool:/c
D RKNN: [04:12:03.429] 4    OutputOperator     FLOAT16  NPU    (1,16,320,320) (1,16,320,320) 0/0/0 0 Out:/d
D RKNN: [04:12:03.429] ----------------------------------------------------------------
W RKNN: [04:12:03.312] Warning: Tensor /x need parameter qtype
""".splitlines()


def test_optable_off_npu_excludes_the_graph_boundary():
    rows = summarize.optable(TABLE)
    assert len(rows) == 5
    assert dict(summarize.off_npu(rows)) == {"AveragePool[INT8]": 1}
    assert q.float16_layers(rows) == ["Conv#2"]
    assert dict(summarize.warnings(TABLE)) == {"W RKNN: Warning: Tensor /x need parameter qtype": 1}


def test_widest_ranks_int8_tensors_by_range():
    w = q.widest(q.parse_cfg(CFG), 2)
    assert [x["name"] for x in w] == ["box0_int8", "cls0_int8"] and w[0]["range"] == approx(28.4)


def test_a_line_inside_the_table_that_is_not_a_row_is_an_error():
    odd = TABLE[:3] + ["D RKNN: [04:12:03.429] 1    Conv  INT8  CPU0   (1,3,640,640) (1,16,320,320)"] + TABLE[3:]
    with pytest.raises(ValueError, match="unreadable"):
        summarize.optable(odd)
    with pytest.raises(ValueError, match="closing"):
        summarize.optable(TABLE[:5])
