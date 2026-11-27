import json

import numpy as np
import pytest

import onnx_proof as op

NAN, INF = float("nan"), float("inf")


def test_stem_carries_flags_size_and_limit():
    meta = {"fuse": True, "logits": True, "size": (640, 384)}
    assert op.variant_stem("v9-t", meta, "origin", "bilinear-cv2", 200) == \
        "v9-t-onnx-fused-logits-origin-bilinear-cv2-640x384-limit200"
    assert op.variant_stem("v9-s", {"fuse": False, "logits": False, "size": (640, 640)}, "origin", "bilinear-cv2", None) == \
        "v9-s-onnx-origin-bilinear-cv2"


def test_stem_tells_a_no_sum_model_apart_and_reads_old_sidecars_as_with_sum():
    base = {"fuse": True, "logits": False, "size": (640, 640)}
    stem = lambda **k: op.variant_stem("v9-t", {**base, **k}, "origin", "bilinear-cv2", None)
    assert stem(with_sum=False) == "v9-t-onnx-fused-nosum-origin-bilinear-cv2"
    assert stem(with_sum=True) == stem() == "v9-t-onnx-fused-origin-bilinear-cv2"
    assert stem(logits=True, with_sum=False) == "v9-t-onnx-fused-logits-origin-bilinear-cv2"  # logits never have a sum


def test_host_sigmoid_saturates_without_nan():
    got = op.sigmoid(np.array([-400, 0, 400], np.float32))
    assert got.tolist() == [0.0, 0.5, 1.0] and got.dtype == np.float32


def gate(theirs, ours):
    return op.tensor_failures(op.accumulate({}, {"box0": np.array(theirs, np.float32)}, {"box0": np.array(ours, np.float32)}))


def test_difference_inside_tolerance_passes_and_outside_fails():
    assert gate([1.0, 2.0], [1.0, 2.0 + 5e-4]) == {}
    assert "box0" in gate([1.0, 2.0], [1.0, 2.0 + 5e-3])


def test_tolerance_is_relative_to_max_of_one_and_the_reference_magnitude():
    # 5e-3 passes against a reference of magnitude 30 (limit 3e-2) and fails against one of 0.5 (limit 1e-3)
    assert gate([30.0, 0.0], [30.0 + 5e-3, 0.0]) == {}
    assert gate([0.5, 0.0], [0.5 + 5e-3, 0.0])


@pytest.mark.parametrize("theirs,ours", [([NAN, 1.0], [1.0, 1.0]), ([1.0, 1.0], [NAN, 1.0]),
                                          ([INF, 1.0], [INF, 1.0]), ([INF, 1.0], [1.0, 1.0]), ([1.0, 1.0], [INF, 1.0])])
def test_non_finite_output_on_either_side_fails_naming_the_tensor(theirs, ours):
    bad = gate(theirs, ours)
    assert list(bad) == ["box0"] and "box0" in bad["box0"] and "non-finite" in bad["box0"]


def test_a_nan_in_a_later_image_is_not_dropped_by_the_running_max():
    worst = op.accumulate({}, {"box0": np.array([1.0], np.float32)}, {"box0": np.array([1.0], np.float32)})
    worst = op.accumulate(worst, {"box0": np.array([NAN], np.float32)}, {"box0": np.array([1.0], np.float32)})
    assert op.tensor_failures(worst)


RUN = {"name": "v9-t", "mode": "origin", "resample": "bilinear-cv2", "size": 640, "limit": None}


def baseline(**k):
    return {**RUN, "AP": 38.21, **k}


def test_baseline_with_the_same_settings_is_accepted_even_without_snap_flag():
    op.check_baseline(baseline(), RUN)
    op.check_baseline(baseline(snap_scores=False), RUN)


@pytest.mark.parametrize("field,value", [("name", "v9-s"), ("mode", "centre"), ("resample", "lanczos-pil"),
                                         ("size", [640, 384]), ("limit", 50), ("snap_scores", True)])
def test_baseline_with_other_settings_is_refused(field, value):
    with pytest.raises(ValueError, match=field):
        op.check_baseline(baseline(**{field: value}), RUN)


def test_ap_gate():
    assert op.ap_failure(38.24, 38.21, 0.05) is None
    assert op.ap_failure(38.21, 38.24, 0.05) is None
    assert "-0.1" in op.ap_failure(38.11, 38.21, 0.05)
    assert op.ap_failure(NAN, 38.21, 0.05)


def test_is_current_needs_the_models_sha_and_still_applies_the_ap_gate(tmp_path):
    path = tmp_path / "r.json"
    meta = {"onnx_sha256": "abc"}
    assert op.is_current(str(path), meta, None, 0.05) == 1  # no result yet
    path.write_text("{ not json")
    assert op.is_current(str(path), meta, None, 0.05) == 1
    path.write_text(json.dumps({"onnx_sha256": "other", "AP": 38.21}))
    assert op.is_current(str(path), meta, None, 0.05) == 1  # a rebuilt model
    path.write_text(json.dumps({"onnx_sha256": "abc", "AP": 38.21}))
    assert op.is_current(str(path), meta, None, 0.05) == 0
    assert op.is_current(str(path), meta, {"AP": 38.21}, 0.05) == 0
    assert op.is_current(str(path), meta, {"AP": 39.0}, 0.05) == 2  # current, but out of tolerance


def test_positive_int_refuses_zero_and_negative():
    import argparse

    assert op.positive_int("8") == 8
    for bad in ("0", "-3"):
        with pytest.raises(argparse.ArgumentTypeError):
            op.positive_int(bad)
