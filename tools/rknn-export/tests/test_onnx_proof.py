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
    return {**RUN, "stage": "pytorch", "AP": 38.21, **k}


def test_baseline_with_the_same_settings_is_accepted_even_without_snap_flag():
    op.check_baseline(baseline(), RUN)
    op.check_baseline(baseline(snap_scores=False), RUN)


@pytest.mark.parametrize("field,value", [("name", "v9-s"), ("mode", "centre"), ("resample", "lanczos-pil"),
                                         ("size", [640, 384]), ("limit", 50), ("snap_scores", True),
                                         ("stage", "onnx"), ("stage", None)])
def test_baseline_with_other_settings_is_refused(field, value):
    with pytest.raises(ValueError, match=field):
        op.check_baseline(baseline(**{field: value}), RUN)


def test_ap_gate():
    assert op.ap_failure(38.24, 38.21, 0.05) is None
    assert op.ap_failure(38.16, 38.21, 0.05) is None  # -0.05000000000000426 in binary
    assert op.ap_failure(38.15, 38.21, 0.05)
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


def result(**k):
    cmp = {"box0": {"max_abs_diff": 1e-4, "max_abs_value": 5.0}}
    return {"onnx_sha256": "abc", "AP": 38.21, "compared": 20, "compare": cmp, **k}


def current(tmp_path, res, compare=20, base=None):
    path = tmp_path / "r.json"
    path.write_text(json.dumps(res))
    return op.is_current(str(path), {"onnx_sha256": "abc"}, base, 0.05, compare)


def test_is_current_wants_a_passing_comparison_over_at_least_the_requested_images(tmp_path):
    assert current(tmp_path, result()) == 0
    assert current(tmp_path, result(), compare=10) == 0
    assert current(tmp_path, result(), compare=21) == 1  # a shorter earlier comparison
    assert current(tmp_path, result(compared=0, compare=None)) == 1  # scoring only
    assert current(tmp_path, result(compared=0, compare=None), compare=None) == 0  # none requested
    assert current(tmp_path, result(), base={"AP": 39.0}) == 2


@pytest.mark.parametrize("res", [{"compared": True}, {"compared": False}, {"compared": None}, {"compared": "20"},
                                 {"compared": 20, "compare": {"box0": {"max_abs_diff": NAN, "max_abs_value": 1.0}}},
                                 {"compared": 20, "compare": {"box0": {"max_abs_diff": 1.0, "max_abs_value": 1.0}}},
                                 {"compared": 20, "compare": {"box0": 1.0}}, {"compared": 20, "compare": {}},
                                 {"compared": 20, "compare": None}])
def test_is_current_treats_old_or_malformed_comparison_records_as_not_current(tmp_path, res):
    assert current(tmp_path, result(**res)) == 1


def test_verify_model_refuses_corrupt_bytes_and_other_checkpoints(tmp_path, monkeypatch):
    model = tmp_path / "m.onnx"
    model.write_bytes(b"bytes")
    meta = {"onnx_sha256": op.sources.sha256_file(str(model)), "checkpoint_sha256": "ck", "mit_commit": "commit"}
    monkeypatch.setattr(op.mit_model, "checkpoint", lambda name: ("w.pt", "ck", "commit"))
    assert op.verify_model(str(model), "v9-t", meta) == ("w.pt", "ck", "commit")
    model.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="does not match"):
        op.verify_model(str(model), "v9-t", meta)
    model.write_bytes(b"bytes")
    with pytest.raises(ValueError, match="does not match"):
        op.verify_model(str(model), "v9-t", {**meta, "checkpoint_sha256": "other"})
    for stale in ({**meta, "mit_commit": "older"}, {k: v for k, v in meta.items() if k != "mit_commit"}):
        with pytest.raises(ValueError, match="does not match"):
            op.verify_model(str(model), "v9-t", stale)


def test_is_current_exits_2_for_a_model_that_differs_from_its_sidecar(tmp_path, monkeypatch, capsys):
    model = tmp_path / "m.onnx"
    model.write_bytes(b"corrupt")
    meta = {"name": "v9-t", "onnx_sha256": "0" * 64, "checkpoint_sha256": "ck", "fuse": False, "logits": False,
            "size": [640, 640]}
    (tmp_path / "m.onnx.json").write_text(json.dumps(meta))
    (tmp_path / "v9-t-onnx-origin-bilinear-cv2.json").write_text(json.dumps(result(onnx_sha256="0" * 64)))
    monkeypatch.setattr(op.mit_model, "checkpoint", lambda name: ("w.pt", "ck", "commit"))
    with pytest.raises(SystemExit) as e:
        op.main(["v9-t", str(model), "--mode", "origin", "--resample", "bilinear-cv2", "--compare", "20",
                 "--out-dir", str(tmp_path), "--is-current"])
    assert e.value.code == 2 and "does not match" in capsys.readouterr().err


SIDECAR = {"checkpoint_sha256": "ck", "mit_commit": "abc"}


def test_baseline_must_record_the_same_checkpoint_and_commit():
    run = {**RUN, **SIDECAR}
    op.check_baseline(baseline(**SIDECAR), run)
    for field, value in (("checkpoint_sha256", "other"), ("mit_commit", "other"),
                         ("checkpoint_sha256", None), ("mit_commit", None)):
        with pytest.raises(ValueError, match=field):
            op.check_baseline(baseline(**{**SIDECAR, field: value}), run)
    with pytest.raises(ValueError, match="mit_commit"):  # a baseline written before it was recorded
        op.check_baseline(baseline(checkpoint_sha256="ck"), run)


def raw(**over):
    out = {n: np.zeros((1, 2, 2, 2), np.float32) for i in range(3) for n in (f"box{i}", f"cls{i}")}
    return {**out, **over}


def test_scoring_outputs_are_ordered_and_squash_only_logit_classes():
    out = raw(cls1=np.full((1, 2, 2, 2), 2.0, np.float32))
    assert op.scoring_outputs(out, False)[3] is out["cls1"]
    got = op.scoring_outputs(out, True)
    assert got[2] is out["box1"] and np.allclose(got[3], 1 / (1 + np.exp(-2.0)))


@pytest.mark.parametrize("value", [NAN, np.inf, -np.inf])
@pytest.mark.parametrize("logits", [False, True])
def test_scoring_refuses_non_finite_raw_outputs_naming_the_tensors(value, logits):
    bad = np.zeros((1, 2, 2, 2), np.float32)
    bad[0, 1, 0, 1] = value
    with pytest.raises(op.NonFinite, match="box0, cls2"):
        op.scoring_outputs(raw(box0=bad, cls2=bad), logits)


def test_stem_marks_a_clamped_export_and_leaves_unclamped_sidecars_alone():
    base = {"fuse": True, "logits": False, "size": (640, 640)}
    stem = lambda **k: op.variant_stem("v9-t", {**base, **k}, "origin", "bilinear-cv2", None)
    assert stem(class_clamp=32.0) == "v9-t-onnx-fused-clamp32-origin-bilinear-cv2"
    assert stem(class_clamp=32.12341) != stem(class_clamp=32.12344)
    assert stem(class_clamp=None) == stem() == "v9-t-onnx-fused-origin-bilinear-cv2"


def test_a_clamped_export_is_not_compared_with_the_unclamped_torch_module(tmp_path):
    model = tmp_path / "m.onnx"
    (tmp_path / "m.onnx.json").write_text(json.dumps({"name": "v9-t", "fuse": True, "logits": False,
                                                       "size": [640, 640], "class_clamp": 32.0}))
    with pytest.raises(SystemExit, match="cannot be compared"):
        op.main(["v9-t", str(model), "--mode", "origin", "--resample", "bilinear-cv2", "--compare", "2"])
