import hashlib
import json
import os
import subprocess
import sys

import pytest

import conftest

sha = lambda b: hashlib.sha256(b).hexdigest()


def run(tmp_path, **over):
    rungs, build = tmp_path / "rungs", tmp_path / "build"
    rungs.mkdir(); build.mkdir()
    (rungs / "yolox_s.onnx").write_bytes(b"twin")
    (rungs / "yolox_s_640_i8.rknn").write_bytes(b"rknn")
    j = {"steps": {"load": "ok", "export": "ok"}, "rknn_sha256": sha(b"rknn"), "source_sha256": sha(b"twin"),
         "toolkit": "2.3.2", "target_platform": "rk3566", "mode": "i8", "quantized_algorithm": "normal",
         "calib_set_id": "c", "calib_policy": "lb640_bgr"} | over
    j = {k: v for k, v in j.items() if v is not None}
    (build / "yolox_s_640_i8.json").write_text(json.dumps(j))
    p = subprocess.run([sys.executable, os.path.join(conftest.TOOL_DIR, "manifest.py"), str(rungs), str(build)],
                       capture_output=True, text=True)
    return p, sorted(f.name for f in rungs.iterdir() if f.name.startswith("MANIFEST"))


def test_matching_hash_writes_both_outputs(tmp_path):
    p, written = run(tmp_path)
    assert p.returncode == 0, p.stderr
    assert written == ["MANIFEST.json", "MANIFEST.sha256"]
    assert (tmp_path / "rungs/MANIFEST.sha256").read_text() == f"{sha(b'twin')}  yolox_s.onnx\n{sha(b'rknn')}  yolox_s_640_i8.rknn\n"


def test_passed_parity_verdict_is_published(tmp_path):
    p, written = run(tmp_path, parity_pass=True)
    assert p.returncode == 0, p.stderr
    assert written == ["MANIFEST.json", "MANIFEST.sha256"]


@pytest.mark.parametrize("over", [{"rknn_sha256": "0" * 64}, {"rknn_sha256": None},
                                  {"steps": {"load": "ok", "export": "ERR boom"}}, {"sim_outputs": "ERR boom"}, {"parity_pass": False}])
def test_json_that_does_not_vouch_for_the_file_writes_nothing(tmp_path, over):
    p, written = run(tmp_path, **over)
    assert p.returncode != 0 and "yolox_s_640_i8.rknn" in p.stderr
    assert written == []


def mit_dir(tmp_path, onnx_sha=None, sidecar=True, rknn_name="yolov9_t_i8.rknn", drop=(), **over):
    rungs, build = tmp_path / "rungs", tmp_path / "build"
    rungs.mkdir(); build.mkdir()
    (rungs / "yolov9_t.onnx").write_bytes(b"twin")
    (rungs / rknn_name).write_bytes(b"rknn")
    j = {"steps": {"load": "ok"}, "rknn_sha256": sha(b"rknn"), "source_sha256": sha(b"twin"), "toolkit": "2.3.2",
         "target_platform": "rk3566", "mode": "i8", "quantized_algorithm": "normal", "calib_set_id": "c",
         "calib_policy": "lb640_bgr"}
    (build / (rknn_name[:-5] + ".json")).write_text(json.dumps(j))
    if sidecar:
        sc = {"name": "v9-t", "onnx_sha256": onnx_sha or sha(b"twin"), "mit_commit": "abc123",
              "checkpoint_sha256": "c" * 64} | over
        (build / "yolov9_t.onnx.json").write_text(json.dumps({k: v for k, v in sc.items() if k not in drop}))
    p = subprocess.run([sys.executable, os.path.join(conftest.TOOL_DIR, "manifest.py"), str(rungs), str(build)],
                       capture_output=True, text=True)
    return p, rungs


def test_mit_sidecar_labels_twin_and_its_rknn(tmp_path):
    p, rungs = mit_dir(tmp_path)
    assert p.returncode == 0, p.stderr
    rows = {r["file"]: r for r in json.loads((rungs / "MANIFEST.json").read_text())}
    src = {"repo": "MultimediaTechLab/YOLO", "commit": "abc123", "checkpoint": "v9-t", "checkpoint_sha256": "c" * 64}
    assert all(r["license"] == "MIT" and r["source"] == src for r in rows.values())


def test_sidecar_for_other_bytes_refuses(tmp_path):
    p, rungs = mit_dir(tmp_path, onnx_sha="0" * 64)
    assert p.returncode != 0 and "yolov9_t.onnx" in p.stderr
    assert not (rungs / "MANIFEST.json").exists()


@pytest.mark.parametrize("field", ["name", "mit_commit", "checkpoint_sha256"])
@pytest.mark.parametrize("how", ["absent", "empty"])
def test_incomplete_sidecar_refuses_naming_the_field(tmp_path, field, how):
    p, rungs = mit_dir(tmp_path, **({"drop": (field,)} if how == "absent" else {field: ""}))
    assert p.returncode != 0 and "yolov9_t.onnx" in p.stderr and field in p.stderr
    assert not (rungs / "MANIFEST.json").exists()


def test_sidecar_naming_an_unsupported_model_refuses(tmp_path):
    p, _ = mit_dir(tmp_path, name="v9-c")
    assert p.returncode != 0 and "name" in p.stderr


def test_yolov9_twin_without_sidecar_refuses(tmp_path):
    p, rungs = mit_dir(tmp_path, sidecar=False)
    assert p.returncode != 0 and "MIT sidecar" in p.stderr and "yolov9_t.onnx" in p.stderr
    assert not (rungs / "MANIFEST.json").exists()


def test_yolov9_rknn_whose_twin_has_no_sidecar_refuses(tmp_path):
    p, rungs = mit_dir(tmp_path, sidecar=False)
    assert "yolov9_t_i8.rknn: MIT sidecar" in p.stderr
    assert not (rungs / "MANIFEST.json").exists()


def test_existing_rows_carry_no_source(tmp_path):
    p, _ = run(tmp_path)
    rows = json.loads((tmp_path / "rungs/MANIFEST.json").read_text())
    assert p.returncode == 0 and all("source" not in r and r["license"] == "Apache-2.0" for r in rows)
