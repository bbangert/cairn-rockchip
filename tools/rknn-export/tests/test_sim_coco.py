import json
import os

import numpy as np
import pytest

import sim_coco
from score_coco import load_detections, save_detections


def test_stems_keep_the_float_naming_and_mark_core():
    st = sim_coco.stems("/o/v9-t_i8-sim", 200)
    assert st == {"reference": "/o/v9-t_i8-sim-origin-bilinear-cv2-limit200",
                  "core": "/o/v9-t_i8-sim-origin-bilinear-cv2-limit200-core"}
    assert sim_coco.stems("/o/x", None)["reference"] == "/o/x-origin-bilinear-cv2"


def test_chunks_merge_back_into_one_result(tmp_path):
    for k, (ids, n) in enumerate((([1, 2], 3), ([3], 0))):
        for r in ("reference", "core"):
            save_detections(str(tmp_path / f"{k}.{r}.npz"), np.full(n, ids[0]), np.zeros(n, int), np.full(n, 0.5),
                            np.zeros((n, 4)), ids)
    assert sim_coco.chunk_done(str(tmp_path), 1) and not sim_coco.chunk_done(str(tmp_path), 2)
    sim_coco.merge(str(tmp_path), 2, "core", str(tmp_path / "all.npz"))
    d = load_detections(str(tmp_path / "all.npz"))
    assert d.images.tolist() == [1, 2, 3] and len(d.score) == 3


def test_a_resume_under_other_arguments_is_refused(tmp_path, monkeypatch):
    onnx = tmp_path / "m.onnx"
    onnx.write_bytes(b"model")
    (tmp_path / "m.onnx.json").write_text(json.dumps({"size": [640, 640], "name": "v9-t"}))
    monkeypatch.setattr(sim_coco.coco_run, "_images", lambda ann, limit: [{"id": 1}])
    stem = sim_coco.stems(str(tmp_path / "m-ort"), 5)["reference"]
    os.makedirs(stem + ".parts")
    with open(stem + ".parts/config.json", "w") as f:
        json.dump({"limit": 9}, f)
    with pytest.raises(SystemExit, match="other arguments"):
        sim_coco.main(["--limit", "5", "--out-dir", str(tmp_path), "onnx", str(onnx)])


SIZE = 64


def outputs(score=0.9):
    out = {}
    for i, stride in enumerate((8, 16, 32)):
        g = SIZE // stride
        out[f"box{i}"] = np.zeros((1, 64, g, g), np.float32)
        cls = np.zeros((1, 80, g, g), np.float32)
        if i == 0:
            cls[0, 7, 3, 3] = score
        out[f"cls{i}"] = cls
    return out


@pytest.fixture
def onnx_run(tmp_path, monkeypatch):
    """sim_coco.main in onnx mode over three fake images, one per chunk, with a session that is a dict."""
    import functools
    from decode_split import placement

    onnx = tmp_path / "m.onnx"
    onnx.write_bytes(b"model")
    (tmp_path / "m.onnx.json").write_text(json.dumps({
        "size": [SIZE, SIZE], "name": "v9-t", "logits": False, "fuse": True, "onnx_sha256": "o",
        "checkpoint_sha256": "c", "mit_commit": "m"}))
    images = [{"id": i, "file_name": f"{i}.jpg", "width": 128, "height": 64} for i in (3, 5, 9)]
    monkeypatch.setattr(sim_coco.coco_run, "_images", lambda ann, limit: images)
    monkeypatch.setattr(sim_coco.onnx_proof, "session", lambda path, threads: None)
    state = {"score": 0.9, "kill_after": None, "chunks": 0}
    monkeypatch.setattr(sim_coco.onnx_proof, "raw_outputs", lambda sess, canvas: outputs(state["score"]))
    real = sim_coco.coco_run.run
    prep = lambda path, size, mode, resample: (np.zeros((SIZE, SIZE, 3), np.uint8), placement(128, 64, size, mode))

    def run(*a, **kw):
        if state["kill_after"] is not None and state["chunks"] >= state["kill_after"]:
            raise KeyboardInterrupt  # the process dies mid-run
        state["chunks"] += 1
        return real(*a, **kw, prepare=prep)

    monkeypatch.setattr(sim_coco.coco_run, "run", run)
    argv = ["--chunk", "1", "--workers", "1", "--out-dir", str(tmp_path / "out"), "onnx", str(onnx)]
    return argv, state, tmp_path / "out" / "m-ort-64x64-origin-bilinear-cv2"  # a non-640 size is in the stem


def test_a_killed_run_resumes_the_missing_chunks_and_merges_all(onnx_run):
    argv, state, stem = onnx_run
    state["kill_after"] = 1
    with pytest.raises(KeyboardInterrupt):
        sim_coco.main(argv)
    parts = str(stem) + ".parts"
    assert sim_coco.chunk_done(parts, 0) and not sim_coco.chunk_done(parts, 1)
    assert not os.path.exists(str(stem) + "-core.meta.json")
    state.update(kill_after=None, chunks=0)
    sim_coco.main(argv)
    assert state["chunks"] == 2
    for suffix in ("", "-core"):
        d = load_detections(f"{stem}{suffix}.npz")
        assert d.images.tolist() == [3, 5, 9] and d.class_id.tolist() == [7, 7, 7]
        meta = json.load(open(f"{stem}{suffix}.meta.json"))
        assert meta["rules"] == ("reference" if not suffix else "core") and meta["stage"] == "onnx"
        assert meta["graph_id"] == json.load(open(parts + "/config.json"))["graph_id"]
    state["chunks"] = 0
    sim_coco.main(argv)  # complete: nothing is rerun
    assert state["chunks"] == 0


def test_a_resume_whose_model_answers_differently_is_refused(onnx_run):
    argv, state, stem = onnx_run
    state["kill_after"] = 1
    with pytest.raises(KeyboardInterrupt):
        sim_coco.main(argv)
    state.update(kill_after=None, score=0.8)
    with pytest.raises(SystemExit, match="outputs differ"):
        sim_coco.main(argv)
    assert not sim_coco.chunk_done(str(stem) + ".parts", 1)
