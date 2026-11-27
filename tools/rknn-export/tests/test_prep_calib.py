import pytest

import prep_calib


@pytest.mark.parametrize("w, h, size, want", [
    (1000, 562, 640, (640, 358)),  # 359.68 floors to 359, forced even
    (1920, 1080, 416, (416, 234)),
    (2560, 1920, 640, (640, 480)),
])
def test_letterbox_content_follows_the_detector_rule(w, h, size, want):
    r = min(size / w, size / h)
    assert (prep_calib.scaled_dim(w, r, size), prep_calib.scaled_dim(h, r, size)) == want


@pytest.mark.parametrize("side, ratio, limit, want", [
    (7, 0.5, 640, 2),   # 3 -> even 2
    (1, 0.1, 640, 2),   # 0 -> floor of 2
    (5, 1.0, 1, 1),     # clamped to limit 1, even 0, raised to 2, capped at 1
    (9, 1.0, 8, 8),     # limit wins over 9 -> 8
])
def test_scaled_dim_even_and_clamp(side, ratio, limit, want):
    assert prep_calib.scaled_dim(side, ratio, limit) == want


def test_subset_id_is_the_set_id_formula_over_a_prefix():
    import hashlib
    frames = [{"npy": f"{i:03d}.npy", "sha256": f"{i:064x}"} for i in range(4)]
    d = hashlib.sha256()
    for f in frames[:2]:
        d.update(f"{f['sha256']}  {f['npy']}\n".encode())
    assert prep_calib.subset_id(frames, 2) == d.hexdigest()[:16]
    assert prep_calib.subset_id(frames) != prep_calib.subset_id(frames, 3)
    assert prep_calib.subset_id([{"file": "000.png", "sha256": "ab"}]) == prep_calib.subset_id([{"npy": "000.png", "sha256": "ab"}])


def test_list_keeps_its_order_and_names_coco_ids(tmp_path):
    lst = tmp_path / "ids.txt"
    lst.write_text("# header\n201446\n9\nframe_01.png\n")
    assert prep_calib.listed("/d", str(lst)) == ["/d/000000201446.jpg", "/d/000000000009.jpg", "/d/frame_01.png"]
    lst.write_text("9\n9\n")
    with pytest.raises(ValueError, match="twice"):
        prep_calib.listed("/d", str(lst))


def test_png_set_is_the_scoring_canvas_and_reads_back_rgb(tmp_path):
    cv2 = pytest.importorskip("cv2")
    import json
    import numpy as np
    import coco_run

    src = tmp_path / "src"
    src.mkdir()
    bgr = np.zeros((48, 96, 3), np.uint8)
    bgr[..., 0], bgr[..., 2] = 20, 200  # blue 20, red 200
    cv2.imwrite(str(src / "a.png"), bgr)
    out = tmp_path / "set"
    assert prep_calib.write_set([str(src / "a.png")], str(out), (64, 64), "png", "rgb", str(src)) == 1
    meta = json.loads((out / "set.json").read_text())
    assert meta["frames"][0]["file"] == "000.png" and (meta["frames"][0]["w"], meta["frames"][0]["h"]) == (96, 48)
    assert meta["id"] == prep_calib.subset_id(meta["frames"])
    want, _ = coco_run.prepare(str(src / "a.png"), (64, 64), "origin", "bilinear-cv2")
    got = cv2.imread(str(out / "000.png"))[:, :, ::-1]  # the toolkit's read: RGB
    assert (got == want).all() and got[0, 0].tolist() == [200, 0, 20] and got[63, 63].tolist() == [114] * 3


def test_convert_calibrates_on_a_recorded_prefix(tmp_path):
    import json
    import types
    import convert

    d = tmp_path / "set"
    d.mkdir()
    frames = [{"file": f"{i:03d}.png", "sha256": f"{i:02x}"} for i in range(5)]
    (d / "set.json").write_text(json.dumps({"id": prep_calib.subset_id(frames), "frames": frames}))
    (d / "dataset.txt").write_text("".join(f"/x/{f['file']}\n" for f in frames))
    a = types.SimpleNamespace(calib=str(d), calib_count=2, name="b")
    ds, rec = convert.calib_dataset(a, str(tmp_path))
    assert open(ds).read().split() == ["/x/000.png", "/x/001.png"]
    assert rec["calib_count"] == 2 and rec["calib_set_id"] == prep_calib.subset_id(frames, 2)
    assert rec["calib_full_set_id"] == prep_calib.subset_id(frames)
    a.calib_count = 6
    with pytest.raises(SystemExit):
        convert.calib_dataset(a, str(tmp_path))


def test_a_set_without_per_frame_hashes_still_takes_a_count(tmp_path):
    import argparse
    import hashlib
    import json
    import convert

    d = tmp_path / "old"
    d.mkdir()
    for n in ("000.npy", "001.npy"):
        (d / n).write_bytes(n.encode())
    (d / "dataset.txt").write_text(f"{d}/000.npy\n{d}/001.npy\n")
    (d / "set.json").write_text(json.dumps({"id": "x", "frames": [{"npy": "000.npy"}, {"npy": "001.npy"}]}))
    a = argparse.Namespace(calib=str(d), calib_count=1, name="m")
    _ds, rec = convert.calib_dataset(a, str(tmp_path))
    sha = hashlib.sha256(b"000.npy").hexdigest()
    assert rec["calib_set_id"] == prep_calib.subset_id([{"npy": "000.npy", "sha256": sha}])


def test_parity_may_share_a_source_with_calibration_frames_the_count_leaves_out(tmp_path):
    import json
    import types
    import convert

    def write(name, srcs):
        d = tmp_path / name
        d.mkdir()
        (d / "set.json").write_text(json.dumps({"frames": [{"src": s} for s in srcs]}))
        return str(d)

    calib, held = write("calib", ["a_00.png", "b_00.png"]), write("held", ["b_01.png"])
    args = lambda n: types.SimpleNamespace(onnx=None, calib=calib, parity=held, out=None, mode="i8", hybrid=False,
                                           ranges=False, float16=None, kind="mit", calib_count=n)
    convert.check_args(args(1))
    for n in (2, None):
        with pytest.raises(SystemExit, match="share sources"):
            convert.check_args(args(n))


def test_a_failure_after_the_export_still_writes_the_record(monkeypatch):
    import types
    import convert

    saved = {}
    res = {"steps": {"build": "ok"}, "_step": lambda k, fn: fn() in (0, None)}
    monkeypatch.setattr(convert, "check_args", lambda a: None)
    monkeypatch.setattr(convert, "build", lambda a: (types.SimpleNamespace(init_runtime=lambda: 0), res))
    monkeypatch.setattr(convert, "post_steps", lambda r, a, res: 1 / 0)
    monkeypatch.setattr(convert, "save", lambda a, res: saved.update(res) or {"steps": res["steps"]})
    with pytest.raises(SystemExit) as e:
        convert.main(["n", "m.onnx", "i8"])
    assert e.value.code == 1 and saved["steps"]["post"].startswith("ERR ZeroDivisionError")
