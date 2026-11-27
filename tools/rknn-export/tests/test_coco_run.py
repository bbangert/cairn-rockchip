import json
import os

import numpy as np
import pytest

import coco_run
from decode_split import BINS, placement
from score_coco import load_detections

SIZE = 64
CELL = 3  # stride-8 cell (3, 3): centre (28, 28), every side 2 bins * 8 px -> box [12, 12, 44, 44]
BOX = (12, 12, 44, 44)
SRC = {"a.jpg": (128, 64), "b.jpg": (64, 64), "c.jpg": (128, 64)}
IDS = {"a.jpg": 30, "b.jpg": 10, "c.jpg": 20}


def tensors(hit):
    out = []
    for stride in (8, 16, 32):
        g = SIZE // stride
        box = np.zeros((1, 4 * BINS, g, g), np.float32)
        cls = np.zeros((1, 80, g, g), np.float32)
        if hit and stride == 8:
            for side in range(4):
                box[0, side * BINS + 2, CELL, CELL] = 40  # softmax collapses onto bin 2
            cls[0, 7, CELL, CELL] = 0.9
        out += [box, cls]
    return out


@pytest.fixture
def ann(tmp_path):
    images = [{"id": IDS[n], "file_name": n, "width": w, "height": h} for n, (w, h) in SRC.items()]
    path = tmp_path / "ann.json"
    path.write_text(json.dumps({"images": images, "annotations": [], "categories": []}))
    return str(path)


def fake_prepare():
    def prepare(path, size, mode_, resample):
        name = path.rsplit("/", 1)[-1]
        # the canvas carries the image id so `infer` can tell which image it was handed
        return np.full((SIZE, SIZE, 3), IDS[name], np.uint8), placement(*SRC[name], size, mode_)
    return prepare


def run(ann, tmp_path, mode="centre", workers=1, limit=None, seen=None, name="out.npz"):
    def infer(canvas):
        if seen is not None:
            seen.append(int(canvas[0, 0, 0]))
        return tensors(hit=canvas[0, 0, 0] != IDS["c.jpg"])  # image 20 has nothing to detect

    out = str(tmp_path / name)
    stats = coco_run.run(infer, "/imgs", ann, SIZE, mode, "bilinear-cv2", out, limit=limit, workers=workers,
                         prepare=fake_prepare())
    return load_detections(out), stats


def test_ascending_id_order_and_limit(ann, tmp_path):
    seen = []
    d, stats = run(ann, tmp_path, seen=seen)
    assert seen == [10, 20, 30]
    assert d.images.tolist() == [10, 20, 30]
    assert stats["images"] == 3

    seen.clear()
    d, _ = run(ann, tmp_path, limit=2, seen=seen, name="lim.npz")
    assert seen == [10, 20]
    assert d.images.tolist() == [10, 20]
    assert set(d.image_id.tolist()) == {10}


def test_image_without_detections_is_still_listed(ann, tmp_path):
    d, _ = run(ann, tmp_path)
    assert 20 in d.images.tolist() and 20 not in d.image_id.tolist()
    assert d.image_id.tolist() == [10, 30] and d.class_id.tolist() == [7, 7]
    assert d.score == pytest.approx([0.9, 0.9])


@pytest.mark.parametrize("mode, a_box, b_box", [
    # 128x64 source: centre scale 0.5, content 64x32 padded 16 top; origin the same without the pad;
    # stretch 0.5 by 1. 64x64 source fits exactly in every mode.
    ("centre", (24, 0, 88, 56), BOX),
    ("origin", (24, 24, 88, 64), BOX),
    ("stretch", (24, 12, 88, 44), BOX),
])
def test_boxes_come_back_in_original_pixels(ann, tmp_path, mode, a_box, b_box):
    d, _ = run(ann, tmp_path, mode=mode)
    boxes = {int(i): b for i, b in zip(d.image_id, d.box)}
    assert boxes[30] == pytest.approx(a_box, abs=1e-3)
    assert boxes[10] == pytest.approx(b_box, abs=1e-3)


def test_pool_and_single_process_write_the_same_detections(ann, tmp_path):
    one, _ = run(ann, tmp_path, workers=1, name="one.npz")
    many, _ = run(ann, tmp_path, workers=2, name="many.npz")
    for field in one._fields:
        assert np.array_equal(getattr(one, field), getattr(many, field)), field


def test_empty_annotation_set_is_an_error(tmp_path):
    p = tmp_path / "a.json"
    p.write_text(json.dumps({"images": []}))
    with pytest.raises(ValueError):
        coco_run.run(lambda c: [], "/x", str(p), SIZE, "centre", "bilinear-cv2", str(tmp_path / "o.npz"), workers=1)


@pytest.mark.parametrize("resample", coco_run.RESAMPLES)
@pytest.mark.parametrize("mode", coco_run.MODES)
def test_prepare_pads_and_keeps_channel_order(tmp_path, mode, resample):
    pytest.importorskip("PIL")
    if resample == "bilinear-cv2":
        pytest.importorskip("cv2")
    from PIL import Image

    path = str(tmp_path / "im.png")
    Image.new("RGB", (128, 64), (200, 100, 50)).save(path)  # lossless, so only the resize can move a value
    canvas, p = coco_run.prepare(path, 64, mode, resample)
    assert canvas.shape == (64, 64, 3) and canvas.dtype == np.uint8
    assert p == placement(128, 64, 64, mode)
    content = canvas[p.pad_top:p.pad_top + p.new_h, p.pad_left:p.pad_left + p.new_w]
    assert np.abs(content.astype(int) - (200, 100, 50)).max() <= 1
    mask = np.ones((64, 64), bool)
    mask[p.pad_top:p.pad_top + p.new_h, p.pad_left:p.pad_left + p.new_w] = False
    assert (canvas[mask] == 114).all()
    assert mask.any() == (mode == "centre" or mode == "origin")


def test_prepare_rejects_unknown_resample(tmp_path):
    pytest.importorskip("PIL")
    from PIL import Image

    path = str(tmp_path / "im.png")
    Image.new("RGB", (8, 8)).save(path)
    with pytest.raises(ValueError):
        coco_run.prepare(path, 64, "centre", "nearest")


def test_workers_see_single_threaded_blas_variables(monkeypatch):
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        monkeypatch.setenv(k, "32")
    with coco_run._worker_pool(2) as pool:
        # os.getenv is picklable and runs in the worker, which spawns on this first submit
        seen = [pool.submit(os.getenv, k).result() for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")]
    assert seen == ["1", "1", "1"]
    assert os.environ["OMP_NUM_THREADS"] == "32"


def test_worker_pool_restores_unset_variables(monkeypatch):
    monkeypatch.delenv("MKL_NUM_THREADS", raising=False)
    with coco_run._worker_pool(1):
        assert os.environ["MKL_NUM_THREADS"] == "1"
    assert "MKL_NUM_THREADS" not in os.environ


@pytest.mark.parametrize("workers", [1, 2])
def test_worker_exception_propagates_and_leaves_no_npz(ann, tmp_path, workers, monkeypatch):
    # a (1, 4, 1, 1) box tensor cannot be decoded at a 64 px input, so the failure comes from _post
    bad = tensors(hit=False)
    bad[0] = np.zeros((1, 4 * BINS, 1, 1), np.float32)
    out = tmp_path / "bad.npz"
    with pytest.raises(Exception):
        coco_run.run(lambda c: bad, "/imgs", ann, SIZE, "centre", "bilinear-cv2", str(out), workers=workers,
                     prepare=fake_prepare())
    assert not out.exists() and not list(tmp_path.glob("*.npz"))


def test_non_square_size_through_prepare_and_run(tmp_path):
    pytest.importorskip("PIL")
    from PIL import Image

    w, h = 96, 64
    path = str(tmp_path / "im.png")
    Image.new("RGB", (192, 64), (200, 100, 50)).save(path)
    canvas, p = coco_run.prepare(path, (w, h), "origin", "lanczos-pil")
    assert canvas.shape == (h, w, 3)
    assert p == placement(192, 64, (w, h), "origin")

    ann = tmp_path / "ann.json"
    ann.write_text(json.dumps({"images": [{"id": 1, "file_name": "im.png", "width": 192, "height": 64}],
                               "annotations": [], "categories": []}))

    def infer(c):
        assert c.shape == (h, w, 3)
        out = []
        for stride in (8, 16, 32):
            gh, gw = h // stride, w // stride
            box = np.zeros((1, 4 * BINS, gh, gw), np.float32)
            cls = np.zeros((1, 80, gh, gw), np.float32)
            if stride == 8:
                for side in range(4):
                    box[0, side * BINS + 2, 3, 5] = 40
                cls[0, 7, 3, 5] = 0.9
            out += [box, cls]
        return out

    npz = str(tmp_path / "o.npz")
    coco_run.run(infer, str(tmp_path), str(ann), (w, h), "origin", "lanczos-pil", npz, workers=1)
    d = load_detections(npz)
    # cell (row 3, col 5): centre (44, 28), half-side 16 -> [28, 12, 60, 44] in canvas px; origin mode
    # scales 192x64 by 0.5 with no padding, so original px are twice that,
    # with y2 clipped to the 64 px image height
    assert d.box[0] == pytest.approx((56, 24, 120, 64), abs=1e-3)


def test_jobs_do_not_alias_the_tensors_infer_returned(ann, tmp_path, monkeypatch):
    reused = tensors(hit=True)
    jobs = []
    post = coco_run._post
    monkeypatch.setattr(coco_run, "_post", lambda job: jobs.append(job) or post(job))
    coco_run.run(lambda c: reused, "/imgs", ann, SIZE, "centre", "bilinear-cv2", str(tmp_path / "o.npz"), workers=1,
                 prepare=fake_prepare())
    assert len(jobs) == len(SRC)
    assert not any(np.shares_memory(a, r) for job in jobs for a, r in zip(job[3], reused))


def test_rule_sets_from_one_pass_and_an_explicit_image_list(ann, tmp_path):
    images = [{"id": 30, "file_name": "a.jpg", "width": 128, "height": 64}]
    outs = {r: str(tmp_path / f"{r}.npz") for r in ("reference", "core")}
    calls = []
    t = coco_run.run(lambda c: calls.append(1) or tensors(True), "/x", ann, SIZE, "origin", "bilinear-cv2", outs,
                     workers=1, prepare=fake_prepare(), images=images)
    assert t["images"] == 1 and len(calls) == 1
    ref, core = (load_detections(outs[r]) for r in ("reference", "core"))
    assert ref.images.tolist() == core.images.tolist() == [30]
    assert ref.class_id.tolist() == core.class_id.tolist() == [7]
    assert core.score.tolist() == pytest.approx([0.9])


def test_unknown_rule_set_is_refused(ann, tmp_path):
    with pytest.raises(ValueError, match="rules"):
        coco_run.run(lambda c: tensors(True), "/x", ann, SIZE, "origin", "bilinear-cv2", {"rust": str(tmp_path / "x.npz")},
                     workers=1, prepare=fake_prepare())
