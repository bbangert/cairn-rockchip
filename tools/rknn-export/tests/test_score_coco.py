import json

import numpy as np
import pytest

import score_coco as sc


def write_ann(path, images, anns, cats=(1,)):
    json.dump({
        "images": [{"id": i, "width": 640, "height": 480} for i in images],
        "annotations": [{"id": n + 1, "image_id": i, "category_id": c, "bbox": b, "area": b[2] * b[3], "iscrowd": 0}
                        for n, (i, c, b) in enumerate(anns)],
        "categories": [{"id": c, "name": f"c{c}"} for c in cats],
    }, open(path, "w"))
    return str(path)


def dets(rows, images):
    """rows: (image_id, class_id, score, x1, y1, x2, y2)."""
    r = rows or [()]
    a = np.array(r, np.float64).reshape(-1, 7) if rows else np.zeros((0, 7))
    return sc.Detections(a[:, 0].astype(np.int64), a[:, 1].astype(np.int64), a[:, 2].astype(np.float32),
                         a[:, 3:].astype(np.float32), np.array(images, np.int64))


BOX_A, BOX_B = [100, 100, 50, 50], [300, 200, 50, 50]  # xywh, 2500 px: COCO "medium"


def test_perfect_detections_score_100(tmp_path):
    ann = write_ann(tmp_path / "a.json", [1, 2], [(1, 1, BOX_A), (2, 1, BOX_B)])
    r = sc.evaluate(ann, dets([(1, 0, 0.9, 100, 100, 150, 150), (2, 0, 0.8, 300, 200, 350, 250)], [1, 2]))
    assert r["AP"] == r["AP50"] == r["AP75"] == r["APm"] == 100.0
    assert r["APs"] is None and r["APl"] is None  # no ground truth of that size
    assert (r["n_images"], r["n_detections"], r["full_set"]) == (2, 2, True)


def test_half_recall_is_51_of_101_points(tmp_path):
    ann = write_ann(tmp_path / "a.json", [1], [(1, 1, BOX_A), (1, 1, BOX_B)])
    r = sc.evaluate(ann, dets([(1, 0, 0.9, 100, 100, 150, 150)], [1]))
    # precision 1 up to recall 0.5, then 0: recall thresholds 0.00..0.50 are 51 of the 101 points
    assert r["AP"] == r["AP50"] == r["AP75"] == 50.50
    assert r["raw"]["AP"] == pytest.approx(51 / 101)


def test_iou_below_every_threshold_but_the_loosest(tmp_path):
    ann = write_ann(tmp_path / "a.json", [1], [(1, 1, BOX_A)])
    # shifted 10 px of 50: IoU = 40*50 / (2*2500 - 2000) = 2/3 -> a hit at IoU .5 and .55..65, a miss from .7
    r = sc.evaluate(ann, dets([(1, 0, 0.9, 110, 100, 160, 150)], [1]))
    assert r["AP50"] == 100.0 and r["AP75"] == 0.0
    assert r["AP"] == pytest.approx(100 * 4 / 10)


def test_category_map():
    m = sc.COCO80_TO_91
    assert len(m) == len(set(m)) == 80
    assert (m[0], m[11], m[79]) == (1, 13, 90)
    assert sorted(set(range(1, 91)) - set(m)) == [12, 26, 29, 30, 45, 66, 68, 69, 71, 83]


def test_dense_class_11_scores_against_category_13(tmp_path):
    ann = write_ann(tmp_path / "a.json", [1], [(1, 13, BOX_A)], cats=(1, 13))
    hit = sc.evaluate(ann, dets([(1, 11, 0.9, 100, 100, 150, 150)], [1]))
    miss = sc.evaluate(ann, dets([(1, 12, 0.9, 100, 100, 150, 150)], [1]))  # dense 12 is category 14
    assert hit["AP"] == 100.0 and miss["AP"] == 0.0


def test_no_detections_scores_zero_where_ground_truth_exists(tmp_path):
    ann = write_ann(tmp_path / "a.json", [1, 2], [(1, 1, BOX_A)])
    r = sc.evaluate(ann, dets([], [1, 2]))
    assert [r[k] for k in ("AP", "AP50", "AP75", "APm")] == [0.0] * 4
    assert r["APs"] is None and r["APl"] is None  # no ground truth of that size, detections or not
    assert (r["n_images"], r["n_detections"], r["full_set"]) == (2, 0, True)


def test_no_images_run_is_an_error(tmp_path):
    ann = write_ann(tmp_path / "a.json", [1], [(1, 1, BOX_A)])
    with pytest.raises(ValueError, match="no images"):
        sc.evaluate(ann, dets([], []))


def test_images_restrict_the_ground_truth_that_counts(tmp_path):
    ann = write_ann(tmp_path / "a.json", [1, 2], [(1, 1, BOX_A), (2, 1, BOX_B)])
    d = dets([(1, 0, 0.9, 100, 100, 150, 150)], [1])
    r = sc.evaluate(ann, d)
    assert r["AP"] == 100.0 and r["full_set"] is False and r["n_images"] == 1
    # same detections, but image 2 was run too and found nothing: its ground truth is now missed
    r2 = sc.evaluate(ann, d._replace(images=np.array([1, 2], np.int64)))
    assert r2["AP"] == 50.50 and r2["full_set"] is True


def test_detections_outside_the_run_set_or_the_annotations_are_refused(tmp_path):
    ann = write_ann(tmp_path / "a.json", [1, 2], [(1, 1, BOX_A)])
    with pytest.raises(ValueError, match="not listed"):
        sc.evaluate(ann, dets([(2, 0, 0.9, 1, 1, 5, 5)], [1]))
    with pytest.raises(ValueError, match="not in the annotation"):
        sc.evaluate(ann, dets([], [1, 99]))
    with pytest.raises(ValueError, match="class_id"):
        sc.evaluate(ann, dets([(1, 80, 0.9, 1, 1, 5, 5)], [1]))


def test_npz_round_trip(tmp_path):
    d = dets([(1, 0, 0.5, 1, 2, 3, 4), (2, 79, 0.25, 5, 6, 7, 8)], [1, 2, 3])
    p = tmp_path / "d.npz"
    sc.save_detections(p, *d)
    back = sc.load_detections(p)
    for a, b in zip(d, back):
        assert a.dtype == b.dtype and a.shape == b.shape and (a == b).all()
    assert back.images.tolist() == [1, 2, 3]  # includes the image with no detections


def test_mismatched_columns_are_refused_not_truncated(tmp_path):
    ann = write_ann(tmp_path / "a.json", [1], [(1, 1, BOX_A)])
    d = dets([(1, 0, 0.9, 100, 100, 150, 150), (1, 0, 0.8, 1, 1, 9, 9)], [1])
    with pytest.raises(ValueError, match="same|length|shapes"):
        sc.evaluate(ann, d._replace(class_id=d.class_id[:1]))
    with pytest.raises(ValueError, match="shapes"):
        sc.evaluate(ann, d._replace(box=d.box[:, :3]))
    with pytest.raises(ValueError, match="shapes"):
        sc.evaluate(ann, d._replace(score=d.score[:, None]))
    with pytest.raises(ValueError, match="shapes"):
        sc.save_detections(tmp_path / "s.npz", d.image_id, d.class_id[:1], d.score, d.box, d.images)
    p = tmp_path / "bad.npz"
    np.savez(p, image_id=d.image_id, class_id=d.class_id[:1], score=d.score, box=d.box, images=d.images)
    with pytest.raises(ValueError, match="shapes"):
        sc.evaluate(ann, sc.load_detections(p))


def test_cli_prints_one_line_and_writes_traceable_json(tmp_path, capsys):
    ann = write_ann(tmp_path / "a.json", [1], [(1, 1, BOX_A), (1, 1, BOX_B)])
    p = tmp_path / "d.npz"
    sc.save_detections(p, *dets([(1, 0, 0.9, 100, 100, 150, 150)], [1]))
    out = tmp_path / "r.json"
    sc.main([ann, str(p), "--out", str(out), "--label", "v9-t pytorch"])
    lines = capsys.readouterr().out.splitlines()
    assert lines == ["v9-t pytorch: AP 50.50 AP50 50.50 AP75 50.50 (1 images, 1 dets)"]
    r = json.loads(out.read_text())
    assert r["label"] == "v9-t pytorch" and r["AP"] == 50.5
    import hashlib
    assert r["dets_sha256"] == hashlib.sha256(p.read_bytes()).hexdigest()
    assert r["ann_sha256"] == hashlib.sha256(open(ann, "rb").read()).hexdigest()
