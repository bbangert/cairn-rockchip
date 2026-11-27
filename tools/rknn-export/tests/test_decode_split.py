import numpy as np
import pytest

import decode_split as ds


def levels(size, fill=0.0, nc=80):
    w, h = (size, size) if isinstance(size, int) else size
    boxes = [np.full((1, 64, h // s, w // s), fill, np.float32) for s in ds.STRIDES]
    clss = [np.zeros((1, nc, h // s, w // s), np.float32) for s in ds.STRIDES]
    return boxes, clss


def one_hot_ltrb(boxes, ltrb):
    # a +50 logit on one bin per side makes the softmax expectation that bin, to float precision
    for b in boxes:
        b[:] = -50
        for side, k in enumerate(ltrb):
            b[0, side * 16 + k] = 50


def test_uniform_logits_are_bin_7_5_on_every_side():
    boxes, clss = levels(64)
    xyxy, _ = ds.decode(boxes, clss, 64)
    # stride 8, anchor (0, 0): centre (4, 4), +-7.5 * 8
    assert xyxy[0] == pytest.approx([4 - 60, 4 - 60, 4 + 60, 4 + 60])
    # stride 32, anchor (1, 0): centre (48, 16), +-7.5 * 32
    assert xyxy[64 + 16 + 1] == pytest.approx([48 - 240, 16 - 240, 48 + 240, 16 + 240])


def test_two_equal_peaks_average_their_bins():
    boxes, clss = levels(64)
    for b in boxes:
        b[:] = -50
        b[0, 2] = b[0, 4] = 50  # left side: bins 2 and 4 -> 3
    xyxy, _ = ds.decode(boxes, clss, 64)
    assert xyxy[0, 0] == pytest.approx(4 - 3 * 8)
    assert xyxy[0, 2] == pytest.approx(4 + 7.5 * 8)  # right side untouched: uniform


def test_channel_is_side_times_16_plus_bin():
    boxes, clss = levels(64)
    for b in boxes:
        b[:] = -100
    boxes[0][0, 6, 0, 0] = 100  # side 0 (left) bin 6; bin * 4 + side would read bin 1 of side 2 (right)
    boxes[0][0, 16 + 3, 0, 1] = 100  # side 1 (top) bin 3
    xyxy, _ = ds.decode(boxes, clss, 64)
    assert xyxy[0] == pytest.approx([4 - 6 * 8, 4 - 7.5 * 8, 4 + 7.5 * 8, 4 + 7.5 * 8])
    assert xyxy[1] == pytest.approx([12 - 7.5 * 8, 4 - 3 * 8, 12 + 7.5 * 8, 4 + 7.5 * 8])


def test_anchors_run_by_level_then_row_major():
    boxes, clss = levels(64)
    one_hot_ltrb(boxes, (0, 0, 0, 0))  # zero distance: xyxy is the anchor centre twice
    xyxy, _ = ds.decode(boxes, clss, 64)
    assert len(xyxy) == 64 + 16 + 4
    centre = {0: (4, 4), 1: (12, 4), 8: (4, 12), 63: (60, 60),
              64: (8, 8), 65: (24, 8), 68: (8, 24), 79: (56, 56),
              80: (16, 16), 81: (48, 16), 82: (16, 48), 83: (48, 48)}
    for i, (x, y) in centre.items():
        assert xyxy[i] == pytest.approx([x, y, x, y]), i


def test_rectangular_input_has_distinct_grid_sizes():
    boxes, clss = levels((640, 384))
    assert [b.shape[2:] for b in boxes] == [(48, 80), (24, 40), (12, 20)]
    one_hot_ltrb(boxes, (2, 3, 4, 5))
    xyxy, scores = ds.decode(boxes, clss, (640, 384))
    assert len(xyxy) == scores.shape[0] == 48 * 80 + 24 * 40 + 12 * 20
    # stride 8 anchor x=70, y=10: centre (564, 84); width and height swapped would put it elsewhere
    assert xyxy[10 * 80 + 70] == pytest.approx([564 - 16, 84 - 24, 564 + 32, 84 + 40])
    # last stride-32 anchor x=19, y=11: centre (624, 368)
    assert xyxy[-1] == pytest.approx([624 - 64, 368 - 96, 624 + 128, 368 + 160])
    # first stride-16 anchor
    assert xyxy[48 * 80] == pytest.approx([8 - 32, 8 - 48, 8 + 64, 8 + 80])


def test_swapped_grid_is_refused():
    boxes, clss = levels((384, 640))
    with pytest.raises(ValueError, match="box\\[0\\]"):
        ds.decode(boxes, clss, (640, 384))


def test_scores_pass_through_per_anchor_and_class():
    boxes, clss = levels(64)
    clss[1][0, 7, 2, 3] = 0.9
    _, scores = ds.decode(boxes, clss, 64)
    i = 64 + 2 * 4 + 3
    assert scores[i, 7] == np.float32(0.9)
    assert scores.sum() == pytest.approx(0.9)


def test_bad_inputs_get_clear_errors():
    boxes, clss = levels(64)
    with pytest.raises(ValueError, match="one box and one cls"):
        ds.decode(boxes[:2], clss[:2], 64)
    with pytest.raises(ValueError, match="cls\\[2\\]"):
        ds.decode(boxes, clss[:2] + [clss[2][:, :, :1]], 64)
    with pytest.raises(ValueError, match="box\\[0\\]"):
        ds.decode([b[:, :60] for b in boxes], clss, 64)
    with pytest.raises(ValueError, match="positive"):
        ds.decode(boxes, clss, (0, 64))


def cands(entries, n):
    """entries: (anchor, class, score); boxes are laid by the caller."""
    s = np.zeros((n, 80), np.float32)
    for a, c, v in entries:
        s[a, c] = v
    return s


def row(*x):
    return np.array(x, np.float32)


def test_same_class_overlap_is_suppressed_only_above_the_threshold():
    box = np.stack([row(0, 0, 10, 10), row(0, 0, 10, 9), row(0, 0, 10, 5)])  # IoU with the first: .9, .5
    s = cands([(0, 3, 0.9), (1, 3, 0.8), (2, 3, 0.7)], 3)
    b, sc, c = ds.nms(box, s)
    assert sc.tolist() == pytest.approx([0.9, 0.7]) and c.tolist() == [3, 3]
    assert b[1].tolist() == [0, 0, 10, 5]


def test_iou_equal_to_the_threshold_survives():
    box = np.stack([row(0, 0, 10, 10), row(0, 0, 10, 5)])  # IoU exactly .5
    s = cands([(0, 1, 0.9), (1, 1, 0.8)], 2)
    assert len(ds.nms(box, s, iou=0.5)[1]) == 2
    assert len(ds.nms(box, s, iou=0.4999)[1]) == 1


def test_different_classes_do_not_suppress_each_other():
    box = np.stack([row(0, 0, 10, 10), row(0, 0, 10, 10)])
    s = cands([(0, 1, 0.9), (1, 2, 0.8)], 2)
    _, sc, c = ds.nms(box, s)
    assert c.tolist() == [1, 2] and len(sc) == 2


def test_one_anchor_scores_every_class_over_conf():
    box = np.stack([row(0, 0, 10, 10)])
    s = cands([(0, 3, 0.9), (0, 5, 0.5), (0, 6, 0.0005)], 1)
    b, sc, c = ds.nms(box, s)
    assert c.tolist() == [3, 5] and sc.tolist() == pytest.approx([0.9, 0.5])
    assert b.shape == (2, 4) and (b == b[0]).all()


def test_score_equal_to_conf_is_excluded():
    box = np.stack([row(0, 0, 10, 10), row(20, 20, 30, 30)])
    s = cands([(0, 1, 0.5), (1, 1, 0.75)], 2)
    _, sc, _ = ds.nms(box, s, conf=0.5)
    assert sc.tolist() == [0.75]


def test_max_det_keeps_the_highest_scores_in_order():
    n = 10
    box = np.stack([row(20 * i, 0, 20 * i + 10, 10) for i in range(n)])
    s = cands([(i, 4, 0.1 * (i + 1)) for i in range(n)], n)
    _, sc, _ = ds.nms(box, s, max_det=3)
    assert sc.tolist() == pytest.approx([1.0, 0.9, 0.8])


def test_max_det_is_a_prefix_of_the_unlimited_result_in_one_class():
    rng = np.random.default_rng(0)
    n = 400
    xy = rng.uniform(0, 600, (n, 2))
    box = np.concatenate([xy, xy + rng.uniform(5, 30, (n, 2))], axis=1).astype(np.float32)
    s = np.zeros((n, 80), np.float32)
    s[:, 3] = np.sort(rng.uniform(0.1, 1, n))[::-1]
    full = ds.nms(box, s, max_det=1000)
    assert len(full[1]) > 50
    for k in (1, 7, 50):
        for lim, ref in zip(ds.nms(box, s, max_det=k), full):
            assert np.array_equal(lim, ref[:k])


def test_greedy_limit_returns_a_prefix():
    rng = np.random.default_rng(1)
    xy = rng.uniform(0, 300, (200, 2))
    box = np.concatenate([xy, xy + rng.uniform(5, 40, (200, 2))], axis=1).astype(np.float32)
    full = ds._greedy(box, 0.5)
    assert len(full) > 10
    for k in (1, 5, len(full), len(full) + 3):
        assert ds._greedy(box, 0.5, k).tolist() == full[:k].tolist()


def test_max_det_across_classes_is_a_global_cut():
    n = 5
    box = np.stack([row(20 * i, 0, 20 * i + 10, 10) for i in range(n)])
    s = cands([(i, 1, 0.5 + 0.01 * i) for i in range(n)] + [(i, 2, 0.1 + 0.01 * i) for i in range(n)], n)
    _, sc, c = ds.nms(box, s, max_det=3)
    assert c.tolist() == [1, 1, 1] and sc.tolist() == pytest.approx([0.54, 0.53, 0.52])


def test_max_nms_drops_candidates_below_the_cap_before_nms():
    # the 0.8 box would survive on its own, but sits past the cap of 2
    box = np.stack([row(0, 0, 10, 10), row(0, 0, 10, 9), row(50, 50, 60, 60)])
    s = cands([(0, 1, 0.9), (1, 1, 0.85), (2, 1, 0.8)], 3)
    _, sc, _ = ds.nms(box, s, max_nms=2)
    assert sc.tolist() == pytest.approx([0.9])


def test_nothing_over_conf_is_an_empty_result():
    b, sc, c = ds.nms(np.zeros((3, 4), np.float32), np.zeros((3, 80), np.float32))
    assert b.shape == (0, 4) and sc.shape == c.shape == (0,) and c.dtype == np.int64


def test_result_is_sorted_by_score_descending():
    rng = np.random.default_rng(0)
    box = rng.uniform(0, 600, (200, 2)).astype(np.float32)
    box = np.concatenate([box, box + rng.uniform(5, 80, (200, 2)).astype(np.float32)], axis=1)
    s = (rng.uniform(0, 1, (200, 80)) * (rng.uniform(0, 1, (200, 80)) > 0.99)).astype(np.float32)
    _, sc, _ = ds.nms(box, s)
    assert len(sc) > 1 and (np.diff(sc) <= 0).all()


@pytest.mark.parametrize("src, size, mode, want", [
    ((640, 480), 640, "centre", (640, 480, 0, 80, 1.0, 1.0)),
    ((640, 480), 640, "origin", (640, 480, 0, 0, 1.0, 1.0)),
    ((640, 480), 640, "stretch", (640, 640, 0, 0, 1.0, 640 / 480)),
    ((500, 375), 640, "centre", (640, 480, 0, 80, 1.28, 1.28)),
    ((500, 375), 640, "origin", (640, 480, 0, 0, 1.28, 1.28)),
    ((500, 375), 640, "stretch", (640, 640, 0, 0, 1.28, 640 / 375)),
    ((427, 640), 640, "centre", (427, 640, 106, 0, 1.0, 1.0)),
    ((427, 640), 640, "origin", (426, 640, 0, 0, 426 / 427, 1.0)),  # core's even floor loses a column
    ((427, 640), 640, "stretch", (640, 640, 0, 0, 640 / 427, 1.0)),
    ((1920, 1080), (640, 384), "centre", (640, 360, 0, 12, 1 / 3, 1 / 3)),
    ((1920, 1080), (640, 384), "origin", (640, 360, 0, 0, 1 / 3, 1 / 3)),
    ((1920, 1080), (640, 384), "stretch", (640, 384, 0, 0, 1 / 3, 384 / 1080)),
    ((1920, 1080), (640, 640), "centre", (640, 360, 0, 140, 1 / 3, 1 / 3)),
    ((1920, 1080), (640, 640), "origin", (640, 360, 0, 0, 1 / 3, 1 / 3)),
    ((1080, 1920), (640, 384), "centre", (216, 384, 212, 0, 0.2, 0.2)),
])
def test_placement_numbers(src, size, mode, want):
    p = ds.placement(*src, size, mode)
    assert (p.src_w, p.src_h) == src
    got = (p.new_w, p.new_h, p.pad_left, p.pad_top)
    assert got == want[:4]
    assert (p.scale_x, p.scale_y) == pytest.approx(want[4:], rel=1e-12)


def test_centre_unmaps_by_the_single_scale_even_when_new_over_src_differs():
    # 333x250 into 640: scale 1.9219, new 640x480 (480.48 truncated); new/src would give 1.9219 and 1.92
    p = ds.placement(333, 250, 640, "centre")
    assert p.scale_x == p.scale_y == min(640 / 333, 640 / 250)
    assert (p.new_w, p.new_h, p.pad_top) == (640, 480, 80)
    out = ds.unmap([[0, 80, 640, 560]], p)[0]  # the content rectangle
    assert out[2] == pytest.approx(333) and out[3] == pytest.approx(480 / p.scale_y) and out[3] < 250


@pytest.mark.parametrize("src", [(640, 480), (500, 375), (427, 640), (1920, 1080)])
@pytest.mark.parametrize("size", [640, (640, 384)])
@pytest.mark.parametrize("mode", ["centre", "origin", "stretch"])
def test_original_box_round_trips_through_the_input(src, size, mode):
    p = ds.placement(*src, size, mode)
    orig = np.array([[src[0] * 0.1, src[1] * 0.2, src[0] * 0.8, src[1] * 0.9]])
    fwd = orig.copy()
    fwd[:, [0, 2]] = fwd[:, [0, 2]] * p.scale_x + p.pad_left
    fwd[:, [1, 3]] = fwd[:, [1, 3]] * p.scale_y + p.pad_top
    assert np.abs(ds.unmap(fwd, p) - orig).max() < 1e-3


def test_unmap_clips_to_the_source_frame():
    p = ds.placement(640, 480, 640, "centre")  # 80 px of pad above and below
    out = ds.unmap([[-10, 10, 700, 600], [5, 5, 20, 70]], p)
    assert out[0].tolist() == [0, 0, 640, 480]
    assert out[1].tolist() == [5, 0, 20, 0]  # wholly inside the top pad


@pytest.mark.parametrize("args", [(0, 480, 640, "centre"), (640, 480, 640, "zoom"), (640, 480, (640, -1), "origin")])
def test_placement_rejects_nonsense(args):
    with pytest.raises(ValueError):
        ds.placement(*args)


@pytest.mark.parametrize("text,want", [("640", (640, 640)), ("640x384", (640, 384)), ("32x64", (32, 64)),
                                       ("1280", (1280, 1280))])
def test_parse_size(text, want):
    assert ds.parse_size(text) == want


@pytest.mark.parametrize("text", ["", "x", "640x", "x384", "640x384x2", "641", "640x380", "0", "0x32", "-32", "6.4e2",
                                  "640 x 384", "abc"])
def test_parse_size_rejects(text):
    with pytest.raises(ValueError):
        ds.parse_size(text)


@pytest.mark.parametrize("size,want", [(640, ""), ((640, 640), ""), ((640, 384), "-640x384"), (384, "-384x384"),
                                       ((1280, 640), "-1280x640")])
def test_size_suffix_only_marks_a_non_default_size(size, want):
    assert ds.size_suffix(size) == want


def test_numpy_integer_sizes_are_sizes():
    assert ds.size_suffix(np.int64(384)) == "-384x384"
    assert ds.size_suffix((np.int64(640), np.int32(384))) == "-640x384"
    boxes, clss = levels((64, 32))
    xyxy, _ = ds.decode(boxes, clss, (np.int64(64), np.int64(32)))
    assert xyxy.shape == (8 * 4 + 4 * 2 + 2, 4)
    assert ds.placement(100, 50, np.int64(64), "stretch").new_w == 64


def test_decode_leaves_a_float64_input_unchanged():
    boxes, clss = levels(64)
    boxes = [b.astype(np.float64) for b in boxes]
    rng = np.random.default_rng(0)
    for b in boxes:
        b[:] = rng.normal(size=b.shape)
    before = [b.copy() for b in boxes]
    ds.decode(boxes, clss, 64)
    for b, want in zip(boxes, before):
        assert np.array_equal(b, want)


# core rules: a 640 square input showing a 640x480 source at the origin, so rows 480.. are pad
P = ds.placement(640, 480, 640, "origin")
WIRE = 0.5e-4 * 640 * 2  # 4 dp normalised, then back to pixels: x0 and the width each round


def core(box, s, **kw):
    return ds.core_nms(np.asarray(box, np.float32).reshape(-1, 4), s, P, 640, **kw)


def test_round_half_away_is_rusts_not_numpys():
    assert ds.round_half_away([0.0005, 0.0015, -0.0005, 0.0025], 3).tolist() == [0.001, 0.002, -0.001, 0.003]
    assert np.round(0.0025, 3) == 0.002  # what it replaces: half to even


def test_core_takes_one_class_per_anchor_the_last_of_a_tie():
    s = np.zeros((1, 80), np.float32)
    s[0, [3, 7]] = 0.5
    _, sc, c = core([[10, 10, 50, 50]], s)
    assert c.tolist() == [7] and sc.tolist() == [0.5]
    s[0, 9] = 0.4  # multi-label would add a second detection on the same anchor
    assert core([[10, 10, 50, 50]], s)[2].tolist() == [7]


def test_core_floor_is_inclusive():
    s = np.zeros((2, 80), np.float32)
    s[0, 1], s[1, 1] = 0.001, 0.0009
    assert len(core([[0, 0, 10, 10], [100, 100, 110, 110]], s, floor=np.float32(0.001))[1]) == 1


def test_core_suppresses_at_equal_iou():
    box = [[0, 0, 10, 10], [0, 0, 10, 4.5]]  # IoU exactly 0.45
    s = cands([(0, 1, 0.9), (1, 1, 0.8)], 2)
    assert len(core(box, s)[1]) == 1
    assert len(ds.nms(np.asarray(box, np.float32), s, iou=0.45)[1]) == 2  # the reference keeps it


def test_core_drops_a_pad_ghost_before_it_can_suppress_a_real_box():
    # the ghost lies wholly in the pad (y >= 480) and overlaps the real edge box by more than 0.45
    box = [[100, 480, 200, 600], [100, 400, 200, 600]]
    s = cands([(0, 0, 0.9), (1, 0, 0.8)], 2)
    b, sc, _ = core(box, s)
    assert sc.tolist() == [0.8]
    assert b[0].tolist() == pytest.approx([100, 400, 200, 480], abs=WIRE)  # clamped to the content


def test_core_cuts_to_max_cand_before_nms_and_max_det_after():
    n = 40
    box = [[15 * i, 0, 15 * i + 10, 10] for i in range(n)]  # disjoint and inside the frame: NMS keeps all
    s = cands([(i, 0, 0.5 + i / 100) for i in range(n)], n)
    assert len(core(box, s)[1]) == 32
    _, sc, _ = core(box, s, max_cand=10)
    assert sc.tolist() == pytest.approx([0.89 - i / 100 for i in range(10)])


def test_core_ties_keep_anchor_order():
    box = [[20 * i, 0, 20 * i + 10, 10] for i in range(5)]
    s = cands([(i, 0, 0.5) for i in range(5)], 5)
    b, _, _ = core(box, s, max_cand=3)
    assert b[:, 0].tolist() == pytest.approx([0, 20, 40], abs=WIRE)


def test_core_refuses_boxes_past_four_inputs():
    s = cands([(0, 0, 0.9), (1, 0, 0.8)], 2)
    assert core([[0, 0, 2561, 10], [0, 0, 2560, 10]], s)[1].tolist() == [0.8]


def test_core_rounds_the_wire():
    s = cands([(0, 0, 0.12345)], 1)
    b, sc, _ = core([[10.00004, 20, 30, 40]], s)
    assert sc.tolist() == [0.123]
    # normalised 4 dp: x0 = 10.00004 / 640 -> 0.0156 -> 9.984 px
    assert b[0, 0] == pytest.approx(0.0156 * 640)


def test_detections_dispatches_and_refuses_unknown_rules():
    s = cands([(0, 0, 0.9)], 1)
    box = np.asarray([[10, 10, 50, 50]], np.float32)
    ref = ds.detections(box, s, P, 640, "reference")
    assert ref[0][0].tolist() == pytest.approx([10, 10, 50, 50])
    assert ds.detections(box, s, P, 640, "core")[1].tolist() == [0.9]
    with pytest.raises(ValueError, match="rules"):
        ds.detections(box, s, P, 640, "rust")


def test_core_reproduces_the_residual_in_heads_rs_nms_doc():
    # heads.rs `nms` doc: 1920x1080 into a 416 input (content rows 0..234), two person boxes x 65..130;
    # 0.9 at y 233..500 (one content row) suppresses 0.8 at y 150..500 (IoU 267 / 350 = 0.7629) and
    # what reaches the wire is [0.1563, 0.9957, 0.1563, 0.0043]
    p = ds.placement(1920, 1080, 416, "origin")
    assert (p.new_w, p.new_h) == (416, 234)
    s = cands([(0, 0, 0.9), (1, 0, 0.8)], 2)
    b, sc, c = ds.core_nms(np.array([[65, 233, 130, 500], [65, 150, 130, 500]], np.float64), s, p, 416)
    assert sc.tolist() == [0.9] and c.tolist() == [0]
    x0, y0 = b[0, 0] / 1920, b[0, 1] / 1080
    assert [round(x0, 4), round(y0, 4), round(b[0, 2] / 1920 - x0, 4), round(b[0, 3] / 1080 - y0, 4)] == \
        [0.1563, 0.9957, 0.1563, 0.0043]  # 65 / 416 = 0.15625: half away from zero, where numpy says 0.1562


def test_core_unprojects_by_scale_then_by_source_as_geometry_rs():
    # 427 px wide filling 640: 4 / (scale * 427) is 0.00625000000000000..., which rounds up; core's
    # 4 / scale / 427 lands just under and rounds down
    p = ds.placement(427, 320, 640, "origin")
    x0, _y0, _w, _h, alive = ds._wire([[4.0, 0.0, 100.0, 100.0]], p)
    assert x0.tolist() == [0.0062] and alive.all()
