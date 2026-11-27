import pytest

import coco_calib


def test_draw_is_seeded_and_ignores_the_files_order():
    ids = list(range(1000, 1100))
    a = coco_calib.draw(ids, 10, seed=1)
    assert a == coco_calib.draw(list(reversed(ids)), 10, seed=1)
    assert a != coco_calib.draw(ids, 10, seed=2)
    assert len(set(a)) == 10 and set(a) <= set(ids)


def test_a_longer_draw_starts_with_the_shorter_one():
    ids = list(range(500))
    assert coco_calib.draw(ids, 200)[:50] == coco_calib.draw(ids, 50)


def test_draw_more_than_there_are_is_refused():
    with pytest.raises(ValueError):
        coco_calib.draw([1, 2], 3)


def test_read_ids_prefix_comments_and_duplicates(tmp_path):
    p = tmp_path / "ids.txt"
    p.write_text("# seed\n5\n3\n9\n")
    assert coco_calib.read_ids(str(p)) == [5, 3, 9]
    assert coco_calib.read_ids(str(p), 2) == [5, 3]
    with pytest.raises(ValueError):
        coco_calib.read_ids(str(p), 4)
    p.write_text("5\n5\n")
    with pytest.raises(ValueError, match="duplicate"):
        coco_calib.read_ids(str(p))


def test_tracked_list_is_500_distinct_ids():
    ids = coco_calib.read_ids(coco_calib.IDS)
    assert len(ids) == 500 == len(set(ids))


def test_fetch_refuses_a_directory_holding_other_files(tmp_path):
    p = tmp_path / "ids.txt"
    p.write_text("5\n")
    d = tmp_path / "imgs"
    d.mkdir()
    (d / "other.jpg").write_bytes(b"x")
    with pytest.raises(SystemExit, match="new or empty"):
        coco_calib.main(["fetch", str(p), str(d)])


def test_fetch_refuses_a_stray_png_beside_its_own_files(tmp_path):
    p = tmp_path / "ids.txt"
    p.write_text("5\n")
    d = tmp_path / "imgs"
    d.mkdir()
    (d / coco_calib.file_name(5)).write_bytes(b"x")
    (d / "MANIFEST.sha256").write_text("")
    (d / "stray.png").write_bytes(b"x")
    with pytest.raises(SystemExit, match="new or empty"):
        coco_calib.main(["fetch", str(p), str(d)])
