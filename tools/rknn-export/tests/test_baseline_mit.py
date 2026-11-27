import numpy as np
import pytest

import baseline_mit as bm


def test_snap_boundaries_and_ceiling():
    p = np.array([0, 1 / 1024, 1 / 512 - 1e-6, 1 / 512, 3 / 512, 0.5, 255 / 256, 1 - 1e-6, 1.0], np.float32)
    got = bm.snap_scores(p)
    # below half a step is 0; a tie goes up; nothing can reach 1, the top 8-bit value is 255/256
    assert got.tolist() == [0, 0, 0, 1 / 256, 2 / 256, 0.5, 255 / 256, 255 / 256, 255 / 256]
    assert got.dtype == np.float32
    assert (got * 256 == np.round(got * 256)).all()


def test_snap_is_idempotent():
    p = np.random.default_rng(0).uniform(size=1000).astype(np.float32)
    once = bm.snap_scores(p)
    assert (bm.snap_scores(once) == once).all()


def test_stems_keep_variants_apart():
    assert bm.variant_stem("v9-t", "origin", "bilinear-cv2", None) == "v9-t-pytorch-origin-bilinear-cv2"
    assert bm.variant_stem("v9-t", "origin", "bilinear-cv2", 200, (640, 384), True) == \
        "v9-t-pytorch-origin-bilinear-cv2-640x384-snap256-limit200"


@pytest.mark.parametrize("limit", ["0", "-5"])
def test_limit_below_one_is_rejected(limit, capsys):
    with pytest.raises(SystemExit):
        bm.main(["v9-t", "--mode", "origin", "--resample", "bilinear-cv2", "--limit", limit])
    assert "below 1" in capsys.readouterr().err


def test_print_stem_matches_variant_stem(capsys):
    assert bm.main(["v9-t", "--mode", "centre", "--resample", "lanczos-pil", "--print-stem"]) is None
    assert capsys.readouterr().out == bm.variant_stem("v9-t", "centre", "lanczos-pil", None) + "\n"
