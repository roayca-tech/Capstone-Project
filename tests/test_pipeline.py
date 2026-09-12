"""Shared crop function and data-audit helpers (no video I/O)."""

import numpy as np

from wgn.preprocess import crop_face, process_video, sample_indices


def test_process_video_calls_shared_crop_face():
    assert "crop_face" in process_video.__code__.co_names


def test_preprocess_external_reuses_crop_face():
    import scripts.preprocess_external as ext

    from wgn.preprocess import crop_face as shared
    assert ext.crop_face is shared
    assert ext.process_video is process_video


def test_crop_face_returns_256_rgb():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[100:300, 200:400] = 200
    face = crop_face(frame, (200, 100, 400, 300), size=256)
    assert face is not None
    assert face.shape == (256, 256, 3)


def test_sample_indices_skips_endpoints_when_possible():
    idx = sample_indices(100, 10)
    assert len(idx) == 10
    assert 0 not in idx
    assert 99 not in idx


def test_verify_data_overlap_helper(tmp_path):
    from scripts.verify_data import assert_no_clip_overlap

    for split, clip in (("train", "033"), ("val", "720"), ("test", "000")):
        d = tmp_path / split / "real" / clip
        d.mkdir(parents=True)
        (d / "000001.png").write_bytes(b"x")
    assert assert_no_clip_overlap(tmp_path) == []

    leak = tmp_path / "test" / "real" / "033"
    leak.mkdir(parents=True)
    (leak / "000001.png").write_bytes(b"x")
    errors = assert_no_clip_overlap(tmp_path)
    assert errors and "033" in errors[0]
