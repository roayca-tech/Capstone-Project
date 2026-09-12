"""Official FF++ JSONs must not leak identities across splits."""

import json
from pathlib import Path

from wgn.preprocess import fake_ids, load_splits, real_ids

SPLIT_DIR = Path(__file__).resolve().parents[1] / "splits"


def test_official_split_sizes():
    splits = load_splits(SPLIT_DIR)
    assert len(splits["train"]) == 360
    assert len(splits["val"]) == 70
    assert len(splits["test"]) == 70
    assert len(real_ids(splits["train"])) == 720
    assert len(real_ids(splits["val"])) == 140
    assert len(real_ids(splits["test"])) == 140


def test_official_splits_have_no_clip_id_overlap():
    splits = load_splits(SPLIT_DIR)
    real = {name: set(real_ids(pairs)) for name, pairs in splits.items()}
    fake = {name: set(fake_ids(pairs)) for name, pairs in splits.items()}
    for kind, ids in (("real", real), ("fake", fake)):
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            overlap = ids[a] & ids[b]
            assert not overlap, f"{kind} {a}∩{b}: {sorted(overlap)[:8]}"


def test_split_files_are_official_lists_of_pairs():
    for name in ("train", "val", "test"):
        pairs = json.loads((SPLIT_DIR / f"{name}.json").read_text())
        assert pairs and all(len(p) == 2 for p in pairs)
