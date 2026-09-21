"""DFDC planning: identity-safe splits and class balance.

The leakage test is the important one. A fake and the real clip it came from
show the same face, so they must never straddle a split boundary.
"""

import json

import pytest

from wgn.dfdc import (
    LABEL_FAKE,
    LABEL_REAL,
    assign_splits,
    build_plan,
    frame_budget,
    group_by_identity,
    load_metadata,
    verify_plan,
)


def make_part(root, part_name, groups):
    """Write a fake DFDC part: `groups` maps real stem -> number of fakes."""
    folder = root / part_name
    folder.mkdir(parents=True)
    meta = {}
    for real, n_fakes in groups.items():
        (folder / f"{real}.mp4").write_bytes(b"\x00")
        meta[f"{real}.mp4"] = {"label": "REAL", "split": "train"}
        for i in range(n_fakes):
            fake = f"{real}_fake{i}"
            (folder / f"{fake}.mp4").write_bytes(b"\x00")
            meta[f"{fake}.mp4"] = {
                "label": "FAKE",
                "split": "train",
                "original": f"{real}.mp4",
            }
    (folder / "metadata.json").write_text(json.dumps(meta))
    return folder


@pytest.fixture
def dfdc_root(tmp_path):
    make_part(tmp_path, "dfdc_train_part_00", {f"real{i:02d}": 5 for i in range(12)})
    make_part(tmp_path, "dfdc_train_part_01", {f"real{i:02d}": 5 for i in range(12, 24)})
    return tmp_path


def test_load_metadata_reads_labels_and_identity_links(dfdc_root):
    clips = load_metadata(dfdc_root)
    assert len(clips) == 24 * 6  # 24 reals + 5 fakes each
    reals = [c for c in clips if c.label == LABEL_REAL]
    fakes = [c for c in clips if c.label == LABEL_FAKE]
    assert len(reals) == 24
    assert len(fakes) == 120
    assert all(c.original is None for c in reals)
    assert all(c.identity == c.original for c in fakes)


def test_load_metadata_skips_videos_absent_from_disk(tmp_path):
    folder = tmp_path / "dfdc_train_part_00"
    folder.mkdir(parents=True)
    (folder / "present.mp4").write_bytes(b"\x00")
    (folder / "metadata.json").write_text(json.dumps({
        "present.mp4": {"label": "REAL"},
        "absent.mp4": {"label": "FAKE", "original": "present.mp4"},
    }))
    clips = load_metadata(tmp_path)
    assert [c.name for c in clips] == ["present"]


def test_fake_groups_with_its_original(dfdc_root):
    groups = group_by_identity(load_metadata(dfdc_root))
    assert len(groups) == 24
    for identity, members in groups.items():
        assert sum(1 for c in members if c.label == LABEL_REAL) == 1
        assert all(c.identity == identity for c in members)


def test_assign_splits_covers_every_group_exactly_once(dfdc_root):
    groups = group_by_identity(load_metadata(dfdc_root))
    split_of = assign_splits(groups, seed=0)
    assert set(split_of) == set(groups)
    assert set(split_of.values()) <= {"train", "val", "test"}


def test_assign_splits_is_deterministic_per_seed(dfdc_root):
    groups = group_by_identity(load_metadata(dfdc_root))
    assert assign_splits(groups, seed=0) == assign_splits(groups, seed=0)
    assert assign_splits(groups, seed=0) != assign_splits(groups, seed=1)


def test_assign_splits_rejects_bad_ratios(dfdc_root):
    groups = group_by_identity(load_metadata(dfdc_root))
    with pytest.raises(ValueError):
        assign_splits(groups, ratios=(0.8, 0.3, 0.1))


def test_plan_has_no_identity_leakage(dfdc_root):
    """The highest-severity silent failure mode."""
    plan = build_plan(dfdc_root, seed=0)
    assert not [e for e in verify_plan(plan) if e.startswith("IDENTITY LEAK")]

    identities = {}
    for split, clips in plan["splits"].items():
        for clip in clips:
            assert identities.setdefault(clip.identity, split) == split


def test_verify_plan_detects_injected_leak(dfdc_root):
    plan = build_plan(dfdc_root, seed=0)
    leaked = plan["splits"]["train"][0]
    plan["splits"]["test"].append(leaked)
    errors = verify_plan(plan)
    assert any(e.startswith("IDENTITY LEAK") for e in errors)


def test_frame_budget_balances_classes():
    # 5 fakes per real, 10 frames each -> real clips need 50 frames
    frames_real, frames_fake = frame_budget(n_real=24, n_fake=120, frames_fake=10)
    assert frames_fake == 10
    assert frames_real == 50
    assert 24 * frames_real == 120 * frames_fake


def test_frame_budget_is_clamped_and_safe_on_empty():
    assert frame_budget(n_real=1, n_fake=100_000)[0] <= 120
    assert frame_budget(n_real=0, n_fake=0) == (10, 10)


def test_plan_budgets_are_class_balanced(dfdc_root):
    plan = build_plan(dfdc_root, seed=0)
    assert not [e for e in verify_plan(plan) if e.startswith("CLASS IMBALANCE")]
    for budget in plan["budgets"].values():
        real, fake = budget["expected_real_frames"], budget["expected_fake_frames"]
        assert max(real, fake) / min(real, fake) <= 1.05


def test_build_plan_requires_data(tmp_path):
    with pytest.raises(RuntimeError):
        build_plan(tmp_path)


def test_prepare_dfdc_reuses_shared_crop_pipeline():
    import scripts.prepare_dfdc as prep

    from wgn.preprocess import process_video
    assert prep.process_video is process_video
