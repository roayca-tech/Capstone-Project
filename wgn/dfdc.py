"""
DFDC (Deepfake Detection Challenge) support.

Kaggle competition layout, one folder per downloaded part:

    dfdc_train_part_00/
        metadata.json
        aagfhgtpmv.mp4
        ...

`metadata.json` maps a filename to its label:

    {"aagfhgtpmv.mp4": {"label": "FAKE", "split": "train", "original": "vudstovrck.mp4"},
     "vudstovrck.mp4": {"label": "REAL", "split": "train"}}

Every fake carries an `original` naming the real clip it was derived from, so a
fake and its source show the same face. Those clips must land in the same split
or the model memorises identities instead of blending artifacts — the same
leakage the paper avoids by using the official FF++ split lists.

DFDC is fake-heavy (roughly 1 real per 5 fakes), so `frame_budget` samples more
frames from each real clip to equalise the classes. This generalises the paper's
10-frames-per-fake / 40-frames-per-real protocol (Sec. 4.2).
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

LABEL_REAL = "real"
LABEL_FAKE = "fake"
SPLITS = ("train", "val", "test")

# Frames per fake clip, matching the paper's FF++ protocol.
FRAMES_FAKE = 10

# Guard rail: never sample more than this from a single real clip, however
# unbalanced the download is. A clip has a few hundred frames at most.
MAX_FRAMES_REAL = 120


@dataclass(frozen=True)
class Clip:
    """One DFDC video with its label and identity link."""

    name: str              # filename stem, e.g. "aagfhgtpmv"
    path: Path
    label: str             # LABEL_REAL or LABEL_FAKE
    original: str | None   # stem of the source real clip, fakes only
    part: str              # containing part folder, e.g. "dfdc_train_part_00"

    @property
    def identity(self) -> str:
        """Group key: a fake belongs to the identity of its source clip."""
        return self.original or self.name


def find_metadata(root: Path) -> list[Path]:
    """Locate every metadata.json under a DFDC download root."""
    return sorted(Path(root).rglob("metadata.json"))


def load_metadata(root: Path) -> list[Clip]:
    """Parse all parts under `root` into Clip records.

    Videos listed in metadata but missing from disk are skipped; partial
    downloads are normal when only a few parts are fetched.
    """
    clips: list[Clip] = []
    for meta_path in find_metadata(root):
        folder = meta_path.parent
        info = json.loads(meta_path.read_text())
        for filename, row in info.items():
            video = folder / filename
            if not video.exists():
                continue
            raw_label = str(row.get("label", "")).upper()
            if raw_label not in {"REAL", "FAKE"}:
                continue
            label = LABEL_REAL if raw_label == "REAL" else LABEL_FAKE
            original = row.get("original")
            clips.append(
                Clip(
                    name=Path(filename).stem,
                    path=video,
                    label=label,
                    original=Path(original).stem if original else None,
                    part=folder.name,
                )
            )
    return clips


def group_by_identity(clips: list[Clip]) -> dict[str, list[Clip]]:
    """Bucket clips so a real clip and every fake derived from it stay together."""
    groups: dict[str, list[Clip]] = {}
    for clip in clips:
        groups.setdefault(clip.identity, []).append(clip)
    return groups


def assign_splits(
    groups: dict[str, list[Clip]],
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 0,
) -> dict[str, str]:
    """Map each identity group to train / val / test.

    Splitting happens at the identity level, never the clip level. Deterministic
    for a given seed so a run is reproducible from its config.
    """
    if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios must be three fractions summing to 1, got {ratios}")

    keys = sorted(groups)
    random.Random(seed).shuffle(keys)

    n = len(keys)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    # Remainder goes to test so every group is assigned exactly once.
    bounds = {
        "train": keys[:n_train],
        "val": keys[n_train:n_train + n_val],
        "test": keys[n_train + n_val:],
    }
    return {key: split for split, members in bounds.items() for key in members}


def frame_budget(
    n_real: int, n_fake: int, frames_fake: int = FRAMES_FAKE
) -> tuple[int, int]:
    """Return (frames_per_real_clip, frames_per_fake_clip) balancing the classes.

    DFDC has far more fakes than reals, so real clips are sampled more densely,
    exactly as the paper samples 40 frames per real FF++ clip against 10 per fake.
    """
    if n_real <= 0 or n_fake <= 0:
        return frames_fake, frames_fake
    target = n_fake * frames_fake
    frames_real = round(target / n_real)
    return max(1, min(frames_real, MAX_FRAMES_REAL)), frames_fake


def build_plan(
    root: Path,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 0,
    frames_fake: int = FRAMES_FAKE,
) -> dict:
    """Full preprocessing plan: per-split clip lists plus per-class frame counts."""
    clips = load_metadata(root)
    if not clips:
        raise RuntimeError(f"No DFDC metadata.json with matching videos under {root}")

    groups = group_by_identity(clips)
    split_of = assign_splits(groups, ratios, seed)

    by_split: dict[str, list[Clip]] = {s: [] for s in SPLITS}
    for identity, members in groups.items():
        by_split[split_of[identity]].extend(members)

    budgets = {}
    for split, members in by_split.items():
        n_real = sum(1 for c in members if c.label == LABEL_REAL)
        n_fake = sum(1 for c in members if c.label == LABEL_FAKE)
        real_frames, fake_frames = frame_budget(n_real, n_fake, frames_fake)
        budgets[split] = {
            "n_real_clips": n_real,
            "n_fake_clips": n_fake,
            "frames_per_real": real_frames,
            "frames_per_fake": fake_frames,
            "expected_real_frames": n_real * real_frames,
            "expected_fake_frames": n_fake * fake_frames,
        }

    return {
        "root": str(root),
        "seed": seed,
        "ratios": list(ratios),
        "n_clips": len(clips),
        "n_identities": len(groups),
        "parts": sorted({c.part for c in clips}),
        "splits": by_split,
        "budgets": budgets,
    }


def verify_plan(plan: dict) -> list[str]:
    """Return a list of problems; empty means the plan is safe to preprocess."""
    errors = []
    seen: dict[str, str] = {}
    for split, clips in plan["splits"].items():
        for clip in clips:
            prior = seen.get(clip.identity)
            if prior and prior != split:
                errors.append(
                    f"IDENTITY LEAK {clip.identity!r} in both {prior} and {split}"
                )
            seen[clip.identity] = split

    for split, budget in plan["budgets"].items():
        real, fake = budget["expected_real_frames"], budget["expected_fake_frames"]
        if real and fake:
            ratio = max(real, fake) / min(real, fake)
            if ratio > 1.05:
                errors.append(
                    f"CLASS IMBALANCE {split}: real={real} fake={fake} ratio={ratio:.2f}"
                )
        elif not real or not fake:
            errors.append(f"EMPTY CLASS {split}: real={real} fake={fake}")
    return errors
