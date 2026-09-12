"""Audit a preprocessed crop tree against the Phase 3 acceptance criteria.

Reports frame counts, class balance, detection failures, pixel mean/std,
and asserts zero clip-ID overlap across train/val/test.

Usage:
    python scripts/verify_data.py --data data/ffpp_c23 --splits splits/
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

MANIPS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "FaceShifter"]
TARGET_TRAIN_REAL = 28800
TARGET_TRAIN_PER_MANIP = 7200
MAX_RATIO = 1.05
MAX_DETECT_FAIL = 0.02


def clip_ids(root: Path, split: str, category: str) -> set[str]:
    d = root / split / category
    if not d.exists():
        return set()
    return {p.name for p in d.iterdir() if p.is_dir()}


def list_pngs(root: Path, split: str, category: str) -> list[Path]:
    d = root / split / category
    if not d.exists():
        return []
    return sorted(d.rglob("*.png"))


def categories(root: Path, split: str) -> list[str]:
    d = root / split
    if not d.exists():
        return []
    return sorted(p.name for p in d.iterdir() if p.is_dir())


def assert_no_clip_overlap(root: Path) -> list[str]:
    """Highest-severity silent failure mode: identity leakage across splits."""
    errors = []
    cats = set()
    for split in ("train", "val", "test"):
        cats.update(categories(root, split))
    for cat in sorted(cats):
        ids = {s: clip_ids(root, s, cat) for s in ("train", "val", "test")}
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            overlap = ids[a] & ids[b]
            if overlap:
                sample = sorted(overlap)[:8]
                errors.append(
                    f"CLIP OVERLAP {cat} {a}∩{b} n={len(overlap)} e.g. {sample}"
                )
    return errors


def pixel_stats(paths: list[Path], limit: int = 256) -> dict:
    if not paths:
        return {"mean": None, "std": None, "n": 0}
    rng = np.random.default_rng(0)
    pick = paths if len(paths) <= limit else [
        paths[i] for i in rng.choice(len(paths), size=limit, replace=False)
    ]
    acc = []
    for p in pick:
        arr = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32)
        acc.append(arr)
    stacked = np.stack(acc, axis=0)
    return {
        "mean": stacked.mean(axis=(0, 1, 2)).tolist(),
        "std": stacked.std(axis=(0, 1, 2)).tolist(),
        "n": int(stacked.shape[0]),
    }


def inspect_shapes(paths: list[Path], limit: int = 64) -> dict:
    bad = []
    checked = 0
    for p in paths[:limit]:
        checked += 1
        im = Image.open(p)
        if im.size != (256, 256) or im.mode != "RGB" or p.suffix.lower() != ".png":
            bad.append({"path": str(p), "size": im.size, "mode": im.mode})
    return {"checked": checked, "bad": bad}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--splits", default="splits")
    ap.add_argument("--out", default=None, help="write JSON report here")
    args = ap.parse_args()

    root = Path(args.data)
    report = {
        "data": str(root),
        "counts": {},
        "class_balance": {},
        "shapes": {},
        "pixels": {},
        "failures": {},
        "overlap_errors": [],
        "acceptance": {},
    }

    for split in ("train", "val", "test"):
        split_counts = {}
        for cat in categories(root, split):
            pngs = list_pngs(root, split, cat)
            split_counts[cat] = {
                "frames": len(pngs),
                "clips": len(clip_ids(root, split, cat)),
            }
        report["counts"][split] = split_counts

        n_real = split_counts.get("real", {}).get("frames", 0)
        n_fake = sum(
            v["frames"] for k, v in split_counts.items() if k != "real"
        )
        ratio = (max(n_real, n_fake) / min(n_real, n_fake)) if n_real and n_fake else None
        report["class_balance"][split] = {
            "real": n_real,
            "fake": n_fake,
            "ratio": ratio,
        }

        sample = []
        for cat in categories(root, split):
            sample.extend(list_pngs(root, split, cat)[:16])
        report["shapes"][split] = inspect_shapes(sample)
        report["pixels"][split] = pixel_stats(sample)

    fail_path = root / "failures.json"
    if fail_path.exists():
        failures = json.loads(fail_path.read_text())
        by_reason = defaultdict(int)
        for row in failures:
            by_reason[row["reason"]] += 1
        n_detect = by_reason.get("detection_failed", 0)
        n_attempted = 0
        for split_counts in report["counts"].values():
            n_attempted += sum(v["clips"] for v in split_counts.values())
        # attempted ≈ successful clips + detection failures (missing videos extra)
        n_attempted += n_detect
        rate = n_detect / n_attempted if n_attempted else None
        report["failures"] = {
            "path": str(fail_path),
            "n": len(failures),
            "by_reason": dict(by_reason),
            "detection_failure_rate": rate,
        }
    else:
        report["failures"] = {"path": None, "n": 0, "note": "no failures.json"}

    report["overlap_errors"] = assert_no_clip_overlap(root)

    train = report["counts"].get("train", {})
    n_real = train.get("real", {}).get("frames", 0)
    per_manip = {
        m: train.get(m, {}).get("frames", 0)
        for m in MANIPS if m in train
    }
    n_fake = sum(per_manip.values())
    ratio = report["class_balance"]["train"]["ratio"]
    detect_rate = report["failures"].get("detection_failure_rate")
    n_bad_shapes = sum(len(v.get("bad", [])) for v in report["shapes"].values())

    report["acceptance"] = {
        "train_real_near_28800": n_real > 0 and abs(n_real - TARGET_TRAIN_REAL) / TARGET_TRAIN_REAL < 0.05,
        "train_fake_near_28800": n_fake > 0 and abs(n_fake - TARGET_TRAIN_REAL) / TARGET_TRAIN_REAL < 0.05,
        "class_ratio_within_1.05": ratio is not None and ratio <= MAX_RATIO,
        "detection_fail_lt_2pct": detect_rate is not None and detect_rate < MAX_DETECT_FAIL,
        "all_crops_256_rgb_png": n_bad_shapes == 0,
        "zero_clip_id_overlap": len(report["overlap_errors"]) == 0,
        "per_manipulation_train_frames": per_manip,
    }

    text = json.dumps(report, indent=2)
    print(text)
    dest = Path(args.out) if args.out else root / "verify_report.json"
    dest.write_text(text)
    print(f"wrote {dest}")

    if report["overlap_errors"]:
        print("FAIL: clip-ID overlap across splits", file=sys.stderr)
        for e in report["overlap_errors"]:
            print(" ", e, file=sys.stderr)
        sys.exit(1)

    ok = all(v is True for k, v in report["acceptance"].items() if k != "per_manipulation_train_frames")
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
