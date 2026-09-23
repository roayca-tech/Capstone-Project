"""MTCNN crop pipeline for Celeb-DF, DFDC, and WildDeepfake.

Reuses `crop_face` / `process_video` from wgn.preprocess — the same function
that produced the FF++ training crops. Do not reimplement cropping here.

Usage:
    python scripts/preprocess_external.py --dataset celebdf \
        --root $CELEBDF --out data/celebdf
    python scripts/preprocess_external.py --dataset dfdc \
        --root $DFDC --out data/dfdc
    python scripts/preprocess_external.py --dataset wilddeepfake \
        --root $WDF --out data/wilddeepfake
    python scripts/preprocess_external.py --dataset hidf \
        --root $HIDF --out data/hidf
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm

from wgn.preprocess import FRAMES_FAKE, crop_face, process_video

# Re-export so a code review can assert a single shared crop function.
assert crop_face is not None
assert "crop_face" in process_video.__code__.co_names


def _iter_videos(folder: Path) -> list[Path]:
    vids = []
    for ext in ("*.mp4", "*.avi", "*.mkv", "*.mov"):
        vids.extend(folder.rglob(ext))
    return sorted(vids)


def _run_list(items, detector, size, failures):
    saved = 0
    for label, clip, src, dest, n_frames in tqdm(items, desc="crops"):
        if not src.exists():
            failures.append({
                "category": label, "clip": clip, "path": str(src),
                "reason": "missing_video", "saved": 0,
            })
            continue
        result = process_video(src, dest, n_frames, detector, size)
        saved += result["saved"]
        if result.get("reason"):
            failures.append({
                "category": label, "clip": clip, "path": str(src),
                **result,
            })
    return saved


def celebdf_items(root: Path, out: Path, n_frames: int):
    """Celeb-DF v2: Celeb-real / YouTube-real / Celeb-synthesis.

    If List_of_testing_videos.txt is present, only those clips are extracted
    into test/{real,fake}. Otherwise every video is written under test/.
    """
    test_list = root / "List_of_testing_videos.txt"
    items = []
    if test_list.exists():
        for line in test_list.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            # official format: "<label> <relpath>" with label 0=fake, 1=real
            # or just a relative path
            if len(parts) >= 2 and parts[0] in {"0", "1"}:
                label_i, rel = int(parts[0]), parts[1]
                label = "real" if label_i == 1 else "fake"
            else:
                rel = parts[-1]
                label = "real" if "real" in rel.lower() else "fake"
            src = root / rel
            clip = Path(rel).stem
            items.append((label, clip, src, out / "test" / label / clip, n_frames))
        return items

    mapping = {
        "Celeb-real": "real",
        "YouTube-real": "real",
        "Celeb-synthesis": "fake",
    }
    for folder, label in mapping.items():
        src_dir = root / folder
        if not src_dir.exists():
            continue
        for src in _iter_videos(src_dir):
            items.append((label, src.stem, src, out / "test" / label / src.stem, n_frames))
    return items


def dfdc_items(root: Path, out: Path, n_frames: int):
    """DFDC: metadata.json next to mp4s, or a flat label in the filename."""
    items = []
    for meta in root.rglob("metadata.json"):
        info = json.loads(meta.read_text())
        for name, row in info.items():
            label = "fake" if str(row.get("label", "")).lower() == "fake" else "real"
            src = meta.parent / name
            clip = Path(name).stem
            items.append((label, clip, src, out / "test" / label / clip, n_frames))
    if items:
        return items
    for src in _iter_videos(root):
        name = src.name.lower()
        label = "fake" if "fake" in name else "real"
        items.append((label, src.stem, src, out / "test" / label / src.stem, n_frames))
    return items


def wilddeepfake_items(root: Path, out: Path, n_frames: int):
    """WildDeepfake: real/ and fake/ trees (train/test optional)."""
    items = []
    for split in ("test", "train", "val"):
        for label in ("real", "fake"):
            d = root / split / label
            if not d.exists():
                d = root / label / split
            if not d.exists():
                continue
            dest_split = "test" if split == "test" else split
            for src in _iter_videos(d):
                items.append(
                    (label, src.stem, src, out / dest_split / label / src.stem, n_frames)
                )
    if items:
        return items
    for label in ("real", "fake"):
        d = root / label
        if not d.exists():
            continue
        for src in _iter_videos(d):
            items.append((label, src.stem, src, out / "test" / label / src.stem, n_frames))
    return items


def hidf_items(root: Path, out: Path, n_frames: int):
    """HiDF videos: Real-vid/ and Fake-vid/ next to each other."""
    items = []
    for folder, label in (("Real-vid", "real"), ("Fake-vid", "fake")):
        src_dir = root / folder
        if not src_dir.exists():
            continue
        for src in _iter_videos(src_dir):
            items.append((label, src.stem, src, out / "test" / label / src.stem, n_frames))
    return items


BUILDERS = {
    "celebdf": celebdf_items,
    "dfdc": dfdc_items,
    "wilddeepfake": wilddeepfake_items,
    "hidf": hidf_items,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(BUILDERS))
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-frames", type=int, default=FRAMES_FAKE)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    from facenet_pytorch import MTCNN

    root, out = Path(args.root), Path(args.out)
    if not root.exists():
        print(f"dataset root not found: {root}", file=sys.stderr)
        sys.exit(1)

    detector = MTCNN(keep_all=True, device=args.device, post_process=False)
    items = BUILDERS[args.dataset](root, out, args.n_frames)
    if not items:
        print(f"no videos discovered under {root}", file=sys.stderr)
        sys.exit(1)

    failures = []
    saved = _run_list(items, detector, args.size, failures)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset": args.dataset,
        "root": str(root),
        "n_videos": len(items),
        "n_frames_saved": saved,
        "n_frames_per_clip": args.n_frames,
        "size": args.size,
        "crop_fn": "wgn.preprocess.crop_face",
        "process_fn": "wgn.preprocess.process_video",
        "n_failures": len(failures),
    }
    (out / "preprocess_manifest.json").write_text(json.dumps(manifest, indent=2))
    (out / "failures.json").write_text(json.dumps(failures, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
