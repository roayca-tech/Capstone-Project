"""
FF++ preprocessing: sample frames from videos, detect + crop faces with MTCNN,
resize to 256x256, write to disk.

Paper protocol (Sec. 4.2):
  - 720 / 140 / 140 clip split per subset (official FF++ split lists)
  - 10 frames per FAKE clip for each of the 4 manipulations
  - 40 frames per REAL clip
  -> 4 * 720 * 10 = 28,800 fake vs 720 * 40 = 28,800 real training frames.
     The 10/40 ratio exists precisely to balance the classes. Do not change it
     without also changing the loss weighting.

Requires: facenet-pytorch (MTCNN), opencv-python, tqdm.

Usage:
    python -m wgn.preprocess \
        --root /data/FaceForensics++ \
        --out  /data/ffpp_crops \
        --compression c23 \
        --split-json /data/ffpp_splits
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

MANIPULATIONS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
FACESHIFTER = "FaceShifter"

# Frames per clip. Real clips are sampled 4x more heavily because there are
# 4x fewer of them (1000 real vs 4000 fake).
FRAMES_FAKE = 10
FRAMES_REAL = 40

# Enlarge the MTCNN box before cropping. Blending seams and boundary artifacts
# live just outside the tight face box, so this margin is not cosmetic.
BBOX_SCALE = 1.3


def load_splits(split_dir: Path):
    """Read the official FF++ train/val/test.json split files.

    Each file is a list of pairs, e.g. [["033","097"], ...]. For real videos the
    clip id is the first element; for fakes the clip name is "033_097".
    """
    splits = {}
    for name in ["train", "val", "test"]:
        with open(split_dir / f"{name}.json") as f:
            pairs = json.load(f)
        splits[name] = pairs
    return splits


def real_ids(pairs):
    ids = set()
    for a, b in pairs:
        ids.add(a)
        ids.add(b)
    return sorted(ids)


def fake_ids(pairs):
    return sorted([f"{a}_{b}" for a, b in pairs] + [f"{b}_{a}" for a, b in pairs])


def sample_indices(total: int, k: int):
    """Evenly spaced frame indices across the clip, avoiding the first/last frame."""
    if total <= 0:
        return []
    if total <= k:
        return list(range(total))
    return np.linspace(0, total - 1, k + 2, dtype=int)[1:-1].tolist()


def crop_face(frame, box, scale=BBOX_SCALE, size=256):
    """Shared crop used by FF++ and every external dataset. Do not reimplement."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1) * scale
    x1 = int(max(0, cx - side / 2))
    y1 = int(max(0, cy - side / 2))
    x2 = int(min(w, cx + side / 2))
    y2 = int(min(h, cy + side / 2))
    if x2 <= x1 or y2 <= y1:
        return None
    face = frame[y1:y2, x1:x2]
    return cv2.resize(face, (size, size), interpolation=cv2.INTER_AREA)


def process_video(path, out_dir, n_frames, detector, size=256):
    """Sample, detect, crop. Returns a result dict; never silently drops a clip."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"saved": 0, "sampled": 0, "reason": "unreadable_video"}

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    wanted = set(sample_indices(total, n_frames))
    if not wanted:
        cap.release()
        return {"saved": 0, "sampled": 0, "reason": "empty_video"}

    out_dir.mkdir(parents=True, exist_ok=True)
    saved, idx = 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in wanted:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            try:
                boxes, probs = detector.detect(rgb)
            except Exception:
                boxes, probs = None, None
            if boxes is not None and len(boxes):
                best = boxes[int(np.argmax(probs))]
                face = crop_face(rgb, best, size=size)
                if face is not None:
                    cv2.imwrite(
                        str(out_dir / f"{idx:06d}.png"),
                        cv2.cvtColor(face, cv2.COLOR_RGB2BGR),
                    )
                    saved += 1
        idx += 1
    cap.release()
    if saved == 0:
        return {"saved": 0, "sampled": len(wanted), "reason": "detection_failed"}
    return {"saved": saved, "sampled": len(wanted), "reason": None}


def _record(failures, split, category, clip, path, result):
    if result.get("reason"):
        failures.append({
            "split": split,
            "category": category,
            "clip": clip,
            "path": str(path),
            "reason": result["reason"],
            "sampled": result.get("sampled", 0),
            "saved": result.get("saved", 0),
        })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="FF++ download root")
    ap.add_argument("--out", required=True, help="output crop directory")
    ap.add_argument("--split-json", required=True, help="dir with train/val/test.json")
    ap.add_argument("--compression", default="c23", choices=["c23", "c40", "raw"])
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument(
        "--manipulations",
        nargs="+",
        default=None,
        help="fake sources to extract (default: the four FF++ manipulations)",
    )
    ap.add_argument(
        "--include-faceshifter",
        action="store_true",
        help="also extract FaceShifter at the same compression level",
    )
    ap.add_argument("--skip-real", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    from facenet_pytorch import MTCNN

    detector = MTCNN(keep_all=True, device=args.device, post_process=False)

    root, out = Path(args.root), Path(args.out)
    splits = load_splits(Path(args.split_json))
    manips = list(args.manipulations or MANIPULATIONS)
    if args.include_faceshifter and FACESHIFTER not in manips:
        manips.append(FACESHIFTER)

    failures = []
    counts = {}

    for split, pairs in splits.items():
        if not args.skip_real:
            src = root / "original_sequences/youtube" / args.compression / "videos"
            key = f"{split}/real"
            counts[key] = 0
            for vid in tqdm(real_ids(pairs), desc=key):
                f = src / f"{vid}.mp4"
                if not f.exists():
                    _record(failures, split, "real", vid, f, {
                        "saved": 0, "sampled": 0, "reason": "missing_video",
                    })
                    continue
                result = process_video(
                    f, out / split / "real" / vid, FRAMES_REAL, detector, args.size
                )
                counts[key] += result["saved"]
                _record(failures, split, "real", vid, f, result)

        for manip in manips:
            src = root / "manipulated_sequences" / manip / args.compression / "videos"
            key = f"{split}/{manip}"
            counts[key] = 0
            for vid in tqdm(fake_ids(pairs), desc=key):
                f = src / f"{vid}.mp4"
                if not f.exists():
                    _record(failures, split, manip, vid, f, {
                        "saved": 0, "sampled": 0, "reason": "missing_video",
                    })
                    continue
                result = process_video(
                    f, out / split / manip / vid, FRAMES_FAKE, detector, args.size
                )
                counts[key] += result["saved"]
                _record(failures, split, manip, vid, f, result)

    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "compression": args.compression,
        "size": args.size,
        "frames_real": FRAMES_REAL,
        "frames_fake": FRAMES_FAKE,
        "manipulations": manips,
        "bbox_scale": BBOX_SCALE,
        "counts": counts,
        "n_failures": len(failures),
        "n_detection_failures": sum(
            1 for r in failures if r["reason"] == "detection_failed"
        ),
    }
    (out / "preprocess_manifest.json").write_text(json.dumps(manifest, indent=2))
    (out / "failures.json").write_text(json.dumps(failures, indent=2))
    print(f"done -> {out}")
    print(json.dumps(manifest, indent=2))
    print(f"logged {len(failures)} clip failures to {out / 'failures.json'}")


if __name__ == "__main__":
    main()
