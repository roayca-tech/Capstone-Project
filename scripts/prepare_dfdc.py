"""Turn a DFDC Kaggle download into the crop tree that wgn.train expects.

Output layout matches the FF++ pipeline, so training needs no new flags:

    out/<split>/real/<clip>/<frame>.png
    out/<split>/fake/<clip>/<frame>.png

Cropping reuses `wgn.preprocess.process_video`, which calls the shared
`crop_face()`. Identity-safe splits come from `wgn.dfdc`.

Usage:
    # inspect the plan without touching any video
    python scripts/prepare_dfdc.py --root ~/data/dfdc --out data/dfdc --dry-run

    # extract crops
    python scripts/prepare_dfdc.py --root ~/data/dfdc --out data/dfdc
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm

from wgn.dfdc import FRAMES_FAKE, LABEL_REAL, build_plan, verify_plan
from wgn.preprocess import process_video


def summarise(plan: dict) -> str:
    lines = [
        f"root        {plan['root']}",
        f"parts       {len(plan['parts'])}: {', '.join(plan['parts'])}",
        f"clips       {plan['n_clips']}",
        f"identities  {plan['n_identities']}",
        f"seed        {plan['seed']}  ratios {plan['ratios']}",
        "",
        f"{'split':<6} {'real':>6} {'fake':>6} {'f/real':>7} {'f/fake':>7} "
        f"{'real frames':>12} {'fake frames':>12}",
    ]
    for split, b in plan["budgets"].items():
        lines.append(
            f"{split:<6} {b['n_real_clips']:>6} {b['n_fake_clips']:>6} "
            f"{b['frames_per_real']:>7} {b['frames_per_fake']:>7} "
            f"{b['expected_real_frames']:>12} {b['expected_fake_frames']:>12}"
        )
    return "\n".join(lines)


def plan_to_json(plan: dict) -> dict:
    """Serialisable view of the plan, including the full clip assignment."""
    return {
        "root": plan["root"],
        "seed": plan["seed"],
        "ratios": plan["ratios"],
        "n_clips": plan["n_clips"],
        "n_identities": plan["n_identities"],
        "parts": plan["parts"],
        "budgets": plan["budgets"],
        "assignment": {
            split: [
                {
                    "clip": c.name,
                    "label": c.label,
                    "identity": c.identity,
                    "part": c.part,
                }
                for c in clips
            ]
            for split, clips in plan["splits"].items()
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dir holding dfdc_train_part_* folders")
    ap.add_argument("--out", required=True, help="output crop tree")
    ap.add_argument("--frames-fake", type=int, default=FRAMES_FAKE,
                    help="frames per fake clip; real clips scale up to balance")
    ap.add_argument("--ratios", type=float, nargs=3, default=(0.7, 0.15, 0.15),
                    metavar=("TRAIN", "VAL", "TEST"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and exit without decoding video")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    root, out = Path(args.root).expanduser(), Path(args.out)
    if not root.exists():
        print(f"DFDC root not found: {root}", file=sys.stderr)
        sys.exit(1)

    plan = build_plan(root, tuple(args.ratios), args.seed, args.frames_fake)
    print(summarise(plan))

    errors = verify_plan(plan)
    if errors:
        print("\nplan problems:", file=sys.stderr)
        for e in errors:
            print(" ", e, file=sys.stderr)
        if any(e.startswith("IDENTITY LEAK") for e in errors):
            sys.exit(1)
        print("\n(class-balance warnings only; continuing)", file=sys.stderr)

    out.mkdir(parents=True, exist_ok=True)
    (out / "dfdc_plan.json").write_text(json.dumps(plan_to_json(plan), indent=2))
    print(f"\nwrote {out / 'dfdc_plan.json'}")

    if args.dry_run:
        print("dry run: no video decoded")
        return

    from facenet_pytorch import MTCNN

    detector = MTCNN(keep_all=True, device=args.device, post_process=False)

    failures = []
    counts = {}
    for split, clips in plan["splits"].items():
        budget = plan["budgets"][split]
        counts[split] = {"real": 0, "fake": 0}
        for clip in tqdm(clips, desc=split):
            n_frames = (
                budget["frames_per_real"] if clip.label == LABEL_REAL
                else budget["frames_per_fake"]
            )
            dest = out / split / clip.label / clip.name
            result = process_video(clip.path, dest, n_frames, detector, args.size)
            counts[split][clip.label] += result["saved"]
            if result.get("reason"):
                failures.append({
                    "split": split,
                    "clip": clip.name,
                    "label": clip.label,
                    "path": str(clip.path),
                    **result,
                })

    manifest = {
        "dataset": "dfdc",
        "root": str(root),
        "size": args.size,
        "seed": args.seed,
        "ratios": list(args.ratios),
        "crop_fn": "wgn.preprocess.crop_face",
        "budgets": plan["budgets"],
        "frames_saved": counts,
        "n_failures": len(failures),
        "n_detection_failures": sum(
            1 for f in failures if f["reason"] == "detection_failed"
        ),
    }
    (out / "preprocess_manifest.json").write_text(json.dumps(manifest, indent=2))
    (out / "failures.json").write_text(json.dumps(failures, indent=2))
    print(json.dumps(manifest, indent=2))
    print(f"logged {len(failures)} clip failures to {out / 'failures.json'}")


if __name__ == "__main__":
    main()
