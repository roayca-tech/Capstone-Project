"""PNG grid of 64 random 256x256 crops per manipulation (and real).

Inspect face-swap sheets by eye: the wrong face is a silent label error.

Usage:
    python scripts/contact_sheet.py --data data/ffpp_c23 --out contact_sheets/
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image

CATEGORIES = ["real", "Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "FaceShifter"]
N = 64
GRID = 8


def collect(root: Path, category: str) -> list[Path]:
    paths = []
    for split in ("train", "val", "test"):
        d = root / split / category
        if d.exists():
            paths.extend(d.rglob("*.png"))
    return paths


def sheet(paths: list[Path], out: Path, n: int = N, cell: int = 128):
    rng = random.Random(0)
    pick = paths if len(paths) <= n else rng.sample(paths, n)
    cols = GRID
    rows = (len(pick) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * cell, rows * cell), (16, 16, 16))
    for i, p in enumerate(pick):
        im = Image.open(p).convert("RGB").resize((cell, cell), Image.Resampling.BILINEAR)
        r, c = divmod(i, cols)
        canvas.paste(im, (c * cell, r * cell))
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    return len(pick)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="contact_sheets")
    ap.add_argument("--split", default=None, help="restrict to one split")
    args = ap.parse_args()

    root = Path(args.data)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    written = {}
    cats = CATEGORIES
    if args.split:
        cats = sorted(
            p.name for p in (root / args.split).iterdir() if p.is_dir()
        ) if (root / args.split).exists() else cats

    for cat in cats:
        if args.split:
            paths = list((root / args.split / cat).rglob("*.png")) if (root / args.split / cat).exists() else []
        else:
            paths = collect(root, cat)
        if not paths:
            print(f"skip {cat}: no frames")
            continue
        dest = out_dir / f"{cat}.png"
        n = sheet(paths, dest)
        written[cat] = {"n": n, "path": str(dest)}
        print(f"{cat}: {n} crops -> {dest}")

    print("Inspect FaceSwap / FaceShifter / Deepfakes sheets: the cropped face "
          "must be the manipulated identity, not a bystander.")
    (out_dir / "index.json").write_text(__import__("json").dumps(written, indent=2))


if __name__ == "__main__":
    main()
