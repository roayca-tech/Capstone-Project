"""Grad-CAM on fused WGN features plus F / A / F⊙A (paper Figure 2).

`model(x, return_attention=True)` already returns logit, A, F, F⊙A.

Six-column layout (one row per source):
    input | F (channel-mean) | A | F⊙A (channel-mean) | Grad-CAM | overlay

Usage:
    python scripts/gradcam.py --ckpt runs/wgn_dfdc_s0/best.pt \
        --data data/dfdc --out figures/attention_dfdc.png --device mps
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from wgn.model import WGN
from wgn.train import MEAN, STD, MANIPULATIONS, build_transforms, pick_device

try:
    from wgn.baselines import CBAMAttention
except ImportError:
    CBAMAttention = None


def denorm(t: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(MEAN, device=t.device).view(3, 1, 1)
    std = torch.tensor(STD, device=t.device).view(3, 1, 1)
    x = (t * std + mean).clamp(0, 1)
    return (x.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


def to_heat(map2d: torch.Tensor, size: int) -> np.ndarray:
    m = map2d.detach().float().cpu()
    m = m - m.min()
    m = m / (m.max() + 1e-6)
    m = F.interpolate(m[None, None], size=(size, size), mode="bilinear", align_corners=False)[0, 0]
    return _colourise(m.numpy())


def _colourise(x: np.ndarray) -> np.ndarray:
    """Simple magenta-yellow heatmap; no matplotlib dependency."""
    x = np.clip(x, 0, 1)
    r = np.clip(1.5 * x, 0, 1)
    g = np.clip(1.5 * x - 0.4, 0, 1)
    b = np.clip(1.0 - x, 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def overlay(rgb: np.ndarray, heat: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    return (rgb.astype(np.float32) * (1 - alpha) + heat.astype(np.float32) * alpha).astype(np.uint8)


def gradcam_fused(model: WGN, x: torch.Tensor) -> torch.Tensor:
    """Grad-CAM over the fused feature map (Eq. 12 output)."""
    activations = {}

    def hook(_m, _i, o):
        activations["feat"] = o
        o.retain_grad()

    handle = model.fuse.register_forward_hook(hook)
    model.zero_grad(set_to_none=True)
    logit = model(x)
    logit.sum().backward()
    handle.remove()
    feat = activations["feat"]
    weights = feat.grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * feat).sum(dim=1, keepdim=True))
    return cam[0, 0]


def categories_for(data: Path, split: str) -> list[str]:
    """DFDC, Celeb-DF, and HiDF are real/fake. FF++ adds manipulation folders."""
    root = data / split
    if not root.is_dir():
        return []
    names = {p.name for p in root.iterdir() if p.is_dir()}
    if names >= {"real", "fake"} and not (names & set(MANIPULATIONS)):
        return ["real", "fake"]
    return [c for c in ["real", *MANIPULATIONS] if c in names]


def pick_images(data: Path, split: str, category: str, n: int) -> list[Path]:
    """One middle frame from each of n clips, so a row is not eight frames of one video."""
    d = data / split / category
    if not d.is_dir():
        return []
    clips = sorted(p for p in d.iterdir() if p.is_dir())
    chosen = []
    if clips:
        step = max(1, len(clips) // n)
        for clip in clips[::step]:
            frames = sorted(clip.glob("*.png"))
            if frames:
                chosen.append(frames[len(frames) // 2])
            if len(chosen) == n:
                break
        return chosen
    frames = sorted(d.rglob("*.png"))
    if not frames:
        return []
    step = max(1, len(frames) // n)
    return frames[::step][:n]


def load_model(args, device):
    attention = None
    wgsa_kwargs = {}
    if args.attention == "cbam":
        attention = CBAMAttention()
    model = WGN(
        backbone=args.backbone,
        pretrained=False,
        fusion=args.fusion,
        attention=attention,
        wgsa_kwargs=wgsa_kwargs,
    ).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="figures/figure2.png")
    ap.add_argument("--backbone", default="mobilevit_s")
    ap.add_argument("--fusion", default="concat")
    ap.add_argument("--attention", default="wgsa", choices=["wgsa", "cbam"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--per-class", type=int, default=2)
    ap.add_argument("--device", default=None, choices=["cpu", "mps", "cuda"])
    args = ap.parse_args()

    device = args.device or pick_device()
    print(f"device {device}")
    model = load_model(args, device)
    tf = build_transforms(train=False, size=args.size)

    rows = categories_for(Path(args.data), args.split)
    cols = []
    missing = []
    for cat in rows:
        paths = pick_images(Path(args.data), args.split, cat, args.per_class)
        if not paths:
            missing.append(cat)
            continue
        for path in paths:
            img = Image.open(path).convert("RGB")
            x = tf(img).unsqueeze(0).to(device)
            x.requires_grad_(True)
            logit, attn, feat, att_feat = model(x, return_attention=True)
            cam = gradcam_fused(model, x)
            rgb = denorm(x[0].detach())
            f_map = to_heat(feat[0].mean(0), args.size)
            a_map = to_heat(attn[0, 0], args.size)
            fa_map = to_heat(att_feat[0].mean(0), args.size)
            g_map = to_heat(cam, args.size)
            ov = overlay(rgb, g_map)
            label = f"{cat}  logit {logit.item():+.2f}"
            cols.append((label, [rgb, f_map, a_map, fa_map, g_map, ov]))
            print(f"{label} attn[min,max]="
                  f"{attn.min().item():.3f},{attn.max().item():.3f} <- {path}")

    if not cols:
        raise SystemExit(f"no images found under {args.data}/{args.split} ({missing})")

    headers = ["input", "F", "A", "F⊙A", "Grad-CAM", "overlay"]
    cell = args.size
    pad = 4
    label_h = 18
    n_rows, n_cols = len(cols), 6
    canvas = Image.new(
        "RGB",
        (n_cols * (cell + pad) + pad, label_h + n_rows * (cell + pad + label_h) + pad),
        (255, 255, 255),
    )
    draw = ImageDraw.Draw(canvas)
    for j, name in enumerate(headers):
        draw.text((pad + j * (cell + pad), 2), name, fill=(0, 0, 0))
    y = label_h
    for cat, images in cols:
        x = pad
        for im in images:
            canvas.paste(Image.fromarray(im), (x, y))
            x += cell + pad
        draw.text((pad, y + cell + 2), cat, fill=(0, 0, 0))
        y += cell + pad + label_h

    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(dest)
    print(f"wrote {dest}  columns={headers}  missing={missing}")
    print("Attention maps should be structured, not uniform or saturated.")


if __name__ == "__main__":
    main()
