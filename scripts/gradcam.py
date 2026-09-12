"""Grad-CAM on fused WGN features plus F / A / F⊙A (paper Figure 2).

`model(x, return_attention=True)` already returns logit, A, F, F⊙A.

Six-column layout (one row per source):
    input | F (channel-mean) | A | F⊙A (channel-mean) | Grad-CAM | overlay

Usage:
    python scripts/gradcam.py --ckpt runs/wgn_c23_s0/best.pt \
        --data data/ffpp_c23 --out figures/figure2.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from wgn.model import WGN
from wgn.train import MEAN, STD, MANIPULATIONS, build_transforms

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


def pick_image(data: Path, split: str, category: str) -> Path | None:
    d = data / split / category
    if not d.exists():
        return None
    pngs = sorted(d.rglob("*.png"))
    return pngs[0] if pngs else None


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
    ckpt = torch.load(args.ckpt, map_location=device)
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
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args, device)
    tf = build_transforms(train=False, size=args.size)

    rows = ["real", *MANIPULATIONS]
    cols = []
    missing = []
    for cat in rows:
        path = pick_image(Path(args.data), args.split, cat)
        if path is None:
            missing.append(cat)
            continue
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
        cols.append((cat, [rgb, f_map, a_map, fa_map, g_map, ov]))
        print(f"{cat}: logit={logit.item():.3f} attn[min,max]="
              f"{attn.min().item():.3f},{attn.max().item():.3f} <- {path}")

    if not cols:
        raise SystemExit(f"no images found under {args.data}/{args.split} ({missing})")

    headers = ["input", "F", "A", "F⊙A", "Grad-CAM", "overlay"]
    cell = args.size
    pad = 4
    label_h = 24
    n_rows, n_cols = len(cols), 6
    canvas = Image.new(
        "RGB",
        (n_cols * (cell + pad) + pad, label_h + n_rows * (cell + pad + label_h) + pad),
        (255, 255, 255),
    )
    # header
    for j, name in enumerate(headers):
        # baked as a thin coloured bar + we print names to stdout; PIL default font is optional
        pass
    y = label_h
    for cat, images in cols:
        x = pad
        for im in images:
            canvas.paste(Image.fromarray(im), (x, y))
            x += cell + pad
        y += cell + pad + label_h

    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(dest)
    print(f"wrote {dest}  columns={headers}  missing={missing}")
    print("Attention maps should be structured, not uniform or saturated.")


if __name__ == "__main__":
    main()
