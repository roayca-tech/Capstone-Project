"""
Training / evaluation for WGN.

Paper setup (Sec. 4.1): Adam, lr 1e-4, cosine annealing, BCE loss, batch 32,
25 epochs, 256x256 input, single RTX A2000 (12 GB).
Ablations (Sec. 5.2) use AdamW for 10 epochs.

Every run is logged to runs/<tag>/ with config.json (full argv, git SHA, seed)
and metrics.json (per-epoch + test). Metrics are frame-level.

Usage:
    python -m wgn.train --data data/ffpp_c23 --tag wgn_c23 --epochs 25 --amp
    python -m wgn.train --data data/ffpp_c23 --tag baseline_c23 --no-wgsa --amp
    python -m wgn.train --data data/ffpp_c23 --train-manip Deepfakes --epochs 25 --amp
"""

import argparse
import json
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .baselines import CBAMAttention
from .model import BaselineNet, WGN, count_parameters

MANIPULATIONS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def build_transforms(train: bool, size: int = 256):
    # Open question 2: augmentation is unspecified. Start with horizontal flip
    # only — blur/sharpen/heavy JPEG destroy the high-frequency evidence WGSA
    # reads. Recorded in runs/DEVIATIONS.md.
    if train:
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def discover_categories(root: Path, split: str):
    d = root / split
    if not d.exists():
        return []
    return sorted(p.name for p in d.iterdir() if p.is_dir() and p.name != "real")


class FFPPFrames(Dataset):
    """Flat frame dataset over the preprocessed crop tree.

    Expected layout:
        root/<split>/real/<clip>/<frame>.png
        root/<split>/<Manipulation>/<clip>/<frame>.png

    Args:
        manips: which fake sources to include. Restrict to one for
            cross-manipulation training.
    """

    def __init__(self, root, split, manips=None, train=False, size=256):
        self.root = Path(root) / split
        self.tf = build_transforms(train, size)
        self.samples = []

        real_dir = self.root / "real"
        if real_dir.exists():
            for p in real_dir.rglob("*.png"):
                self.samples.append((p, 0))
        for m in (manips or MANIPULATIONS):
            d = self.root / m
            if d.exists():
                for p in d.rglob("*.png"):
                    self.samples.append((p, 1))

        if not self.samples:
            raise RuntimeError(f"No frames found under {self.root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = Image.open(path).convert("RGB")
        return self.tf(img), torch.tensor(label, dtype=torch.float32)


def build_model(args):
    if args.no_wgsa:
        return BaselineNet(backbone=args.backbone, pretrained=args.pretrained)
    wgsa_kwargs = {
        "pooling": args.pooling,
        "norm": args.norm,
        "use_ll": args.use_ll,
        "gating": args.gating,
        "bands": args.bands,
    }
    attention = CBAMAttention() if args.attention == "cbam" else None
    return WGN(
        backbone=args.backbone,
        pretrained=args.pretrained,
        fusion=args.fusion,
        attention=attention,
        wgsa_kwargs=wgsa_kwargs,
    )


@torch.no_grad()
def evaluate(model, loader, device):
    """Frame-level ACC (%) and AUC (%). Video-level aggregation is not used."""
    model.eval()
    scores, labels = [], []
    for x, y in loader:
        logit = model(x.to(device, non_blocking=True)).squeeze(1)
        scores.append(torch.sigmoid(logit).float().cpu().numpy())
        labels.append(y.numpy())
    s = np.concatenate(scores)
    y = np.concatenate(labels)
    acc = ((s > 0.5).astype(int) == y).mean()
    auc = roc_auc_score(y, s) if len(np.unique(y)) > 1 else float("nan")
    return acc * 100, auc * 100


def dump_json(path: Path, payload):
    path.write_text(json.dumps(payload, indent=2, default=str))


def evaluate_test_splits(model, args, device, dl_kwargs, manips):
    results = {}
    for m in manips:
        ds = FFPPFrames(args.data, "test", [m], train=False, size=args.size)
        loader = DataLoader(ds, shuffle=False, **dl_kwargs)
        acc, auc = evaluate(model, loader, device)
        results[m] = {"acc": acc, "auc": auc, "n": len(ds), "level": "frame"}
        print(f"test/{m:<16} acc {acc:.2f} | auc {auc:.2f}  (frame-level, n={len(ds)})")

    try:
        ds_all = FFPPFrames(args.data, "test", manips, train=False, size=args.size)
        acc, auc = evaluate(model, DataLoader(ds_all, shuffle=False, **dl_kwargs), device)
        results["all"] = {"acc": acc, "auc": auc, "n": len(ds_all), "level": "frame"}
        print(f"test/{'all':<16} acc {acc:.2f} | auc {auc:.2f}  (frame-level, n={len(ds_all)})")
    except RuntimeError:
        pass
    return results


def default_tag(args) -> str:
    if args.no_wgsa:
        kind = "baseline"
    elif args.attention == "cbam":
        kind = "cbam"
    else:
        kind = "wgn"
    manip = (args.train_manip or "all").lower()
    return f"{kind}_{manip}_s{args.seed}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--backbone", default="mobilevit_s")
    ap.add_argument("--fusion", default="concat", choices=["concat", "add", "multiply"])
    ap.add_argument("--train-manip", default=None,
                    help="restrict training fakes to one manipulation")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--optimizer", default="adam", choices=["adam", "adamw"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--amp", action="store_true", help="mixed precision (fits 12 GB)")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None, help="run folder name under --out")
    ap.add_argument("--out", default="runs", help="parent directory for run logs")
    ap.add_argument("--ckpt", default=None, help="checkpoint to load (eval or resume)")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--no-wgsa", action="store_true", help="C0 / Table 1 backbone baseline")
    ap.add_argument("--attention", default="wgsa", choices=["wgsa", "cbam"])
    ap.add_argument("--pooling", default="avgmax", choices=["avgmax", "mean"])
    ap.add_argument("--norm", default="minmax", choices=["minmax", "sigmoid"])
    ap.add_argument("--use-ll", dest="use_ll", action="store_true", default=True)
    ap.add_argument("--no-use-ll", dest="use_ll", action="store_false")
    ap.add_argument("--gating", dest="gating", action="store_true", default=True)
    ap.add_argument("--no-gating", dest="gating", action="store_false")
    ap.add_argument("--bands", default=None, help="e.g. hh for ablation C7")
    ap.add_argument("--pretrained", dest="pretrained", action="store_true", default=True)
    ap.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    args = ap.parse_args()

    if args.tag is None:
        args.tag = default_tag(args)

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = bool(args.amp and device == "cuda")

    manips = [args.train_manip] if args.train_manip else None
    if manips is None:
        discovered = discover_categories(Path(args.data), "train")
        manips = discovered or MANIPULATIONS

    out_dir = Path(args.out) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.eval_only:
        train_ds = None
        val_ds = None
        print("eval-only: skipping dataset load for train/val")
    else:
        train_ds = FFPPFrames(args.data, "train", manips, train=True, size=args.size)
        val_ds = FFPPFrames(args.data, "val", manips, train=False, size=args.size)
        print(f"train {len(train_ds)} | val {len(val_ds)}")

    dl = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=device == "cuda")
    train_dl = DataLoader(train_ds, shuffle=True, drop_last=True, **dl) if train_ds else None
    val_dl = DataLoader(val_ds, shuffle=False, **dl) if val_ds else None

    model = build_model(args).to(device)
    n_params = count_parameters(model)
    print(f"params: {n_params:.2f} M")

    config = {
        "tag": args.tag,
        "seed": args.seed,
        "git_sha": git_sha(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "amp": use_amp,
        "num_params_m": n_params,
        "metric_level": "frame",
        "train_manips": manips,
        "train_size": len(train_ds) if train_ds else 0,
        "val_size": len(val_ds) if val_ds else 0,
        "args": vars(args),
    }
    dump_json(out_dir / "config.json", config)

    loaded_ckpt = None
    if args.ckpt:
        loaded_ckpt = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(loaded_ckpt["model"])
        print(f"loaded checkpoint {args.ckpt}")

    if args.eval_only:
        eval_manips = discover_categories(Path(args.data), "test") or MANIPULATIONS
        test = evaluate_test_splits(model, args, device, dl, eval_manips)
        dump_json(out_dir / "metrics.json", {
            "frame_level": True,
            "epochs": [],
            "best": {
                "epoch": loaded_ckpt.get("epoch"),
                "acc": loaded_ckpt.get("acc"),
                "auc": loaded_ckpt.get("auc"),
            } if loaded_ckpt else None,
            "test": test,
        })
        return

    opt_cls = torch.optim.Adam if args.optimizer == "adam" else torch.optim.AdamW
    opt = opt_cls(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best = -1.0
    best_auc = float("nan")
    epoch_logs = []
    ckpt_path = out_dir / "best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for x, y in train_dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = crit(model(x).squeeze(1), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += loss.item() * x.size(0)
        sched.step()

        acc, auc = evaluate(model, val_dl, device)
        loss_epoch = running / len(train_ds)
        print(f"epoch {epoch:02d} | loss {loss_epoch:.4f} "
              f"| val acc {acc:.2f} | val auc {auc:.2f}  (frame-level)")
        epoch_logs.append({
            "epoch": epoch,
            "loss": loss_epoch,
            "val_acc": acc,
            "val_auc": auc,
            "lr": sched.get_last_lr()[0],
        })
        dump_json(out_dir / "metrics.json", {
            "frame_level": True,
            "epochs": epoch_logs,
            "best": {"epoch": epoch, "acc": best, "auc": best_auc} if best >= 0 else None,
        })

        # Paper selects the checkpoint with the best validation accuracy (Sec. 5.1).
        if acc > best:
            best = acc
            best_auc = auc
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "acc": acc,
                    "auc": auc,
                    "tag": args.tag,
                    "seed": args.seed,
                },
                ckpt_path,
            )

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    print(f"\nbest val acc {ckpt['acc']:.2f} (epoch {ckpt['epoch']})  (frame-level)")

    eval_manips = discover_categories(Path(args.data), "test") or MANIPULATIONS
    test = evaluate_test_splits(model, args, device, dl, eval_manips)
    dump_json(out_dir / "metrics.json", {
        "frame_level": True,
        "epochs": epoch_logs,
        "best": {
            "epoch": ckpt["epoch"],
            "acc": ckpt["acc"],
            "auc": ckpt.get("auc"),
        },
        "test": test,
    })


if __name__ == "__main__":
    main()
