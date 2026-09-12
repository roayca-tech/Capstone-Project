"""
Training / evaluation for WGN.

Paper setup (Sec. 4.1): Adam, lr 1e-4, cosine annealing, BCE loss, batch 32,
25 epochs, 256x256 input, single RTX A2000 (12 GB).
Ablations (Sec. 5.2) use AdamW for 10 epochs.

Usage:
    # in-domain, all four manipulations
    python -m wgn.train --data /data/ffpp_crops --epochs 25

    # cross-manipulation: train on DF only, evaluate on all four
    python -m wgn.train --data /data/ffpp_crops --train-manip Deepfakes --epochs 25
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .model import WGN, count_parameters

MANIPULATIONS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def build_transforms(train: bool, size: int = 256):
    if train:
        # Keep augmentation mild. Aggressive blur/sharpening destroys exactly the
        # high-frequency evidence WGSA is built to read.
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.1, 0.1, 0.1, 0.02),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
        ])
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


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

        for p in (self.root / "real").rglob("*.png"):
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


@torch.no_grad()
def evaluate(model, loader, device):
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
    ap.add_argument("--out", default="checkpoints")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    manips = [args.train_manip] if args.train_manip else MANIPULATIONS

    train_ds = FFPPFrames(args.data, "train", manips, train=True)
    val_ds = FFPPFrames(args.data, "val", manips, train=False)
    print(f"train {len(train_ds)} | val {len(val_ds)}")

    dl = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)
    train_dl = DataLoader(train_ds, shuffle=True, drop_last=True, **dl)
    val_dl = DataLoader(val_ds, shuffle=False, **dl)

    model = WGN(backbone=args.backbone, pretrained=True, fusion=args.fusion).to(device)
    print(f"params: {count_parameters(model):.2f} M")

    opt_cls = torch.optim.Adam if args.optimizer == "adam" else torch.optim.AdamW
    opt = opt_cls(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.train_manip or "all"
    best = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for x, y in train_dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.amp):
                loss = crit(model(x).squeeze(1), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += loss.item() * x.size(0)
        sched.step()

        acc, auc = evaluate(model, val_dl, device)
        print(f"epoch {epoch:02d} | loss {running/len(train_ds):.4f} "
              f"| val acc {acc:.2f} | val auc {auc:.2f}")

        # Paper selects the checkpoint with the best validation accuracy.
        if acc > best:
            best = acc
            torch.save({"model": model.state_dict(), "epoch": epoch, "acc": acc},
                       out_dir / f"wgn_{args.backbone}_{tag}_best.pt")

    # Cross-manipulation: evaluate the best checkpoint on each test subset.
    ckpt = torch.load(out_dir / f"wgn_{args.backbone}_{tag}_best.pt",
                      map_location=device)
    model.load_state_dict(ckpt["model"])
    print(f"\nbest val acc {ckpt['acc']:.2f} (epoch {ckpt['epoch']})")
    for m in MANIPULATIONS:
        ds = FFPPFrames(args.data, "test", [m], train=False)
        acc, auc = evaluate(model, DataLoader(ds, shuffle=False, **dl), device)
        print(f"test/{m:<16} acc {acc:.2f} | auc {auc:.2f}")


if __name__ == "__main__":
    main()
