"""
Wavelet-Guided Spatial Attention (WGSA).

Implements Eqs. (2)-(10) of Ghosh & Naskar, "WGN: Wavelet-Guided Network for
Efficient and Generalised Deepfake Detection" (CVPRW 2026).

Pipeline:
    F (C,H,W)
      -> channel pool (avg+max)/2            -> F_g (1,H,W)      Eq. 2
      -> fixed Haar DWT (stride-2 conv)      -> LL,LH,HL,HH      Eq. 3-5
      -> spatial means                       -> s in R^4         Eq. 6
      -> tiny MLP + sigmoid                  -> w in (0,1)^4     Eq. 7
      -> reweight subbands                                       Eq. 8
      -> IDWT (transposed conv)              -> A_raw (1,H,W)    Eq. 9
      -> per-sample min-max normalisation    -> A in [0,1]       Eq. 10
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


BAND_INDEX = {"ll": 0, "lh": 1, "hl": 2, "hh": 3}


def haar_kernels() -> torch.Tensor:
    """Return the 4 orthonormal 2x2 Haar analysis kernels, shape (4, 1, 2, 2).

    h = [1, 1] / sqrt(2)   (low-pass)
    g = [1, -1] / sqrt(2)  (high-pass)
    K_XY = outer(x, y)  ->  each kernel has entries +/- 1/2
    """
    h = torch.tensor([1.0, 1.0]) / (2.0 ** 0.5)
    g = torch.tensor([1.0, -1.0]) / (2.0 ** 0.5)

    k_ll = torch.outer(h, h)   # 1/2 * [[ 1,  1], [ 1,  1]]
    k_lh = torch.outer(h, g)   # 1/2 * [[ 1, -1], [ 1, -1]]
    k_hl = torch.outer(g, h)   # 1/2 * [[ 1,  1], [-1, -1]]
    k_hh = torch.outer(g, g)   # 1/2 * [[ 1, -1], [-1,  1]]

    return torch.stack([k_ll, k_lh, k_hl, k_hh], dim=0).unsqueeze(1)


class HaarDWT(nn.Module):
    """Single-level 2D Haar DWT via a fixed stride-2 convolution (Eq. 5).

    Kernels are registered as non-trainable buffers: zero learnable parameters,
    but fully differentiable w.r.t. the input.
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("weight", haar_kernels(), persistent=False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, 1, H, W) -> (B, 4, H/2, W/2), channel order [LL, LH, HL, HH]
        return F.conv2d(z, self.weight, stride=2)


class HaarIDWT(nn.Module):
    """Inverse Haar transform via transposed convolution (Eq. 9)."""

    def __init__(self):
        super().__init__()
        self.register_buffer("weight", haar_kernels(), persistent=False)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        # y: (B, 4, H/2, W/2) -> (B, 1, H, W)
        return F.conv_transpose2d(y, self.weight, stride=2)


def parse_bands(bands: str | None) -> list[str] | None:
    """Parse a comma-separated band list such as 'hh' or 'lh,hl,hh'."""
    if bands is None:
        return None
    names = [n.strip().lower() for n in bands.split(",") if n.strip()]
    unknown = [n for n in names if n not in BAND_INDEX]
    if not names or unknown:
        raise ValueError(
            f"Invalid bands={bands!r}; expected a subset of ll,lh,hl,hh."
        )
    return names


class WGSA(nn.Module):
    """Wavelet-Guided Spatial Attention.

    Args:
        hidden: width of the gating MLP (paper: Linear(4, 32) -> ReLU -> Linear(32, 4)).
        pooling: 'avgmax' (paper default, Eq. 2) or 'mean' (ablation C4).
        norm: 'minmax' (paper default, Eq. 10) or 'sigmoid' (ablation C5).
        use_ll: keep the LL approximation band. False -> ablation C6.
        gating: enable the learnable band gates. False -> ablation C3.
        bands: if set, keep only the named subbands (e.g. 'hh' for ablation C7).
            Channel order is [LL, LH, HL, HH]. Combined with use_ll=False if both
            are given (LL is always dropped when use_ll is False).
        eps: stabiliser in the min-max denominator.

    Returns the attention map A of shape (B, 1, H, W) in [0, 1].
    """

    def __init__(
        self,
        hidden: int = 32,
        pooling: str = "avgmax",
        norm: str = "minmax",
        use_ll: bool = True,
        gating: bool = True,
        bands: str | None = None,
        eps: float = 1e-6,
    ):
        super().__init__()
        assert pooling in {"avgmax", "mean"}
        assert norm in {"minmax", "sigmoid"}

        self.pooling = pooling
        self.norm = norm
        self.use_ll = use_ll
        self.gating = gating
        self.bands = parse_bands(bands)
        self.eps = eps

        self.dwt = HaarDWT()
        self.idwt = HaarIDWT()

        # Eq. 7 — tiny gating MLP; 4 -> hidden -> 4, sigmoid to (0,1)^4
        self.gate = nn.Sequential(
            nn.Linear(4, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 4),
            nn.Sigmoid(),
        )

    def channel_pool(self, feat: torch.Tensor) -> torch.Tensor:
        """Eq. 2 — collapse C channels into a single guide map."""
        if self.pooling == "mean":
            return feat.mean(dim=1, keepdim=True)
        avg = feat.mean(dim=1, keepdim=True)
        mx = feat.amax(dim=1, keepdim=True)
        return 0.5 * (avg + mx)

    def band_keep_mask(self, ref: torch.Tensor) -> torch.Tensor:
        """Return a length-4 0/1 mask over [LL, LH, HL, HH]."""
        if self.bands is None:
            keep = torch.ones(4, device=ref.device, dtype=ref.dtype)
        else:
            keep = torch.zeros(4, device=ref.device, dtype=ref.dtype)
            for name in self.bands:
                keep[BAND_INDEX[name]] = 1.0
        if not self.use_ll:
            keep[0] = 0.0
        return keep

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        b, _, h, w = feat.shape
        if h % 2 or w % 2:
            raise ValueError(
                f"WGSA needs even spatial dims for a single-level DWT, got {h}x{w}."
            )

        guide = self.channel_pool(feat)              # (B, 1, H, W)
        bands = self.dwt(guide)                      # (B, 4, H/2, W/2)

        # Eq. 6 — compact 4-D frequency signature
        s = bands.mean(dim=(2, 3))                   # (B, 4)

        # Eq. 7-8 — content-aware subband gating
        if self.gating:
            w_gate = self.gate(s)                    # (B, 4) in (0,1)
        else:
            w_gate = torch.ones_like(s)              # ablation C3

        if not self.use_ll:                          # ablation C6
            w_gate = w_gate * torch.tensor(
                [0.0, 1.0, 1.0, 1.0], device=w_gate.device, dtype=w_gate.dtype
            )

        if self.bands is not None:                   # ablation C7
            w_gate = w_gate * self.band_keep_mask(w_gate)

        bands = bands * w_gate[:, :, None, None]

        # Eq. 9 — reconstruct the raw attention map
        a_raw = self.idwt(bands)                     # (B, 1, H, W)

        # Eq. 10 — per-sample min-max normalisation
        if self.norm == "sigmoid":                   # ablation C5
            return torch.sigmoid(a_raw)

        flat = a_raw.view(b, -1)
        a_min = flat.min(dim=1).values.view(b, 1, 1, 1)
        a_max = flat.max(dim=1).values.view(b, 1, 1, 1)
        return (a_raw - a_min) / (a_max - a_min + self.eps)
