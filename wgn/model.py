"""
WGN: backbone -> WGSA -> attention modulation -> concat fusion -> GAP -> logit.

Implements Eqs. (1), (11)-(14) of the paper. Backbone-agnostic: anything that
maps (B,3,256,256) -> (B,C,H,W) with even H,W works.
"""

import torch
import torch.nn as nn

from .wgsa import WGSA


def build_backbone(name: str = "mobilevit_s", pretrained: bool = True):
    """Return (feature_extractor, channel_count) using timm's features_only API.

    mobilevit_s @ 256x256 -> (B, 640, 8, 8), ~4.94M params without the classifier,
    which matches the 'Backbone only' row of Table 6.
    """
    import timm

    model = timm.create_model(
        name, pretrained=pretrained, features_only=True, out_indices=(-1,)
    )
    channels = model.feature_info.channels()[-1]
    return model, channels


class WGN(nn.Module):
    """Wavelet-Guided Network.

    Args:
        backbone: timm model name, or an nn.Module returning (B,C,H,W).
        num_channels: required only when passing a custom nn.Module backbone.
        fusion: 'concat' (paper, Eq. 12), 'add' (C8), or 'multiply' (C9).
        num_classes: 1 for a BCE-with-logits real/fake head.
        attention: optional module with WGSA's (B,C,H,W)->(B,1,H,W) interface.
            Used by ablation C2 (CBAMAttention). Defaults to WGSA.
        wgsa_kwargs: forwarded to WGSA when `attention` is None.
    """

    def __init__(
        self,
        backbone="mobilevit_s",
        pretrained: bool = True,
        num_channels: int = None,
        fusion: str = "concat",
        num_classes: int = 1,
        attention: nn.Module = None,
        wgsa_kwargs: dict = None,
    ):
        super().__init__()
        assert fusion in {"concat", "add", "multiply"}

        if isinstance(backbone, str):
            self.backbone, c = build_backbone(backbone, pretrained)
            self._timm = True
        else:
            if num_channels is None:
                raise ValueError("num_channels is required for a custom backbone.")
            self.backbone, c = backbone, num_channels
            self._timm = False

        self.channels = c
        self.fusion_mode = fusion
        self.wgsa = attention if attention is not None else WGSA(**(wgsa_kwargs or {}))

        # Eq. 12 — 1x1 conv over [F || F_att], then BN + ReLU
        in_ch = 2 * c if fusion == "concat" else c
        self.fuse = nn.Sequential(
            nn.Conv2d(in_ch, c, kernel_size=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )

        # Eq. 14 — linear classifier on the GAP'd representation
        self.classifier = nn.Linear(c, num_classes)

    def extract(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone(x)
        if self._timm or isinstance(out, (list, tuple)):
            out = out[-1]
        return out

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        feat = self.extract(x)                       # Eq. 1  (B, C, H, W)
        attn = self.wgsa(feat)                       # (B, 1, H, W)
        att_feat = feat * attn                       # Eq. 11

        if self.fusion_mode == "concat":
            fused = self.fuse(torch.cat([feat, att_feat], dim=1))   # Eq. 12
        elif self.fusion_mode == "add":
            fused = self.fuse(feat + att_feat)
        else:
            fused = self.fuse(att_feat)

        z = fused.mean(dim=(2, 3))                   # Eq. 13
        logit = self.classifier(z)                   # Eq. 14

        if return_attention:
            return logit, attn, feat, att_feat
        return logit


class BaselineNet(nn.Module):
    """Backbone-only detector: GAP -> Linear. No WGSA, no fusion (C0 / Table 1).

    The head is deliberately fusion-free so the WGN vs backbone comparison is
    not confounded by the extra 1x1 conv.
    """

    def __init__(
        self,
        backbone="mobilevit_s",
        pretrained: bool = True,
        num_channels: int = None,
        num_classes: int = 1,
    ):
        super().__init__()
        if isinstance(backbone, str):
            self.backbone, c = build_backbone(backbone, pretrained)
            self._timm = True
        else:
            if num_channels is None:
                raise ValueError("num_channels is required for a custom backbone.")
            self.backbone, c = backbone, num_channels
            self._timm = False

        self.channels = c
        self.classifier = nn.Linear(c, num_classes)

    def extract(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone(x)
        if self._timm or isinstance(out, (list, tuple)):
            out = out[-1]
        return out

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        feat = self.extract(x)
        logit = self.classifier(feat.mean(dim=(2, 3)))
        if return_attention:
            ones = torch.ones(
                feat.size(0), 1, feat.size(2), feat.size(3),
                device=feat.device, dtype=feat.dtype,
            )
            return logit, ones, feat, feat
        return logit


def count_parameters(model: nn.Module) -> float:
    """Trainable parameter count, in millions."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
