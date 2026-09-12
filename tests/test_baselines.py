"""C0 / C2 contracts: BaselineNet is fusion-free; CBAM drops into WGN."""

import torch
import torch.nn as nn

from wgn.baselines import CBAMAttention
from wgn.model import BaselineNet, WGN, count_parameters


class StubBackbone(nn.Module):
    def __init__(self, channels=640, stride=32):
        super().__init__()
        self.net = nn.Conv2d(3, channels, stride, stride=stride)

    def forward(self, x):
        return self.net(x)


def test_cbam_output_shape_and_range():
    a = CBAMAttention()(torch.randn(4, 640, 8, 8))
    assert a.shape == (4, 1, 8, 8)
    assert a.min() >= 0.0 and a.max() <= 1.0


def test_cbam_drops_into_wgn():
    m = WGN(
        backbone=StubBackbone(),
        num_channels=640,
        pretrained=False,
        attention=CBAMAttention(),
    )
    x = torch.randn(2, 3, 256, 256)
    assert m(x).shape == (2, 1)
    added = count_parameters(m) - count_parameters(m.backbone)
    assert 0.80 <= added <= 0.84, f"expected ~0.82M added, got {added:.3f}M"


def test_baseline_is_fusion_free():
    m = BaselineNet(backbone=StubBackbone(), num_channels=640, pretrained=False)
    assert not hasattr(m, "fuse")
    assert not hasattr(m, "wgsa")
    x = torch.randn(2, 3, 256, 256)
    assert m(x).shape == (2, 1)
    added = count_parameters(m) - count_parameters(m.backbone)
    assert added < 0.01  # Linear(640, 1) only


def test_baseline_return_attention_is_neutral():
    m = BaselineNet(backbone=StubBackbone(), num_channels=640, pretrained=False).eval()
    logit, attn, feat, att_feat = m(torch.randn(2, 3, 256, 256), return_attention=True)
    assert logit.shape == (2, 1)
    assert torch.allclose(attn, torch.ones_like(attn))
    assert torch.allclose(feat, att_feat)
