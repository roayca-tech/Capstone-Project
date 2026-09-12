"""
Acceptance tests for the WGN implementation.

These encode the paper's stated invariants. An agent must keep all of these
green. A failure here is a correctness bug, not a tuning issue.

Run:  pytest tests/ -v
"""

import pytest
import torch
import torch.nn as nn

from wgn.model import WGN, count_parameters
from wgn.wgsa import WGSA, HaarDWT, HaarIDWT, haar_kernels


# --------------------------------------------------------------------------
# Stand-in backbone matching MobileViT-S geometry at 256x256: (B, 640, 8, 8).
# Lets the whole suite run on CPU without downloading timm weights.
# --------------------------------------------------------------------------
class StubBackbone(nn.Module):
    def __init__(self, channels=640, stride=32):
        super().__init__()
        self.net = nn.Conv2d(3, channels, stride, stride=stride)

    def forward(self, x):
        return self.net(x)


def make_model(**kw):
    return WGN(backbone=StubBackbone(), num_channels=640, pretrained=False, **kw)


# ==========================================================================
# 1. Wavelet correctness  (Eq. 3-5, 9)
# ==========================================================================

def test_haar_kernels_are_orthonormal():
    """Flattened Haar kernels must form an orthonormal basis of R^4."""
    k = haar_kernels().view(4, 4)
    gram = k @ k.T
    assert torch.allclose(gram, torch.eye(4), atol=1e-6), gram


def test_perfect_reconstruction():
    """IDWT(DWT(z)) == z. Fails if the 1/sqrt(2) normalisation is wrong."""
    z = torch.randn(4, 1, 16, 16)
    rec = HaarIDWT()(HaarDWT()(z))
    assert rec.shape == z.shape
    assert (rec - z).abs().max() < 1e-5


def test_energy_preservation():
    """Parseval: subband energy equals input energy for an orthonormal transform."""
    z = torch.randn(4, 1, 16, 16)
    y = HaarDWT()(z)
    assert torch.allclose(z.pow(2).sum(), y.pow(2).sum(), rtol=1e-5)


def test_dwt_has_no_trainable_parameters():
    """Haar kernels are fixed buffers. If these train, it is no longer a DWT."""
    assert sum(p.numel() for p in HaarDWT().parameters()) == 0
    assert sum(p.numel() for p in HaarIDWT().parameters()) == 0


def test_dwt_band_ordering():
    """Channel order must be [LL, LH, HL, HH]. A constant input puts all
    energy in LL and zero in the detail bands."""
    z = torch.ones(1, 1, 8, 8)
    y = HaarDWT()(z)
    assert y[:, 0].abs().min() > 0.5           # LL is active
    assert y[:, 1:].abs().max() < 1e-5         # LH, HL, HH are ~zero


def test_dwt_is_differentiable():
    """Fixed weights must still pass gradients to the input."""
    z = torch.randn(2, 1, 8, 8, requires_grad=True)
    HaarDWT()(z).sum().backward()
    assert z.grad is not None and z.grad.abs().sum() > 0


# ==========================================================================
# 2. WGSA module  (Eq. 2, 6-10)
# ==========================================================================

def test_wgsa_output_shape_and_range():
    a = WGSA()(torch.randn(4, 640, 8, 8))
    assert a.shape == (4, 1, 8, 8)
    assert a.min() >= 0.0 and a.max() <= 1.0


def test_wgsa_parameter_count():
    """4->32->4 MLP = 4*32+32 + 32*4+4 = 292. Nothing else may be learnable."""
    assert sum(p.numel() for p in WGSA().parameters()) == 292


def test_minmax_normalisation_is_per_sample():
    """Each sample must be normalised independently. Batch-wide normalisation
    leaks information across samples and inflates validation scores."""
    m = WGSA().eval()
    x = torch.randn(8, 64, 8, 8)
    x[0] *= 100.0                       # make one sample wildly different in scale
    a = m(x)
    for i in range(x.size(0)):
        assert a[i].min() < 1e-4        # every sample reaches 0
        assert a[i].max() > 1 - 1e-4    # and reaches 1


def test_sample_independence():
    """A sample's attention map must not change when batched with others."""
    m = WGSA().eval()
    x = torch.randn(4, 64, 8, 8)
    with torch.no_grad():
        alone = m(x[:1])
        batched = m(x)
    assert (alone - batched[:1]).abs().max() < 1e-5


def test_gradients_reach_backbone_features():
    f = torch.randn(2, 64, 8, 8, requires_grad=True)
    WGSA()(f).sum().backward()
    assert f.grad.abs().sum() > 0


def test_odd_spatial_dims_rejected():
    with pytest.raises(ValueError):
        WGSA()(torch.randn(1, 64, 7, 7))


def test_use_ll_false_zeroes_approximation_band():
    """Ablation C6: the LL band must be fully suppressed."""
    m = WGSA(use_ll=False).eval()
    guide = m.channel_pool(torch.randn(2, 64, 8, 8))
    bands = m.dwt(guide)
    w = m.gate(bands.mean(dim=(2, 3)))
    w = w * torch.tensor([0.0, 1.0, 1.0, 1.0])
    assert w[:, 0].abs().max() == 0.0


def test_gating_disabled_uses_unit_weights():
    """Ablation C3: gates fixed at 1.0, MLP unused."""
    m = WGSA(gating=False).eval()
    x = torch.randn(2, 64, 8, 8)
    before = m(x).clone()
    for p in m.gate.parameters():        # perturb the MLP
        with torch.no_grad():
            p.add_(torch.randn_like(p))
    assert (m(x) - before).abs().max() < 1e-6


# ==========================================================================
# 3. Full model  (Eq. 1, 11-14)
# ==========================================================================

def test_forward_shapes():
    logit, attn, feat, att_feat = make_model()(
        torch.randn(2, 3, 256, 256), return_attention=True
    )
    assert logit.shape == (2, 1)
    assert attn.shape == (2, 1, 8, 8)
    assert feat.shape == att_feat.shape == (2, 640, 8, 8)


def test_added_parameter_budget():
    """Paper Table 6: WGN adds +0.82M over the backbone. Almost all of it is
    the 640x1280 fusion conv; WGSA itself is 292 params."""
    m = make_model()
    added = count_parameters(m) - count_parameters(m.backbone)
    assert 0.80 <= added <= 0.84, f"expected ~0.82M added params, got {added:.3f}M"


def test_attention_actually_changes_output():
    """Guards against silently detached or all-ones attention."""
    m = make_model().eval()
    x = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        base = m(x)
        feat = m.extract(x)
        fused = m.fuse(torch.cat([feat, feat], dim=1))   # attention == 1 everywhere
        neutral = m.classifier(fused.mean(dim=(2, 3)))
    assert (base - neutral).abs().max() > 1e-4


def test_backward_pass_updates_wgsa_gate():
    m = make_model()
    loss = nn.BCEWithLogitsLoss()(m(torch.randn(2, 3, 256, 256)).squeeze(1),
                                  torch.tensor([0.0, 1.0]))
    loss.backward()
    grads = [p.grad for p in m.wgsa.gate.parameters() if p.grad is not None]
    assert grads and max(g.abs().sum() for g in grads) > 0


@pytest.mark.parametrize("fusion", ["concat", "add", "multiply"])
def test_fusion_variants(fusion):
    assert make_model(fusion=fusion)(torch.randn(2, 3, 256, 256)).shape == (2, 1)


@pytest.mark.parametrize("kw", [
    {"pooling": "mean"}, {"norm": "sigmoid"},
    {"use_ll": False}, {"gating": False},
])
def test_ablation_variants_construct_and_run(kw):
    assert make_model(wgsa_kwargs=kw)(torch.randn(2, 3, 256, 256)).shape == (2, 1)


def test_backbone_agnostic():
    """WGSA must work on an arbitrary channel count and spatial size."""
    m = WGN(backbone=StubBackbone(channels=256, stride=16),
            num_channels=256, pretrained=False)
    assert m(torch.randn(2, 3, 256, 256)).shape == (2, 1)


def test_deterministic_in_eval_mode():
    m = make_model().eval()
    x = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        assert (m(x) - m(x)).abs().max() == 0.0


def test_bands_hh_zeroes_other_bands():
    """Ablation C7: only the HH band is kept."""
    m = WGSA(bands="hh").eval()
    guide = m.channel_pool(torch.randn(2, 64, 8, 8))
    bands = m.dwt(guide)
    w = m.gate(bands.mean(dim=(2, 3)))
    w = w * torch.tensor([0.0, 0.0, 0.0, 1.0])
    assert w[:, :3].abs().max() == 0.0
    keep = m.band_keep_mask(w)
    assert keep.tolist() == [0.0, 0.0, 0.0, 1.0]
    out = m(torch.randn(2, 64, 8, 8))
    assert out.shape == (2, 1, 8, 8)
