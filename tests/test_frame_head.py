import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  
import torch.nn.functional as F  

from sqa.frame_head import PAD_LOGIT, FaultFrameHead, LayerWeightedSum, topk_presence  # noqa: E402
from sqa.postprocess import presence_from_frames 

H = 16


def _head(**kw):
    torch.manual_seed(0)
    kw.setdefault("hidden_dim", 32)
    return FaultFrameHead(H, **kw).eval()


def _lengths_to_mask(lengths, t=None):
    t = t or max(lengths)
    return torch.arange(t)[None, :] < torch.tensor(lengths)[:, None]



def test_output_shape_and_logits_not_probs():
    head = _head()
    out = head(torch.randn(2, 37, H))
    assert out.shape == (2, 37, 3)
    assert (out < 0).any()


def test_layer_weighted_input():
    head = _head(num_input_layers=4)
    assert head(torch.randn(2, 4, 10, H)).shape == (2, 10, 3)
    with pytest.raises(ValueError):
        head(torch.randn(2, 3, 10, H))
    with pytest.raises(ValueError):
        _head()(torch.randn(2, 4, 10, H))


def test_layer_weights_start_uniform():
    lws = LayerWeightedSum(4)
    assert torch.allclose(lws.weights, torch.full((4,), 0.25))


@pytest.mark.parametrize("kw", [{"kernel_size": 4}, {"prior_prob": 0.0}, {"n_blocks": 2, "dilations": [1]}])
def test_invalid_config_raises(kw):
    with pytest.raises(ValueError):
        FaultFrameHead(H, **kw)


def test_wrong_input_dim_raises():
    with pytest.raises(ValueError):
        _head()(torch.randn(1, 10, H + 1))


def test_valid_mask_contract():
    head = _head()
    h = torch.randn(2, 10, H)
    with pytest.raises(ValueError):
        head(h, torch.ones(2, 9, dtype=torch.bool))
    with pytest.raises(ValueError):
        head(h, torch.full((2, 10), 0.7))
    a = head(h, torch.ones(2, 10, dtype=torch.bool))
    b = head(h, torch.ones(2, 10))
    assert torch.equal(a, b)


@pytest.mark.parametrize("t", [1, 2, 3])
def test_very_short_sequences(t):
    out = _head()(torch.randn(1, t, H))
    assert out.shape == (1, t, 3) and torch.isfinite(out).all()


def test_initial_probs_follow_prior():
    head = _head(prior_prob=0.05)
    p = head.predict_proba(torch.randn(4, 200, H))
    assert abs(p.mean().item() - 0.05) < 0.02
    assert (p > 0.5).float().mean().item() < 0.01



def test_padding_invariance():
    head = _head(n_blocks=3, dilations=[1, 2, 4])
    torch.manual_seed(1)
    short, long = torch.randn(1, 23, H), torch.randn(1, 50, H)
    alone = head(short)

    batch = torch.zeros(2, 50, H)
    batch[0, :23] = short[0]
    batch[0, 23:] = 1e3 * torch.randn(27, H)
    batch[1] = long[0]
    mask = _lengths_to_mask([23, 50])
    out = head(batch, mask)

    assert torch.allclose(out[0, :23], alone[0], atol=1e-5)
    assert torch.allclose(out[1], head(long)[0], atol=1e-5)
    assert torch.all(out[0, 23:] == PAD_LOGIT)


def test_receptive_field_is_declared_and_respected():
    head = _head(n_blocks=2, kernel_size=5, dilations=[1, 2])
    rf = head.receptive_field_frames
    assert rf == 1 + 4 * 1 + 4 * 2
    h = torch.randn(1, 60, H)
    h2 = h.clone()
    t0 = 30
    h2[0, t0, 0] += 5.0  # 1 chiều: dịch đều mọi chiều sẽ bị LayerNorm triệt tiêu
    diff = (head(h2) - head(h)).abs().amax(dim=-1)[0]
    changed = torch.nonzero(diff > 1e-6).flatten()
    assert changed.numel() > 0
    half = rf // 2
    assert changed.min() >= t0 - half and changed.max() <= t0 + half
    assert diff[t0] > 0


def test_no_leakage_between_batch_items():
    head = _head()
    h = torch.randn(2, 20, H)
    base = head(h)
    h2 = h.clone()
    h2[1] += 3.0 * torch.randn_like(h2[1])
    assert torch.allclose(head(h2)[0], base[0], atol=1e-6)



def test_gradients_finite_and_zero_on_padding():
    head = FaultFrameHead(H, hidden_dim=32).train()
    h = torch.randn(3, 40, H, requires_grad=True)
    mask = _lengths_to_mask([40, 7, 1])
    logits = head(h, mask)
    target = torch.zeros_like(logits)
    loss = (F.binary_cross_entropy_with_logits(logits, target, reduction="none") * mask[..., None]).sum()
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in head.parameters() if p.grad is not None)
    assert torch.all(h.grad[1, 7:] == 0) and torch.all(h.grad[2, 1:] == 0)


def test_can_overfit_synthetic_localization():
    g = torch.Generator().manual_seed(0)
    b, t = 8, 120
    h = torch.randn(b, t, H, generator=g)
    target = torch.zeros(b, t, 3)
    for i in range(b):
        for c in range(3):
            if (i + c) % 2 == 0:
                s = int(torch.randint(0, t - 30, (1,), generator=g))
                L = int(torch.randint(3, 30, (1,), generator=g))
                target[i, s:s + L, c] = 1
                h[i, s:s + L, c] += 3.0
    lengths = [t - 10 * (i % 3) for i in range(b)]
    mask = _lengths_to_mask(lengths, t)
    target = target * mask[..., None]

    torch.manual_seed(0)
    head = FaultFrameHead(H, hidden_dim=32, dropout=0.0)
    opt = torch.optim.Adam(head.parameters(), lr=3e-3)
    for _ in range(300):
        opt.zero_grad()
        logits = head(h, mask)
        loss = (F.binary_cross_entropy_with_logits(logits, target, reduction="none") * mask[..., None]).sum() / mask.sum()
        loss.backward()
        opt.step()

    pred = (head.eval().predict_proba(h, mask) > 0.5) & mask[..., None]
    gt = target.bool()
    inter = (pred & gt).sum().item()
    union = (pred | gt).sum().item()
    assert inter / union >= 0.9



def test_topk_presence_matches_numpy_reference():
    rng = np.random.default_rng(0)
    for t in [0, 1, 2, 3, 4, 10, 49, 50, 51, 70, 90, 333]:
        probs = rng.uniform(size=(t, 3)).astype(np.float32)
        tpad = max(t, 1) + 5
        x = torch.full((1, tpad, 3), 0.999)
        x[0, :t] = torch.from_numpy(probs)
        mask = _lengths_to_mask([t], tpad)
        got = topk_presence(x, mask)[0].numpy()
        for c in range(3):
            ref = presence_from_frames(probs[:, c]) if t > 0 else 0.0
            assert got[c] == pytest.approx(ref, abs=1e-6), (t, c)


def test_topk_presence_differentiable_and_no_nan_for_empty_sample():
    x = torch.rand(2, 30, 3, requires_grad=True)
    mask = _lengths_to_mask([30, 0], 30)
    p = topk_presence(x, mask)
    assert torch.all(p[1] == 0)
    p.sum().backward()
    assert torch.isfinite(x.grad).all()
    assert torch.all(x.grad[1] == 0)
    assert (x.grad[0] > 0).sum() == 3 * 3



class _FakeBackbone(nn.Module):
    hidden_size = H

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(1, H)

    def forward(self, wav, attention_mask=None):
        b, n = wav.shape
        t = (n - 400) // 320 + 1
        frames = wav[:, : t * 320].reshape(b, t, 320).mean(-1, keepdim=True)
        return self.proj(frames)

    def frame_valid_mask(self, attention_mask):
        lengths = attention_mask.long().sum(1)
        t = (attention_mask.shape[1] - 400) // 320 + 1
        valid = torch.clamp((lengths - 400) // 320 + 1, min=0)
        return torch.arange(t)[None] < valid[:, None]


def test_sqamodel_uses_frame_head_with_mask():
    from sqa.model import SQAModel

    torch.manual_seed(0)
    model = SQAModel(backbone=_FakeBackbone(), fault_head_kwargs={"hidden_dim": 32}).eval()
    wav = torch.randn(2, 16_000)
    am = torch.zeros(2, 16_000, dtype=torch.long)
    am[0, :] = 1
    am[1, :8_000] = 1
    q, probs, h = model(wav, attention_mask=am)
    _, logits, _ = model(wav, attention_mask=am, return_fault_logits=True)
    assert q.shape == (2,) and probs.shape == (2, h.shape[1], 3)
    assert torch.allclose(probs, torch.sigmoid(logits))
    n_valid = (8_000 - 400) // 320 + 1
    assert torch.all(logits[1, n_valid:] == PAD_LOGIT)
    assert torch.all((probs >= 0) & (probs <= 1))