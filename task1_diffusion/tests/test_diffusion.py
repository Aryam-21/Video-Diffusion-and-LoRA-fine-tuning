"""Sanity tests ("check gradients and shape preservation").  Run from the repo root:

    python -m pytest task1_diffusion/tests -q          or          python -m task1_diffusion.tests.test_diffusion
"""
import torch

from common.schedules import NoiseSchedule
from task1_diffusion.diffusion import GaussianDiffusion
from task1_diffusion.metrics import psnr, ssim
from task1_diffusion.models import build_unet
from task1_diffusion.preprocessing import preprocess_batch


def _shapes():
    return {"image": (4, 1, 16, 16), "video": (2, 1, 4, 16, 16)}


def test_schedule_is_valid():
    for name in ("linear", "cosine"):
        s = NoiseSchedule(name, 200)
        assert (s.betas > 0).all() and (s.betas < 1).all()
        assert (s.alpha_bar[1:] < s.alpha_bar[:-1]).all(), "alpha_bar must decrease monotonically"
        assert s.alpha_bar[0] > 0.99 and s.alpha_bar[-1] < 0.01, "start ~clean, end ~pure noise"


def test_forward_process_statistics():
    s = NoiseSchedule("cosine", 200)
    d = GaussianDiffusion(s)
    x0 = torch.randn(512, 1, 8, 8).clamp(-1, 1)
    xt = d.q_sample(x0, torch.full((512,), 199))
    assert abs(xt.std().item() - 1.0) < 0.1, "at t=T-1 the sample should be ~N(0,1)"
    t = torch.full((512,), 50)
    noise = torch.randn_like(x0)
    assert torch.allclose(d.predict_x0(d.q_sample(x0, t, noise), t, noise), x0, atol=1e-4), "x0 recovery from true eps"


def test_shape_preservation_and_gradients():
    for task, shape in _shapes().items():
        net = build_unet(task, num_classes=5, base=16, mults=(1, 2))
        for m in net.modules():                         # zero-init layers would give zero grads at init: perturb them
            if isinstance(m, (torch.nn.Conv2d, torch.nn.Conv3d)) and m.weight.abs().sum() == 0:
                torch.nn.init.normal_(m.weight, std=0.02)
        x = torch.randn(shape)
        t = torch.randint(0, 200, (shape[0],))
        y = torch.randint(0, 5, (shape[0],))
        out = net(x, t, y)
        assert out.shape == x.shape, f"{task}: output must have the input's shape"
        out.square().mean().backward()
        for n, p in net.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(), f"{task}: bad grad in {n}"
        assert sum(p.grad.abs().sum() for p in net.parameters()) > 0


def test_can_overfit_one_batch():
    torch.manual_seed(0)
    net = build_unet("image", 0, base=16, mults=(1, 2))
    d = GaussianDiffusion(NoiseSchedule("cosine", 100), 0, 0.0)
    x = torch.rand(8, 1, 16, 16) * 2 - 1
    opt = torch.optim.Adam(net.parameters(), 2e-3)
    first = None
    for i in range(60):
        loss, plain = d.training_losses(net, x)
        opt.zero_grad(); loss.backward(); opt.step()
        first = plain.item() if first is None else first
    assert plain.item() < first, "loss should go down when training on a single batch"


def test_samplers_run_and_are_finite():
    net = build_unet("video", 3, base=16, mults=(1, 2)).eval()
    d = GaussianDiffusion(NoiseSchedule("linear", 20), 3)
    y = torch.tensor([0, 1])
    for x in (d.ddim_sample(net, (2, 1, 4, 16, 16), steps=5, y=y, guidance=2.0),
              d.ddpm_sample(net, (2, 1, 4, 16, 16), y=y)):
        assert x.shape == (2, 1, 4, 16, 16) and torch.isfinite(x).all()


def test_metrics_and_preprocessing():
    x = torch.rand(2, 1, 16, 16) * 2 - 1
    assert psnr(x, x).min() > 90 and torch.allclose(ssim(x, x), torch.ones(2), atol=1e-4)
    assert ssim(x, torch.randn_like(x)).mean() < 0.5
    rgb = torch.rand(3, 40, 60)
    y = preprocess_batch(rgb, 32)
    assert y.shape == (1, 32, 32) and y.min() >= -1.001 and y.max() <= 1.001


def test_min_snr_weights_are_bounded():
    """weight = min(SNR, gamma)/SNR  must be in (0, 1], equal to 1 wherever SNR <= gamma, and smaller for the cleanest steps."""
    s = NoiseSchedule("cosine", 200)
    w = s.snr.clamp(max=5.0) / s.snr
    assert (w > 0).all() and (w <= 1 + 1e-6).all()
    assert torch.allclose(w[s.snr <= 5.0], torch.ones_like(w[s.snr <= 5.0]))
    assert w[0] < 0.05, "almost-clean timesteps must be strongly down-weighted"


def test_classifier_free_guidance_formula():
    net = build_unet("image", num_classes=3, base=16, mults=(1, 2)).eval()
    for m in net.modules():
        if isinstance(m, torch.nn.Conv2d) and m.weight.abs().sum() == 0:
            torch.nn.init.normal_(m.weight, std=0.05)
    d = GaussianDiffusion(NoiseSchedule("linear", 50), 3)
    x, t, y = torch.randn(2, 1, 16, 16), torch.tensor([10, 20]), torch.tensor([0, 2])
    with torch.no_grad():
        e_c, e_u = net(x, t, y), net(x, t, torch.full_like(y, 3))
        assert torch.allclose(d.model_eps(net, x, t, y, guidance=1.0), e_c, atol=1e-5)
        assert torch.allclose(d.model_eps(net, x, t, y, guidance=3.0), e_u + 3.0 * (e_c - e_u), atol=1e-4)


def test_ddim_can_start_from_a_partially_noised_sample():
    """This path is what the denoising test in evaluate.py relies on."""
    net = build_unet("image", 0, base=16, mults=(1, 2)).eval()
    d = GaussianDiffusion(NoiseSchedule("cosine", 100), 0)
    x0 = torch.rand(3, 1, 16, 16) * 2 - 1
    t = torch.full((3,), 30)
    out = d.ddim_sample(net, x0.shape, steps=10, x_T=d.q_sample(x0, t), t_start=30)
    assert out.shape == x0.shape and torch.isfinite(out).all()


def test_synthetic_videos_stay_inside_the_frame():
    """Regression test: 24-frame clips used to run out of the 32x32 frame (the label then no longer matched the content)."""
    from task1_diffusion.synthetic import generate_moving_shapes
    for frames in (8, 24):
        v, _ = generate_moving_shapes(128, frames, 32, seed=0)
        assert ((v > 0.5).reshape(len(v), frames, -1).sum(-1) >= 20).all(), f"shape left the frame with {frames} frames"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
