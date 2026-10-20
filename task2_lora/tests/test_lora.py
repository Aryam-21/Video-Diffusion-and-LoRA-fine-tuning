"""Unit tests for our hand-written LoRA. Pure PyTorch: no GPU, no diffusers, no model download (runs in seconds).

    python -m pytest task2_lora/tests -q
"""
import copy

import pytest
import torch
import torch.nn as nn

from task2_lora.lora import (LoRALinear, inject_lora, load_lora_state_dict, lora_modules, lora_parameters,
                             lora_state_dict, merge_lora, set_lora_scale)


class Attn(nn.Module):
    def __init__(self, d=8):
        super().__init__()
        self.to_q, self.to_k, self.to_v = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
        self.to_out = nn.ModuleList([nn.Linear(d, d), nn.Dropout(0.0)])      # same layout as diffusers attention blocks

    def forward(self, x):
        return self.to_out[0](self.to_q(x) + self.to_k(x) + self.to_v(x))


class Net(nn.Module):
    def __init__(self, d=8):
        super().__init__()
        self.proj_in, self.attn1, self.attn2 = nn.Linear(d, d), Attn(d), Attn(d)
        self.not_to_q = nn.Linear(d, d)                  # ends with "to_q" but is NOT an attention projection

    def forward(self, x):
        return self.not_to_q(self.attn2(self.attn1(self.proj_in(x))))


def _make(r=4, seed=0):
    torch.manual_seed(seed)
    base = Net()
    net = copy.deepcopy(base)
    names = inject_lora(net, r=r, alpha=r)
    return base, net, names


def _randomise_B(net):
    for m in lora_modules(net):
        nn.init.normal_(m.B, std=0.1)


def test_starts_exactly_at_base_model():                  # B = 0  ->  step 0 == pretrained model
    base, net, _ = _make()
    x = torch.randn(5, 8)
    assert torch.allclose(base(x), net(x), atol=1e-6)


def test_only_lora_matrices_are_trainable():
    _, net, names = _make()
    trainable = {n for n, p in net.named_parameters() if p.requires_grad}
    assert trainable and all(n.endswith((".A", ".B")) for n in trainable)
    assert len(names) == 8 and len(lora_parameters(net)) == 16          # 2 blocks x 4 projections, each with A and B
    assert not any("proj_in" in n or "not_to_q" in n for n in trainable)


def test_name_matching_respects_boundaries():
    _, net, names = _make()
    assert not isinstance(net.not_to_q, LoRALinear) and not isinstance(net.proj_in, LoRALinear)
    assert "attn1.to_out.0" in names


def test_gradients_at_init_flow_only_into_B():
    """With B=0: dL/dB != 0 but dL/dA = 0 (A is multiplied by B). That is why training still starts - B moves first."""
    _, net, _ = _make()
    net(torch.randn(16, 8)).square().mean().backward()
    for m in lora_modules(net):
        assert m.B.grad.abs().sum() > 0
        assert m.A.grad.abs().sum() == 0
    assert all(p.grad is None for n, p in net.named_parameters() if not p.requires_grad)      # frozen weights get no grad


def test_merge_equals_unmerged_output():
    _, net, _ = _make()
    _randomise_B(net)
    x = torch.randn(5, 8)
    before = net(x)
    merge_lora(net)
    assert torch.allclose(before, net(x), atol=1e-5)


def test_scale_zero_is_base_model_and_scale_is_monotone():
    base, net, _ = _make()
    _randomise_B(net)
    x = torch.randn(5, 8)
    set_lora_scale(net, 0.0)
    assert torch.allclose(base(x), net(x), atol=1e-6)
    set_lora_scale(net, 1.0)
    d1 = (net(x) - base(x)).norm()
    set_lora_scale(net, 2.0)
    assert (net(x) - base(x)).norm() > d1


def test_state_dict_round_trip():
    _, net, _ = _make()
    _randomise_B(net)
    state = lora_state_dict(net)
    assert len(state) == 16 and all(k.endswith((".A", ".B")) for k in state)
    _, fresh, _ = _make()                                  # same base weights (same seed), LoRA at init
    load_lora_state_dict(fresh, state)
    x = torch.randn(5, 8)
    assert torch.allclose(net(x), fresh(x), atol=1e-6)


def test_lora_can_fit_a_target_and_param_count_formula():
    """r*(in+out) parameters per adapted layer; and a few steps of Adam on A,B must reduce a regression loss."""
    base, net, names = _make(r=2)
    assert sum(p.numel() for p in lora_parameters(net)) == len(names) * 2 * (8 + 8)
    x, y = torch.randn(64, 8), torch.randn(64, 8)
    opt = torch.optim.Adam(lora_parameters(net), 1e-2)
    first = None
    for _ in range(80):
        loss = (net(x) - y).square().mean()
        opt.zero_grad(); loss.backward(); opt.step()
        first = loss.item() if first is None else first
    assert loss.item() < first


def test_inject_raises_when_nothing_matches():
    with pytest.raises(RuntimeError):
        inject_lora(nn.Sequential(nn.Linear(4, 4)), targets=("to_q",))
