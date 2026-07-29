import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from layers.TemporalFusion_layers import SpectralBranch
from layers.TemporalFusion_layers import HigherOrderInteractionBlock


def _make_sine_batch(freqs, length=32, d_model=4):
    t = torch.arange(length, dtype=torch.float32)
    waves = []
    for freq in freqs:
        sample = torch.sin(2 * torch.pi * freq * t / length).view(length, 1).repeat(1, d_model)
        waves.append(sample)
    return torch.stack(waves, dim=0)


def test_fft_top_amplitude_high_bin():
    branch = SpectralBranch(d_model=4, modes=2, mode_select='top_amplitude', dropout=0.0)
    x = _make_sine_batch([10], length=32, d_model=4).requires_grad_(True)

    out = branch(x)

    assert tuple(out.shape) == tuple(x.shape)
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert branch.weight_real.grad is not None
    assert branch.weight_real.grad[:, :2].abs().sum().item() > 0.0


def test_fft_selection_batch_permutation_invariant():
    branch = SpectralBranch(d_model=4, modes=2, mode_select='top_amplitude', dropout=0.0).eval()
    x = _make_sine_batch([2, 10], length=32, d_model=4)
    perm = torch.tensor([1, 0], dtype=torch.long)

    with torch.no_grad():
        out = branch(x)
        out_perm = branch(x.index_select(0, perm))

    assert torch.allclose(out_perm, out.index_select(0, perm), atol=1e-6)


def test_fft_selection_batch_composition_invariant():
    branch = SpectralBranch(d_model=4, modes=2, mode_select='top_amplitude', dropout=0.0).eval()
    base = _make_sine_batch([10], length=32, d_model=4)
    extra = _make_sine_batch([2], length=32, d_model=4)

    with torch.no_grad():
        base_out = branch(base)
        combined_out = branch(torch.cat([base, extra], dim=0))

    assert torch.allclose(base_out[0], combined_out[0], atol=1e-6)


def test_fft_top_amplitude_gradients_are_finite():
    branch = SpectralBranch(d_model=3, modes=3, mode_select='top_amplitude', dropout=0.0)
    x = torch.randn(2, 24, 3, requires_grad=True)

    out = branch(x)
    loss = out.square().mean()
    loss.backward()

    assert torch.isfinite(out).all()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert branch.weight_real.grad is not None
    assert branch.weight_imag.grad is not None
    assert torch.isfinite(branch.weight_real.grad).all()
    assert torch.isfinite(branch.weight_imag.grad).all()


def test_fft_learned_selector_varies_across_frequency_bins():
    branch = SpectralBranch(d_model=2, modes=4, mode_select='learned', dropout=0.0)
    with torch.no_grad():
        branch.freq_mask_logits.zero_()
        branch.freq_mask_logits[0, 0] = torch.tensor([-12.0, 12.0, -12.0, 12.0])
        branch.freq_mask_logits[0, 1] = torch.tensor([12.0, -12.0, 12.0, -12.0])

    mask_logits, _, _ = branch._interpolate_learned_spectral_params(n_freqs=9)
    soft_mask = torch.sigmoid(mask_logits)

    assert soft_mask.shape == (1, 2, 9)
    assert soft_mask[0, 0].std().item() > 0.05
    assert soft_mask[0, 1].std().item() > 0.05
    assert not torch.allclose(soft_mask[0, 0], soft_mask[0, 1])


def test_fft_learned_runtime_lengths_and_checkpoint_roundtrip():
    branch = SpectralBranch(d_model=2, modes=4, mode_select='learned', dropout=0.0)
    with torch.no_grad():
        branch.freq_mask_logits[0, 0] = torch.tensor([-4.0, -1.0, 1.0, 4.0])
        branch.freq_mask_logits[0, 1] = torch.tensor([4.0, 1.0, -1.0, -4.0])

    short_summary = branch.summarize_learned_mask(n_freqs=5)
    long_summary = branch.summarize_learned_mask(n_freqs=9)
    reloaded = SpectralBranch(d_model=2, modes=4, mode_select='learned', dropout=0.0)
    reloaded.load_state_dict(branch.state_dict())
    reloaded_summary = reloaded.summarize_learned_mask(n_freqs=9)

    assert short_summary is not None
    assert long_summary is not None
    assert short_summary["fft_learned_mask_std"] > 0.0
    assert long_summary["fft_learned_mask_std"] > 0.0
    assert short_summary["fft_learned_mask_peak_bin_mean"] != long_summary["fft_learned_mask_peak_bin_mean"]
    assert reloaded_summary == long_summary


def test_fft_learned_mode_gradients_are_finite():
    branch = SpectralBranch(d_model=3, modes=5, mode_select='learned', dropout=0.0)
    x = torch.randn(2, 24, 3, requires_grad=True)

    out = branch(x)
    loss = out.square().mean()
    loss.backward()

    assert torch.isfinite(out).all()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert branch.weight_real.grad is not None
    assert branch.weight_imag.grad is not None
    assert branch.freq_mask_logits.grad is not None
    assert torch.isfinite(branch.weight_real.grad).all()
    assert torch.isfinite(branch.weight_imag.grad).all()
    assert torch.isfinite(branch.freq_mask_logits.grad).all()


def test_higher_order_two_forward_backward():
    block = HigherOrderInteractionBlock(d_model=8, interaction_order=2, interaction_rank=4, dropout=0.0)
    x = torch.randn(2, 7, 8, requires_grad=True)

    out, payload = block(x, return_payload=True)
    loss = out.square().mean()
    loss.backward()

    assert tuple(out.shape) == tuple(x.shape)
    assert tuple(payload["interaction_gates"].shape) == (2, 7, 1)
    assert torch.isfinite(out).all()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_higher_order_two_has_gate_gradient():
    block = HigherOrderInteractionBlock(d_model=8, interaction_order=2, interaction_rank=4, dropout=0.0)
    x = torch.randn(2, 7, 8, requires_grad=True)

    out = block(x)
    out.mean().backward()

    assert block.gate_projection.weight.grad is not None
    assert block.gate_projection.weight.grad.abs().sum().item() > 0.0


def test_higher_order_three_forward_backward():
    block = HigherOrderInteractionBlock(d_model=8, interaction_order=3, interaction_rank=4, dropout=0.0)
    x = torch.randn(2, 7, 8, requires_grad=True)

    out, payload = block(x, return_payload=True)
    loss = out.square().mean()
    loss.backward()

    assert tuple(out.shape) == tuple(x.shape)
    assert tuple(payload["interaction_gates"].shape) == (2, 7, 2)
    assert torch.isfinite(out).all()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_higher_order_terms_match_gate_count():
    order2 = HigherOrderInteractionBlock(d_model=8, interaction_order=2, interaction_rank=4, dropout=0.0)
    order3 = HigherOrderInteractionBlock(d_model=8, interaction_order=3, interaction_rank=4, dropout=0.0)
    assert order2.gate_projection.out_features == 1
    assert order3.gate_projection.out_features == 2


def test_higher_order_gate_logits_change_output():
    block = HigherOrderInteractionBlock(d_model=8, interaction_order=3, interaction_rank=4, dropout=0.0).eval()
    x = torch.randn(1, 5, 8)

    with torch.no_grad():
        block.gate_projection.weight.zero_()
        block.gate_projection.bias.fill_(-12.0)
        suppressed = block(x)
        block.gate_projection.bias.fill_(12.0)
        activated = block(x)

    assert not torch.allclose(suppressed, activated)
