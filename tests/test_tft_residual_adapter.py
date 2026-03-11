import io
import os
import pickle
import random

import numpy as np
import pytest
import torch
from torch import nn

from layers.TemporalFusion_layers import (
    ExtensionResidualAdapter,
    temporarily_zero_extensions,
)


def _numpy_rng_bytes():
    return pickle.dumps(np.random.get_state())


def test_neutral_adapter_is_exact_and_preserves_shared_gradient_while_strength_learns():
    adapter = ExtensionResidualAdapter(mode='neutral')
    branch = nn.Linear(4, 4, bias=False)
    with torch.no_grad():
        branch.weight.copy_(torch.eye(4))

    shared_input = torch.arange(1.0, 9.0).reshape(2, 4).requires_grad_()
    base_output = shared_input * 2.0
    combined = adapter(base_output, lambda: branch(base_output))

    assert torch.equal(combined, base_output)
    combined.sum().backward()

    reference_input = shared_input.detach().clone().requires_grad_()
    (reference_input * 2.0).sum().backward()
    assert torch.equal(shared_input.grad, reference_input.grad)
    assert adapter.residual_strength.grad is not None
    assert adapter.residual_strength.grad.item() == pytest.approx(base_output.detach().sum().item())
    assert branch.weight.grad is not None
    assert torch.count_nonzero(branch.weight.grad).item() == 0


@pytest.mark.parametrize("nonfinite", [float('nan'), float('inf'), float('-inf')])
def test_exact_zero_adapter_quarantines_nonfinite_delta_and_cannot_self_activate(
    nonfinite,
):
    adapter = ExtensionResidualAdapter(mode='neutral')
    base = torch.arange(1.0, 7.0).reshape(2, 3).requires_grad_()
    delta = torch.ones_like(base)
    delta[0, 1] = nonfinite

    combined = adapter(base, delta)

    assert torch.equal(combined, base)
    assert torch.isfinite(combined).all()
    combined.sum().backward()
    assert torch.equal(base.grad, torch.ones_like(base))
    assert adapter.residual_strength.grad is not None
    assert adapter.residual_strength.grad.item() == 0.0
    # The raw diagnostic remains visibly non-finite rather than disguising a
    # broken branch as a legitimate zero-valued delta.
    assert not torch.isfinite(adapter.diagnostics()['delta_rms'])


def test_nonfinite_delta_with_active_strength_fails_closed_explicitly():
    adapter = ExtensionResidualAdapter(mode='small_residual')
    base = torch.ones(2, 3)
    delta = torch.full_like(base, float('nan'))

    with pytest.raises(ValueError, match='non-finite extension delta'):
        adapter(base, delta)


def test_nonzero_scalar_strength_changes_output_and_activates_branch_gradients():
    adapter = ExtensionResidualAdapter(mode='neutral')
    branch = nn.Linear(3, 3, bias=False)
    with torch.no_grad():
        adapter.residual_strength.fill_(0.25)
        branch.weight.copy_(torch.eye(3))

    base = torch.arange(1.0, 7.0).reshape(2, 3)
    delta = branch(base)
    combined = adapter(base, delta)
    assert torch.equal(combined, base * 1.25)

    combined.sum().backward()
    assert torch.count_nonzero(branch.weight.grad).item() > 0
    assert adapter.residual_strength.grad.item() != 0.0


def test_channel_strength_and_initialization_modes_have_declared_arithmetic():
    neutral = ExtensionResidualAdapter(3, strength_type='channel', mode='neutral')
    small = ExtensionResidualAdapter(
        3,
        strength_type='channel',
        mode='small_residual',
        small_residual_strength=0.05,
    )
    legacy = ExtensionResidualAdapter(mode='legacy')

    assert neutral.residual_strength.shape == (3,)
    assert torch.equal(neutral.residual_strength, torch.zeros(3))
    assert torch.equal(small.residual_strength, torch.full((3,), 0.05))
    assert legacy.residual_strength.shape == torch.Size([])
    assert legacy.residual_strength.item() == 1.0

    with torch.no_grad():
        small.residual_strength.copy_(torch.tensor([0.0, 0.5, -0.25]))
    base = torch.ones(2, 4, 3)
    delta = torch.full_like(base, 2.0)
    expected = base + delta * torch.tensor([0.0, 0.5, -0.25])
    assert torch.equal(small(base, delta), expected)
    assert torch.equal(legacy(base, delta), base + delta)


@pytest.mark.parametrize(
    ('kwargs', 'message'),
    [
        ({'strength_type': 'matrix'}, 'strength_type'),
        ({'mode': 'off'}, 'mode'),
        ({'strength_type': 'channel'}, 'channels'),
        ({'channels': 0, 'strength_type': 'channel'}, 'channels'),
        ({'channels': True, 'strength_type': 'channel'}, 'channels'),
        ({'mode': 'small_residual', 'small_residual_strength': 0.0}, 'nonzero'),
        ({'mode': 'small_residual', 'small_residual_strength': float('nan')}, 'finite'),
        ({'mode': 'legacy', 'legacy_strength': float('inf')}, 'finite'),
        ({'extension_name': ''}, 'extension_name'),
    ],
)
def test_constructor_rejects_ambiguous_or_invalid_contracts(kwargs, message):
    with pytest.raises(ValueError, match=message):
        ExtensionResidualAdapter(**kwargs)


def test_forward_validates_shape_dtype_layout_and_channel_contract():
    adapter = ExtensionResidualAdapter(3, strength_type='channel')
    base = torch.ones(2, 3)

    with pytest.raises(ValueError, match='identical shapes'):
        adapter(base, torch.ones(2, 2))
    with pytest.raises(ValueError, match='same dtype'):
        adapter(base, torch.ones(2, 3, dtype=torch.float64))
    with pytest.raises(ValueError, match='last dimension'):
        adapter(torch.ones(2, 4), torch.ones(2, 4))
    with pytest.raises(ValueError, match='floating-point'):
        adapter(torch.ones(2, 3, dtype=torch.int64), torch.ones(2, 3, dtype=torch.int64))
    with pytest.raises(ValueError, match='at least one dimension'):
        adapter(torch.tensor(1.0), torch.tensor(1.0))
    with pytest.raises(ValueError, match='non-empty'):
        adapter(torch.empty(0, 3), torch.empty(0, 3))
    with pytest.raises(TypeError, match='base_output'):
        adapter([1.0], torch.ones(1))
    with pytest.raises(TypeError, match='callable result'):
        adapter(base, lambda: 'not a tensor')

    sparse = base.to_sparse()
    with pytest.raises(ValueError, match='strided'):
        adapter(sparse, sparse)


def test_neutral_lazy_delta_restores_torch_python_and_numpy_rng_even_on_dropout():
    adapter = ExtensionResidualAdapter(mode='neutral')
    base = torch.ones(64, 8)
    observed_callback_randomness = []

    torch.manual_seed(314159)
    random.seed(271828)
    np.random.seed(161803)
    torch_before = torch.get_rng_state().clone()
    python_before = random.getstate()
    numpy_before = _numpy_rng_bytes()

    def stochastic_delta():
        observed_callback_randomness.append((random.random(), float(np.random.rand())))
        return torch.nn.functional.dropout(base, p=0.5, training=True)

    combined = adapter.combine_lazy(base, stochastic_delta)
    assert torch.equal(combined, base)
    assert len(observed_callback_randomness) == 1
    assert torch.equal(torch.get_rng_state(), torch_before)
    assert random.getstate() == python_before
    assert _numpy_rng_bytes() == numpy_before


def test_lazy_delta_rng_is_restored_when_callback_raises():
    adapter = ExtensionResidualAdapter(mode='neutral')
    base = torch.ones(2, 3)
    torch.manual_seed(91)
    random.seed(92)
    np.random.seed(93)
    torch_before = torch.get_rng_state().clone()
    python_before = random.getstate()
    numpy_before = _numpy_rng_bytes()

    def failing_delta():
        torch.rand(4)
        random.random()
        np.random.rand()
        raise RuntimeError('branch failed')

    with pytest.raises(RuntimeError, match='branch failed'):
        adapter(base, failing_delta)
    assert torch.equal(torch.get_rng_state(), torch_before)
    assert random.getstate() == python_before
    assert _numpy_rng_bytes() == numpy_before


def test_diagnostics_report_raw_and_effective_strength_and_true_rms_values():
    adapter = ExtensionResidualAdapter(mode='legacy', legacy_strength=0.25)
    base = torch.full((2, 3), 2.0)
    delta = torch.full((2, 3), 4.0)
    combined, diagnostics = adapter(base, delta, return_diagnostics=True)

    assert torch.equal(combined, torch.full((2, 3), 3.0))
    assert set(diagnostics) == {
        'raw_residual_strength',
        'effective_residual_strength',
        'residual_strength',
        'base_rms',
        'delta_rms',
        'combined_minus_base_rms',
    }
    assert diagnostics['raw_residual_strength'].tolist() == pytest.approx([0.25])
    assert diagnostics['effective_residual_strength'].tolist() == pytest.approx([0.25])
    assert diagnostics['residual_strength'].tolist() == pytest.approx([0.25])
    assert diagnostics['base_rms'].item() == pytest.approx(2.0)
    assert diagnostics['delta_rms'].item() == pytest.approx(4.0)
    assert diagnostics['combined_minus_base_rms'].item() == pytest.approx(1.0)

    # Returned diagnostics are copies and cannot mutate the adapter's cache.
    diagnostics['base_rms'].zero_()
    assert adapter.diagnostics()['base_rms'].item() == pytest.approx(2.0)


class _AdapterContainer(nn.Module):
    def __init__(self, duplicate_names=False):
        super().__init__()
        self.left_adapter = ExtensionResidualAdapter(
            mode='legacy', legacy_strength=0.3, extension_name='left'
        )
        self.right_adapter = ExtensionResidualAdapter(
            mode='legacy',
            legacy_strength=0.7,
            extension_name='left' if duplicate_names else 'right',
        )


def test_temporary_zero_context_is_selective_nested_nonpersistent_and_restores():
    model = _AdapterContainer()
    state_before = {key: value.clone() for key, value in model.state_dict().items()}
    base = torch.ones(2, 2)
    delta = torch.ones_like(base)

    with temporarily_zero_extensions(model, names='left') as yielded:
        assert yielded is model
        assert model.left_adapter.is_temporarily_zeroed
        assert not model.right_adapter.is_temporarily_zeroed
        assert torch.equal(model.left_adapter(base, delta), base)
        assert torch.equal(model.right_adapter(base, delta), base + 0.7 * delta)
        left_diagnostics = model.left_adapter.diagnostics()
        assert left_diagnostics['raw_residual_strength'].item() == pytest.approx(0.3)
        assert left_diagnostics['effective_residual_strength'].item() == 0.0

        with temporarily_zero_extensions(model, names=['right_adapter']):
            assert model.left_adapter.is_temporarily_zeroed
            assert model.right_adapter.is_temporarily_zeroed
            assert torch.equal(model.right_adapter(base, delta), base)

        assert model.left_adapter.is_temporarily_zeroed
        assert not model.right_adapter.is_temporarily_zeroed
        assert set(model.state_dict()) == set(state_before)
        for key, value in state_before.items():
            assert torch.equal(model.state_dict()[key], value)

    assert not model.left_adapter.is_temporarily_zeroed
    assert not model.right_adapter.is_temporarily_zeroed
    assert torch.equal(model.left_adapter(base, delta), base + 0.3 * delta)


def test_temporary_zero_context_restores_all_adapters_after_exception():
    model = _AdapterContainer()
    with pytest.raises(RuntimeError, match='counterfactual failed'):
        with temporarily_zero_extensions(model):
            assert model.left_adapter.is_temporarily_zeroed
            assert model.right_adapter.is_temporarily_zeroed
            raise RuntimeError('counterfactual failed')
    assert not model.left_adapter.is_temporarily_zeroed
    assert not model.right_adapter.is_temporarily_zeroed


def test_temporary_zero_context_rejects_unknown_names_before_mutation():
    model = _AdapterContainer()
    with pytest.raises(ValueError, match='Unknown extension'):
        with temporarily_zero_extensions(model, names=['left', 'missing']):
            pass
    assert not model.left_adapter.is_temporarily_zeroed
    assert not model.right_adapter.is_temporarily_zeroed


def test_semantic_extension_name_selects_all_matching_adapters_but_qualified_name_is_exact():
    ambiguous = _AdapterContainer(duplicate_names=True)
    base = torch.ones(2, 2)
    delta = torch.ones_like(base)

    with temporarily_zero_extensions(ambiguous, names='left'):
        assert ambiguous.left_adapter.is_temporarily_zeroed
        assert ambiguous.right_adapter.is_temporarily_zeroed
        assert torch.equal(ambiguous.left_adapter(base, delta), base)
        assert torch.equal(ambiguous.right_adapter(base, delta), base)
    assert not ambiguous.left_adapter.is_temporarily_zeroed
    assert not ambiguous.right_adapter.is_temporarily_zeroed

    with temporarily_zero_extensions(ambiguous, names='left_adapter'):
        assert ambiguous.left_adapter.is_temporarily_zeroed
        assert not ambiguous.right_adapter.is_temporarily_zeroed
    assert not ambiguous.left_adapter.is_temporarily_zeroed
    assert not ambiguous.right_adapter.is_temporarily_zeroed


def test_state_dict_round_trip_preserves_channel_strength_not_override_or_diagnostics():
    adapter = ExtensionResidualAdapter(
        3,
        strength_type='channel',
        mode='small_residual',
        extension_name='serializable',
    )
    with torch.no_grad():
        adapter.residual_strength.copy_(torch.tensor([0.2, -0.4, 0.8]))
    base = torch.ones(2, 3)
    delta = torch.full_like(base, 2.0)
    expected = adapter(base, delta)

    buffer = io.BytesIO()
    with temporarily_zero_extensions(adapter):
        torch.save(adapter.state_dict(), buffer)
        assert adapter.is_temporarily_zeroed
    buffer.seek(0)

    restored = ExtensionResidualAdapter(3, strength_type='channel', mode='neutral')
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    assert set(restored.state_dict()) == {'residual_strength'}
    assert not restored.is_temporarily_zeroed
    assert restored.diagnostics() is None
    assert torch.equal(restored.residual_strength, adapter.residual_strength)
    assert torch.equal(restored(base, delta), expected)


def test_float64_and_cpu_autocast_preserve_dtype_and_exact_neutral_output():
    double_adapter = ExtensionResidualAdapter(mode='neutral').double()
    double_base = torch.randn(2, 3, dtype=torch.float64)
    assert torch.equal(double_adapter(double_base, double_base.square()), double_base)

    base_layer = nn.Linear(4, 4, bias=False)
    delta_layer = nn.Linear(4, 4, bias=False)
    with torch.no_grad():
        base_layer.weight.copy_(torch.eye(4))
        delta_layer.weight.copy_(torch.eye(4))
    adapter = ExtensionResidualAdapter(mode='neutral')
    inputs = torch.ones(2, 4)
    with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
        base = base_layer(inputs)
        combined = adapter(base, lambda: delta_layer(base))
    assert base.dtype == torch.bfloat16
    assert combined.dtype == torch.bfloat16
    assert torch.equal(combined, base)
    combined.float().sum().backward()
    assert adapter.residual_strength.grad is not None
    assert adapter.residual_strength.grad.item() != 0.0
    assert torch.count_nonzero(delta_layer.weight.grad).item() == 0


@pytest.mark.skipif(
    os.environ.get('TFT_RUN_ACCELERATOR_TESTS') != '1',
    reason='set TFT_RUN_ACCELERATOR_TESTS=1 on a validated CUDA/ROCm worker',
)
def test_accelerator_autocast_and_lazy_rng_neutrality():
    if not torch.cuda.is_available():
        pytest.skip('CUDA/ROCm device unavailable')
    device = torch.device('cuda')
    base_layer = nn.Linear(4, 4, bias=False).to(device)
    delta_layer = nn.Sequential(nn.Dropout(0.5), nn.Linear(4, 4, bias=False)).to(device)
    adapter = ExtensionResidualAdapter(mode='neutral').to(device)
    inputs = torch.ones(8, 4, device=device)
    cpu_state = torch.get_rng_state().clone()
    accelerator_state = torch.cuda.get_rng_state(device).clone()

    with torch.autocast(device_type='cuda', dtype=torch.float16):
        base = base_layer(inputs)
        combined = adapter(base, lambda: delta_layer(base))
    assert combined.dtype == base.dtype
    assert torch.equal(combined, base)
    assert torch.equal(torch.get_rng_state(), cpu_state)
    assert torch.equal(torch.cuda.get_rng_state(device), accelerator_state)


def test_adapter_on_wrong_device_fails_with_actionable_message():
    adapter = ExtensionResidualAdapter(mode='neutral')
    base = torch.ones(2, 3, device='meta')
    with pytest.raises(ValueError, match='Move the adapter'):
        adapter(base, base.clone())
