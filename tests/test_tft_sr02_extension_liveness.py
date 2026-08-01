"""Isolated enabled-behavior checks for every SR02 residual extension."""

import pytest
import torch

from tests.test_tft_sr02_model_neutrality import (
    EXTENSION_CASES,
    _adapters,
    _inputs,
    _paired_models,
)


@pytest.mark.parametrize(
    ("extension_name", "variant_overrides"),
    EXTENSION_CASES,
    ids=[case[0] for case in EXTENSION_CASES],
)
def test_each_nonzero_extension_changes_predictions_and_reaches_branch_parameters(
    extension_name,
    variant_overrides,
):
    reference, variant = _paired_models(variant_overrides)
    reference.eval()
    variant.eval()

    matching_adapters = [
        adapter
        for adapter in _adapters(variant).values()
        if adapter.extension_name == extension_name
    ]
    assert matching_adapters
    with torch.no_grad():
        for adapter in matching_adapters:
            adapter.residual_strength.fill_(0.2)

    inputs = _inputs(requires_grad=True)
    with torch.no_grad():
        reference_output = reference(
            *(value.detach() for value in inputs)
        )
    variant_output = variant(*inputs)
    assert not torch.equal(variant_output.detach(), reference_output)

    generator = torch.Generator().manual_seed(271828)
    probe = torch.randn(
        variant_output.shape,
        generator=generator,
        device=variant_output.device,
        dtype=variant_output.dtype,
    )
    (variant_output * probe).sum().backward()

    reference_parameter_names = set(dict(reference.named_parameters()))
    optional_branch_parameters = [
        parameter
        for name, parameter in variant.named_parameters()
        if name not in reference_parameter_names
        and not name.endswith("residual_adapter.residual_strength")
    ]
    assert optional_branch_parameters
    assert any(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        and torch.count_nonzero(parameter.grad).item() > 0
        for parameter in optional_branch_parameters
    )
