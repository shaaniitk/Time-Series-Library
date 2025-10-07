import os
import sys
import importlib.util
import torch


def _load_mixture_nll_loss(repo_root: str):
    mdn_decoder_path = os.path.join(repo_root, 'layers', 'modular', 'decoder', 'mixture_density_decoder.py')
    spec = importlib.util.spec_from_file_location("mixture_density_decoder", mdn_decoder_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MixtureNLLLoss


def extract_mdn_params(outputs):
    if isinstance(outputs, tuple) and len(outputs) >= 3:
        means, log_stds, log_weights = outputs[0], outputs[1], outputs[2]
        return (means, log_stds, log_weights)
    return None


def mdn_expected_value(means, log_weights):
    probs = torch.softmax(log_weights, dim=-1)
    # Broadcast probs over target axis for multivariate case
    if means.dim() == 4 and probs.dim() == 3:
        probs = probs.unsqueeze(-2)
    return (probs * means).sum(dim=-1)


def test_mdn_loss_unpacked_tuple():
    # Ensure repo root on path
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    MixtureNLLLoss = _load_mixture_nll_loss(repo_root)

    # Shapes
    B, T, num_targets, K = 2, 5, 4, 3

    # Synthetic MDN parameters
    means = torch.randn(B, T, num_targets, K)
    log_stds = torch.randn(B, T, num_targets, K)
    log_weights = torch.randn(B, T, K)

    # Simulate model output with extra trailing item
    outputs = (means, log_stds, log_weights, {'aux': 'extra'})

    mdn_params = extract_mdn_params(outputs)
    assert mdn_params is not None

    # Expected-value decoding for metrics (not used in loss)
    pred = mdn_expected_value(mdn_params[0], mdn_params[2])
    assert pred.shape == (B, T, num_targets)

    # Targets
    targets = torch.randn(B, T, num_targets)

    # Loss should accept the MDN params 3-tuple without unpack errors
    criterion = MixtureNLLLoss()
    loss = criterion(mdn_params, targets)
    assert loss.ndim == 0 and torch.isfinite(loss)