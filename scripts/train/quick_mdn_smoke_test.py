import os
import sys
import torch
import importlib.util

# Ensure repo root is on sys.path
CURRENT_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

def extract_mdn_params(outputs):
    if isinstance(outputs, tuple) and len(outputs) >= 3:
        means, log_stds, log_weights = outputs[0], outputs[1], outputs[2]
        return (means, log_stds, log_weights)
    return None

def mdn_expected_value(means, log_weights):
    probs = torch.softmax(log_weights, dim=-1)
    if means.dim() == 4 and probs.dim() == 3:
        probs = probs.unsqueeze(-2)
    return (probs * means).sum(dim=-1)

# Load MixtureNLLLoss directly from file to avoid heavy package imports
mdn_decoder_path = os.path.join(REPO_ROOT, 'layers', 'modular', 'decoder', 'mixture_density_decoder.py')
spec = importlib.util.spec_from_file_location("mixture_density_decoder", mdn_decoder_path)
mixture_density_decoder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mixture_density_decoder)
MixtureNLLLoss = mixture_density_decoder.MixtureNLLLoss

def main():
    # Shapes
    B, T, num_targets, K = 2, 5, 4, 3

    # Create synthetic MDN parameters
    means = torch.randn(B, T, num_targets, K)
    log_stds = torch.randn(B, T, num_targets, K)
    log_weights = torch.randn(B, T, K)

    # Simulate model output with extra trailing item
    outputs = (means, log_stds, log_weights, {'aux': 'extra'})

    # Extract MDN params and compute expected value
    mdn_params = extract_mdn_params(outputs)
    assert mdn_params is not None, "Failed to extract MDN params"

    pred = mdn_expected_value(mdn_params[0], mdn_params[2])
    # Expected shape: [B, T, num_targets]
    assert pred.shape == (B, T, num_targets), f"Unexpected prediction shape: {pred.shape}"

    # Create synthetic targets
    targets = torch.randn(B, T, num_targets)

    # Compute loss using proper MDN tuple
    criterion = MixtureNLLLoss()
    loss = criterion(mdn_params, targets)
    print("MDN smoke test OK | Loss:", float(loss.item()))

if __name__ == "__main__":
    main()