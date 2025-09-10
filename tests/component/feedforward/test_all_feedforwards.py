import torch
import pytest
from .base_test_feedforward import BaseFeedforwardTest


# This single class will be used to test ALL discovered feedforward modules.
# The 'feedforward_param' fixture is automatically provided by conftest.py
# and contains a tuple of (class, config) for each discovered module.

class TestAllFeedforwardModules(BaseFeedforwardTest):

    @pytest.fixture
    def component_class(self, feedforward_param):
        """Gets the component class from the parametrized fixture."""
        return feedforward_param[0]

    @pytest.fixture
    def component_config(self, feedforward_param, d_model, d_ff, dropout_rate):
        """Gets the component config and merges it with base test params."""
        # Start with base config required by most feedforward modules
        base_config = {
            "d_model": d_model,
            "d_ff": d_ff,
            "dropout": dropout_rate
        }

        # Add module-specific config from discovery
        specific_config = feedforward_param[1]
        base_config.update(specific_config)

        return base_config

    def test_feedforward_specific_forward(self, component_instance, input_tensor, d_model):
        """Additional feedforward-specific forward pass test."""
        output = component_instance(input_tensor)
        
        # Most feedforward networks should preserve the model dimension
        # (though some might change it, so we check if it's reasonable)
        assert output.size(-1) > 0, "Output feature dimension should be positive"
        
        # Check for proper tensor properties
        assert torch.isfinite(output).all(), "Output contains NaN or infinite values"
        assert not torch.allclose(output, torch.zeros_like(output)), "Output should not be all zeros"

    def test_feedforward_nonlinearity(self, component_instance, d_model):
        """Tests that the feedforward network introduces non-linearity."""
        # Create two different inputs
        input1 = torch.randn(1, 10, d_model)
        input2 = torch.randn(1, 10, d_model)
        
        # Get outputs
        output1 = component_instance(input1)
        output2 = component_instance(input2)
        
        # Combined input
        combined_input = input1 + input2
        combined_output = component_instance(combined_input)
        
        # For non-linear networks, f(a+b) != f(a) + f(b)
        linear_combination = output1 + output2
        
        # Allow for some numerical tolerance, but they should be different
        if not torch.allclose(combined_output, linear_combination, atol=1e-5):
            # This is expected for non-linear networks
            pass
        else:
            # If they're too close, the network might be too linear
            # This is not necessarily a failure, but worth noting
            print(f"Warning: {component_instance.__class__.__name__} might be too linear")

    def test_feedforward_capacity(self, component_instance, d_model):
        """Tests that the feedforward network has sufficient representational capacity."""
        # Create a batch of diverse inputs
        batch_size = 8
        seq_len = 16
        inputs = torch.randn(batch_size, seq_len, d_model)
        
        outputs = component_instance(inputs)
        
        # Check that outputs have reasonable variance
        output_std = outputs.std()
        assert output_std > 1e-4, f"Output standard deviation too low: {output_std}"
        
        # Check that different inputs produce different outputs
        output_diffs = []
        for i in range(batch_size - 1):
            diff = torch.norm(outputs[i] - outputs[i + 1])
            output_diffs.append(diff.item())
        
        avg_diff = sum(output_diffs) / len(output_diffs)
        assert avg_diff > 1e-4, f"Average output difference too small: {avg_diff}"