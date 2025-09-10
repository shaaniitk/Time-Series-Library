import torch
import pytest
from .base_test_output import BaseOutputTest


# This single class will be used to test ALL discovered output modules.
# The 'output_param' fixture is automatically provided by conftest.py
# and contains a tuple of (class, config) for each discovered module.

class TestAllOutputModules(BaseOutputTest):

    @pytest.fixture
    def component_class(self, output_param):
        """Gets the component class from the parametrized fixture."""
        return output_param[0]

    @pytest.fixture
    def component_config(self, output_param, d_model, output_dim, pred_len):
        """Gets the component config and merges it with base test params."""
        # Start with base config required by most output modules
        base_config = {
            "d_model": d_model,
            "output_dim": output_dim,
            "pred_len": pred_len
        }

        # Add module-specific config from discovery
        specific_config = output_param[1]
        base_config.update(specific_config)

        return base_config

    def test_output_specific_forward(self, component_instance, input_tensor, batch_size, output_dim):
        """Additional output-specific forward pass test."""
        output = component_instance(input_tensor)
        
        # Check basic output properties
        assert output.shape[0] == batch_size, "Batch dimension should be preserved"
        assert torch.isfinite(output).all(), "Output contains NaN or infinite values"

    def test_forecasting_output_shape(self, component_instance, input_tensor, batch_size, pred_len):
        """Tests output shape for forecasting tasks."""
        output = component_instance(input_tensor)
        
        # For forecasting, we expect the output to have prediction length dimension
        # The exact shape depends on the specific output layer implementation
        assert output.shape[0] == batch_size, "Batch dimension should be preserved"
        
        # Check if output has reasonable dimensions for forecasting
        if len(output.shape) == 3:
            # Format: (batch, pred_len, features) or (batch, seq_len, features)
            assert output.shape[1] > 0, "Sequence/prediction dimension should be positive"
            assert output.shape[2] > 0, "Feature dimension should be positive"
        elif len(output.shape) == 2:
            # Format: (batch, features)
            assert output.shape[1] > 0, "Feature dimension should be positive"

    def test_output_value_distribution(self, component_instance, input_tensor):
        """Tests that output values have reasonable distribution."""
        output = component_instance(input_tensor)
        
        # Check for reasonable variance (not all same values)
        if output.numel() > 1:
            output_std = output.std()
            assert output_std > 1e-6, f"Output standard deviation too low: {output_std}"
        
        # Check for reasonable range (not extremely large values)
        output_abs_max = output.abs().max()
        assert output_abs_max < 1e6, f"Output values too large: {output_abs_max}"

    def test_output_consistency_across_calls(self, component_instance, input_tensor):
        """Tests that multiple forward passes with same input produce same output."""
        component_instance.eval()
        
        with torch.no_grad():
            output1 = component_instance(input_tensor)
            output2 = component_instance(input_tensor)
            output3 = component_instance(input_tensor)
        
        torch.testing.assert_close(output1, output2, msg="First and second outputs should be identical")
        torch.testing.assert_close(output2, output3, msg="Second and third outputs should be identical")

    def test_output_scaling_behavior(self, component_instance, d_model, batch_size, seq_len):
        """Tests how output scales with input magnitude."""
        # Test with different input scales
        scales = [0.1, 1.0, 10.0]
        base_input = torch.randn(batch_size, seq_len, d_model)
        
        outputs = []
        for scale in scales:
            scaled_input = base_input * scale
            output = component_instance(scaled_input)
            outputs.append(output)
        
        # Check that outputs are different for different input scales
        for i in range(len(outputs) - 1):
            assert not torch.allclose(outputs[i], outputs[i + 1], atol=1e-5), \
                f"Outputs should differ for different input scales ({scales[i]} vs {scales[i + 1]})"

    def test_output_with_extreme_inputs(self, component_instance, input_tensor):
        """Tests output layer behavior with extreme input values."""
        # Test with very small inputs
        small_input = input_tensor * 1e-6
        try:
            small_output = component_instance(small_input)
            assert torch.isfinite(small_output).all(), "Output should be finite for small inputs"
        except Exception as e:
            pytest.fail(f"Failed with small inputs: {e}")
        
        # Test with moderately large inputs (avoid overflow)
        large_input = input_tensor * 10
        try:
            large_output = component_instance(large_input)
            assert torch.isfinite(large_output).all(), "Output should be finite for large inputs"
        except Exception as e:
            pytest.fail(f"Failed with large inputs: {e}")