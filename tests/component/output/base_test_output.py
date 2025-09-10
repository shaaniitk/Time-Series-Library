import torch
import pytest
from ..base_test_component import BaseComponentTest

class BaseOutputTest(BaseComponentTest):
    """
    An abstract test suite for all output components. It inherits from
    BaseComponentTest and adds output-specific tests.
    """

    # --- Fixtures to provide data for output tests ---
    @pytest.fixture
    def batch_size(self):
        return 2

    @pytest.fixture
    def seq_len(self):
        return 32

    @pytest.fixture
    def d_model(self):
        return 64

    @pytest.fixture
    def output_dim(self):
        return 1  # Common for forecasting tasks

    @pytest.fixture
    def pred_len(self):
        return 24  # Prediction length for forecasting

    @pytest.fixture
    def num_classes(self):
        return 10  # For classification tasks

    @pytest.fixture
    def input_tensor(self, batch_size, seq_len, d_model):
        """Standard input tensor for output layer testing."""
        return torch.randn(batch_size, seq_len, d_model)

    @pytest.fixture
    def encoder_output(self, batch_size, seq_len, d_model):
        """Encoder output tensor for output layer testing."""
        return torch.randn(batch_size, seq_len, d_model)
        
    # --- Output-Specific Tests ---
    
    def test_forward_pass_runs(self, component_instance, input_tensor):
        """Tests that the forward pass executes without errors."""
        try:
            output = component_instance(input_tensor)
        except Exception as e:
            pytest.fail(f"Forward pass failed with an exception: {e}")

    def test_output_shape_is_correct(self, component_instance, input_tensor, batch_size):
        """Tests if the output tensor has the expected shape."""
        output = component_instance(input_tensor)
        
        # Output should maintain batch dimension
        assert output.shape[0] == batch_size, "Batch dimension mismatch"
        assert len(output.shape) >= 2, "Output should be at least 2D (batch, features)"

    def test_output_is_tensor(self, component_instance, input_tensor):
        """Tests that the output is a valid tensor."""
        output = component_instance(input_tensor)
        assert isinstance(output, torch.Tensor), "Output must be a torch.Tensor"

    def test_gradient_flow(self, component_instance, input_tensor):
        """Tests that gradients can flow through the output layer."""
        input_tensor.requires_grad_(True)
        output = component_instance(input_tensor)
        loss = output.sum()
        
        try:
            loss.backward()
            assert input_tensor.grad is not None, "Gradients should flow to input"
        except Exception as e:
            pytest.fail(f"Gradient computation failed: {e}")

    def test_different_batch_sizes(self, component_instance, seq_len, d_model):
        """Tests that the output layer works with different batch sizes."""
        for batch_size in [1, 3, 5]:
            input_tensor = torch.randn(batch_size, seq_len, d_model)
            try:
                output = component_instance(input_tensor)
                assert output.shape[0] == batch_size, f"Failed for batch size {batch_size}"
            except Exception as e:
                pytest.fail(f"Failed for batch size {batch_size}: {e}")

    def test_output_range_reasonable(self, component_instance, input_tensor):
        """Tests that output values are in a reasonable range."""
        output = component_instance(input_tensor)
        
        # Check for NaN or infinite values
        assert torch.isfinite(output).all(), "Output contains NaN or infinite values"
        
        # Check that output is not all zeros (unless it's a valid initialization)
        if output.numel() > 1:
            assert not torch.allclose(output, torch.zeros_like(output)), "Output should not be all zeros"

    def test_deterministic_output(self, component_instance, input_tensor):
        """Tests that the output is deterministic given the same input."""
        component_instance.eval()
        
        with torch.no_grad():
            output1 = component_instance(input_tensor)
            output2 = component_instance(input_tensor)
            
        torch.testing.assert_close(output1, output2, msg="Outputs should be identical for same input")

    def test_parameter_count(self, component_instance):
        """Tests that the output layer has trainable parameters."""
        param_count = sum(p.numel() for p in component_instance.parameters() if p.requires_grad)
        assert param_count > 0, "Output layer should have trainable parameters"

    def test_different_sequence_lengths(self, component_instance, batch_size, d_model):
        """Tests that the output layer works with different sequence lengths."""
        for seq_len in [8, 16, 32, 64]:
            input_tensor = torch.randn(batch_size, seq_len, d_model)
            try:
                output = component_instance(input_tensor)
                assert output.shape[0] == batch_size, f"Failed for sequence length {seq_len}"
            except Exception as e:
                pytest.fail(f"Failed for sequence length {seq_len}: {e}")

    def test_output_dtype_consistency(self, component_instance, input_tensor):
        """Tests that output dtype is consistent with input dtype."""
        # Test with float32
        input_float32 = input_tensor.float()
        output_float32 = component_instance(input_float32)
        
        # Test with float64
        input_float64 = input_tensor.double()
        component_instance = component_instance.double()
        output_float64 = component_instance(input_float64)
        
        assert output_float32.dtype == torch.float32, "Output dtype should match input (float32)"
        assert output_float64.dtype == torch.float64, "Output dtype should match input (float64)"