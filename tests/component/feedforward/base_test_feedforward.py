import torch
import pytest
from ..base_test_component import BaseComponentTest

class BaseFeedforwardTest(BaseComponentTest):
    """
    An abstract test suite for all feedforward components. It inherits from
    BaseComponentTest and adds feedforward-specific tests.
    """

    # --- Fixtures to provide data for feedforward tests ---
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
    def d_ff(self):
        return 256

    @pytest.fixture
    def dropout_rate(self):
        return 0.1

    @pytest.fixture
    def input_tensor(self, batch_size, seq_len, d_model):
        """Standard input tensor for feedforward testing."""
        return torch.randn(batch_size, seq_len, d_model)

    @pytest.fixture
    def single_vector(self, batch_size, d_model):
        """Single vector input for feedforward testing."""
        return torch.randn(batch_size, d_model)
        
    # --- Feedforward-Specific Tests ---
    
    def test_forward_pass_runs(self, component_instance, input_tensor):
        """Tests that the forward pass executes without errors."""
        try:
            output = component_instance(input_tensor)
        except Exception as e:
            pytest.fail(f"Forward pass failed with an exception: {e}")

    def test_output_shape_preservation(self, component_instance, input_tensor):
        """Tests if the output tensor preserves input shape (except possibly last dim)."""
        output = component_instance(input_tensor)
        
        # Output should preserve batch and sequence dimensions
        assert output.shape[:-1] == input_tensor.shape[:-1], "Batch and sequence dimensions should be preserved"
        assert len(output.shape) == len(input_tensor.shape), "Output should have same number of dimensions as input"

    def test_output_is_tensor(self, component_instance, input_tensor):
        """Tests that the output is a valid tensor."""
        output = component_instance(input_tensor)
        assert isinstance(output, torch.Tensor), "Output must be a torch.Tensor"

    def test_gradient_flow(self, component_instance, input_tensor):
        """Tests that gradients can flow through the feedforward network."""
        input_tensor.requires_grad_(True)
        output = component_instance(input_tensor)
        loss = output.sum()
        
        try:
            loss.backward()
            assert input_tensor.grad is not None, "Gradients should flow to input"
        except Exception as e:
            pytest.fail(f"Gradient computation failed: {e}")

    def test_different_input_shapes(self, component_instance, d_model):
        """Tests that the feedforward works with different input shapes."""
        test_shapes = [
            (1, d_model),      # Single sample
            (3, d_model),      # Batch without sequence
            (2, 16, d_model),  # Batch with sequence
            (1, 32, d_model),  # Single sample with sequence
        ]
        
        for shape in test_shapes:
            input_tensor = torch.randn(*shape)
            try:
                output = component_instance(input_tensor)
                assert output.shape[:-1] == input_tensor.shape[:-1], f"Failed for shape {shape}"
            except Exception as e:
                pytest.fail(f"Failed for input shape {shape}: {e}")

    def test_training_vs_eval_mode(self, component_instance, input_tensor):
        """Tests behavior difference between training and eval modes."""
        # Test in training mode
        component_instance.train()
        train_output = component_instance(input_tensor)
        
        # Test in eval mode
        component_instance.eval()
        with torch.no_grad():
            eval_output = component_instance(input_tensor)
        
        # Both should be valid tensors
        assert isinstance(train_output, torch.Tensor), "Training output should be tensor"
        assert isinstance(eval_output, torch.Tensor), "Eval output should be tensor"
        assert train_output.shape == eval_output.shape, "Output shapes should match between modes"

    def test_activation_functions(self, component_instance, input_tensor):
        """Tests that activation functions produce reasonable outputs."""
        output = component_instance(input_tensor)
        
        # Check for NaN or infinite values
        assert torch.isfinite(output).all(), "Output contains NaN or infinite values"
        
        # For most activations, output should have some variation
        if output.numel() > 1:
            assert output.std() > 1e-6, "Output should have some variation (not all zeros/constants)"

    def test_parameter_count(self, component_instance):
        """Tests that the feedforward network has trainable parameters."""
        param_count = sum(p.numel() for p in component_instance.parameters() if p.requires_grad)
        assert param_count > 0, "Feedforward network should have trainable parameters"