import torch
import pytest
from ..base_test_component import BaseComponentTest

class BaseBackboneTest(BaseComponentTest):
    """
    An abstract test suite for all backbone components. It inherits from
    BaseComponentTest and adds backbone-specific tests.
    """

    # --- Fixtures to provide data for backbone tests ---
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
    def vocab_size(self):
        return 1000

    @pytest.fixture
    def input_tensor(self, batch_size, seq_len, d_model):
        """Standard input tensor for backbone testing."""
        return torch.randn(batch_size, seq_len, d_model)

    @pytest.fixture
    def input_ids(self, batch_size, seq_len, vocab_size):
        """Token IDs for transformer-based backbones."""
        return torch.randint(0, vocab_size, (batch_size, seq_len))

    @pytest.fixture
    def attention_mask(self, batch_size, seq_len):
        """Attention mask for transformer-based backbones."""
        return torch.ones(batch_size, seq_len)
        
    # --- Backbone-Specific Tests ---
    
    def test_forward_pass_runs(self, component_instance, input_tensor):
        """Tests that the forward pass executes without errors."""
        try:
            output = component_instance(input_tensor)
        except Exception as e:
            pytest.fail(f"Forward pass failed with an exception: {e}")

    def test_output_shape_is_correct(self, component_instance, input_tensor, batch_size, seq_len):
        """Tests if the output tensor has the expected shape."""
        output = component_instance(input_tensor)
        
        # Output should maintain batch and sequence dimensions
        assert output.shape[0] == batch_size, "Batch dimension mismatch"
        assert output.shape[1] == seq_len, "Sequence length dimension mismatch"
        assert len(output.shape) == 3, "Output should be 3D (batch, seq, features)"

    def test_output_is_tensor(self, component_instance, input_tensor):
        """Tests that the output is a valid tensor."""
        output = component_instance(input_tensor)
        assert isinstance(output, torch.Tensor), "Output must be a torch.Tensor"

    def test_gradient_flow(self, component_instance, input_tensor):
        """Tests that gradients can flow through the backbone."""
        input_tensor.requires_grad_(True)
        output = component_instance(input_tensor)
        loss = output.sum()
        
        try:
            loss.backward()
            assert input_tensor.grad is not None, "Gradients should flow to input"
        except Exception as e:
            pytest.fail(f"Gradient computation failed: {e}")

    def test_different_batch_sizes(self, component_instance, seq_len, d_model):
        """Tests that the backbone works with different batch sizes."""
        for batch_size in [1, 3, 5]:
            input_tensor = torch.randn(batch_size, seq_len, d_model)
            try:
                output = component_instance(input_tensor)
                assert output.shape[0] == batch_size, f"Failed for batch size {batch_size}"
            except Exception as e:
                pytest.fail(f"Failed for batch size {batch_size}: {e}")

    def test_eval_mode_consistency(self, component_instance, input_tensor):
        """Tests that the backbone produces consistent outputs in eval mode."""
        component_instance.eval()
        
        with torch.no_grad():
            output1 = component_instance(input_tensor)
            output2 = component_instance(input_tensor)
            
        # Outputs should be identical in eval mode (no dropout, etc.)
        torch.testing.assert_close(output1, output2, msg="Outputs should be identical in eval mode")