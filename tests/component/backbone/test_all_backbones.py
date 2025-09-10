import pytest
from .base_test_backbone import BaseBackboneTest


# This single class will be used to test ALL discovered backbone modules.
# The 'backbone_param' fixture is automatically provided by conftest.py
# and contains a tuple of (class, config) for each discovered module.

class TestAllBackboneModules(BaseBackboneTest):

    @pytest.fixture
    def component_class(self, backbone_param):
        """Gets the component class from the parametrized fixture."""
        return backbone_param[0]

    @pytest.fixture
    def component_config(self, backbone_param, d_model, vocab_size):
        """Gets the component config and merges it with base test params."""
        # Start with base config required by most backbone modules
        base_config = {"d_model": d_model}
        
        # Add vocab_size for transformer-based backbones if needed
        if hasattr(backbone_param[0], '__name__') and 'transformer' in backbone_param[0].__name__.lower():
            base_config["vocab_size"] = vocab_size

        # Add module-specific config from discovery
        specific_config = backbone_param[1]
        base_config.update(specific_config)

        return base_config

    def test_backbone_specific_forward(self, component_instance, input_tensor):
        """Additional backbone-specific forward pass test."""
        output = component_instance(input_tensor)
        
        # Ensure output has proper dimensionality
        assert output.dim() == 3, "Backbone output should be 3D"
        assert output.size(-1) > 0, "Feature dimension should be positive"
        
        # Check for NaN or infinite values
        assert torch.isfinite(output).all(), "Output contains NaN or infinite values"