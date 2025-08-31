import pytest, inspect
from .base_test_attention import BaseAttentionTest


# This single class will be used to test ALL discovered attention modules.
# The 'attention_param' fixture is automatically provided by conftest.py
# and contains a tuple of (class, config) for each discovered module.

class TestAllAttentionModules(BaseAttentionTest):

    @pytest.fixture
    def component_class(self, attention_param):
        """Gets the component class from the parametrized fixture."""
        return attention_param[0]

    @pytest.fixture
    def component_config(self, attention_param, d_model, n_heads):
        """Gets the component config and merges it with base test params."""
        # Start with base config required by all attention modules
        base_config = {"d_model": d_model, "n_heads": n_heads}

        # Add module-specific config from discovery
        specific_config = attention_param[1]
        base_config.update(specific_config)

        return base_config
