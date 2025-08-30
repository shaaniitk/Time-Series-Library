import torch
import pytest

class BaseComponentTest:
    """
    The absolute base class for all component tests in the library.
    code Code
    IGNORE_WHEN_COPYING_START
    IGNORE_WHEN_COPYING_END

        
    It establishes a minimal contract that every component (Attention,
    Encoder, Loss, etc.) must fulfill.
    """

    @pytest.fixture
    def component_instance(self, component_class, component_config):
        """A pytest fixture to provide an instance of the component under test."""
        return component_class(**component_config)

    def test_is_torch_nn_module(self, component_instance):
        """Ensures the component is a valid torch.nn.Module."""
        assert isinstance(component_instance, torch.nn.Module), \
            f"{component_instance.__class__.__name__} must inherit from torch.nn.Module"

    def test_is_torchscript_compatible(self, component_instance):
        """
        Tests if the component can be successfully JIT-scripted. This is a
        critical check for deployment and performance.
        """
        try:
            torch.jit.script(component_instance)
        except Exception as e:
            pytest.fail(f"Component could not be torch-scripted. Error: {e}")

    