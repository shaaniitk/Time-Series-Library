import unittest
from pathlib import Path
import sys
import torch

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from tests.models.sanity_test_mixin import SanityTestMixin
from models.HierarchicalEnhancedAutoformer import Model as HierarchicalEnhancedAutoformerModel
from utils.logger import logger

class TestHierarchicalEnhancedAutoformerSine(unittest.TestCase, SanityTestMixin):
    def test_sine_convergence(self):
        logger.info("Running Sine Convergence Test for HierarchicalEnhancedAutoformer")
        # HierarchicalEnhancedAutoformer has defaults for its specific args (n_levels, wavelet_type etc.).
        # The SanityTestMixin's Args object will serve as the base 'configs'.
        model_specific_config_overrides = {
            'n_levels': 2, # Override default (3) as it might be too much for tiny model
            'd_model': 32,
            'd_ff': 64
        }
        loss_config_pinball = {
            'loss_name': 'pinball',
            'quantile_levels': [0.1, 0.5, 0.9]
        }
        mse, _, _ = self.run_sanity_test(
            HierarchicalEnhancedAutoformerModel,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            epochs=20, # Hierarchical models might also need a bit more
            model_config=model_specific_config_overrides,
            loss_config=loss_config_pinball
        )
        logger.info(f"HierarchicalEnhancedAutoformer Sine Test Final MSE: {mse:.6f}")
        # Allow slightly higher MSE due to hierarchical complexity with tiny model
        self.assertLess(mse, 0.2, "HierarchicalEnhancedAutoformer did not converge well on synthetic sine wave data with Pinball loss (MSE > 0.2)")

if __name__ == '__main__':
    # Set up logging for direct execution
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    unittest.main()