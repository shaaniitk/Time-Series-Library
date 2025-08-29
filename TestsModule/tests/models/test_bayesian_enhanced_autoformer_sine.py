import unittest
from pathlib import Path
import sys
import torch

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from tests.models.sanity_test_mixin import SanityTestMixin
from models.BayesianEnhancedAutoformer import Model as BayesianEnhancedAutoformerModel
from utils.logger import logger

class TestBayesianEnhancedAutoformerSine(unittest.TestCase, SanityTestMixin):
    def test_sine_convergence(self):
        logger.info("Running Sine Convergence Test for BayesianEnhancedAutoformer")
        # BayesianEnhancedAutoformer has defaults for its specific args (like kl_weight, quantile_levels=None).
        # The SanityTestMixin's Args object will serve as the base 'configs'.
        model_specific_config_overrides = {
            'kl_weight': 1e-5, # Ensure KL loss is active and weighted
            # For Bayesian model, d_model and d_ff might need to be slightly larger
            # for Pinball loss to converge well, but let's try with mixin defaults first.
            'd_model': 32, # Example override
            'd_ff': 64    # Example override
        }
        loss_config_pinball = {
            'loss_name': 'pinball',
            'quantile_levels': [0.1, 0.5, 0.9]
        }
        mse, _, _ = self.run_sanity_test(
            BayesianEnhancedAutoformerModel,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            epochs=20, # Bayesian models might need a bit more epochs or careful LR
            model_config=model_specific_config_overrides,
            loss_config=loss_config_pinball
        )

        self.assertLess(mse, 0.2, "Bayesian did not converge well on synthetic sine wave data with Pinball loss (MSE > 0.2)")

if __name__ == '__main__':
    # Set up logging for direct execution
    import logging

    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')

    unittest.main()