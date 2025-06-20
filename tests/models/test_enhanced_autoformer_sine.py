import unittest
from pathlib import Path
import sys
import torch

# Add project root to path to allow direct execution of this test file
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from tests.models.sanity_test_mixin import SanityTestMixin
from models.EnhancedAutoformer import Model as EnhancedAutoformerModel
from utils.logger import logger

class TestEnhancedAutoformerSine(unittest.TestCase, SanityTestMixin):
    def test_sine_convergence(self):
        logger.info("Running Sine Convergence Test for EnhancedAutoformer")
        # model_config can be used to pass specific architectural overrides
        # if the defaults in SanityTestMixin are not suitable.
        # For this basic sine test, the mixin's defaults are generally fine.
        model_specific_config_overrides = {
            'd_model': 32, 
            'd_ff': 64
        }

        # Run without quantile levels first
        mse_no_quantiles, _, _ = self.run_sanity_test(
            EnhancedAutoformerModel,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            epochs=15, # Slightly more epochs for robust convergence check
            model_config=model_specific_config_overrides,
            loss_config={'loss_name': 'mse'} # No quantiles
        )
        logger.info(f"EnhancedAutoformer Sine Test (No Quantiles) Final MSE: {mse_no_quantiles:.6f}")
        self.assertLess(mse_no_quantiles, 0.15, "EnhancedAutoformer did not converge without quantiles (MSE > 0.15)")

        # Run with quantile levels
        loss_config_pinball = {
            'loss_name': 'pinball',
            'quantile_levels': [0.1, 0.5, 0.9]
        }

        mse_with_quantiles, _, _ = self.run_sanity_test(
            EnhancedAutoformerModel,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            epochs=15, # Slightly more epochs for robust convergence check
            model_config=model_specific_config_overrides,
            loss_config=loss_config_pinball
        )

        logger.info(f"EnhancedAutoformer Sine Test Final MSE: {mse_with_quantiles:.6f}")
        self.assertLess(mse_with_quantiles, 0.15, "EnhancedAutoformer did not converge on synthetic sine wave data with Pinball loss (MSE > 0.15)")

if __name__ == '__main__':
    # Set up logging for direct execution
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    unittest.main()
