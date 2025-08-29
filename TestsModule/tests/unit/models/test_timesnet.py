import unittest
import sys
import os
import torch

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from models.TimesNet import Model as TimesNet
from tests.models.sanity_test_mixin import SanityTestMixin
from utils.logger import logger

class TestTimesNet(unittest.TestCase, SanityTestMixin):
    def setUp(self):
        logger.info("Setting up TimesNet test case")
        class Args:
            seq_len = 24
            pred_len = 12
            d_model = 16
            e_layers = 2
            d_ff = 32
            enc_in = 4
            c_out = 4
            embed = 'timeF'
            freq = 'h'
            dropout = 0.1
            task_name = 'long_term_forecast'
            features = 'M'
            label_len = 12
            top_k = 2
            num_kernels = 6
        self.args = Args()
        self.model = TimesNet(self.args)

    def test_forward_shape(self):
        logger.info("Testing TimesNet forward shape")
        batch_size = 2
        x_enc = torch.randn(batch_size, self.args.seq_len, self.args.enc_in)
        x_mark_enc = torch.randn(batch_size, self.args.seq_len, 4)
        x_dec = torch.randn(batch_size, self.args.pred_len, self.args.c_out)
        x_mark_dec = torch.randn(batch_size, self.args.pred_len, 4)
        out = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        self.assertEqual(out.shape, (batch_size, self.args.pred_len, self.args.c_out))

    def test_sanity(self):
        logger.info("Running TimesNet sanity test")
        logger.info("="*50)
        logger.info("TIMESNET SANITY TEST")
        logger.info("="*50)
        
        mse, y_pred, y_true = self.run_sanity_test(TimesNet)
        
        logger.info("="*50)
        logger.info(f" TimesNet Sanity Test MSE: {mse:.6f}")
        logger.info("="*50)
        
        # Relaxed threshold for sanity test
        self.assertTrue(mse < 2.0, f"Sanity test failed: MSE {mse:.6f} >= 2.0")

    def test_parameter_count(self):
        logger.info("Testing TimesNet parameter count")
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        
        # Basic sanity check - should have reasonable number of parameters
        self.assertGreater(total_params, 1000)
        self.assertLess(total_params, 1000000)  # Should be reasonable for test config

if __name__ == '__main__':
    unittest.main()
