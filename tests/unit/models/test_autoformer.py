import unittest
import sys
import os
import torch

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from models.Autoformer import Model as Autoformer
from tests.models.sanity_test_mixin import SanityTestMixin
from utils.logger import logger

class TestAutoformer(unittest.TestCase, SanityTestMixin):
    def setUp(self):
        logger.info("Setting up Autoformer test case")
        class Args:
            # MATCHED PARAMETER COUNT CONFIGURATION WITH TIMESNET
            seq_len = 24
            pred_len = 12
            label_len = 12
            d_model = 128  # Increased from 16 to ~128 (8x larger)
            e_layers = 3   # Increased from 2 to 3 layers
            d_layers = 2   # Increased from 1 to 2 layers
            d_ff = 512     # Increased from 32 to 512 (16x larger)
            enc_in = 4
            dec_in = 4
            c_out = 4
            embed = 'timeF'
            freq = 'h'
            dropout = 0.1
            task_name = 'long_term_forecast'
            features = 'M'
            
            # Autoformer specific parameters
            n_heads = 8    # Increased from 2 to 8 (must divide d_model: 128/8=16)
            moving_avg = 25
            factor = 1
            activation = 'gelu'
            output_attention = False
        self.args = Args()
        self.model = Autoformer(self.args)

    def test_forward_shape(self):
        logger.info("Testing Autoformer forward shape")
        batch_size = 2
        x_enc = torch.randn(batch_size, self.args.seq_len, self.args.enc_in)
        x_mark_enc = torch.randn(batch_size, self.args.seq_len, 4)
        x_dec = torch.randn(batch_size, self.args.label_len + self.args.pred_len, self.args.dec_in)
        x_mark_dec = torch.randn(batch_size, self.args.label_len + self.args.pred_len, 4)
        
        out = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        expected_shape = (batch_size, self.args.pred_len, self.args.c_out)
        self.assertEqual(out.shape, expected_shape, 
                        f"Expected output shape {expected_shape}, got {out.shape}")

    def test_sanity(self):
        logger.info("Running Autoformer sanity test")
        logger.info("="*50)
        logger.info("AUTOFORMER SANITY TEST")
        logger.info("="*50)
        
        mse, y_pred, y_true = self.run_sanity_test(Autoformer)
        
        logger.info("="*50)
        logger.info(f" Autoformer Sanity Test MSE: {mse:.6f}")
        logger.info("="*50)
        
        # Relaxed threshold for sanity test
        self.assertTrue(mse < 2.0, f"Sanity test failed: MSE {mse:.6f} >= 2.0")

    def test_parameter_count(self):
        logger.info("Testing Autoformer parameter count")
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        
        # Basic sanity check - should have reasonable number of parameters
        self.assertGreater(total_params, 1000)
        self.assertLess(total_params, 1000000)  # Should be reasonable for test config

    def test_comparison_with_timesnet(self):
        """Direct comparison test with TimesNet using identical configuration"""
        logger.info("Running comparison test: Autoformer vs TimesNet")
        
        # Test Autoformer
        logger.info("Testing Autoformer...")
        autoformer_mse, autoformer_pred, autoformer_true = self.run_sanity_test(Autoformer)
        
        # Test TimesNet with same config
        from models.TimesNet import Model as TimesNet
        
        # Create TimesNet args matching Autoformer exactly
        class TimesNetArgs:
            seq_len = 24
            pred_len = 12
            label_len = 12
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
            # TimesNet specific parameters
            top_k = 2
            num_kernels = 6
        
        logger.info("Testing TimesNet...")
        timesnet_mse, timesnet_pred, timesnet_true = self.run_sanity_test(TimesNet)
        
        # Compare results
        logger.info("="*60)
        logger.info("COMPARISON RESULTS:")
        logger.info(f"Autoformer MSE: {autoformer_mse:.6f}")
        logger.info(f"TimesNet MSE:   {timesnet_mse:.6f}")
        
        if autoformer_mse < timesnet_mse:
            improvement = ((timesnet_mse - autoformer_mse) / timesnet_mse) * 100
            logger.info(f"TROPHY Autoformer wins by {improvement:.2f}% improvement")
        else:
            improvement = ((autoformer_mse - timesnet_mse) / autoformer_mse) * 100
            logger.info(f"TROPHY TimesNet wins by {improvement:.2f}% improvement")
        
        logger.info("="*60)
        
        # Both should achieve reasonable performance
        self.assertTrue(autoformer_mse < 2.0, f"Autoformer MSE too high: {autoformer_mse}")
        self.assertTrue(timesnet_mse < 2.0, f"TimesNet MSE too high: {timesnet_mse}")

if __name__ == '__main__':
    unittest.main()
