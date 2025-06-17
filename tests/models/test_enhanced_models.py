import unittest
import torch
import numpy as np
from tests.models.sanity_test_mixin import SanityTestMixin
from utils.logger import logger

# Import the enhanced models
from models.EnhancedAutoformer import EnhancedAutoformer
from models.BayesianEnhancedAutoformer import BayesianEnhancedAutoformer  
from models.HierarchicalEnhancedAutoformer import HierarchicalEnhancedAutoformer
from models.Autoformer import Model as Autoformer


class TestEnhancedModels(unittest.TestCase, SanityTestMixin):
    """
    Comprehensive unit tests for the three enhanced Autoformer models:
    1. EnhancedAutoformer
    2. BayesianEnhancedAutoformer  
    3. HierarchicalEnhancedAutoformer
    """
    
    def setUp(self):
        logger.info("Setting up Enhanced Models test suite")
        
        # Standard configuration for all models
        class Args:
            # Core parameters
            seq_len = 24
            pred_len = 12
            label_len = 12
            d_model = 64  # Reasonable size for testing
            e_layers = 2
            d_layers = 1
            d_ff = 256
            enc_in = 4
            dec_in = 4  # Should equal c_out for target features only
            c_out = 4
            embed = 'timeF'
            freq = 'h'
            dropout = 0.1
            task_name = 'long_term_forecast'
            features = 'M'
            
            # Autoformer specific parameters
            n_heads = 8
            moving_avg = 25
            factor = 1
            activation = 'gelu'
            output_attention = False
            
            # Enhanced model parameters
            adaptive_correlation = True
            learnable_decomposition = True
            curriculum_learning = True
            
            # Bayesian parameters
            n_mc_samples = 5  # Reduced for testing
            prior_std = 1.0
            kl_weight = 0.01
            
            # Hierarchical parameters
            n_levels = 3
            wavelet_type = 'db4'
            use_multiwavelet = True
            
        self.args = Args()
        
    def test_enhanced_autoformer_shape(self):
        """Test EnhancedAutoformer forward pass and output shape"""
        logger.info("Testing EnhancedAutoformer forward shape")
        
        model = EnhancedAutoformer(self.args)
        batch_size = 2
        
        x_enc = torch.randn(batch_size, self.args.seq_len, self.args.enc_in)
        x_mark_enc = torch.randn(batch_size, self.args.seq_len, 4)
        x_dec = torch.randn(batch_size, self.args.label_len + self.args.pred_len, self.args.dec_in)
        x_mark_dec = torch.randn(batch_size, self.args.label_len + self.args.pred_len, 4)
        
        out = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        expected_shape = (batch_size, self.args.pred_len, self.args.c_out)
        
        self.assertEqual(out.shape, expected_shape, 
                        f"EnhancedAutoformer: Expected output shape {expected_shape}, got {out.shape}")
        logger.info(f"✓ EnhancedAutoformer output shape: {out.shape}")
        
    def test_bayesian_autoformer_shape(self):
        """Test BayesianEnhancedAutoformer forward pass and output shape"""
        logger.info("Testing BayesianEnhancedAutoformer forward shape")
        
        model = BayesianEnhancedAutoformer(self.args)
        batch_size = 2
        
        x_enc = torch.randn(batch_size, self.args.seq_len, self.args.enc_in)
        x_mark_enc = torch.randn(batch_size, self.args.seq_len, 4)
        x_dec = torch.randn(batch_size, self.args.label_len + self.args.pred_len, self.args.dec_in)
        x_mark_dec = torch.randn(batch_size, self.args.label_len + self.args.pred_len, 4)
        
        out = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        expected_shape = (batch_size, self.args.pred_len, self.args.c_out)
        
        self.assertEqual(out.shape, expected_shape, 
                        f"BayesianEnhancedAutoformer: Expected output shape {expected_shape}, got {out.shape}")
        logger.info(f"✓ BayesianEnhancedAutoformer output shape: {out.shape}")
        
    def test_hierarchical_autoformer_shape(self):
        """Test HierarchicalEnhancedAutoformer forward pass and output shape"""
        logger.info("Testing HierarchicalEnhancedAutoformer forward shape")
        
        model = HierarchicalEnhancedAutoformer(self.args)
        batch_size = 2
        
        x_enc = torch.randn(batch_size, self.args.seq_len, self.args.enc_in)
        x_mark_enc = torch.randn(batch_size, self.args.seq_len, 4)
        x_dec = torch.randn(batch_size, self.args.label_len + self.args.pred_len, self.args.dec_in)
        x_mark_dec = torch.randn(batch_size, self.args.label_len + self.args.pred_len, 4)
        
        out = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        expected_shape = (batch_size, self.args.pred_len, self.args.c_out)
        
        self.assertEqual(out.shape, expected_shape, 
                        f"HierarchicalEnhancedAutoformer: Expected output shape {expected_shape}, got {out.shape}")
        logger.info(f"✓ HierarchicalEnhancedAutoformer output shape: {out.shape}")
        
    def test_enhanced_autoformer_sanity(self):
        """Run sanity test for EnhancedAutoformer"""
        logger.info("="*60)
        logger.info("ENHANCED AUTOFORMER SANITY TEST")
        logger.info("="*60)
        
        # Custom config for enhanced features
        model_config = {
            'd_model': 32,  # Smaller for faster testing
            'e_layers': 2,
            'd_layers': 1,
            'd_ff': 128,
            'n_heads': 4
        }
        
        mse, y_pred, y_true = self.run_sanity_test(
            EnhancedAutoformer, 
            epochs=8,  # Reduced epochs for testing
            model_config=model_config
        )
        
        logger.info("="*60)
        logger.info(f"✓ EnhancedAutoformer Sanity Test MSE: {mse:.6f}")
        logger.info("="*60)
        
        # Should achieve good performance
        self.assertTrue(mse < 2.0, f"EnhancedAutoformer sanity test failed: MSE {mse:.6f} >= 2.0")
        
    def test_bayesian_autoformer_sanity(self):
        """Run sanity test for BayesianEnhancedAutoformer"""
        logger.info("="*60)
        logger.info("BAYESIAN ENHANCED AUTOFORMER SANITY TEST")
        logger.info("="*60)
        
        # Custom config for Bayesian model
        model_config = {
            'd_model': 32,  # Smaller for faster testing
            'e_layers': 2,
            'd_layers': 1,
            'd_ff': 128,
            'n_heads': 4
        }
        
        mse, y_pred, y_true = self.run_sanity_test(
            BayesianEnhancedAutoformer, 
            epochs=8,  # Reduced epochs for testing
            model_config=model_config
        )
        
        logger.info("="*60)
        logger.info(f"✓ BayesianEnhancedAutoformer Sanity Test MSE: {mse:.6f}")
        logger.info("="*60)
        
        # Bayesian models may have slightly higher MSE due to regularization
        self.assertTrue(mse < 3.0, f"BayesianEnhancedAutoformer sanity test failed: MSE {mse:.6f} >= 3.0")
        
    def test_hierarchical_autoformer_sanity(self):
        """Run sanity test for HierarchicalEnhancedAutoformer"""
        logger.info("="*60)
        logger.info("HIERARCHICAL ENHANCED AUTOFORMER SANITY TEST")
        logger.info("="*60)
        
        # Custom config for hierarchical model
        model_config = {
            'd_model': 32,  # Smaller for faster testing
            'e_layers': 2,
            'd_layers': 1,
            'd_ff': 128,
            'n_heads': 4
        }
        
        mse, y_pred, y_true = self.run_sanity_test(
            HierarchicalEnhancedAutoformer, 
            epochs=8,  # Reduced epochs for testing
            model_config=model_config
        )
        
        logger.info("="*60)
        logger.info(f"✓ HierarchicalEnhancedAutoformer Sanity Test MSE: {mse:.6f}")
        logger.info("="*60)
        
        # Should achieve good performance
        self.assertTrue(mse < 2.0, f"HierarchicalEnhancedAutoformer sanity test failed: MSE {mse:.6f} >= 2.0")
        
    def test_parameter_counts(self):
        """Compare parameter counts across all models"""
        logger.info("Testing parameter counts for all models")
        
        models = {
            'Autoformer': Autoformer(self.args),
            'EnhancedAutoformer': EnhancedAutoformer(self.args),
            'BayesianEnhancedAutoformer': BayesianEnhancedAutoformer(self.args),
            'HierarchicalEnhancedAutoformer': HierarchicalEnhancedAutoformer(self.args)
        }
        
        logger.info("="*50)
        logger.info("PARAMETER COUNT COMPARISON")
        logger.info("="*50)
        
        for name, model in models.items():
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            logger.info(f"{name:25}: {total_params:8,} total, {trainable_params:8,} trainable")
                 # Basic sanity checks
        self.assertGreater(total_params, 1000, f"{name} has too few parameters")
        # HierarchicalEnhancedAutoformer has more parameters due to multi-resolution encoders
        max_params = 100000000 if "Hierarchical" in name else 5000000
        self.assertLess(total_params, max_params, f"{name} has too many parameters for test config")
            
        logger.info("="*50)
        
    def test_enhanced_vs_original_comparison(self):
        """Direct comparison between original and enhanced models"""
        logger.info("="*70)
        logger.info("ENHANCED MODELS VS ORIGINAL AUTOFORMER COMPARISON")
        logger.info("="*70)
        
        # Reduced config for faster comparison testing
        model_config = {
            'd_model': 32,
            'e_layers': 2, 
            'd_layers': 1,
            'd_ff': 128,
            'n_heads': 4
        }
        
        # Test all models
        results = {}
        
        logger.info("Testing Original Autoformer...")
        autoformer_mse, _, _ = self.run_sanity_test(Autoformer, epochs=6, model_config=model_config)
        results['Original Autoformer'] = autoformer_mse
        
        logger.info("Testing Enhanced Autoformer...")
        enhanced_mse, _, _ = self.run_sanity_test(EnhancedAutoformer, epochs=6, model_config=model_config)
        results['Enhanced Autoformer'] = enhanced_mse
        
        logger.info("Testing Bayesian Enhanced Autoformer...")
        bayesian_mse, _, _ = self.run_sanity_test(BayesianEnhancedAutoformer, epochs=6, model_config=model_config)
        results['Bayesian Enhanced'] = bayesian_mse
        
        logger.info("Testing Hierarchical Enhanced Autoformer...")
        hierarchical_mse, _, _ = self.run_sanity_test(HierarchicalEnhancedAutoformer, epochs=6, model_config=model_config)
        results['Hierarchical Enhanced'] = hierarchical_mse
        
        # Display results
        logger.info("="*70)
        logger.info("FINAL COMPARISON RESULTS:")
        logger.info("="*70)
        
        sorted_results = sorted(results.items(), key=lambda x: x[1])
        
        for i, (name, mse) in enumerate(sorted_results):
            rank_emoji = ["🥇", "🥈", "🥉", "🏅"][i] if i < 4 else "📊"
            logger.info(f"{rank_emoji} {name:25}: MSE = {mse:.6f}")
            
        # Calculate improvements
        original_mse = results['Original Autoformer']
        logger.info("-" * 70)
        logger.info("IMPROVEMENTS OVER ORIGINAL:")
        
        for name, mse in results.items():
            if name != 'Original Autoformer':
                if mse < original_mse:
                    improvement = ((original_mse - mse) / original_mse) * 100
                    logger.info(f"✅ {name:25}: {improvement:+6.2f}% improvement")
                else:
                    degradation = ((mse - original_mse) / original_mse) * 100
                    logger.info(f"❌ {name:25}: {degradation:+6.2f}% degradation")
                    
        logger.info("="*70)
        
        # All models should achieve reasonable performance
        for name, mse in results.items():
            if 'Bayesian' in name:
                threshold = 3.0  # Bayesian models may have higher MSE due to regularization
            else:
                threshold = 2.5
                
            self.assertTrue(mse < threshold, 
                          f"{name} MSE too high: {mse:.6f} >= {threshold}")
    
    def test_model_robustness(self):
        """Test model robustness with different input scenarios"""
        logger.info("Testing model robustness")
        
        models = [
            ('EnhancedAutoformer', EnhancedAutoformer),
            ('BayesianEnhancedAutoformer', BayesianEnhancedAutoformer),
            ('HierarchicalEnhancedAutoformer', HierarchicalEnhancedAutoformer)
        ]
        
        for name, ModelClass in models:
            logger.info(f"Testing robustness for {name}")
            model = ModelClass(self.args)
            model.eval()
            
            # Test with different batch sizes
            for batch_size in [1, 3, 8]:
                x_enc = torch.randn(batch_size, self.args.seq_len, self.args.enc_in)
                x_mark_enc = torch.randn(batch_size, self.args.seq_len, 4)
                x_dec = torch.randn(batch_size, self.args.label_len + self.args.pred_len, self.args.dec_in)
                x_mark_dec = torch.randn(batch_size, self.args.label_len + self.args.pred_len, 4)
                
                with torch.no_grad():
                    out = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
                    
                expected_shape = (batch_size, self.args.pred_len, self.args.c_out)
                self.assertEqual(out.shape, expected_shape, 
                               f"{name} failed with batch_size={batch_size}")
                
            logger.info(f"✓ {name} robustness test passed")


if __name__ == '__main__':
    # Configure test runner for detailed output
    unittest.TextTestRunner(verbosity=2).run(unittest.TestLoader().loadTestsFromTestCase(TestEnhancedModels))
