import unittest
from unittest.mock import patch
import torch
from argparse import Namespace

from models.MambaHierarchical import MambaHierarchical
from models.ImprovedMambaHierarchical import ImprovedMambaHierarchical

class TestModelFallbacks(unittest.TestCase):
    def setUp(self):
        self.configs = Namespace(
            task_name='long_term_forecast',
            seq_len=96,
            label_len=48,
            pred_len=24,
            enc_in=44,
            dec_in=4,
            c_out=4,
            d_model=64,
            num_targets=4,
            num_covariates=40,
            covariate_family_size=4,
            wavelet_type='db4',
            wavelet_levels=3,
            use_trend_decomposition=False,
            mamba_d_state=16,
            mamba_d_conv=4,
            mamba_expand=2,
            attention_heads=8,
            hierarchical_attention_heads=8,
            cross_attention_heads=8,
            fusion_strategy='weighted_concat',
            use_family_attention=True,
            use_moe=False,
            dropout=0.1,
            trend_kernel_size=25,
            use_autoregressive=True,
            num_experts=4,
        )
        self.x_enc = torch.randn(2, 96, 44)
        self.x_dec = torch.randn(2, 72, 4)
        self.x_mark_enc = torch.randn(2, 96, 4)
        self.x_mark_dec = torch.randn(2, 72, 4)

    @patch('models.MambaHierarchical.TargetProcessor.__call__', side_effect=Exception("Mock Processor Failure"))
    def test_mambahierarchical_target_fallback(self, mock_processor):
        """
        Tests if MambaHierarchical correctly uses the fallback when the target processor fails.
        Effectiveness Check: Ensures the model doesn't crash and produces an output of the
        correct shape, proving the fallback path is functional.
        """
        model = MambaHierarchical(self.configs)
        try:
            output = model(self.x_enc, self.x_mark_enc, self.x_dec, self.x_mark_dec)
            if isinstance(output, tuple):
                output = output[0]
            self.assertEqual(output.shape, (2, self.configs.pred_len, self.configs.c_out))
        except Exception as e:
            self.fail(f"MambaHierarchical forward pass failed during fallback: {e}")

    @patch('models.MambaHierarchical.CovariateProcessor.__call__', side_effect=Exception("Mock Processor Failure"))
    def test_mambahierarchical_covariate_fallback(self, mock_processor):
        """Tests MambaHierarchical covariate processor fallback."""
        model = MambaHierarchical(self.configs)
        try:
            output = model(self.x_enc, self.x_mark_enc, self.x_dec, self.x_mark_dec)
            if isinstance(output, tuple):
                output = output[0]
            self.assertEqual(output.shape, (2, self.configs.pred_len, self.configs.c_out))
        except Exception as e:
            self.fail(f"MambaHierarchical forward pass failed during fallback: {e}")

    @patch('models.ImprovedMambaHierarchical.EnhancedTargetProcessor.__call__', side_effect=Exception("Mock Processor Failure"))
    def test_improvedmambahierarchical_target_fallback(self, mock_processor):
        """Tests ImprovedMambaHierarchical target processor fallback."""
        model = ImprovedMambaHierarchical(self.configs)
        try:
            output = model(self.x_enc, self.x_mark_enc, self.x_dec, self.x_mark_dec)
            self.assertEqual(output.shape, (2, self.configs.pred_len, self.configs.c_out))
        except Exception as e:
            self.fail(f"ImprovedMambaHierarchical forward pass failed during fallback: {e}")

    @patch('models.ImprovedMambaHierarchical.CovariateProcessor.__call__', side_effect=Exception("Mock Processor Failure"))
    def test_improvedmambahierarchical_covariate_fallback(self, mock_processor):
        """Tests ImprovedMambaHierarchical covariate processor fallback."""
        model = ImprovedMambaHierarchical(self.configs)
        try:
            output = model(self.x_enc, self.x_mark_enc, self.x_dec, self.x_mark_dec)
            self.assertEqual(output.shape, (2, self.configs.pred_len, self.configs.c_out))
        except Exception as e:
            self.fail(f"ImprovedMambaHierarchical forward pass failed during fallback: {e}")

if __name__ == '__main__':
    unittest.main()