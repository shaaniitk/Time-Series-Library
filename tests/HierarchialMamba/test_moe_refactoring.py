import unittest
import torch
from argparse import Namespace

from models.ImprovedMambaHierarchical import ImprovedMambaHierarchical

class TestMoERefactoring(unittest.TestCase):
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
            mamba_d_state=16,
            mamba_d_conv=4,
            mamba_expand=2,
            attention_heads=8,
            hierarchical_attention_heads=8,
            cross_attention_heads=8,
            fusion_strategy='weighted_concat',
            use_family_attention=True,
            use_moe=True,  # Enable MoE for this test
            dropout=0.1,
            trend_kernel_size=25,
            use_autoregressive=True,
            num_experts=4,
            covariate_families=None,
        )
        self.x_enc = torch.randn(2, 96, 44)
        self.x_dec = torch.randn(2, 72, 4)
        self.x_mark_enc = torch.randn(2, 96, 4)
        self.x_mark_dec = torch.randn(2, 72, 4)

    def test_moe_on_fused_context(self):
        """
        Tests if the refactored model correctly applies MoE to a single fused context.
        Effectiveness Check: Ensures the model runs without error, produces the correct
        output shape, and potentially returns an auxiliary loss, confirming the
        new architecture is functional.
        """
        model = ImprovedMambaHierarchical(self.configs)
        model.eval()
        
        try:
            output = model(self.x_enc, self.x_mark_enc, self.x_dec, self.x_mark_dec)
            aux_loss = model.get_auxiliary_loss()

            self.assertEqual(output.shape, (2, self.configs.pred_len, self.configs.c_out))
            # In eval mode, aux_loss might be None, but if it exists, it should be a tensor.
            if aux_loss is not None:
                self.assertIsInstance(aux_loss, torch.Tensor)
        except Exception as e:
            self.fail(f"Model forward pass failed after MoE refactoring: {e}")

if __name__ == '__main__':
    unittest.main()