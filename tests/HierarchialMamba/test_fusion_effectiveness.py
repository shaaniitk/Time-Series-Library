import unittest
import torch
from argparse import Namespace
from copy import deepcopy

from models.ImprovedMambaHierarchical import ImprovedMambaHierarchical

class TestImprovedMambaHierarchicalFusion(unittest.TestCase):
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
            use_moe=False,  # Disable MoE for this specific test to isolate fusion
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

    def test_fusion_is_effective_after_fix(self):
        """
        Tests if the DualCrossAttention output is actually used after the fix.
        Effectiveness Check: Compares the output of a normal forward pass with a pass
        where the attention mechanism is programmatically bypassed. If the outputs
        are different, it proves the fusion mechanism is active and influential.
        """
        # --- Run with the fusion mechanism active ---
        model = ImprovedMambaHierarchical(self.configs)
        model.eval()
        with torch.no_grad():
            output_with_fusion = model(self.x_enc, self.x_mark_enc, self.x_dec, self.x_mark_dec)

        # --- Run with the fusion mechanism programmatically bypassed ---
        # Create a deepcopy to ensure weights are identical
        model_bypassed = deepcopy(model)
        model_bypassed.eval()

        # Mock the cross-attention to return the original contexts, simulating the old bug
        def bypass_attention(target_context, covariate_context, *args, **kwargs):
            # Return original contexts and a simple average for the fused_context
            return (target_context + covariate_context) / 2, target_context, covariate_context

        model_bypassed.dual_cross_attention.forward = bypass_attention

        with torch.no_grad():
            output_without_fusion = model_bypassed(self.x_enc, self.x_mark_enc, self.x_dec, self.x_mark_dec)

        # --- Assert that the outputs are different ---
        # If the fusion step is effective, the outputs should not be identical.
        self.assertFalse(
            torch.allclose(output_with_fusion, output_without_fusion, atol=1e-6),
            "The output with and without fusion are identical. The cross-attention context is still not being used effectively."
        )
        
        # Also check that the shapes are correct
        self.assertEqual(output_with_fusion.shape, (2, self.configs.pred_len, self.configs.c_out))
        self.assertEqual(output_without_fusion.shape, (2, self.configs.pred_len, self.configs.c_out))

if __name__ == '__main__':
    unittest.main()

