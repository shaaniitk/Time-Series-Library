
import unittest
import torch
from argparse import Namespace
from models.modular_autoformer import ModularAutoformer
from configs.autoformer import (
    standard_config,
    enhanced_config,
    fixed_config,
    enhanced_fixed_config,
    bayesian_enhanced_config,
    hierarchical_config,
    quantile_bayesian_config,
)

class TestModularAutoformerIntegration(unittest.TestCase):

    def setUp(self):
        self.num_targets = 7
        self.num_covariates = 3
        self.seq_len = 96
        self.pred_len = 24
        self.label_len = 48
        self.batch_size = 2

        self.configs_to_test = {
            "standard": standard_config.get_standard_autoformer_config,
            "fixed": fixed_config.get_fixed_autoformer_config,
            "enhanced": enhanced_config.get_enhanced_autoformer_config,
            "enhanced_fixed": enhanced_fixed_config.get_enhanced_fixed_autoformer_config,
            "bayesian_enhanced": bayesian_enhanced_config.get_bayesian_enhanced_autoformer_config,
            "hierarchical": hierarchical_config.get_hierarchical_autoformer_config,
            "quantile_bayesian": quantile_bayesian_config.get_quantile_bayesian_autoformer_config,
        }

    def test_model_instantiation_and_forward_pass(self):
        for name, config_func in self.configs_to_test.items():
            with self.subTest(name=name):
                print(f"--- [DEBUG] Testing configuration: {name}")
                
                # Get config
                config = config_func(num_targets=self.num_targets, num_covariates=self.num_covariates, 
                                     seq_len=self.seq_len, pred_len=self.pred_len, label_len=self.label_len,
                                     quantile_levels=[0.1, 0.5, 0.9]) # Explicitly pass 3 quantiles
                
                print(f"--- [DEBUG] Config object created: {config}")

                # Instantiate model
                model = ModularAutoformer(config)
                model.eval()

                # Create dummy data
                x_enc = torch.randn(self.batch_size, self.seq_len, config.enc_in)
                x_mark_enc = torch.randn(self.batch_size, self.seq_len, 4)
                
                dec_input_len = self.label_len + self.pred_len
                x_dec = torch.randn(self.batch_size, dec_input_len, config.dec_in)
                x_mark_dec = torch.randn(self.batch_size, dec_input_len, 4)

                # Test forward pass
                with torch.no_grad():
                    output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
                
                expected_c_out = config.c_out
                if 'quantiles' in config.loss_params:
                    expected_c_out = config.c_out_evaluation * len(config.loss_params['quantiles'])
                
                print(f"--- [DEBUG] Expected shape: {(self.batch_size, self.pred_len, expected_c_out)}")
                
                self.assertEqual(output.shape, (self.batch_size, self.pred_len, expected_c_out))
                print(f"--- {name} configuration passed ---")

if __name__ == '__main__':
    unittest.main()
