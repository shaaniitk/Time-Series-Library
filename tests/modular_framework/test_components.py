
import unittest
import torch
from layers.modular.decomposition import get_decomposition_component
from layers.modular.attention import get_attention_component
from layers.modular.encoder import get_encoder_component
from layers.modular.decoder import get_decoder_component
from layers.modular.sampling import get_sampling_component
from layers.modular.output_heads import get_output_head_component

class TestModularComponents(unittest.TestCase):

    def setUp(self):
        self.batch_size = 2
        self.seq_len = 96
        self.d_model = 16
        self.c_out = 7
        self.n_heads = 4
        self.d_ff = 32
        self.e_layers = 2
        self.d_layers = 1
        self.num_quantiles = 3

        self.dummy_input = torch.randn(self.batch_size, self.seq_len, self.d_model)
        self.dummy_cross_input = torch.randn(self.batch_size, self.seq_len, self.d_model)

    def test_decomposition_components(self):
        for name in ["series_decomp", "stable_decomp", "learnable_decomp", "wavelet_decomp"]:
            with self.subTest(name=name):
                params = {'kernel_size': 25} if 'wavelet' not in name else {'seq_len': self.seq_len, 'd_model': self.d_model}
                if name == 'learnable_decomp':
                    params = {'input_dim': self.d_model}
                
                component = get_decomposition_component(name, **params)
                seasonal, trend = component(self.dummy_input)
                self.assertEqual(seasonal.shape, self.dummy_input.shape)
                self.assertEqual(trend.shape, self.dummy_input.shape)

    def test_attention_components(self):
        for name in ["autocorrelation_layer"]:
            with self.subTest(name=name):
                params = {'d_model': self.d_model, 'n_heads': self.n_heads, 'dropout': 0.1, 'factor': 1}
                component = get_attention_component(name, **params)
                output, attn_weights = component(self.dummy_input, self.dummy_input, self.dummy_input, attn_mask=None)
                self.assertEqual(output.shape, self.dummy_input.shape)

    def test_encoder_decoder_components(self):
        attention_comp = get_attention_component('autocorrelation_layer', d_model=self.d_model, n_heads=self.n_heads, dropout=0.1)
        decomp_comp = get_decomposition_component('series_decomp', kernel_size=25)
        
        for name in ["standard", "enhanced", "stable"]:
            with self.subTest(name=name):
                encoder = get_encoder_component(name, e_layers=self.e_layers, d_model=self.d_model, n_heads=self.n_heads, d_ff=self.d_ff, dropout=0.1, activation='relu', attention_comp=attention_comp, decomp_comp=decomp_comp)
                output, _ = encoder(self.dummy_input)
                self.assertEqual(output.shape, self.dummy_input.shape)

                decoder = get_decoder_component(name, d_layers=self.d_layers, d_model=self.d_model, c_out=self.c_out, n_heads=self.n_heads, d_ff=self.d_ff, dropout=0.1, activation='relu', self_attention_comp=attention_comp, cross_attention_comp=attention_comp, decomp_comp=decomp_comp)
                
                # Provide an initial trend tensor with the correct output dimension
                initial_trend = torch.zeros(self.batch_size, self.seq_len, self.c_out)
                seasonal, trend = decoder(self.dummy_input, self.dummy_cross_input, trend=initial_trend)
                self.assertEqual(seasonal.shape, self.dummy_input.shape)

    def test_sampling_components(self):
        class MockModel(torch.nn.Module):
            def __init__(self, c_out, seq_len, batch_size):
                super().__init__()
                self.c_out = c_out
                self.seq_len = seq_len
                self.batch_size = batch_size
            def forward(self, *args):
                return torch.randn(self.batch_size, self.seq_len, self.c_out)

        mock_model = MockModel(self.c_out, self.seq_len, self.batch_size)

        for name in ["deterministic", "bayesian", "dropout"]:
            with self.subTest(name=name):
                params = {}
                if name != "deterministic":
                    params['n_samples'] = 2
                
                component = get_sampling_component(name, **params)
                result = component(mock_model.forward, None, None, None, None)
                self.assertIn('prediction', result)
                self.assertEqual(result['prediction'].shape, (self.batch_size, self.seq_len, self.c_out))

    def test_output_head_components(self):
        for name in ["standard", "quantile"]:
            with self.subTest(name=name):
                params = {'d_model': self.d_model, 'c_out': self.c_out}
                if name == 'quantile':
                    params['num_quantiles'] = self.num_quantiles
                
                component = get_output_head_component(name, **params)
                output = component(self.dummy_input)
                
                expected_dim = self.c_out * self.num_quantiles if name == 'quantile' else self.c_out
                self.assertEqual(output.shape, (self.batch_size, self.seq_len, expected_dim))

if __name__ == '__main__':
    unittest.main()
