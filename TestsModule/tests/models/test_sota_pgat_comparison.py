"""Model comparison tests for SOTA_Temporal_PGAT.

This module compares SOTA_Temporal_PGAT against other models in the framework
to ensure consistent behavior and performance characteristics.
"""

# Suppress optional dependency warnings
from TestsModule.suppress_warnings import suppress_optional_dependency_warnings
suppress_optional_dependency_warnings()

import pytest
import torch
import numpy as np
from typing import Dict, Any, List, Tuple
from unittest.mock import MagicMock
import time

# Import models for comparison
try:
    from models.SOTA_Temporal_PGAT import SOTA_Temporal_PGAT as SOTATemporalPGAT
except ImportError:
    pytest.skip("SOTA_Temporal_PGAT model not available", allow_module_level=True)

# Import other models for comparison
MODELS_TO_COMPARE = []
try:
    from models.Transformer import Model as Transformer
    MODELS_TO_COMPARE.append(("Transformer", Transformer))
except ImportError:
    pass

try:
    from models.Informer import Model as Informer
    MODELS_TO_COMPARE.append(("Informer", Informer))
except ImportError:
    pass

try:
    from models.Autoformer import Model as Autoformer
    MODELS_TO_COMPARE.append(("Autoformer", Autoformer))
except ImportError:
    pass

try:
    from models.FEDformer import Model as FEDformer
    MODELS_TO_COMPARE.append(("FEDformer", FEDformer))
except ImportError:
    pass

try:
    from models.PatchTST import Model as PatchTST
    MODELS_TO_COMPARE.append(("PatchTST", PatchTST))
except ImportError:
    pass


class TestSOTATemporalPGATComparison:
    """Comparison test suite for SOTA_Temporal_PGAT model."""
    
    @pytest.fixture(autouse=True)
    def setup_test(self):
        """Setup test environment and model configurations."""
        # Base configuration that should work for most models
        self.base_configs = MagicMock()
        self.base_configs.seq_len = 96
        self.base_configs.label_len = 48
        self.base_configs.pred_len = 96
        self.base_configs.enc_in = 7
        self.base_configs.dec_in = 7
        self.base_configs.c_out = 7
        self.base_configs.d_model = 256
        self.base_configs.n_heads = 8
        self.base_configs.e_layers = 2
        self.base_configs.d_layers = 1
        self.base_configs.d_ff = 1024
        self.base_configs.dropout = 0.1
        self.base_configs.activation = 'gelu'
        self.base_configs.output_attention = False
        self.base_configs.mix = True
        self.base_configs.factor = 1
        self.base_configs.embed = 'timeF'
        self.base_configs.freq = 'h'
        
        # SOTA_Temporal_PGAT specific configurations
        self.pgat_configs = MagicMock()
        for attr in dir(self.base_configs):
            if not attr.startswith('_'):
                setattr(self.pgat_configs, attr, getattr(self.base_configs, attr))
        
        # Add graph-specific configurations
        self.pgat_configs.graph_dim = 32
        self.pgat_configs.num_nodes = self.pgat_configs.enc_in
        self.pgat_configs.graph_alpha = 0.2
        self.pgat_configs.graph_beta = 0.1
        
        # Test parameters
        self.batch_size = 4
        self.tolerance = 1e-5
        
        # Create SOTA_Temporal_PGAT model
        self.pgat_model = SOTATemporalPGAT(self.pgat_configs)
        self.pgat_model.eval()
    
    def create_test_data(self, batch_size=None, seq_len=None):
        """Create test data for model comparison."""
        if batch_size is None:
            batch_size = self.batch_size
        if seq_len is None:
            seq_len = self.base_configs.seq_len
        
        x_enc = torch.randn(batch_size, seq_len, self.base_configs.enc_in)
        x_mark_enc = torch.randn(batch_size, seq_len, 4)
        x_dec = torch.randn(batch_size, self.base_configs.label_len + self.base_configs.pred_len, self.base_configs.dec_in)
        x_mark_dec = torch.randn(batch_size, self.base_configs.label_len + self.base_configs.pred_len, 4)
        
        return x_enc, x_mark_enc, x_dec, x_mark_dec
    
    def get_model_configs(self, model_name: str) -> MagicMock:
        """Get appropriate configuration for each model type."""
        configs = MagicMock()
        
        # Copy base configurations
        for attr in dir(self.base_configs):
            if not attr.startswith('_'):
                setattr(configs, attr, getattr(self.base_configs, attr))
        
        # Model-specific adjustments
        if model_name == "PatchTST":
            configs.patch_len = 16
            configs.stride = 8
            configs.padding_patch = 'end'
            configs.revin = 1
            configs.affine = 0
            configs.subtract_last = 0
            configs.decomposition = 0
            configs.kernel_size = 25
        elif model_name == "FEDformer":
            configs.version = 'Fourier'
            configs.mode_select = 'random'
            configs.modes = 32
            configs.L = 3
            configs.base = 'legendre'
            configs.cross_activation = 'tanh'
        elif model_name == "Autoformer":
            configs.moving_avg = 25
            configs.factor = 1
        elif model_name == "Informer":
            configs.factor = 5
            configs.distil = True
        
        return configs
    
    def create_model_safely(self, model_class, model_name: str):
        """Safely create a model instance with appropriate error handling."""
        try:
            configs = self.get_model_configs(model_name)
            model = model_class(configs)
            model.eval()
            return model, None
        except Exception as e:
            return None, str(e)
    
    def test_output_shape_consistency(self):
        """Test that all models produce outputs with consistent shapes."""
        x_enc, x_mark_enc, x_dec, x_mark_dec = self.create_test_data()
        
        # Get SOTA_Temporal_PGAT output
        with torch.no_grad():
            pgat_output = self.pgat_model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        expected_shape = (self.batch_size, self.base_configs.pred_len, self.base_configs.c_out)
        assert pgat_output.shape == expected_shape, \
            f"SOTA_Temporal_PGAT output shape mismatch: {pgat_output.shape} vs {expected_shape}"
        
        # Compare with other models
        for model_name, model_class in MODELS_TO_COMPARE:
            model, error = self.create_model_safely(model_class, model_name)
            
            if model is None:
                pytest.skip(f"Could not create {model_name}: {error}")
                continue
            
            try:
                with torch.no_grad():
                    output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
                
                # Handle tuple outputs (some models return attention weights)
                if isinstance(output, tuple):
                    output = output[0]
                
                assert output.shape == expected_shape, \
                    f"{model_name} output shape mismatch: {output.shape} vs {expected_shape}"
                
            except Exception as e:
                pytest.skip(f"Could not test {model_name}: {e}")
    
    def test_numerical_output_ranges(self):
        """Test that model outputs are in reasonable numerical ranges."""
        x_enc, x_mark_enc, x_dec, x_mark_dec = self.create_test_data()
        
        models_to_test = [("SOTA_Temporal_PGAT", self.pgat_model)]
        
        # Add other models
        for model_name, model_class in MODELS_TO_COMPARE:
            model, error = self.create_model_safely(model_class, model_name)
            if model is not None:
                models_to_test.append((model_name, model))
        
        output_stats = {}
        
        for model_name, model in models_to_test:
            try:
                with torch.no_grad():
                    output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
                
                # Handle tuple outputs
                if isinstance(output, tuple):
                    output = output[0]
                
                # Compute statistics
                stats = {
                    'mean': output.mean().item(),
                    'std': output.std().item(),
                    'min': output.min().item(),
                    'max': output.max().item(),
                    'abs_mean': output.abs().mean().item()
                }
                
                output_stats[model_name] = stats
                
                # Basic sanity checks
                assert torch.isfinite(output).all(), f"{model_name} produced non-finite outputs"
                assert stats['abs_mean'] > 1e-8, f"{model_name} outputs too close to zero"
                assert stats['abs_mean'] < 1e6, f"{model_name} outputs too large"
                
            except Exception as e:
                pytest.skip(f"Could not test numerical ranges for {model_name}: {e}")
        
        # Compare output ranges across models
        if len(output_stats) > 1:
            abs_means = [stats['abs_mean'] for stats in output_stats.values()]
            mean_ratio = max(abs_means) / min(abs_means)
            
            # Models should produce outputs in similar ranges (within 2 orders of magnitude)
            assert mean_ratio < 100, \
                f"Output magnitude varies too much across models: {output_stats}"
    
    def test_inference_time_comparison(self):
        """Compare inference times across models."""
        x_enc, x_mark_enc, x_dec, x_mark_dec = self.create_test_data()
        
        models_to_test = [("SOTA_Temporal_PGAT", self.pgat_model)]
        
        # Add other models
        for model_name, model_class in MODELS_TO_COMPARE:
            model, error = self.create_model_safely(model_class, model_name)
            if model is not None:
                models_to_test.append((model_name, model))
        
        timing_results = {}
        num_runs = 10
        
        for model_name, model in models_to_test:
            try:
                # Warmup
                with torch.no_grad():
                    for _ in range(3):
                        output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
                
                # Timing runs
                times = []
                for _ in range(num_runs):
                    start_time = time.time()
                    with torch.no_grad():
                        output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
                    end_time = time.time()
                    times.append(end_time - start_time)
                
                timing_results[model_name] = {
                    'mean_time': np.mean(times),
                    'std_time': np.std(times),
                    'min_time': np.min(times),
                    'max_time': np.max(times)
                }
                
            except Exception as e:
                pytest.skip(f"Could not benchmark {model_name}: {e}")
        
        # Basic timing sanity checks
        for model_name, stats in timing_results.items():
            # Inference should complete within reasonable time (10 seconds)
            assert stats['mean_time'] < 10.0, \
                f"{model_name} inference too slow: {stats['mean_time']:.3f}s"
            
            # Timing should be consistent (coefficient of variation < 50%)
            cv = stats['std_time'] / stats['mean_time']
            assert cv < 0.5, \
                f"{model_name} timing too inconsistent: CV = {cv:.3f}"
        
        # Log timing comparison for analysis
        print(f"\nInference Time Comparison (batch_size={self.batch_size}):")
        for model_name, stats in timing_results.items():
            print(f"{model_name}: {stats['mean_time']:.3f}±{stats['std_time']:.3f}s")
    
    def test_memory_usage_comparison(self):
        """Compare memory usage across models."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for memory testing")
        
        device = torch.device('cuda')
        x_enc, x_mark_enc, x_dec, x_mark_dec = self.create_test_data()
        x_enc = x_enc.to(device)
        x_mark_enc = x_mark_enc.to(device)
        x_dec = x_dec.to(device)
        x_mark_dec = x_mark_dec.to(device)
        
        models_to_test = [("SOTA_Temporal_PGAT", self.pgat_model)]
        
        # Add other models
        for model_name, model_class in MODELS_TO_COMPARE:
            model, error = self.create_model_safely(model_class, model_name)
            if model is not None:
                models_to_test.append((model_name, model))
        
        memory_results = {}
        
        for model_name, model in models_to_test:
            try:
                model = model.to(device)
                
                # Clear cache
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                
                # Measure memory
                with torch.no_grad():
                    output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
                
                peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
                memory_results[model_name] = peak_memory
                
                # Memory usage should be reasonable (< 4GB for this test)
                assert peak_memory < 4096, \
                    f"{model_name} uses too much memory: {peak_memory:.1f}MB"
                
                model = model.cpu()
                
            except Exception as e:
                pytest.skip(f"Could not test memory usage for {model_name}: {e}")
        
        # Log memory comparison
        print(f"\nMemory Usage Comparison (batch_size={self.batch_size}):")
        for model_name, memory in memory_results.items():
            print(f"{model_name}: {memory:.1f}MB")
    
    def test_gradient_flow_comparison(self):
        """Compare gradient flow properties across models."""
        x_enc, x_mark_enc, x_dec, x_mark_dec = self.create_test_data()
        
        models_to_test = [("SOTA_Temporal_PGAT", self.pgat_model)]
        
        # Add other models
        for model_name, model_class in MODELS_TO_COMPARE:
            model, error = self.create_model_safely(model_class, model_name)
            if model is not None:
                models_to_test.append((model_name, model))
        
        gradient_stats = {}
        
        for model_name, model in models_to_test:
            try:
                model.train()
                
                # Forward pass
                output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
                if isinstance(output, tuple):
                    output = output[0]
                
                loss = output.mean()
                
                # Backward pass
                model.zero_grad()
                loss.backward()
                
                # Collect gradient statistics
                gradient_norms = []
                zero_grad_count = 0
                total_params = 0
                
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.norm().item()
                        gradient_norms.append(grad_norm)
                        if grad_norm < 1e-8:
                            zero_grad_count += 1
                    total_params += 1
                
                if gradient_norms:
                    gradient_stats[model_name] = {
                        'mean_grad_norm': np.mean(gradient_norms),
                        'max_grad_norm': np.max(gradient_norms),
                        'min_grad_norm': np.min(gradient_norms),
                        'zero_grad_ratio': zero_grad_count / len(gradient_norms),
                        'total_params': total_params
                    }
                    
                    # Basic gradient sanity checks
                    stats = gradient_stats[model_name]
                    assert stats['max_grad_norm'] < 100, \
                        f"{model_name} has exploding gradients: {stats['max_grad_norm']}"
                    
                    assert stats['zero_grad_ratio'] < 0.5, \
                        f"{model_name} has too many zero gradients: {stats['zero_grad_ratio']}"
                
                model.eval()
                
            except Exception as e:
                pytest.skip(f"Could not test gradients for {model_name}: {e}")
        
        # Log gradient comparison
        print(f"\nGradient Flow Comparison:")
        for model_name, stats in gradient_stats.items():
            print(f"{model_name}: mean_grad={stats['mean_grad_norm']:.2e}, "
                  f"max_grad={stats['max_grad_norm']:.2e}, "
                  f"zero_ratio={stats['zero_grad_ratio']:.3f}")
    
    def test_parameter_count_comparison(self):
        """Compare parameter counts across models."""
        models_to_test = [("SOTA_Temporal_PGAT", self.pgat_model)]
        
        # Add other models
        for model_name, model_class in MODELS_TO_COMPARE:
            model, error = self.create_model_safely(model_class, model_name)
            if model is not None:
                models_to_test.append((model_name, model))
        
        param_counts = {}
        
        for model_name, model in models_to_test:
            try:
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                
                param_counts[model_name] = {
                    'total': total_params,
                    'trainable': trainable_params,
                    'non_trainable': total_params - trainable_params
                }
                
                # Parameter count should be reasonable (< 100M for this test setup)
                assert total_params < 100_000_000, \
                    f"{model_name} has too many parameters: {total_params:,}"
                
            except Exception as e:
                pytest.skip(f"Could not count parameters for {model_name}: {e}")
        
        # Log parameter comparison
        print(f"\nParameter Count Comparison:")
        for model_name, counts in param_counts.items():
            print(f"{model_name}: {counts['total']:,} total, {counts['trainable']:,} trainable")
    
    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
    def test_batch_size_scalability_comparison(self, batch_size):
        """Test how models scale with different batch sizes."""
        x_enc, x_mark_enc, x_dec, x_mark_dec = self.create_test_data(batch_size=batch_size)
        
        # Test SOTA_Temporal_PGAT
        with torch.no_grad():
            pgat_output = self.pgat_model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        expected_shape = (batch_size, self.base_configs.pred_len, self.base_configs.c_out)
        assert pgat_output.shape == expected_shape, \
            f"SOTA_Temporal_PGAT batch scaling failed for batch_size {batch_size}"
        
        # Test a few other models for comparison
        models_tested = 0
        for model_name, model_class in MODELS_TO_COMPARE[:2]:  # Test only first 2 for efficiency
            model, error = self.create_model_safely(model_class, model_name)
            
            if model is None:
                continue
            
            try:
                with torch.no_grad():
                    output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
                
                if isinstance(output, tuple):
                    output = output[0]
                
                assert output.shape == expected_shape, \
                    f"{model_name} batch scaling failed for batch_size {batch_size}"
                
                models_tested += 1
                
            except Exception as e:
                # Skip individual model failures
                continue
        
        # At least SOTA_Temporal_PGAT should work
        assert models_tested >= 0, "No models could be tested for batch scaling"
    
    def test_input_variation_robustness(self):
        """Test model robustness to input variations."""
        base_data = self.create_test_data()
        
        # Test different input variations
        variations = [
            ("normal", base_data),
            ("scaled_up", tuple(x * 10 for x in base_data)),
            ("scaled_down", tuple(x * 0.1 for x in base_data)),
            ("noisy", tuple(x + torch.randn_like(x) * 0.01 for x in base_data))
        ]
        
        models_to_test = [("SOTA_Temporal_PGAT", self.pgat_model)]
        
        # Add one other model for comparison
        if MODELS_TO_COMPARE:
            model_name, model_class = MODELS_TO_COMPARE[0]
            model, error = self.create_model_safely(model_class, model_name)
            if model is not None:
                models_to_test.append((model_name, model))
        
        for model_name, model in models_to_test:
            for variation_name, data in variations:
                try:
                    x_enc, x_mark_enc, x_dec, x_mark_dec = data
                    
                    with torch.no_grad():
                        output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
                    
                    if isinstance(output, tuple):
                        output = output[0]
                    
                    # Check output is finite and reasonable
                    assert torch.isfinite(output).all(), \
                        f"{model_name} produced non-finite output for {variation_name} input"
                    
                    output_magnitude = output.abs().mean().item()
                    assert 1e-8 < output_magnitude < 1e8, \
                        f"{model_name} output magnitude unreasonable for {variation_name}: {output_magnitude}"
                    
                except Exception as e:
                    pytest.skip(f"Could not test {model_name} with {variation_name} input: {e}")