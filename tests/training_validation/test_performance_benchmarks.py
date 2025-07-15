import pytest
import torch
import torch.nn as nn
import numpy as np
from types import SimpleNamespace
import sys
import os
import time
import psutil
import gc
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.Autoformer_Fixed import Model as AutoformerFixed
from models.EnhancedAutoformer_Fixed import EnhancedAutoformer


class TestPerformanceBenchmarks:
    """Test performance benchmarks for Autoformer models"""
    
    @pytest.fixture
    def base_config(self):
        config = SimpleNamespace()
        config.task_name = 'long_term_forecast'
        config.seq_len = 96
        config.label_len = 48
        config.pred_len = 24
        config.enc_in = 7
        config.dec_in = 7
        config.c_out = 7
        config.d_model = 64
        config.n_heads = 8
        config.e_layers = 2
        config.d_layers = 1
        config.d_ff = 256
        config.moving_avg = 25
        config.factor = 1
        config.dropout = 0.1
        config.activation = 'gelu'
        config.embed = 'timeF'
        config.freq = 'h'
        config.norm_type = 'LayerNorm'
        return config
    
    def measure_memory_usage(self, func):
        """Measure memory usage of a function"""
        # Force garbage collection
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Get baseline memory usage
        process = psutil.Process(os.getpid())
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run function
        result = func()
        
        # Force garbage collection again
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Get peak memory usage
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = peak_memory - baseline_memory
        
        return result, memory_used
    
    def test_inference_speed(self, base_config):
        """Test inference speed with different sequence lengths"""
        model = AutoformerFixed(base_config)
        model.eval()
        
        batch_size = 16
        seq_lengths = [96, 192, 384]
        times = {}
        
        for seq_len in seq_lengths:
            # Update config
            base_config.seq_len = seq_len
            
            # Create inputs
            x_enc = torch.randn(batch_size, seq_len, base_config.enc_in)
            x_mark_enc = torch.randn(batch_size, seq_len, 4)
            x_dec = torch.randn(batch_size, base_config.label_len + base_config.pred_len, base_config.dec_in)
            x_mark_dec = torch.randn(batch_size, base_config.label_len + base_config.pred_len, 4)
            
            # Warmup
            with torch.no_grad():
                _ = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            
            # Measure time
            start_time = time.time()
            with torch.no_grad():
                for _ in range(5):
                    _ = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            end_time = time.time()
            
            avg_time = (end_time - start_time) / 5
            times[seq_len] = avg_time
            
            print(f"Inference time for seq_len={seq_len}: {avg_time:.4f}s")
        
        # Verify that longer sequences take more time
        assert times[96] < times[384], "Longer sequences should take more time"
        
        # Check that time complexity is reasonable (should be less than quadratic)
        time_ratio = times[384] / times[96]
        seq_ratio = 384 / 96
        assert time_ratio < seq_ratio ** 2, "Time complexity should be less than quadratic"
    
    def test_memory_usage(self, base_config):
        """Test memory usage with different model sizes"""
        model_configs = [
            # Small model
            {"d_model": 32, "n_heads": 4, "e_layers": 1, "d_layers": 1, "d_ff": 64},
            # Medium model
            {"d_model": 64, "n_heads": 8, "e_layers": 2, "d_layers": 1, "d_ff": 256},
            # Large model
            {"d_model": 128, "n_heads": 8, "e_layers": 3, "d_layers": 2, "d_ff": 512}
        ]
        
        batch_size = 16
        seq_len = 96
        memory_usages = []
        
        for config_updates in model_configs:
            # Update config
            for key, value in config_updates.items():
                setattr(base_config, key, value)
            
            # Create model and inputs
            def create_and_run_model():
                model = AutoformerFixed(base_config)
                model.eval()
                
                x_enc = torch.randn(batch_size, seq_len, base_config.enc_in)
                x_mark_enc = torch.randn(batch_size, seq_len, 4)
                x_dec = torch.randn(batch_size, base_config.label_len + base_config.pred_len, base_config.dec_in)
                x_mark_dec = torch.randn(batch_size, base_config.label_len + base_config.pred_len, 4)
                
                with torch.no_grad():
                    output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
                
                return output
            
            # Measure memory usage
            _, memory_used = self.measure_memory_usage(create_and_run_model)
            memory_usages.append(memory_used)
            
            print(f"Memory usage for {config_updates}: {memory_used:.2f} MB")
        
        # Verify that larger models use more memory
        assert memory_usages[0] < memory_usages[2], "Larger models should use more memory"
    
    def test_batch_size_scaling(self, base_config):
        """Test performance scaling with batch size"""
        model = AutoformerFixed(base_config)
        model.eval()
        
        seq_len = 96
        batch_sizes = [1, 4, 16, 32]
        times = {}
        
        for batch_size in batch_sizes:
            # Create inputs
            x_enc = torch.randn(batch_size, seq_len, base_config.enc_in)
            x_mark_enc = torch.randn(batch_size, seq_len, 4)
            x_dec = torch.randn(batch_size, base_config.label_len + base_config.pred_len, base_config.dec_in)
            x_mark_dec = torch.randn(batch_size, base_config.label_len + base_config.pred_len, 4)
            
            # Warmup
            with torch.no_grad():
                _ = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            
            # Measure time
            start_time = time.time()
            with torch.no_grad():
                for _ in range(3):
                    _ = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            end_time = time.time()
            
            avg_time = (end_time - start_time) / 3
            times[batch_size] = avg_time
            
            print(f"Inference time for batch_size={batch_size}: {avg_time:.4f}s")
        
        # Verify that larger batches take more time
        assert times[1] < times[32], "Larger batches should take more time"
        
        # Check that batch processing is efficient (should be sublinear)
        time_ratio = times[32] / times[1]
        batch_ratio = 32 / 1
        assert time_ratio < batch_ratio, "Batch processing should be efficient"
    
    def test_model_comparison(self, base_config):
        """Compare performance between standard and enhanced Autoformer"""
        # Create models
        standard_model = AutoformerFixed(base_config)
        enhanced_model = EnhancedAutoformer(base_config)
        
        batch_size = 16
        seq_len = 96
        
        # Create inputs
        x_enc = torch.randn(batch_size, seq_len, base_config.enc_in)
        x_mark_enc = torch.randn(batch_size, seq_len, 4)
        x_dec = torch.randn(batch_size, base_config.label_len + base_config.pred_len, base_config.dec_in)
        x_mark_dec = torch.randn(batch_size, base_config.label_len + base_config.pred_len, 4)
        
        # Measure standard model performance
        standard_model.eval()
        start_time = time.time()
        with torch.no_grad():
            for _ in range(5):
                _ = standard_model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        standard_time = (time.time() - start_time) / 5
        
        # Measure enhanced model performance
        enhanced_model.eval()
        start_time = time.time()
        with torch.no_grad():
            for _ in range(5):
                _ = enhanced_model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        enhanced_time = (time.time() - start_time) / 5
        
        print(f"Standard model time: {standard_time:.4f}s")
        print(f"Enhanced model time: {enhanced_time:.4f}s")
        
        # Enhanced model should be within reasonable performance range
        time_ratio = enhanced_time / standard_time
        assert 0.5 < time_ratio < 2.0, "Enhanced model should have reasonable performance"
    
    def test_parameter_count(self, base_config):
        """Test parameter count for different model configurations"""
        model_configs = [
            # Small model
            {"d_model": 32, "n_heads": 4, "e_layers": 1, "d_layers": 1, "d_ff": 64},
            # Medium model
            {"d_model": 64, "n_heads": 8, "e_layers": 2, "d_layers": 1, "d_ff": 256},
            # Large model
            {"d_model": 128, "n_heads": 8, "e_layers": 3, "d_layers": 2, "d_ff": 512}
        ]
        
        param_counts = []
        
        for config_updates in model_configs:
            # Update config
            for key, value in config_updates.items():
                setattr(base_config, key, value)
            
            # Create model
            model = AutoformerFixed(base_config)
            
            # Count parameters
            param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            param_counts.append(param_count)
            
            print(f"Parameter count for {config_updates}: {param_count:,}")
        
        # Verify that larger models have more parameters
        assert param_counts[0] < param_counts[1] < param_counts[2], "Parameter count should increase with model size"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])