"""
Performance and Benchmarking Tests for Modular Autoformer Framework

This file contains comprehensive performance tests, benchmarks, and
comparisons between modular and original implementations.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import time
import psutil
import gc
import sys
import os
from unittest.mock import Mock, patch
from memory_profiler import profile
import matplotlib.pyplot as plt
from types import SimpleNamespace

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestPerformanceBenchmarks:
    """Test performance characteristics of modular vs original models"""
    
    @pytest.fixture
    def benchmark_config(self):
        """Configuration for performance benchmarking"""
        return {
            'model_params': {
                'seq_len': 96,
                'pred_len': 24,
                'enc_in': 7,
                'dec_in': 7,
                'c_out': 7,
                'd_model': 512,
                'n_heads': 8,
                'e_layers': 2,
                'd_layers': 1,
                'dropout': 0.1
            },
            'test_sizes': [
                {'batch_size': 8, 'seq_len': 96},
                {'batch_size': 16, 'seq_len': 96},
                {'batch_size': 32, 'seq_len': 96},
                {'batch_size': 8, 'seq_len': 192},
                {'batch_size': 16, 'seq_len': 192}
            ],
            'n_runs': 10  # Number of runs for averaging
        }
    
    @pytest.fixture
    def performance_tracker(self):
        """Helper class for tracking performance metrics"""
        
        class PerformanceTracker:
            def __init__(self):
                self.reset()
                
            def reset(self):
                self.times = []
                self.memory_usage = []
                self.gpu_memory = []
                
            def start_timing(self):
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                self.start_time = time.perf_counter()
                self.start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                if torch.cuda.is_available():
                    self.start_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
                else:
                    self.start_gpu_memory = 0
                    
            def end_timing(self):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end_time = time.perf_counter()
                end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                if torch.cuda.is_available():
                    end_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
                else:
                    end_gpu_memory = 0
                
                elapsed_time = end_time - self.start_time
                memory_delta = end_memory - self.start_memory
                gpu_memory_delta = end_gpu_memory - self.start_gpu_memory
                
                self.times.append(elapsed_time)
                self.memory_usage.append(memory_delta)
                self.gpu_memory.append(gpu_memory_delta)
                
                return {
                    'time': elapsed_time,
                    'memory_delta': memory_delta,
                    'gpu_memory_delta': gpu_memory_delta
                }
            
            def get_stats(self):
                if not self.times:
                    return {}
                    
                return {
                    'time_mean': np.mean(self.times),
                    'time_std': np.std(self.times),
                    'time_min': np.min(self.times),
                    'time_max': np.max(self.times),
                    'memory_mean': np.mean(self.memory_usage),
                    'memory_std': np.std(self.memory_usage),
                    'gpu_memory_mean': np.mean(self.gpu_memory),
                    'gpu_memory_std': np.std(self.gpu_memory)
                }
        
        return PerformanceTracker()
    
    def test_forward_pass_performance(self, benchmark_config, performance_tracker):
        """Benchmark forward pass performance"""
        
        class MockOriginalModel:
            """Mock of original monolithic model"""
            def __init__(self, config):
                self.config = config
                # Simulate layers with actual PyTorch modules for realistic performance
                self.layers = nn.ModuleList([
                    nn.Linear(config['d_model'], config['d_model']) 
                    for _ in range(config['e_layers'] + config['d_layers'])
                ])
                self.projection = nn.Linear(config['d_model'], config['c_out'])
                
            def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
                batch_size = x_enc.shape[0]
                
                # Simulate processing through layers
                x = torch.randn(batch_size, self.config['d_model'])
                for layer in self.layers:
                    x = layer(x)
                    x = torch.relu(x)
                
                # Project to output
                output = self.projection(x)
                output = output.unsqueeze(1).repeat(1, self.config['pred_len'], 1)
                
                return output
        
        class MockModularModel:
            """Mock of modular model"""
            def __init__(self, config):
                self.config = config
                # Simulate component-based architecture
                self.decomposition = nn.Conv1d(1, 1, kernel_size=25, padding=12)
                self.attention = nn.MultiheadAttention(config['d_model'], config['n_heads'])
                self.encoder_layers = nn.ModuleList([
                    nn.Linear(config['d_model'], config['d_model']) 
                    for _ in range(config['e_layers'])
                ])
                self.decoder_layers = nn.ModuleList([
                    nn.Linear(config['d_model'], config['d_model']) 
                    for _ in range(config['d_layers'])
                ])
                self.projection = nn.Linear(config['d_model'], config['c_out'])
                
            def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
                batch_size = x_enc.shape[0]
                
                # Simulate decomposition
                decomp_input = x_enc.mean(dim=-1, keepdim=True).transpose(1, 2)
                decomp_output = self.decomposition(decomp_input)
                
                # Simulate attention
                x = torch.randn(batch_size, self.config['d_model'])
                attn_output, _ = self.attention(x.unsqueeze(1), x.unsqueeze(1), x.unsqueeze(1))
                x = attn_output.squeeze(1)
                
                # Process through encoder layers
                for layer in self.encoder_layers:
                    x = layer(x)
                    x = torch.relu(x)
                
                # Process through decoder layers
                for layer in self.decoder_layers:
                    x = layer(x)
                    x = torch.relu(x)
                
                # Project to output
                output = self.projection(x)
                output = output.unsqueeze(1).repeat(1, self.config['pred_len'], 1)
                
                return output
        
        # Test different model sizes
        results = {}
        
        for test_config in benchmark_config['test_sizes']:
            batch_size = test_config['batch_size']
            seq_len = test_config['seq_len']
            
            # Create test data
            x_enc = torch.randn(batch_size, seq_len, benchmark_config['model_params']['enc_in'])
            x_mark_enc = torch.randn(batch_size, seq_len, 4)
            x_dec = torch.randn(batch_size, 48 + benchmark_config['model_params']['pred_len'], 
                              benchmark_config['model_params']['dec_in'])
            x_mark_dec = torch.randn(batch_size, 48 + benchmark_config['model_params']['pred_len'], 4)
            
            # Test original model
            original_model = MockOriginalModel(benchmark_config['model_params'])
            original_model.eval()
            
            performance_tracker.reset()
            for _ in range(benchmark_config['n_runs']):
                performance_tracker.start_timing()
                with torch.no_grad():
                    _ = original_model.forward(x_enc, x_mark_enc, x_dec, x_mark_dec)
                performance_tracker.end_timing()
            
            original_stats = performance_tracker.get_stats()
            
            # Test modular model
            modular_model = MockModularModel(benchmark_config['model_params'])
            modular_model.eval()
            
            performance_tracker.reset()
            for _ in range(benchmark_config['n_runs']):
                performance_tracker.start_timing()
                with torch.no_grad():
                    _ = modular_model.forward(x_enc, x_mark_enc, x_dec, x_mark_dec)
                performance_tracker.end_timing()
            
            modular_stats = performance_tracker.get_stats()
            
            # Store results
            test_name = f"batch_{batch_size}_seq_{seq_len}"
            results[test_name] = {
                'original': original_stats,
                'modular': modular_stats,
                'config': test_config
            }
            
            # Calculate performance ratio
            time_ratio = modular_stats['time_mean'] / original_stats['time_mean']
            memory_ratio = (modular_stats['memory_mean'] / original_stats['memory_mean'] 
                          if original_stats['memory_mean'] != 0 else 1.0)
            
            print(f"Test {test_name}:")
            print(f"  Time ratio (modular/original): {time_ratio:.3f}")
            print(f"  Memory ratio (modular/original): {memory_ratio:.3f}")
            
            # Performance should be reasonable (within 2x of original)
            assert time_ratio < 2.0, f"Modular model too slow: {time_ratio}x slower"
            
        return results
    
    def test_memory_efficiency(self, benchmark_config):
        """Test memory efficiency of modular components"""
        
        class MemoryEfficientComponent:
            """Component designed for memory efficiency"""
            def __init__(self, config):
                self.config = config
                self.cached_computations = {}
                
            def forward_with_caching(self, x, cache_key=None):
                if cache_key and cache_key in self.cached_computations:
                    return self.cached_computations[cache_key]
                
                # Simulate computation
                result = torch.randn_like(x)
                
                if cache_key:
                    self.cached_computations[cache_key] = result
                
                return result
            
            def clear_cache(self):
                self.cached_computations.clear()
            
            def get_memory_usage(self):
                total_elements = sum(tensor.numel() for tensor in self.cached_computations.values())
                return total_elements * 4  # Assuming float32 (4 bytes per element)
        
        # Test memory efficiency with caching
        component = MemoryEfficientComponent(benchmark_config['model_params'])
        
        # Test without caching
        x = torch.randn(32, 96, 512)
        
        initial_memory = psutil.Process().memory_info().rss
        
        for i in range(10):
            _ = component.forward_with_caching(x)
        
        no_cache_memory = psutil.Process().memory_info().rss
        
        # Test with caching
        component.clear_cache()
        
        for i in range(10):
            cache_key = f"computation_{i % 3}"  # Only 3 unique keys, so caching should help
            _ = component.forward_with_caching(x, cache_key)
        
        with_cache_memory = psutil.Process().memory_info().rss
        
        # Calculate memory usage
        no_cache_delta = no_cache_memory - initial_memory
        with_cache_delta = with_cache_memory - no_cache_memory
        
        print(f"Memory without caching: {no_cache_delta / 1024 / 1024:.2f} MB")
        print(f"Memory with caching: {with_cache_delta / 1024 / 1024:.2f} MB")
        print(f"Cache memory usage: {component.get_memory_usage() / 1024 / 1024:.2f} MB")
        
        # Caching should reduce overall memory growth for repeated computations
        # (This is a simplified test - real caching benefits depend on usage patterns)
        assert component.get_memory_usage() > 0  # Cache should contain something
    
    def test_scalability_analysis(self, benchmark_config, performance_tracker):
        """Test how performance scales with input size"""
        
        class ScalableModel:
            def __init__(self, config):
                self.linear = nn.Linear(config['d_model'], config['c_out'])
                
            def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
                batch_size, seq_len, features = x_enc.shape
                
                # O(n) operation
                output = self.linear(x_enc.mean(dim=1))
                output = output.unsqueeze(1).repeat(1, 24, 1)
                
                return output
        
        # Test scaling with different input sizes
        scaling_results = []
        
        model = ScalableModel(benchmark_config['model_params'])
        model.eval()
        
        # Test different sequence lengths
        seq_lengths = [48, 96, 192, 384, 768]
        batch_size = 16
        
        for seq_len in seq_lengths:
            x_enc = torch.randn(batch_size, seq_len, 7)
            x_mark_enc = torch.randn(batch_size, seq_len, 4)
            x_dec = torch.randn(batch_size, 72, 7)
            x_mark_dec = torch.randn(batch_size, 72, 4)
            
            performance_tracker.reset()
            for _ in range(5):  # Fewer runs for scaling test
                performance_tracker.start_timing()
                with torch.no_grad():
                    _ = model.forward(x_enc, x_mark_enc, x_dec, x_mark_dec)
                performance_tracker.end_timing()
            
            stats = performance_tracker.get_stats()
            scaling_results.append({
                'seq_len': seq_len,
                'time_mean': stats['time_mean'],
                'memory_mean': stats['memory_mean']
            })
        
        # Analyze scaling behavior
        for i, result in enumerate(scaling_results):
            print(f"Seq length {result['seq_len']}: {result['time_mean']:.4f}s, "
                  f"{result['memory_mean']:.2f}MB")
        
        # Check that scaling is reasonable (should be roughly linear or better)
        if len(scaling_results) >= 3:
            times = [r['time_mean'] for r in scaling_results]
            seq_lens = [r['seq_len'] for r in scaling_results]
            
            # Calculate scaling factor (time ratio vs size ratio)
            time_ratio = times[-1] / times[0]
            size_ratio = seq_lens[-1] / seq_lens[0]
            scaling_factor = time_ratio / size_ratio
            
            print(f"Scaling factor: {scaling_factor:.3f} (1.0 = linear scaling)")
            
            # Should scale better than quadratic
            assert scaling_factor < size_ratio, "Scaling worse than quadratic"
    
    def test_component_overhead_analysis(self):
        """Analyze overhead introduced by modular architecture"""
        
        class DirectImplementation:
            """Direct implementation without modularity"""
            def __init__(self):
                self.weight = nn.Parameter(torch.randn(512, 512))
                
            def forward(self, x):
                return torch.matmul(x, self.weight)
        
        class ModularImplementation:
            """Modular implementation with component abstractions"""
            def __init__(self):
                self.component_registry = {}
                self.component_registry['linear'] = nn.Linear(512, 512)
                
            def get_component(self, name):
                return self.component_registry[name]
            
            def forward(self, x):
                linear_component = self.get_component('linear')
                return linear_component(x)
        
        # Benchmark both implementations
        x = torch.randn(32, 512)
        
        direct_model = DirectImplementation()
        modular_model = ModularImplementation()
        
        # Warm up
        for _ in range(10):
            _ = direct_model.forward(x)
            _ = modular_model.forward(x)
        
        # Benchmark direct implementation
        start_time = time.perf_counter()
        for _ in range(1000):
            _ = direct_model.forward(x)
        direct_time = time.perf_counter() - start_time
        
        # Benchmark modular implementation
        start_time = time.perf_counter()
        for _ in range(1000):
            _ = modular_model.forward(x)
        modular_time = time.perf_counter() - start_time
        
        overhead_ratio = modular_time / direct_time
        overhead_percent = (overhead_ratio - 1.0) * 100
        
        print(f"Direct implementation: {direct_time:.4f}s")
        print(f"Modular implementation: {modular_time:.4f}s")
        print(f"Overhead: {overhead_percent:.2f}%")
        
        # Overhead should be minimal (less than 50%)
        assert overhead_ratio < 1.5, f"Too much overhead: {overhead_percent:.2f}%"


class TestConcurrencyAndParallelism:
    """Test concurrent and parallel execution capabilities"""
    
    def test_component_parallel_execution(self):
        """Test parallel execution of independent components"""
        
        import threading
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        class ParallelizableComponent:
            def __init__(self, component_id):
                self.component_id = component_id
                self.execution_count = 0
                
            def process(self, data):
                """Simulate component processing"""
                self.execution_count += 1
                
                # Simulate computation time
                computation_time = 0.01  # 10ms
                time.sleep(computation_time)
                
                # Return processed data with component ID
                return {
                    'component_id': self.component_id,
                    'data': data * 2,  # Simple transformation
                    'execution_count': self.execution_count
                }
        
        class ParallelComponentExecutor:
            def __init__(self, components):
                self.components = components
                
            def execute_sequential(self, data):
                """Execute components sequentially"""
                results = []
                start_time = time.perf_counter()
                
                for component in self.components:
                    result = component.process(data)
                    results.append(result)
                
                end_time = time.perf_counter()
                return results, end_time - start_time
            
            def execute_parallel(self, data):
                """Execute components in parallel"""
                results = []
                start_time = time.perf_counter()
                
                with ThreadPoolExecutor(max_workers=len(self.components)) as executor:
                    # Submit all component tasks
                    future_to_component = {
                        executor.submit(component.process, data): component 
                        for component in self.components
                    }
                    
                    # Collect results as they complete
                    for future in as_completed(future_to_component):
                        result = future.result()
                        results.append(result)
                
                end_time = time.perf_counter()
                return results, end_time - start_time
        
        # Test parallel execution
        components = [ParallelizableComponent(i) for i in range(4)]
        executor = ParallelComponentExecutor(components)
        
        test_data = torch.randn(100, 512)
        
        # Test sequential execution
        sequential_results, sequential_time = executor.execute_sequential(test_data)
        
        # Test parallel execution
        parallel_results, parallel_time = executor.execute_parallel(test_data)
        
        # Verify results
        assert len(sequential_results) == 4
        assert len(parallel_results) == 4
        
        # Verify all components were executed
        sequential_ids = {r['component_id'] for r in sequential_results}
        parallel_ids = {r['component_id'] for r in parallel_results}
        
        assert sequential_ids == {0, 1, 2, 3}
        assert parallel_ids == {0, 1, 2, 3}
        
        # Parallel execution should be faster
        speedup = sequential_time / parallel_time
        print(f"Sequential time: {sequential_time:.4f}s")
        print(f"Parallel time: {parallel_time:.4f}s")
        print(f"Speedup: {speedup:.2f}x")
        
        # Should achieve some speedup (accounting for overhead)
        assert speedup > 1.5, f"Insufficient speedup: {speedup:.2f}x"
    
    def test_batch_processing_parallelism(self):
        """Test parallel processing of batches"""
        
        class BatchProcessor:
            def __init__(self, model):
                self.model = model
                
            def process_batch(self, batch_data, batch_id):
                """Process a single batch"""
                x_enc, x_mark_enc, x_dec, x_mark_dec = batch_data
                
                # Simulate processing time
                processing_start = time.perf_counter()
                
                # Simple model forward pass
                with torch.no_grad():
                    result = torch.randn(x_enc.shape[0], 24, 7)  # Mock output
                
                processing_time = time.perf_counter() - processing_start
                
                return {
                    'batch_id': batch_id,
                    'result': result,
                    'processing_time': processing_time,
                    'batch_size': x_enc.shape[0]
                }
            
            def process_batches_sequential(self, batches):
                """Process batches sequentially"""
                results = []
                total_start = time.perf_counter()
                
                for i, batch in enumerate(batches):
                    result = self.process_batch(batch, i)
                    results.append(result)
                
                total_time = time.perf_counter() - total_start
                return results, total_time
            
            def process_batches_parallel(self, batches):
                """Process batches in parallel"""
                results = []
                total_start = time.perf_counter()
                
                with ThreadPoolExecutor(max_workers=4) as executor:
                    # Submit batch processing tasks
                    future_to_batch = {
                        executor.submit(self.process_batch, batch, i): i 
                        for i, batch in enumerate(batches)
                    }
                    
                    # Collect results
                    for future in as_completed(future_to_batch):
                        result = future.result()
                        results.append(result)
                
                total_time = time.perf_counter() - total_start
                
                # Sort results by batch_id to maintain order
                results.sort(key=lambda x: x['batch_id'])
                
                return results, total_time
        
        # Create test batches
        n_batches = 8
        batch_size = 16
        batches = []
        
        for _ in range(n_batches):
            x_enc = torch.randn(batch_size, 96, 7)
            x_mark_enc = torch.randn(batch_size, 96, 4)
            x_dec = torch.randn(batch_size, 72, 7)
            x_mark_dec = torch.randn(batch_size, 72, 4)
            batches.append((x_enc, x_mark_enc, x_dec, x_mark_dec))
        
        # Test batch processing
        mock_model = Mock()
        processor = BatchProcessor(mock_model)
        
        # Sequential processing
        seq_results, seq_time = processor.process_batches_sequential(batches)
        
        # Parallel processing
        par_results, par_time = processor.process_batches_parallel(batches)
        
        # Verify results
        assert len(seq_results) == n_batches
        assert len(par_results) == n_batches
        
        # Check batch order is preserved in parallel processing
        for i in range(n_batches):
            assert par_results[i]['batch_id'] == i
        
        # Calculate speedup
        speedup = seq_time / par_time
        print(f"Sequential batch processing: {seq_time:.4f}s")
        print(f"Parallel batch processing: {par_time:.4f}s")
        print(f"Batch processing speedup: {speedup:.2f}x")
        
        # Should achieve some speedup
        assert speedup > 1.2, f"Insufficient batch processing speedup: {speedup:.2f}x"


class TestResourceUtilization:
    """Test resource utilization and optimization"""
    
    def test_memory_pooling(self):
        """Test memory pooling for efficient memory usage"""
        
        class MemoryPool:
            def __init__(self, pool_size=10):
                self.pool = []
                self.pool_size = pool_size
                self.allocated_count = 0
                self.reused_count = 0
                
            def get_tensor(self, shape, dtype=torch.float32):
                """Get tensor from pool or create new one"""
                # Look for compatible tensor in pool
                for i, tensor in enumerate(self.pool):
                    if (tensor.shape == shape and 
                        tensor.dtype == dtype and 
                        not tensor.is_leaf):  # Not in use
                        
                        # Remove from pool and return
                        reused_tensor = self.pool.pop(i)
                        reused_tensor.zero_()  # Clear data
                        self.reused_count += 1
                        return reused_tensor
                
                # Create new tensor if none available
                new_tensor = torch.zeros(shape, dtype=dtype)
                self.allocated_count += 1
                return new_tensor
            
            def return_tensor(self, tensor):
                """Return tensor to pool"""
                if len(self.pool) < self.pool_size:
                    self.pool.append(tensor.detach())
                
            def get_stats(self):
                """Get pool statistics"""
                return {
                    'pool_size': len(self.pool),
                    'allocated_count': self.allocated_count,
                    'reused_count': self.reused_count,
                    'reuse_ratio': self.reused_count / max(1, self.allocated_count + self.reused_count)
                }
        
        # Test memory pooling
        pool = MemoryPool(pool_size=5)
        
        # Simulate tensor allocation and deallocation pattern
        tensors_in_use = []
        
        # Phase 1: Allocate several tensors
        for i in range(8):
            tensor = pool.get_tensor((32, 96, 512))
            tensors_in_use.append(tensor)
        
        # Phase 2: Return some tensors to pool
        for i in range(0, 6, 2):  # Return every other tensor
            pool.return_tensor(tensors_in_use[i])
        
        # Phase 3: Allocate more tensors (should reuse from pool)
        for i in range(5):
            tensor = pool.get_tensor((32, 96, 512))
            tensors_in_use.append(tensor)
        
        # Check pool statistics
        stats = pool.get_stats()
        
        print(f"Memory pool stats: {stats}")
        
        # Should have achieved some reuse
        assert stats['reused_count'] > 0, "No tensor reuse occurred"
        assert stats['reuse_ratio'] > 0.2, f"Low reuse ratio: {stats['reuse_ratio']:.2f}"
    
    def test_computation_caching(self):
        """Test caching of intermediate computations"""
        
        class ComputationCache:
            def __init__(self, max_size=100):
                self.cache = {}
                self.max_size = max_size
                self.hit_count = 0
                self.miss_count = 0
                
            def _generate_key(self, inputs):
                """Generate cache key from inputs"""
                # Simple hash-based key (in practice, would need more sophisticated hashing)
                if isinstance(inputs, torch.Tensor):
                    return hash(inputs.data_ptr()) % 10000  # Simplified
                elif isinstance(inputs, (list, tuple)):
                    return hash(tuple(self._generate_key(inp) for inp in inputs)) % 10000
                else:
                    return hash(str(inputs)) % 10000
            
            def get_or_compute(self, key, computation_fn, *args, **kwargs):
                """Get cached result or compute and cache"""
                cache_key = self._generate_key((key, args, tuple(sorted(kwargs.items()))))
                
                if cache_key in self.cache:
                    self.hit_count += 1
                    return self.cache[cache_key]
                else:
                    # Compute result
                    result = computation_fn(*args, **kwargs)
                    
                    # Cache result if there's space
                    if len(self.cache) < self.max_size:
                        self.cache[cache_key] = result
                    
                    self.miss_count += 1
                    return result
            
            def get_stats(self):
                """Get cache statistics"""
                total_requests = self.hit_count + self.miss_count
                hit_rate = self.hit_count / max(1, total_requests)
                
                return {
                    'cache_size': len(self.cache),
                    'hit_count': self.hit_count,
                    'miss_count': self.miss_count,
                    'hit_rate': hit_rate
                }
        
        # Test computation caching
        cache = ComputationCache(max_size=20)
        
        def expensive_computation(x, factor=1.0):
            """Simulate expensive computation"""
            time.sleep(0.001)  # 1ms delay
            return x * factor + torch.randn_like(x) * 0.1
        
        # Test caching behavior
        test_tensor = torch.randn(10, 10)
        
        # First computation - should be cache miss
        start_time = time.perf_counter()
        result1 = cache.get_or_compute('test_op', expensive_computation, test_tensor, factor=2.0)
        first_time = time.perf_counter() - start_time
        
        # Second computation with same inputs - should be cache hit
        start_time = time.perf_counter()
        result2 = cache.get_or_compute('test_op', expensive_computation, test_tensor, factor=2.0)
        second_time = time.perf_counter() - start_time
        
        # Third computation with different inputs - should be cache miss
        start_time = time.perf_counter()
        result3 = cache.get_or_compute('test_op2', expensive_computation, test_tensor, factor=3.0)
        third_time = time.perf_counter() - start_time
        
        # Check cache statistics
        stats = cache.get_stats()
        
        print(f"Cache stats: {stats}")
        print(f"First computation: {first_time:.4f}s")
        print(f"Second computation (cached): {second_time:.4f}s")
        print(f"Third computation: {third_time:.4f}s")
        
        # Verify caching behavior
        assert stats['hit_count'] >= 1, "No cache hits occurred"
        assert stats['miss_count'] >= 2, "Insufficient cache misses"
        assert stats['hit_rate'] > 0, "Zero hit rate"
        
        # Cached computation should be faster
        assert second_time < first_time * 0.5, "Cached computation not significantly faster"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
