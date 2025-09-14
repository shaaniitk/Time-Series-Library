"""Performance benchmarks for SOTA_Temporal_PGAT model.

This module provides comprehensive performance testing including:
- Inference time benchmarks
- Memory usage profiling
- Throughput measurements
- Scalability analysis
- Resource utilization monitoring
"""

# Suppress optional dependency warnings
from TestsModule.suppress_warnings import suppress_optional_dependency_warnings
suppress_optional_dependency_warnings()

import pytest
import torch
import numpy as np
import time
import psutil
import gc
from typing import Dict, Any, List, Tuple, Optional
from unittest.mock import MagicMock
from contextlib import contextmanager
import threading

# Import the model
try:
    from models.SOTA_Temporal_PGAT import SOTA_Temporal_PGAT as SOTATemporalPGAT
except ImportError:
    pytest.skip("SOTA_Temporal_PGAT model not available", allow_module_level=True)


class PerformanceProfiler:
    """Utility class for performance profiling."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.start_time = None
        self.end_time = None
        self.peak_memory = 0
        self.start_memory = 0
        self.cpu_percent = []
        self.gpu_memory_allocated = []
        self.gpu_memory_cached = []
        self.monitoring = False
        self.monitor_thread = None
    
    @contextmanager
    def profile(self, monitor_resources=True):
        """Context manager for profiling code execution."""
        self.reset()
        
        # Start monitoring
        if monitor_resources:
            self.start_monitoring()
        
        # Record start state
        self.start_time = time.time()
        self.start_memory = psutil.Process().memory_info().rss / 1024**2  # MB
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        try:
            yield self
        finally:
            # Record end state
            self.end_time = time.time()
            
            if torch.cuda.is_available():
                self.peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
            
            # Stop monitoring
            if monitor_resources:
                self.stop_monitoring()
    
    def start_monitoring(self):
        """Start resource monitoring in background thread."""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_resources)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
    
    def _monitor_resources(self):
        """Monitor system resources in background."""
        while self.monitoring:
            try:
                # CPU usage
                self.cpu_percent.append(psutil.cpu_percent())
                
                # GPU memory if available
                if torch.cuda.is_available():
                    self.gpu_memory_allocated.append(torch.cuda.memory_allocated() / 1024**2)
                    self.gpu_memory_cached.append(torch.cuda.memory_reserved() / 1024**2)
                
                time.sleep(0.1)  # Sample every 100ms
            except Exception:
                break
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get collected performance metrics."""
        metrics = {
            'execution_time': self.end_time - self.start_time if self.start_time and self.end_time else 0,
            'peak_gpu_memory_mb': self.peak_memory,
            'start_memory_mb': self.start_memory
        }
        
        if self.cpu_percent:
            metrics.update({
                'avg_cpu_percent': np.mean(self.cpu_percent),
                'max_cpu_percent': np.max(self.cpu_percent),
                'cpu_samples': len(self.cpu_percent)
            })
        
        if self.gpu_memory_allocated:
            metrics.update({
                'avg_gpu_memory_mb': np.mean(self.gpu_memory_allocated),
                'max_gpu_memory_mb': np.max(self.gpu_memory_allocated),
                'avg_gpu_cached_mb': np.mean(self.gpu_memory_cached),
                'max_gpu_cached_mb': np.max(self.gpu_memory_cached)
            })
        
        return metrics


class TestSOTATemporalPGATPerformance:
    """Performance benchmark suite for SOTA_Temporal_PGAT model."""
    
    @pytest.fixture(autouse=True)
    def setup_test(self):
        """Setup test environment and model configuration."""
        # Create model configuration
        self.configs = MagicMock()
        self.configs.seq_len = 96
        self.configs.label_len = 48
        self.configs.pred_len = 96
        self.configs.enc_in = 7
        self.configs.dec_in = 7
        self.configs.c_out = 7
        self.configs.d_model = 256
        self.configs.n_heads = 8
        self.configs.e_layers = 2
        self.configs.d_layers = 1
        self.configs.d_ff = 1024
        self.configs.dropout = 0.1
        self.configs.activation = 'gelu'
        self.configs.output_attention = False
        self.configs.mix = True
        
        # Graph-specific configurations
        self.configs.graph_dim = 32
        self.configs.num_nodes = self.configs.enc_in
        self.configs.graph_alpha = 0.2
        self.configs.graph_beta = 0.1
        
        # Performance test parameters
        self.warmup_runs = 5
        self.benchmark_runs = 20
        self.profiler = PerformanceProfiler()
        
        # Create model
        self.model = SOTATemporalPGAT(self.configs)
        self.model.eval()
        
        # Move to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
    
    def create_test_data(self, batch_size: int, seq_len: Optional[int] = None, device: Optional[torch.device] = None):
        """Create test data for performance benchmarking."""
        if seq_len is None:
            seq_len = self.configs.seq_len
        if device is None:
            device = self.device
        
        x_enc = torch.randn(batch_size, seq_len, self.configs.enc_in, device=device)
        x_mark_enc = torch.randn(batch_size, seq_len, 4, device=device)
        x_dec = torch.randn(batch_size, self.configs.label_len + self.configs.pred_len, self.configs.dec_in, device=device)
        x_mark_dec = torch.randn(batch_size, self.configs.label_len + self.configs.pred_len, 4, device=device)
        
        return x_enc, x_mark_enc, x_dec, x_mark_dec
    
    def warmup_model(self, batch_size: int = 4):
        """Warmup model to ensure stable performance measurements."""
        x_enc, x_mark_enc, x_dec, x_mark_dec = self.create_test_data(batch_size)
        
        with torch.no_grad():
            for _ in range(self.warmup_runs):
                _ = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    def benchmark_inference_time(self, batch_size: int, num_runs: Optional[int] = None) -> Dict[str, float]:
        """Benchmark inference time for given batch size."""
        if num_runs is None:
            num_runs = self.benchmark_runs
        
        self.warmup_model(batch_size)
        x_enc, x_mark_enc, x_dec, x_mark_dec = self.create_test_data(batch_size)
        
        times = []
        
        for _ in range(num_runs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start_time = time.time()
            
            with torch.no_grad():
                _ = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end_time = time.time()
            times.append(end_time - start_time)
        
        return {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'median_time': np.median(times),
            'p95_time': np.percentile(times, 95),
            'p99_time': np.percentile(times, 99)
        }
    
    def test_inference_time_single_batch(self):
        """Test inference time for single batch."""
        batch_size = 1
        results = self.benchmark_inference_time(batch_size)
        
        # Assertions for reasonable performance
        assert results['mean_time'] < 5.0, f"Single batch inference too slow: {results['mean_time']:.3f}s"
        assert results['std_time'] < results['mean_time'] * 0.5, "Inference time too variable"
        
        print(f"\nSingle Batch Inference (batch_size={batch_size}):")
        print(f"  Mean: {results['mean_time']:.3f}±{results['std_time']:.3f}s")
        print(f"  Range: [{results['min_time']:.3f}, {results['max_time']:.3f}]s")
        print(f"  P95: {results['p95_time']:.3f}s, P99: {results['p99_time']:.3f}s")
    
    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16, 32])
    def test_batch_size_scaling(self, batch_size):
        """Test how inference time scales with batch size."""
        try:
            results = self.benchmark_inference_time(batch_size, num_runs=10)
            
            # Performance should scale reasonably with batch size
            time_per_sample = results['mean_time'] / batch_size
            
            # Time per sample should decrease with larger batches (batching efficiency)
            if batch_size > 1:
                single_batch_results = self.benchmark_inference_time(1, num_runs=5)
                single_time_per_sample = single_batch_results['mean_time']
                
                efficiency_ratio = single_time_per_sample / time_per_sample
                assert efficiency_ratio > 0.5, f"Poor batching efficiency for batch_size {batch_size}: {efficiency_ratio:.2f}"
            
            # Total time should not grow too quickly
            assert results['mean_time'] < batch_size * 2.0, f"Inference time grows too quickly with batch size {batch_size}"
            
            print(f"\nBatch Size {batch_size}:")
            print(f"  Total time: {results['mean_time']:.3f}±{results['std_time']:.3f}s")
            print(f"  Time per sample: {time_per_sample:.3f}s")
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                pytest.skip(f"Out of memory for batch_size {batch_size}")
            else:
                raise
    
    @pytest.mark.parametrize("seq_len", [24, 48, 96, 192, 336, 720])
    def test_sequence_length_scaling(self, seq_len):
        """Test how inference time scales with sequence length."""
        batch_size = 4
        
        try:
            # Update config for this test
            original_seq_len = self.configs.seq_len
            self.configs.seq_len = seq_len
            
            # Create new model with updated config
            test_model = SOTATemporalPGAT(self.configs).to(self.device)
            test_model.eval()
            
            # Warmup
            x_enc, x_mark_enc, x_dec, x_mark_dec = self.create_test_data(batch_size, seq_len)
            with torch.no_grad():
                for _ in range(3):
                    _ = test_model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            
            # Benchmark
            times = []
            for _ in range(10):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                start_time = time.time()
                with torch.no_grad():
                    _ = test_model(x_enc, x_mark_enc, x_dec, x_mark_dec)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                times.append(time.time() - start_time)
            
            mean_time = np.mean(times)
            
            # Time should scale reasonably with sequence length
            # For transformer-based models, expect roughly O(n^2) or better
            time_per_token = mean_time / seq_len
            
            assert mean_time < seq_len * 0.1, f"Inference time grows too quickly with seq_len {seq_len}: {mean_time:.3f}s"
            
            print(f"\nSequence Length {seq_len}:")
            print(f"  Total time: {mean_time:.3f}±{np.std(times):.3f}s")
            print(f"  Time per token: {time_per_token:.4f}s")
            
            # Restore original config
            self.configs.seq_len = original_seq_len
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                pytest.skip(f"Out of memory for seq_len {seq_len}")
            else:
                raise
    
    def test_memory_usage_profiling(self):
        """Profile memory usage during inference."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for memory profiling")
        
        batch_sizes = [1, 4, 8, 16]
        memory_results = {}
        
        for batch_size in batch_sizes:
            try:
                # Clear cache
                torch.cuda.empty_cache()
                gc.collect()
                
                x_enc, x_mark_enc, x_dec, x_mark_dec = self.create_test_data(batch_size)
                
                with self.profiler.profile(monitor_resources=True):
                    with torch.no_grad():
                        for _ in range(5):
                            output = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)
                
                metrics = self.profiler.get_metrics()
                memory_results[batch_size] = metrics
                
                # Memory usage should be reasonable
                peak_memory = metrics['peak_gpu_memory_mb']
                assert peak_memory < 8192, f"Memory usage too high for batch_size {batch_size}: {peak_memory:.1f}MB"
                
                print(f"\nMemory Usage (batch_size={batch_size}):")
                print(f"  Peak GPU memory: {peak_memory:.1f}MB")
                if 'avg_gpu_memory_mb' in metrics:
                    print(f"  Average GPU memory: {metrics['avg_gpu_memory_mb']:.1f}MB")
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"\nOut of memory for batch_size {batch_size}")
                    break
                else:
                    raise
        
        # Check memory scaling
        if len(memory_results) > 1:
            batch_sizes_tested = sorted(memory_results.keys())
            memory_growth_rates = []
            
            for i in range(1, len(batch_sizes_tested)):
                prev_batch = batch_sizes_tested[i-1]
                curr_batch = batch_sizes_tested[i]
                
                prev_memory = memory_results[prev_batch]['peak_gpu_memory_mb']
                curr_memory = memory_results[curr_batch]['peak_gpu_memory_mb']
                
                batch_ratio = curr_batch / prev_batch
                memory_ratio = curr_memory / prev_memory
                
                growth_rate = memory_ratio / batch_ratio
                memory_growth_rates.append(growth_rate)
            
            avg_growth_rate = np.mean(memory_growth_rates)
            
            # Memory should scale roughly linearly with batch size
            assert 0.5 < avg_growth_rate < 2.0, f"Memory scaling seems unreasonable: {avg_growth_rate:.2f}"
    
    def test_throughput_measurement(self):
        """Measure model throughput (samples per second)."""
        batch_size = 8
        duration = 10.0  # seconds
        
        self.warmup_model(batch_size)
        x_enc, x_mark_enc, x_dec, x_mark_dec = self.create_test_data(batch_size)
        
        start_time = time.time()
        samples_processed = 0
        
        with torch.no_grad():
            while time.time() - start_time < duration:
                _ = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)
                samples_processed += batch_size
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
        
        actual_duration = time.time() - start_time
        throughput = samples_processed / actual_duration
        
        # Throughput should be reasonable
        assert throughput > 1.0, f"Throughput too low: {throughput:.2f} samples/sec"
        
        print(f"\nThroughput Measurement:")
        print(f"  Duration: {actual_duration:.1f}s")
        print(f"  Samples processed: {samples_processed}")
        print(f"  Throughput: {throughput:.2f} samples/sec")
        print(f"  Batch size: {batch_size}")
    
    def test_cpu_vs_gpu_performance(self):
        """Compare CPU vs GPU performance if both available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for CPU vs GPU comparison")
        
        batch_size = 4
        
        # Test on CPU
        cpu_model = SOTATemporalPGAT(self.configs).cpu()
        cpu_model.eval()
        
        x_enc_cpu, x_mark_enc_cpu, x_dec_cpu, x_mark_dec_cpu = self.create_test_data(batch_size, device=torch.device('cpu'))
        
        # Warmup CPU
        with torch.no_grad():
            for _ in range(3):
                _ = cpu_model(x_enc_cpu, x_mark_enc_cpu, x_dec_cpu, x_mark_dec_cpu)
        
        # Benchmark CPU
        cpu_times = []
        for _ in range(5):
            start_time = time.time()
            with torch.no_grad():
                _ = cpu_model(x_enc_cpu, x_mark_enc_cpu, x_dec_cpu, x_mark_dec_cpu)
            cpu_times.append(time.time() - start_time)
        
        cpu_mean_time = np.mean(cpu_times)
        
        # Test on GPU
        gpu_results = self.benchmark_inference_time(batch_size, num_runs=5)
        gpu_mean_time = gpu_results['mean_time']
        
        # GPU should be faster (or at least not much slower)
        speedup = cpu_mean_time / gpu_mean_time
        
        print(f"\nCPU vs GPU Performance:")
        print(f"  CPU time: {cpu_mean_time:.3f}±{np.std(cpu_times):.3f}s")
        print(f"  GPU time: {gpu_mean_time:.3f}±{gpu_results['std_time']:.3f}s")
        print(f"  Speedup: {speedup:.2f}x")
        
        # GPU should provide some benefit (at least not be much slower)
        assert speedup > 0.5, f"GPU performance unexpectedly poor: {speedup:.2f}x speedup"
    
    def test_model_size_analysis(self):
        """Analyze model size and parameter distribution."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Analyze parameter distribution by component
        param_breakdown = {}
        
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                module_params = sum(p.numel() for p in module.parameters())
                if module_params > 0:
                    param_breakdown[name] = module_params
        
        # Model size in MB (assuming float32)
        model_size_mb = total_params * 4 / (1024 ** 2)
        
        print(f"\nModel Size Analysis:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Model size: {model_size_mb:.2f}MB")
        
        # Show top parameter-heavy components
        sorted_components = sorted(param_breakdown.items(), key=lambda x: x[1], reverse=True)[:10]
        print(f"  Top components by parameter count:")
        for name, count in sorted_components:
            percentage = (count / total_params) * 100
            print(f"    {name}: {count:,} ({percentage:.1f}%)")
        
        # Reasonable size constraints
        assert total_params < 50_000_000, f"Model too large: {total_params:,} parameters"
        assert model_size_mb < 200, f"Model file too large: {model_size_mb:.2f}MB"
    
    def test_gradient_computation_performance(self):
        """Test performance of gradient computation."""
        batch_size = 4
        self.model.train()
        
        x_enc, x_mark_enc, x_dec, x_mark_dec = self.create_test_data(batch_size)
        
        # Warmup
        for _ in range(3):
            self.model.zero_grad()
            output = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            loss = output.mean()
            loss.backward()
        
        # Benchmark gradient computation
        forward_times = []
        backward_times = []
        
        for _ in range(10):
            self.model.zero_grad()
            
            # Forward pass
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start_time = time.time()
            output = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            loss = output.mean()
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            forward_time = time.time() - start_time
            forward_times.append(forward_time)
            
            # Backward pass
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start_time = time.time()
            loss.backward()
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            backward_time = time.time() - start_time
            backward_times.append(backward_time)
        
        forward_mean = np.mean(forward_times)
        backward_mean = np.mean(backward_times)
        total_mean = forward_mean + backward_mean
        
        print(f"\nGradient Computation Performance:")
        print(f"  Forward pass: {forward_mean:.3f}±{np.std(forward_times):.3f}s")
        print(f"  Backward pass: {backward_mean:.3f}±{np.std(backward_times):.3f}s")
        print(f"  Total (forward + backward): {total_mean:.3f}s")
        print(f"  Backward/Forward ratio: {backward_mean/forward_mean:.2f}")
        
        # Backward pass should not be excessively slower than forward
        assert backward_mean < forward_mean * 5, f"Backward pass too slow: {backward_mean/forward_mean:.2f}x forward time"
        
        self.model.eval()
    
    def test_concurrent_inference(self):
        """Test performance under concurrent inference requests."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for concurrent testing")
        
        batch_size = 2
        num_threads = 4
        requests_per_thread = 5
        
        results = []
        threads = []
        
        def inference_worker(worker_id):
            """Worker function for concurrent inference."""
            worker_times = []
            
            # Each worker gets its own data
            x_enc, x_mark_enc, x_dec, x_mark_dec = self.create_test_data(batch_size)
            
            for _ in range(requests_per_thread):
                start_time = time.time()
                
                with torch.no_grad():
                    _ = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                worker_times.append(time.time() - start_time)
            
            results.append({
                'worker_id': worker_id,
                'times': worker_times,
                'mean_time': np.mean(worker_times)
            })
        
        # Start all threads
        start_time = time.time()
        
        for i in range(num_threads):
            thread = threading.Thread(target=inference_worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        total_time = time.time() - start_time
        
        # Analyze results
        all_times = []
        for result in results:
            all_times.extend(result['times'])
        
        mean_concurrent_time = np.mean(all_times)
        total_requests = num_threads * requests_per_thread
        effective_throughput = total_requests / total_time
        
        print(f"\nConcurrent Inference Performance:")
        print(f"  Threads: {num_threads}")
        print(f"  Requests per thread: {requests_per_thread}")
        print(f"  Total requests: {total_requests}")
        print(f"  Total time: {total_time:.3f}s")
        print(f"  Mean request time: {mean_concurrent_time:.3f}s")
        print(f"  Effective throughput: {effective_throughput:.2f} requests/sec")
        
        # Compare with single-threaded performance
        single_thread_results = self.benchmark_inference_time(batch_size, num_runs=5)
        single_thread_time = single_thread_results['mean_time']
        
        concurrency_overhead = mean_concurrent_time / single_thread_time
        
        print(f"  Single-thread time: {single_thread_time:.3f}s")
        print(f"  Concurrency overhead: {concurrency_overhead:.2f}x")
        
        # Concurrency overhead should be reasonable
        assert concurrency_overhead < 3.0, f"Concurrency overhead too high: {concurrency_overhead:.2f}x"
        assert len(results) == num_threads, "Not all worker threads completed successfully"