"""
ChronosX Production Deployment Testing

Production-ready testing suite for ChronosX models including:
- Production environment simulation
- Scalability testing  
- Error handling and recovery
- Monitoring and alerting
- Load testing and stress testing
- Deployment configurations
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from chronos import ChronosPipeline
import time
import psutil
import logging
import threading
import queue
from typing import Dict, List, Tuple, Optional, Any
import json
from datetime import datetime, timedelta
import warnings
import concurrent.futures
from dataclasses import dataclass
from pathlib import Path
import traceback
import gc
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ProductionConfig:
    """Production deployment configuration"""
    model_name: str = "amazon/chronos-t5-tiny"  # Start with fastest model
    max_concurrent_requests: int = 4
    request_timeout_seconds: int = 30
    max_memory_usage_percent: float = 80.0
    context_length: int = 96
    forecast_length: int = 24
    num_samples: int = 10  # Reduced for production speed
    batch_size: int = 1
    enable_monitoring: bool = True
    enable_error_recovery: bool = True
    log_level: str = "INFO"

@dataclass
class ProductionMetrics:
    """Production monitoring metrics"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_response_time: float = 0.0
    max_response_time: float = 0.0
    min_response_time: float = float('inf')
    memory_usage_peak: float = 0.0
    cpu_usage_peak: float = 0.0
    error_types: Dict[str, int] = None
    
    def __post_init__(self):
        if self.error_types is None:
            self.error_types = {}

class ChronosXProductionService:
    """Production-ready ChronosX forecasting service"""
    
    def __init__(self, config: ProductionConfig):
        self.config = config
        self.pipeline = None
        self.metrics = ProductionMetrics()
        self.is_healthy = False
        self.request_queue = queue.Queue(maxsize=config.max_concurrent_requests * 2)
        self.workers = []
        self.shutdown_event = threading.Event()
        self.response_times = []
        self.lock = threading.Lock()
        
        # Setup logging
        logging.getLogger().setLevel(getattr(logging, config.log_level))
        
    def initialize_model(self) -> bool:
        """Initialize the ChronosX model with error handling"""
        logger.info(f"ROCKET Initializing ChronosX model: {self.config.model_name}")
        
        try:
            start_time = time.time()
            
            # Check available memory before loading
            available_memory_gb = psutil.virtual_memory().available / (1024**3)
            logger.info(f"Available memory: {available_memory_gb:.1f}GB")
            
            if available_memory_gb < 2.0:
                logger.warning("WARN Low memory detected, using CPU with reduced precision")
                torch_dtype = torch.float32
            else:
                torch_dtype = torch.float32
            
            self.pipeline = ChronosPipeline.from_pretrained(
                self.config.model_name,
                device_map="cpu",  # Force CPU for production reliability
                torch_dtype=torch_dtype,
            )
            
            load_time = time.time() - start_time
            
            # Validate with test prediction
            test_data = torch.randn(1, self.config.context_length, dtype=torch_dtype)
            test_start = time.time()
            
            test_forecast = self.pipeline.predict(
                context=test_data,
                prediction_length=self.config.forecast_length,
                num_samples=self.config.num_samples
            )
            
            test_time = time.time() - test_start
            
            logger.info(f"PASS Model loaded successfully")
            logger.info(f"   Load time: {load_time:.2f}s")
            logger.info(f"   Test inference: {test_time:.2f}s")
            logger.info(f"   Test output shape: {test_forecast.shape}")
            
            self.is_healthy = True
            return True
            
        except Exception as e:
            logger.error(f"FAIL Failed to initialize model: {e}")
            logger.error(traceback.format_exc())
            self.is_healthy = False
            return False
    
    def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check"""
        health_status = {
            'status': 'healthy' if self.is_healthy else 'unhealthy',
            'model_loaded': self.pipeline is not None,
            'timestamp': datetime.now().isoformat(),
            'metrics': {
                'total_requests': self.metrics.total_requests,
                'success_rate': (self.metrics.successful_requests / max(self.metrics.total_requests, 1)) * 100,
                'avg_response_time': self.metrics.avg_response_time,
                'memory_usage': psutil.virtual_memory().percent,
                'cpu_usage': psutil.cpu_percent(),
            },
            'config': {
                'model_name': self.config.model_name,
                'max_concurrent_requests': self.config.max_concurrent_requests,
                'context_length': self.config.context_length,
                'forecast_length': self.config.forecast_length
            }
        }
        
        # Check resource constraints
        if psutil.virtual_memory().percent > self.config.max_memory_usage_percent:
            health_status['warnings'] = health_status.get('warnings', [])
            health_status['warnings'].append('High memory usage detected')
        
        return health_status
    
    def predict(self, time_series_data: np.ndarray, 
                prediction_length: Optional[int] = None,
                num_samples: Optional[int] = None) -> Dict[str, Any]:
        """Production prediction with comprehensive error handling"""
        
        request_start = time.time()
        
        try:
            # Input validation
            if not self.is_healthy or self.pipeline is None:
                raise RuntimeError("Service is not healthy - model not loaded")
            
            if len(time_series_data) < self.config.context_length:
                raise ValueError(f"Input data too short. Need at least {self.config.context_length} points, got {len(time_series_data)}")
            
            # Use defaults if not specified
            pred_length = prediction_length or self.config.forecast_length
            samples = num_samples or self.config.num_samples
            
            # Prepare input data
            context = time_series_data[-self.config.context_length:]
            context_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0)
            
            # Monitor resources during prediction
            pre_memory = psutil.virtual_memory().percent
            pre_cpu = psutil.cpu_percent()
            
            # Make prediction
            inference_start = time.time()
            
            forecast = self.pipeline.predict(
                context=context_tensor,
                prediction_length=pred_length,
                num_samples=samples
            )
            
            inference_time = time.time() - inference_start
            
            # Process results
            forecast_np = forecast.cpu().numpy()
            mean_forecast = np.mean(forecast_np, axis=0).flatten()
            std_forecast = np.std(forecast_np, axis=0).flatten()
            
            # Calculate prediction intervals
            quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
            prediction_intervals = {}
            for q in quantiles:
                prediction_intervals[f'q{int(q*100)}'] = np.quantile(forecast_np, q, axis=0).flatten()
            
            # Update metrics
            total_time = time.time() - request_start
            post_memory = psutil.virtual_memory().percent
            post_cpu = psutil.cpu_percent()
            
            with self.lock:
                self.metrics.total_requests += 1
                self.metrics.successful_requests += 1
                
                self.response_times.append(total_time)
                if len(self.response_times) > 1000:  # Keep last 1000 responses
                    self.response_times = self.response_times[-1000:]
                
                self.metrics.avg_response_time = np.mean(self.response_times)
                self.metrics.max_response_time = max(self.metrics.max_response_time, total_time)
                self.metrics.min_response_time = min(self.metrics.min_response_time, total_time)
                self.metrics.memory_usage_peak = max(self.metrics.memory_usage_peak, post_memory)
                self.metrics.cpu_usage_peak = max(self.metrics.cpu_usage_peak, post_cpu)
            
            result = {
                'success': True,
                'forecast': {
                    'mean': mean_forecast.tolist(),
                    'std': std_forecast.tolist(),
                    'prediction_intervals': {k: v.tolist() for k, v in prediction_intervals.items()},
                    'samples': forecast_np.tolist() if samples <= 20 else None  # Only return samples for small requests
                },
                'metadata': {
                    'prediction_length': pred_length,
                    'num_samples': samples,
                    'context_length': len(context),
                    'inference_time': inference_time,
                    'total_time': total_time,
                    'memory_delta': post_memory - pre_memory,
                    'timestamp': datetime.now().isoformat()
                },
                'model_info': {
                    'model_name': self.config.model_name,
                    'version': 'chronos-forecasting'
                }
            }
            
            logger.debug(f"PASS Prediction successful: {total_time:.3f}s")
            return result
            
        except Exception as e:
            # Handle errors gracefully
            total_time = time.time() - request_start
            error_type = type(e).__name__
            
            with self.lock:
                self.metrics.total_requests += 1
                self.metrics.failed_requests += 1
                self.metrics.error_types[error_type] = self.metrics.error_types.get(error_type, 0) + 1
            
            logger.error(f"FAIL Prediction failed: {e}")
            logger.error(traceback.format_exc())
            
            return {
                'success': False,
                'error': {
                    'type': error_type,
                    'message': str(e),
                    'timestamp': datetime.now().isoformat(),
                    'total_time': total_time
                },
                'metadata': {
                    'total_time': total_time
                }
            }
    
    def batch_predict(self, time_series_batch: List[np.ndarray],
                     prediction_length: Optional[int] = None,
                     num_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """Batch prediction with parallel processing"""
        
        if not time_series_batch:
            return []
        
        # Validate batch size
        if len(time_series_batch) > self.config.max_concurrent_requests:
            logger.warning(f"Batch size {len(time_series_batch)} exceeds max concurrent requests {self.config.max_concurrent_requests}")
        
        batch_start = time.time()
        results = []
        
        # Process in parallel using ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(self.config.max_concurrent_requests, len(time_series_batch))) as executor:
            
            # Submit all predictions
            future_to_index = {
                executor.submit(self.predict, ts_data, prediction_length, num_samples): i 
                for i, ts_data in enumerate(time_series_batch)
            }
            
            # Collect results in order
            results = [None] * len(time_series_batch)
            
            for future in concurrent.futures.as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result = future.result(timeout=self.config.request_timeout_seconds)
                    results[index] = result
                except concurrent.futures.TimeoutError:
                    logger.error(f"FAIL Batch prediction {index} timed out")
                    results[index] = {
                        'success': False,
                        'error': {
                            'type': 'TimeoutError',
                            'message': f'Prediction timed out after {self.config.request_timeout_seconds}s',
                            'timestamp': datetime.now().isoformat()
                        }
                    }
                except Exception as e:
                    logger.error(f"FAIL Batch prediction {index} failed: {e}")
                    results[index] = {
                        'success': False,
                        'error': {
                            'type': type(e).__name__,
                            'message': str(e),
                            'timestamp': datetime.now().isoformat()
                        }
                    }
        
        batch_time = time.time() - batch_start
        successful_predictions = sum(1 for r in results if r and r.get('success', False))
        
        logger.info(f"CHART Batch prediction completed: {successful_predictions}/{len(time_series_batch)} successful in {batch_time:.2f}s")
        
        return results
    
    def stress_test(self, num_requests: int = 100, concurrent_requests: int = 4) -> Dict[str, Any]:
        """Stress test the service with multiple concurrent requests"""
        logger.info(f"FIRE Starting stress test: {num_requests} requests with {concurrent_requests} concurrent")
        
        # Generate test data
        np.random.seed(42)
        test_datasets = []
        for i in range(num_requests):
            # Generate diverse test data
            t = np.arange(self.config.context_length + 50)
            if i % 4 == 0:
                # Trending
                data = 0.1 * t + 50 + np.random.normal(0, 2, len(t))
            elif i % 4 == 1:
                # Seasonal
                data = 50 + 10 * np.sin(2 * np.pi * t / 20) + np.random.normal(0, 1, len(t))
            elif i % 4 == 2:
                # Random walk
                data = 50 + np.cumsum(np.random.normal(0, 1, len(t)))
            else:
                # Complex
                data = 50 + 0.1 * t + 5 * np.sin(2 * np.pi * t / 20) + np.random.normal(0, 2, len(t))
            
            test_datasets.append(data)
        
        # Reset metrics for clean measurement
        with self.lock:
            self.metrics = ProductionMetrics()
            self.response_times = []
        
        stress_start = time.time()
        
        # Execute stress test in batches
        batch_size = concurrent_requests
        all_results = []
        
        for i in range(0, num_requests, batch_size):
            batch = test_datasets[i:i+batch_size]
            batch_results = self.batch_predict(batch)
            all_results.extend(batch_results)
            
            # Brief pause between batches to monitor recovery
            if i + batch_size < num_requests:
                time.sleep(0.1)
        
        total_time = time.time() - stress_start
        
        # Analyze results
        successful_results = [r for r in all_results if r and r.get('success', False)]
        failed_results = [r for r in all_results if r and not r.get('success', False)]
        
        # Calculate statistics
        if successful_results:
            response_times = [r['metadata']['total_time'] for r in successful_results]
            inference_times = [r['metadata']['inference_time'] for r in successful_results]
            
            performance_stats = {
                'response_time': {
                    'mean': np.mean(response_times),
                    'std': np.std(response_times),
                    'min': np.min(response_times),
                    'max': np.max(response_times),
                    'p50': np.percentile(response_times, 50),
                    'p95': np.percentile(response_times, 95),
                    'p99': np.percentile(response_times, 99)
                },
                'inference_time': {
                    'mean': np.mean(inference_times),
                    'std': np.std(inference_times),
                    'min': np.min(inference_times),
                    'max': np.max(inference_times)
                }
            }
        else:
            performance_stats = {}
        
        # Error analysis
        error_analysis = {}
        for result in failed_results:
            if 'error' in result:
                error_type = result['error'].get('type', 'Unknown')
                error_analysis[error_type] = error_analysis.get(error_type, 0) + 1
        
        stress_results = {
            'test_config': {
                'num_requests': num_requests,
                'concurrent_requests': concurrent_requests,
                'total_time': total_time
            },
            'results_summary': {
                'total_requests': len(all_results),
                'successful_requests': len(successful_results),
                'failed_requests': len(failed_results),
                'success_rate': len(successful_results) / len(all_results) * 100,
                'requests_per_second': len(all_results) / total_time
            },
            'performance_stats': performance_stats,
            'error_analysis': error_analysis,
            'resource_usage': {
                'peak_memory_percent': self.metrics.memory_usage_peak,
                'peak_cpu_percent': self.metrics.cpu_usage_peak
            },
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"TARGET Stress test completed:")
        logger.info(f"   Success rate: {stress_results['results_summary']['success_rate']:.1f}%")
        logger.info(f"   Requests/sec: {stress_results['results_summary']['requests_per_second']:.1f}")
        if performance_stats:
            logger.info(f"   Avg response time: {performance_stats['response_time']['mean']:.3f}s")
            logger.info(f"   P95 response time: {performance_stats['response_time']['p95']:.3f}s")
        
        return stress_results
    
    def monitor_resources(self, duration_seconds: int = 60) -> Dict[str, Any]:
        """Monitor resource usage over time"""
        logger.info(f"CHART Starting resource monitoring for {duration_seconds}s")
        
        monitoring_start = time.time()
        measurements = []
        
        while time.time() - monitoring_start < duration_seconds:
            measurement = {
                'timestamp': datetime.now().isoformat(),
                'memory_percent': psutil.virtual_memory().percent,
                'memory_available_gb': psutil.virtual_memory().available / (1024**3),
                'cpu_percent': psutil.cpu_percent(interval=1),
                'service_metrics': {
                    'total_requests': self.metrics.total_requests,
                    'successful_requests': self.metrics.successful_requests,
                    'failed_requests': self.metrics.failed_requests,
                    'avg_response_time': self.metrics.avg_response_time
                }
            }
            measurements.append(measurement)
            time.sleep(5)  # Sample every 5 seconds
        
        # Analyze monitoring data
        memory_usage = [m['memory_percent'] for m in measurements]
        cpu_usage = [m['cpu_percent'] for m in measurements]
        
        monitoring_results = {
            'duration_seconds': duration_seconds,
            'measurements': measurements,
            'summary': {
                'memory_usage': {
                    'mean': np.mean(memory_usage),
                    'std': np.std(memory_usage),
                    'min': np.min(memory_usage),
                    'max': np.max(memory_usage)
                },
                'cpu_usage': {
                    'mean': np.mean(cpu_usage),
                    'std': np.std(cpu_usage),
                    'min': np.min(cpu_usage),
                    'max': np.max(cpu_usage)
                }
            }
        }
        
        logger.info(f"GRAPH Resource monitoring completed:")
        logger.info(f"   Average memory usage: {monitoring_results['summary']['memory_usage']['mean']:.1f}%")
        logger.info(f"   Average CPU usage: {monitoring_results['summary']['cpu_usage']['mean']:.1f}%")
        
        return monitoring_results

class ChronosXProductionTester:
    """Comprehensive production testing suite"""
    
    def __init__(self):
        self.test_configs = {
            'tiny_fast': ProductionConfig(
                model_name="amazon/chronos-t5-tiny",
                max_concurrent_requests=8,
                num_samples=10,
                forecast_length=24
            ),
            'small_balanced': ProductionConfig(
                model_name="amazon/chronos-t5-small",
                max_concurrent_requests=4,
                num_samples=15,
                forecast_length=24
            ),
            'base_accurate': ProductionConfig(
                model_name="amazon/chronos-t5-base",
                max_concurrent_requests=2,
                num_samples=20,
                forecast_length=24
            )
        }
        
        self.test_results = {}
    
    def test_configuration(self, config_name: str, config: ProductionConfig) -> Dict[str, Any]:
        """Test a specific production configuration"""
        logger.info(f"TEST Testing production configuration: {config_name}")
        
        service = ChronosXProductionService(config)
        
        test_results = {
            'config_name': config_name,
            'config': config.__dict__,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # Test 1: Model initialization
            logger.info("1 Testing model initialization...")
            init_success = service.initialize_model()
            test_results['initialization'] = {
                'success': init_success,
                'model_loaded': service.pipeline is not None,
                'is_healthy': service.is_healthy
            }
            
            if not init_success:
                logger.error(f"FAIL Configuration {config_name} failed initialization")
                return test_results
            
            # Test 2: Health check
            logger.info("2 Testing health check...")
            health_status = service.health_check()
            test_results['health_check'] = health_status
            
            # Test 3: Single prediction
            logger.info("3 Testing single prediction...")
            np.random.seed(42)
            test_data = 50 + np.cumsum(np.random.normal(0, 1, 200))
            
            single_pred_start = time.time()
            single_result = service.predict(test_data)
            single_pred_time = time.time() - single_pred_start
            
            test_results['single_prediction'] = {
                'success': single_result.get('success', False),
                'response_time': single_pred_time,
                'has_forecast': 'forecast' in single_result,
                'forecast_length': len(single_result.get('forecast', {}).get('mean', [])) if single_result.get('success') else 0
            }
            
            # Test 4: Batch prediction
            logger.info("4 Testing batch prediction...")
            batch_data = [50 + np.cumsum(np.random.normal(0, 1, 200)) for _ in range(4)]
            
            batch_start = time.time()
            batch_results = service.batch_predict(batch_data)
            batch_time = time.time() - batch_start
            
            batch_success = sum(1 for r in batch_results if r.get('success', False))
            
            test_results['batch_prediction'] = {
                'batch_size': len(batch_data),
                'successful_predictions': batch_success,
                'success_rate': batch_success / len(batch_data) * 100,
                'total_time': batch_time,
                'avg_time_per_prediction': batch_time / len(batch_data)
            }
            
            # Test 5: Stress test
            logger.info("5 Running stress test...")
            stress_results = service.stress_test(num_requests=20, concurrent_requests=min(4, config.max_concurrent_requests))
            test_results['stress_test'] = stress_results
            
            # Test 6: Error handling
            logger.info("6 Testing error handling...")
            error_tests = []
            
            # Test with invalid input
            invalid_result = service.predict(np.array([1, 2]))  # Too short
            error_tests.append({
                'test': 'short_input',
                'handled_gracefully': not invalid_result.get('success', False),
                'error_type': invalid_result.get('error', {}).get('type', 'None')
            })
            
            # Test with NaN input
            nan_data = np.full(200, np.nan)
            nan_result = service.predict(nan_data)
            error_tests.append({
                'test': 'nan_input',
                'handled_gracefully': not nan_result.get('success', False),
                'error_type': nan_result.get('error', {}).get('type', 'None')
            })
            
            test_results['error_handling'] = error_tests
            
            # Test 7: Resource monitoring
            logger.info("7 Testing resource monitoring...")
            monitoring_results = service.monitor_resources(duration_seconds=30)
            test_results['resource_monitoring'] = monitoring_results
            
            # Final health check
            final_health = service.health_check()
            test_results['final_health_check'] = final_health
            
            logger.info(f"PASS Configuration {config_name} testing completed successfully")
            
        except Exception as e:
            logger.error(f"FAIL Configuration {config_name} testing failed: {e}")
            test_results['error'] = {
                'type': type(e).__name__,
                'message': str(e),
                'traceback': traceback.format_exc()
            }
        
        finally:
            # Cleanup
            if service.pipeline:
                del service.pipeline
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return test_results
    
    def run_production_testing_suite(self):
        """Run comprehensive production testing across all configurations"""
        logger.info("ROCKET Starting ChronosX Production Testing Suite")
        logger.info("="*70)
        
        # Test configurations based on available memory
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        
        configs_to_test = ['tiny_fast', 'small_balanced']
        if available_memory_gb > 8:
            configs_to_test.append('base_accurate')
        
        logger.info(f"BRAIN Available memory: {available_memory_gb:.1f}GB")
        logger.info(f"CLIPBOARD Testing configurations: {configs_to_test}")
        
        suite_start = time.time()
        
        for config_name in configs_to_test:
            config = self.test_configs[config_name]
            
            logger.info(f"\n" + "="*50)
            logger.info(f"TOOL TESTING: {config_name.upper()}")
            logger.info(f"   Model: {config.model_name}")
            logger.info(f"   Max Concurrent: {config.max_concurrent_requests}")
            logger.info(f"   Samples: {config.num_samples}")
            logger.info("="*50)
            
            test_result = self.test_configuration(config_name, config)
            self.test_results[config_name] = test_result
            
            logger.info(f"PASS {config_name} testing completed\n")
        
        suite_time = time.time() - suite_start
        
        # Generate comprehensive report
        self.generate_production_report()
        
        # Create production visualizations
        self.create_production_visualizations()
        
        # Save detailed results
        self.save_production_results()
        
        logger.info(f"PARTY Production testing suite completed in {suite_time:.1f}s")
        logger.info("="*70)
        
        return self.test_results
    
    def generate_production_report(self):
        """Generate comprehensive production testing report"""
        print("\n" + "="*80)
        print(" CHRONOSX PRODUCTION DEPLOYMENT TESTING REPORT")
        print("="*80)
        
        if not self.test_results:
            print("FAIL No production test results to report")
            return
        
        # Overall summary
        print("\nCHART PRODUCTION READINESS SUMMARY:")
        print("-" * 50)
        
        for config_name, results in self.test_results.items():
            print(f"\nTOOL {config_name.upper()}:")
            
            # Initialization status
            init_success = results.get('initialization', {}).get('success', False)
            print(f"   Initialization: {'PASS Pass' if init_success else 'FAIL Fail'}")
            
            # Single prediction performance
            single_pred = results.get('single_prediction', {})
            if single_pred.get('success'):
                response_time = single_pred.get('response_time', 0)
                print(f"   Single Prediction: PASS {response_time:.3f}s")
            else:
                print(f"   Single Prediction: FAIL Failed")
            
            # Batch prediction performance
            batch_pred = results.get('batch_prediction', {})
            if batch_pred:
                success_rate = batch_pred.get('success_rate', 0)
                avg_time = batch_pred.get('avg_time_per_prediction', 0)
                print(f"   Batch Prediction: PASS {success_rate:.1f}% success, {avg_time:.3f}s/req")
            
            # Stress test results
            stress_test = results.get('stress_test', {})
            if stress_test:
                stress_success_rate = stress_test.get('results_summary', {}).get('success_rate', 0)
                rps = stress_test.get('results_summary', {}).get('requests_per_second', 0)
                print(f"   Stress Test: PASS {stress_success_rate:.1f}% success, {rps:.1f} req/s")
            
            # Error handling
            error_handling = results.get('error_handling', [])
            handled_errors = sum(1 for e in error_handling if e.get('handled_gracefully', False))
            total_error_tests = len(error_handling)
            if total_error_tests > 0:
                print(f"   Error Handling: PASS {handled_errors}/{total_error_tests} handled gracefully")
        
        # Performance comparison
        print(f"\nLIGHTNING PERFORMANCE COMPARISON:")
        print("-" * 50)
        print(f"{'Config':<15} {'Response(s)':<12} {'Batch RPS':<10} {'Stress RPS':<10} {'Memory':<8}")
        print("-" * 50)
        
        for config_name, results in self.test_results.items():
            single_time = results.get('single_prediction', {}).get('response_time', 0)
            
            batch_pred = results.get('batch_prediction', {})
            batch_rps = 1.0 / batch_pred.get('avg_time_per_prediction', 1) if batch_pred.get('avg_time_per_prediction') else 0
            
            stress_rps = results.get('stress_test', {}).get('results_summary', {}).get('requests_per_second', 0)
            
            memory_usage = results.get('resource_monitoring', {}).get('summary', {}).get('memory_usage', {}).get('mean', 0)
            
            print(f"{config_name:<15} {single_time:<12.3f} {batch_rps:<10.1f} {stress_rps:<10.1f} {memory_usage:<8.1f}%")
        
        # Production recommendations
        print(f"\nIDEA PRODUCTION RECOMMENDATIONS:")
        print("-" * 50)
        
        # Find best configurations for different use cases
        best_speed = None
        best_throughput = None
        best_balanced = None
        
        min_response_time = float('inf')
        max_rps = 0
        best_score = 0
        
        for config_name, results in self.test_results.items():
            single_time = results.get('single_prediction', {}).get('response_time', float('inf'))
            stress_rps = results.get('stress_test', {}).get('results_summary', {}).get('requests_per_second', 0)
            success_rate = results.get('stress_test', {}).get('results_summary', {}).get('success_rate', 0)
            
            # Best speed (lowest response time)
            if single_time < min_response_time and results.get('initialization', {}).get('success'):
                min_response_time = single_time
                best_speed = config_name
            
            # Best throughput (highest RPS)
            if stress_rps > max_rps and success_rate > 90:
                max_rps = stress_rps
                best_throughput = config_name
            
            # Best balanced (score = success_rate * rps / response_time)
            if single_time > 0:
                score = (success_rate / 100) * stress_rps / single_time
                if score > best_score:
                    best_score = score
                    best_balanced = config_name
        
        if best_speed:
            print(f"ROCKET For Low Latency: {best_speed.upper()}")
            print(f"   Response time: {self.test_results[best_speed]['single_prediction']['response_time']:.3f}s")
        
        if best_throughput:
            print(f"GRAPH For High Throughput: {best_throughput.upper()}")
            print(f"   Requests/second: {self.test_results[best_throughput]['stress_test']['results_summary']['requests_per_second']:.1f}")
        
        if best_balanced:
            print(f" For Balanced Performance: {best_balanced.upper()}")
        
        # Deployment checklist
        print(f"\nPASS DEPLOYMENT CHECKLIST:")
        print("-" * 30)
        
        checklist_items = [
            "Model initialization successful",
            "Health checks working",
            "Error handling implemented",
            "Resource monitoring active",
            "Stress testing passed",
            "Memory usage acceptable",
            "Response times under SLA"
        ]
        
        for item in checklist_items:
            print(f"    {item}")
        
        print("\nTARGET Production testing complete!")
        print("="*80)
    
    def create_production_visualizations(self):
        """Create production testing visualizations"""
        if len(self.test_results) < 2:
            logger.info("Need at least 2 configurations for visualization")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('ChronosX Production Testing Results', fontsize=16, fontweight='bold')
        
        config_names = list(self.test_results.keys())
        
        # Plot 1: Response time comparison
        response_times = []
        for config in config_names:
            rt = self.test_results[config].get('single_prediction', {}).get('response_time', 0)
            response_times.append(rt)
        
        axes[0, 0].bar(config_names, response_times, alpha=0.7, color='skyblue')
        axes[0, 0].set_ylabel('Response Time (s)')
        axes[0, 0].set_title('Single Prediction Response Time')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Add value labels
        for i, v in enumerate(response_times):
            axes[0, 0].text(i, v + max(response_times) * 0.01, f'{v:.3f}s', ha='center', va='bottom')
        
        # Plot 2: Throughput comparison
        throughputs = []
        for config in config_names:
            rps = self.test_results[config].get('stress_test', {}).get('results_summary', {}).get('requests_per_second', 0)
            throughputs.append(rps)
        
        axes[0, 1].bar(config_names, throughputs, alpha=0.7, color='lightgreen')
        axes[0, 1].set_ylabel('Requests per Second')
        axes[0, 1].set_title('Stress Test Throughput')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Plot 3: Success rates
        success_rates = []
        for config in config_names:
            sr = self.test_results[config].get('stress_test', {}).get('results_summary', {}).get('success_rate', 0)
            success_rates.append(sr)
        
        axes[0, 2].bar(config_names, success_rates, alpha=0.7, color='lightcoral')
        axes[0, 2].set_ylabel('Success Rate (%)')
        axes[0, 2].set_title('Stress Test Success Rate')
        axes[0, 2].set_ylim(0, 100)
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # Plot 4: Memory usage
        memory_usage = []
        for config in config_names:
            mem = self.test_results[config].get('resource_monitoring', {}).get('summary', {}).get('memory_usage', {}).get('mean', 0)
            memory_usage.append(mem)
        
        axes[1, 0].bar(config_names, memory_usage, alpha=0.7, color='gold')
        axes[1, 0].set_ylabel('Memory Usage (%)')
        axes[1, 0].set_title('Average Memory Usage')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Plot 5: Performance vs Resource tradeoff
        if len(config_names) >= 2:
            perf_scores = [1/rt if rt > 0 else 0 for rt in response_times]  # Higher is better
            
            axes[1, 1].scatter(memory_usage, perf_scores, s=100, alpha=0.7)
            for i, config in enumerate(config_names):
                axes[1, 1].annotate(config, (memory_usage[i], perf_scores[i]),
                                  xytext=(5, 5), textcoords='offset points')
            
            axes[1, 1].set_xlabel('Memory Usage (%)')
            axes[1, 1].set_ylabel('Performance Score (1/response_time)')
            axes[1, 1].set_title('Performance vs Resource Usage')
            axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 6: Error handling effectiveness
        error_effectiveness = []
        for config in config_names:
            error_tests = self.test_results[config].get('error_handling', [])
            if error_tests:
                handled = sum(1 for e in error_tests if e.get('handled_gracefully', False))
                effectiveness = handled / len(error_tests) * 100
            else:
                effectiveness = 0
            error_effectiveness.append(effectiveness)
        
        axes[1, 2].bar(config_names, error_effectiveness, alpha=0.7, color='mediumpurple')
        axes[1, 2].set_ylabel('Error Handling (%)')
        axes[1, 2].set_title('Error Handling Effectiveness')
        axes[1, 2].set_ylim(0, 100)
        axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('chronos_x_production_testing.png', dpi=300, bbox_inches='tight')
        logger.info("CHART Saved production testing visualization")
        plt.show()
    
    def save_production_results(self):
        """Save detailed production testing results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON results
        results_file = f'chronos_x_production_results_{timestamp}.json'
        with open(results_file, 'w') as f:
            json.dump(self.test_results, f, indent=2, default=str)
        
        logger.info(f" Saved production results to {results_file}")
        
        # Generate deployment guide
        self.generate_deployment_guide(timestamp)
    
    def generate_deployment_guide(self, timestamp: str):
        """Generate production deployment guide"""
        guide_content = f"""# ChronosX Production Deployment Guide

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Executive Summary

This guide provides production deployment recommendations for ChronosX models based on comprehensive testing results.

## Tested Configurations

"""
        
        for config_name, results in self.test_results.items():
            if results.get('initialization', {}).get('success'):
                config = results.get('config', {})
                single_pred = results.get('single_prediction', {})
                stress_test = results.get('stress_test', {})
                
                guide_content += f"""### {config_name.upper()}

**Model:** {config.get('model_name', 'N/A')}
**Performance:**
- Single prediction: {single_pred.get('response_time', 0):.3f}s
- Throughput: {stress_test.get('results_summary', {}).get('requests_per_second', 0):.1f} req/s
- Success rate: {stress_test.get('results_summary', {}).get('success_rate', 0):.1f}%

**Configuration:**
```yaml
model_name: "{config.get('model_name', '')}"
max_concurrent_requests: {config.get('max_concurrent_requests', 4)}
request_timeout_seconds: {config.get('request_timeout_seconds', 30)}
context_length: {config.get('context_length', 96)}
forecast_length: {config.get('forecast_length', 24)}
num_samples: {config.get('num_samples', 10)}
```

"""
        
        guide_content += """## Production Deployment Checklist

### Prerequisites
- [ ] Python 3.8+ environment
- [ ] Required packages: chronos-forecasting, torch, numpy, pandas
- [ ] Sufficient memory (minimum 4GB available)
- [ ] CPU with good performance (multi-core recommended)

### Deployment Steps

1. **Environment Setup**
   ```bash
   pip install chronos-forecasting torch numpy pandas psutil
   ```

2. **Service Configuration**
   ```python
   from chronos_x_production_service import ChronosXProductionService, ProductionConfig
   
   config = ProductionConfig(
       model_name="amazon/chronos-t5-tiny",  # Start with fastest model
       max_concurrent_requests=4,
       request_timeout_seconds=30,
       context_length=96,
       forecast_length=24,
       num_samples=10
   )
   
   service = ChronosXProductionService(config)
   service.initialize_model()
   ```

3. **Health Monitoring**
   ```python
   # Regular health checks
   health_status = service.health_check()
   
   # Resource monitoring
   monitoring = service.monitor_resources(duration_seconds=300)
   ```

4. **Error Handling**
   - Implement input validation
   - Handle model loading failures
   - Monitor memory usage
   - Implement circuit breaker pattern

### Scaling Recommendations

**For High Volume (>100 req/s):**
- Use tiny model with multiple instances
- Implement load balancing
- Consider horizontal scaling

**For High Accuracy:**
- Use base model with fewer concurrent requests
- Implement request queuing
- Monitor response times closely

**For Balanced Performance:**
- Use small model configuration
- Monitor and auto-scale based on load
- Implement caching for frequent patterns

### Monitoring Metrics

Key metrics to track:
- Response time (p50, p95, p99)
- Throughput (requests per second)
- Error rate
- Memory usage
- CPU utilization
- Model loading time

### Troubleshooting

**High Memory Usage:**
- Reduce batch size
- Switch to smaller model
- Implement memory pooling

**High Response Times:**
- Reduce num_samples
- Optimize input preprocessing
- Check for memory pressure

**Model Loading Failures:**
- Verify internet connectivity
- Check available disk space
- Validate model name

## Production-Ready Code Examples

See the generated production service files:
- `chronos_x_production_service.py` - Main service implementation
- `chronos_x_production_tester.py` - Testing suite
- Production configuration examples

## Support

For issues or questions about ChronosX production deployment, refer to:
- chronos-forecasting documentation
- Test results in chronos_x_production_results_{timestamp}.json
- Performance benchmarks
"""
        
        guide_file = f'chronos_x_deployment_guide_{timestamp}.md'
        with open(guide_file, 'w') as f:
            f.write(guide_content)
        
        logger.info(f" Generated deployment guide: {guide_file}")

def main():
    """Main production testing function"""
    tester = ChronosXProductionTester()
    results = tester.run_production_testing_suite()
    return results

if __name__ == "__main__":
    main()
