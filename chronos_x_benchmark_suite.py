"""
ChronosX Performance Benchmarking Suite

Comprehensive benchmarking of ChronosX variants against traditional models
with detailed performance metrics, statistical analysis, and recommendations.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from chronos import ChronosPipeline
import time
import psutil
import logging
from typing import Dict, List, Tuple, Optional
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy import stats
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChronosXBenchmarkSuite:
    """Comprehensive benchmarking suite for ChronosX models"""
    
    def __init__(self):
        self.chronos_models = {
            'chronos_tiny': 'amazon/chronos-t5-tiny',
            'chronos_small': 'amazon/chronos-t5-small',
            'chronos_base': 'amazon/chronos-t5-base',
        }
        
        # Traditional baseline models for comparison
        self.baseline_models = ['naive_seasonal', 'linear_trend', 'moving_average']
        
        self.benchmark_results = {}
        self.statistical_tests = {}
        
    def generate_benchmark_datasets(self) -> Dict[str, Dict]:
        """Generate diverse datasets for comprehensive benchmarking"""
        np.random.seed(42)
        datasets = {}
        
        # Dataset 1: Trending time series
        t = np.arange(1000)
        trend = 0.05 * t + 50
        noise = np.random.normal(0, 2, 1000)
        trending = trend + noise
        datasets['trending'] = {
            'data': trending,
            'description': 'Linear trending data with noise',
            'characteristics': 'trend+noise'
        }
        
        # Dataset 2: Seasonal time series
        seasonal = 50 + 10 * np.sin(2 * np.pi * t / 50) + 5 * np.sin(2 * np.pi * t / 12)
        seasonal += np.random.normal(0, 1, 1000)
        datasets['seasonal'] = {
            'data': seasonal,
            'description': 'Multi-seasonal data with weekly and daily patterns',
            'characteristics': 'seasonal+noise'
        }
        
        # Dataset 3: Non-stationary financial-like
        returns = np.random.normal(0.001, 0.02, 1000)
        financial = 100 * np.exp(np.cumsum(returns))
        datasets['financial'] = {
            'data': financial,
            'description': 'Random walk financial time series',
            'characteristics': 'non-stationary+volatility'
        }
        
        # Dataset 4: Complex multi-component
        complex_ts = (trend * 0.5 + 
                     seasonal + 
                     5 * np.sin(2 * np.pi * t / 200) +  # Long cycle
                     np.random.normal(0, 3, 1000))
        datasets['complex'] = {
            'data': complex_ts,
            'description': 'Multi-component: trend + seasonality + cycles',
            'characteristics': 'trend+seasonal+cyclical+noise'
        }
        
        # Dataset 5: Real synthetic data if available
        try:
            df = pd.read_csv('synthetic_timeseries.csv')
            real_data = df.iloc[:1000, 1].values
            datasets['real_synthetic'] = {
                'data': real_data,
                'description': 'Real synthetic time series from project',
                'characteristics': 'real_project_data'
            }
            logger.info("✅ Loaded real synthetic data for benchmarking")
        except:
            logger.info("ℹ️ Could not load real synthetic data, using generated only")
        
        return datasets
    
    def implement_baseline_model(self, model_name: str, train_data: np.ndarray, 
                                forecast_length: int) -> np.ndarray:
        """Implement baseline forecasting models"""
        
        if model_name == 'naive_seasonal':
            # Simple seasonal naive (repeat last seasonal period)
            season_length = min(50, len(train_data) // 4)  # Assume weekly seasonality
            if len(train_data) >= season_length:
                seasonal_pattern = train_data[-season_length:]
                forecast = np.tile(seasonal_pattern, (forecast_length // season_length) + 1)[:forecast_length]
            else:
                forecast = np.full(forecast_length, train_data[-1])
            
        elif model_name == 'linear_trend':
            # Simple linear trend extrapolation
            if len(train_data) >= 10:
                x = np.arange(len(train_data))
                y = train_data
                slope, intercept = np.polyfit(x, y, 1)
                future_x = np.arange(len(train_data), len(train_data) + forecast_length)
                forecast = slope * future_x + intercept
            else:
                forecast = np.full(forecast_length, train_data[-1])
                
        elif model_name == 'moving_average':
            # Moving average with trend
            window = min(20, len(train_data) // 4)
            if len(train_data) >= window:
                ma = np.mean(train_data[-window:])
                # Add simple trend
                trend = (train_data[-1] - train_data[-min(window, len(train_data))]) / min(window, len(train_data))
                forecast = ma + trend * np.arange(1, forecast_length + 1)
            else:
                forecast = np.full(forecast_length, np.mean(train_data))
        
        else:
            # Fallback to naive
            forecast = np.full(forecast_length, train_data[-1])
        
        return forecast
    
    def benchmark_chronos_model(self, model_name: str, model_path: str,
                               dataset: np.ndarray, dataset_name: str,
                               context_length: int = 96, forecast_length: int = 24,
                               num_samples: int = 20) -> Dict:
        """Benchmark a specific ChronosX model"""
        
        logger.info(f"🔬 Benchmarking {model_name} on {dataset_name}")
        
        try:
            # Load model
            load_start = time.time()
            pipeline = ChronosPipeline.from_pretrained(
                model_path,
                device_map="cpu",
                torch_dtype=torch.float32,
            )
            load_time = time.time() - load_start
            
            # Prepare multiple test windows
            total_length = len(dataset)
            test_windows = []
            
            # Create multiple test windows for robust evaluation
            if total_length > context_length + forecast_length + 100:
                # Multiple windows from different parts of the series
                step_size = (total_length - context_length - forecast_length) // 5
                for i in range(5):
                    start_idx = i * step_size
                    end_context = start_idx + context_length
                    end_forecast = end_context + forecast_length
                    
                    if end_forecast <= total_length:
                        test_windows.append({
                            'context': dataset[start_idx:end_context],
                            'truth': dataset[end_context:end_forecast],
                            'window_id': i
                        })
            else:
                # Single window from end
                test_windows.append({
                    'context': dataset[-(context_length + forecast_length):-forecast_length],
                    'truth': dataset[-forecast_length:],
                    'window_id': 0
                })
            
            # Benchmark across all windows
            window_results = []
            inference_times = []
            
            for window in test_windows:
                context_tensor = torch.tensor(window['context'], dtype=torch.float32).unsqueeze(0)
                
                # Time inference
                inference_start = time.time()
                forecast = pipeline.predict(
                    context=context_tensor,
                    prediction_length=forecast_length,
                    num_samples=num_samples
                )
                inference_time = time.time() - inference_start
                inference_times.append(inference_time)
                
                # Process results
                forecast_np = forecast.cpu().numpy()
                mean_forecast = np.mean(forecast_np, axis=0).flatten()
                std_forecast = np.std(forecast_np, axis=0).flatten()
                
                # Calculate metrics
                true_values = window['truth']
                min_len = min(len(mean_forecast), len(true_values))
                pred_vals = mean_forecast[:min_len]
                true_vals = true_values[:min_len]
                
                window_metrics = {
                    'mae': mean_absolute_error(true_vals, pred_vals),
                    'rmse': np.sqrt(mean_squared_error(true_vals, pred_vals)),
                    'mape': np.mean(np.abs((pred_vals - true_vals) / (np.abs(true_vals) + 1e-8))) * 100,
                    'smape': np.mean(2 * np.abs(pred_vals - true_vals) / (np.abs(pred_vals) + np.abs(true_vals) + 1e-8)) * 100,
                    'correlation': np.corrcoef(pred_vals, true_vals)[0, 1] if len(pred_vals) > 1 else 0,
                    'mean_uncertainty': np.mean(std_forecast[:min_len]),
                    'forecast_values': pred_vals,
                    'uncertainty_values': std_forecast[:min_len],
                    'true_values': true_vals
                }
                
                window_results.append(window_metrics)
            
            # Aggregate results across windows
            aggregated_metrics = {}
            for metric in ['mae', 'rmse', 'mape', 'smape', 'correlation', 'mean_uncertainty']:
                values = [w[metric] for w in window_results if not np.isnan(w[metric])]
                if values:
                    aggregated_metrics[f'{metric}_mean'] = np.mean(values)
                    aggregated_metrics[f'{metric}_std'] = np.std(values)
                    aggregated_metrics[f'{metric}_median'] = np.median(values)
                else:
                    aggregated_metrics[f'{metric}_mean'] = np.nan
                    aggregated_metrics[f'{metric}_std'] = np.nan
                    aggregated_metrics[f'{metric}_median'] = np.nan
            
            # Performance metrics
            avg_inference_time = np.mean(inference_times)
            std_inference_time = np.std(inference_times)
            
            result = {
                'model_name': model_name,
                'dataset_name': dataset_name,
                'success': True,
                'load_time': load_time,
                'avg_inference_time': avg_inference_time,
                'std_inference_time': std_inference_time,
                'num_test_windows': len(test_windows),
                'aggregated_metrics': aggregated_metrics,
                'window_results': window_results,
                'context_length': context_length,
                'forecast_length': forecast_length,
                'num_samples': num_samples
            }
            
            logger.info(f"✅ {model_name} on {dataset_name}: "
                      f"MAE={aggregated_metrics['mae_mean']:.3f}±{aggregated_metrics['mae_std']:.3f}, "
                      f"Time={avg_inference_time:.2f}±{std_inference_time:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ {model_name} on {dataset_name} failed: {e}")
            return {
                'model_name': model_name,
                'dataset_name': dataset_name,
                'success': False,
                'error': str(e)
            }
    
    def benchmark_baseline_model(self, model_name: str, dataset: np.ndarray, 
                                dataset_name: str, context_length: int = 96, 
                                forecast_length: int = 24) -> Dict:
        """Benchmark baseline model"""
        
        logger.info(f"📊 Benchmarking baseline {model_name} on {dataset_name}")
        
        try:
            # Prepare test windows (same as ChronosX)
            total_length = len(dataset)
            test_windows = []
            
            if total_length > context_length + forecast_length + 100:
                step_size = (total_length - context_length - forecast_length) // 5
                for i in range(5):
                    start_idx = i * step_size
                    end_context = start_idx + context_length
                    end_forecast = end_context + forecast_length
                    
                    if end_forecast <= total_length:
                        test_windows.append({
                            'context': dataset[start_idx:end_context],
                            'truth': dataset[end_context:end_forecast],
                            'window_id': i
                        })
            else:
                test_windows.append({
                    'context': dataset[-(context_length + forecast_length):-forecast_length],
                    'truth': dataset[-forecast_length:],
                    'window_id': 0
                })
            
            # Benchmark across windows
            window_results = []
            inference_times = []
            
            for window in test_windows:
                # Time inference
                inference_start = time.time()
                forecast = self.implement_baseline_model(model_name, window['context'], forecast_length)
                inference_time = time.time() - inference_start
                inference_times.append(inference_time)
                
                # Calculate metrics
                true_values = window['truth']
                min_len = min(len(forecast), len(true_values))
                pred_vals = forecast[:min_len]
                true_vals = true_values[:min_len]
                
                window_metrics = {
                    'mae': mean_absolute_error(true_vals, pred_vals),
                    'rmse': np.sqrt(mean_squared_error(true_vals, pred_vals)),
                    'mape': np.mean(np.abs((pred_vals - true_vals) / (np.abs(true_vals) + 1e-8))) * 100,
                    'smape': np.mean(2 * np.abs(pred_vals - true_vals) / (np.abs(pred_vals) + np.abs(true_vals) + 1e-8)) * 100,
                    'correlation': np.corrcoef(pred_vals, true_vals)[0, 1] if len(pred_vals) > 1 else 0,
                    'forecast_values': pred_vals,
                    'true_values': true_vals
                }
                
                window_results.append(window_metrics)
            
            # Aggregate results
            aggregated_metrics = {}
            for metric in ['mae', 'rmse', 'mape', 'smape', 'correlation']:
                values = [w[metric] for w in window_results if not np.isnan(w[metric])]
                if values:
                    aggregated_metrics[f'{metric}_mean'] = np.mean(values)
                    aggregated_metrics[f'{metric}_std'] = np.std(values)
                    aggregated_metrics[f'{metric}_median'] = np.median(values)
                else:
                    aggregated_metrics[f'{metric}_mean'] = np.nan
                    aggregated_metrics[f'{metric}_std'] = np.nan
                    aggregated_metrics[f'{metric}_median'] = np.nan
            
            result = {
                'model_name': model_name,
                'dataset_name': dataset_name,
                'success': True,
                'load_time': 0.0,  # No loading for baselines
                'avg_inference_time': np.mean(inference_times),
                'std_inference_time': np.std(inference_times),
                'num_test_windows': len(test_windows),
                'aggregated_metrics': aggregated_metrics,
                'window_results': window_results,
                'context_length': context_length,
                'forecast_length': forecast_length
            }
            
            logger.info(f"✅ {model_name} on {dataset_name}: "
                      f"MAE={aggregated_metrics['mae_mean']:.3f}±{aggregated_metrics['mae_std']:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Baseline {model_name} on {dataset_name} failed: {e}")
            return {
                'model_name': model_name,
                'dataset_name': dataset_name,
                'success': False,
                'error': str(e)
            }
    
    def run_comprehensive_benchmark(self):
        """Run comprehensive benchmark across all models and datasets"""
        logger.info("🚀 Starting comprehensive ChronosX benchmark suite...")
        
        # Generate datasets
        datasets = self.generate_benchmark_datasets()
        logger.info(f"📊 Generated {len(datasets)} benchmark datasets")
        
        # Determine which ChronosX models to test based on memory
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        chronos_to_test = ['chronos_tiny', 'chronos_small']
        
        if available_memory_gb > 8:
            chronos_to_test.append('chronos_base')
        
        logger.info(f"🧠 Testing ChronosX models: {chronos_to_test} (Memory: {available_memory_gb:.1f}GB)")
        
        all_results = {}
        
        # Benchmark ChronosX models
        for model_name in chronos_to_test:
            model_path = self.chronos_models[model_name]
            
            for dataset_name, dataset_info in datasets.items():
                result_key = f"{model_name}_{dataset_name}"
                
                result = self.benchmark_chronos_model(
                    model_name, model_path, 
                    dataset_info['data'], dataset_name
                )
                
                all_results[result_key] = result
                
                # Clear memory
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Benchmark baseline models
        for model_name in self.baseline_models:
            for dataset_name, dataset_info in datasets.items():
                result_key = f"{model_name}_{dataset_name}"
                
                result = self.benchmark_baseline_model(
                    model_name, dataset_info['data'], dataset_name
                )
                
                all_results[result_key] = result
        
        self.benchmark_results = all_results
        
        # Perform statistical analysis
        self.perform_statistical_analysis()
        
        # Generate comprehensive report
        self.generate_benchmark_report()
        
        # Create visualizations
        self.create_benchmark_visualizations()
        
        # Save results
        self.save_benchmark_results()
        
        return all_results
    
    def perform_statistical_analysis(self):
        """Perform statistical significance tests"""
        logger.info("📈 Performing statistical analysis...")
        
        successful_results = {k: v for k, v in self.benchmark_results.items() if v.get('success', False)}
        
        # Group by dataset
        datasets = list(set([r['dataset_name'] for r in successful_results.values()]))
        models = list(set([r['model_name'] for r in successful_results.values()]))
        
        statistical_tests = {}
        
        for dataset in datasets:
            dataset_results = {k: v for k, v in successful_results.items() 
                             if v['dataset_name'] == dataset}
            
            if len(dataset_results) < 2:
                continue
            
            # Compare models pairwise
            model_maes = {}
            for result_key, result in dataset_results.items():
                model_name = result['model_name']
                mae_values = [w['mae'] for w in result['window_results']]
                model_maes[model_name] = mae_values
            
            # Statistical tests between models
            dataset_tests = {}
            model_names = list(model_maes.keys())
            
            for i, model1 in enumerate(model_names):
                for j, model2 in enumerate(model_names[i+1:], i+1):
                    if len(model_maes[model1]) > 1 and len(model_maes[model2]) > 1:
                        # Wilcoxon signed-rank test (non-parametric)
                        try:
                            statistic, p_value = stats.wilcoxon(model_maes[model1], model_maes[model2])
                            dataset_tests[f"{model1}_vs_{model2}"] = {
                                'test': 'wilcoxon',
                                'statistic': statistic,
                                'p_value': p_value,
                                'significant': p_value < 0.05,
                                'model1_mean': np.mean(model_maes[model1]),
                                'model2_mean': np.mean(model_maes[model2]),
                                'better_model': model1 if np.mean(model_maes[model1]) < np.mean(model_maes[model2]) else model2
                            }
                        except:
                            # Fall back to t-test
                            try:
                                statistic, p_value = stats.ttest_rel(model_maes[model1], model_maes[model2])
                                dataset_tests[f"{model1}_vs_{model2}"] = {
                                    'test': 'paired_ttest',
                                    'statistic': statistic,
                                    'p_value': p_value,
                                    'significant': p_value < 0.05,
                                    'model1_mean': np.mean(model_maes[model1]),
                                    'model2_mean': np.mean(model_maes[model2]),
                                    'better_model': model1 if np.mean(model_maes[model1]) < np.mean(model_maes[model2]) else model2
                                }
                            except:
                                continue
            
            statistical_tests[dataset] = dataset_tests
        
        self.statistical_tests = statistical_tests
    
    def generate_benchmark_report(self):
        """Generate comprehensive benchmark report"""
        print("\n" + "="*80)
        print("🏆 CHRONOSX COMPREHENSIVE BENCHMARK REPORT")
        print("="*80)
        
        successful_results = {k: v for k, v in self.benchmark_results.items() if v.get('success', False)}
        
        if not successful_results:
            print("❌ No successful benchmark results to report")
            return
        
        # Overall performance summary
        print("\n📊 OVERALL PERFORMANCE SUMMARY:")
        print("-" * 80)
        print(f"{'Model':<20} {'Dataset':<15} {'MAE':<8} {'RMSE':<8} {'MAPE':<8} {'Time(s)':<8}")
        print("-" * 80)
        
        # Sort by MAE performance
        sorted_results = sorted(successful_results.items(), 
                              key=lambda x: x[1]['aggregated_metrics']['mae_mean'])
        
        for result_key, result in sorted_results:
            model = result['model_name']
            dataset = result['dataset_name']
            mae = result['aggregated_metrics']['mae_mean']
            rmse = result['aggregated_metrics']['rmse_mean']
            mape = result['aggregated_metrics']['mape_mean']
            time_taken = result['avg_inference_time']
            
            print(f"{model:<20} {dataset:<15} {mae:<8.3f} {rmse:<8.3f} "
                  f"{mape:<8.1f} {time_taken:<8.3f}")
        
        # Model ranking by dataset
        datasets = list(set([r['dataset_name'] for r in successful_results.values()]))
        
        print(f"\n🥇 MODEL RANKINGS BY DATASET:")
        print("-" * 50)
        
        for dataset in datasets:
            dataset_results = [(k, v) for k, v in successful_results.items() 
                             if v['dataset_name'] == dataset]
            dataset_results.sort(key=lambda x: x[1]['aggregated_metrics']['mae_mean'])
            
            print(f"\n{dataset.upper()}:")
            for i, (result_key, result) in enumerate(dataset_results[:5], 1):
                model = result['model_name']
                mae = result['aggregated_metrics']['mae_mean']
                print(f"  {i}. {model:<20} MAE: {mae:.3f}")
        
        # Speed analysis
        print(f"\n⚡ SPEED ANALYSIS:")
        print("-" * 50)
        
        # Group by model type
        chronos_results = [(k, v) for k, v in successful_results.items() 
                          if 'chronos' in v['model_name']]
        baseline_results = [(k, v) for k, v in successful_results.items() 
                           if 'chronos' not in v['model_name']]
        
        if chronos_results:
            chronos_times = [r[1]['avg_inference_time'] for r in chronos_results]
            print(f"ChronosX Models:")
            print(f"  Average Inference Time: {np.mean(chronos_times):.3f}±{np.std(chronos_times):.3f}s")
            
            # Fastest ChronosX
            fastest_chronos = min(chronos_results, key=lambda x: x[1]['avg_inference_time'])
            print(f"  Fastest: {fastest_chronos[1]['model_name']} "
                  f"({fastest_chronos[1]['avg_inference_time']:.3f}s)")
        
        if baseline_results:
            baseline_times = [r[1]['avg_inference_time'] for r in baseline_results]
            print(f"\nBaseline Models:")
            print(f"  Average Inference Time: {np.mean(baseline_times):.3f}±{np.std(baseline_times):.3f}s")
        
        # Statistical significance
        if self.statistical_tests:
            print(f"\n📈 STATISTICAL SIGNIFICANCE ANALYSIS:")
            print("-" * 50)
            
            for dataset, tests in self.statistical_tests.items():
                significant_tests = {k: v for k, v in tests.items() if v['significant']}
                
                if significant_tests:
                    print(f"\n{dataset.upper()} - Significant Differences (p<0.05):")
                    for test_name, test_result in significant_tests.items():
                        better_model = test_result['better_model']
                        p_value = test_result['p_value']
                        print(f"  {test_name}: {better_model} is significantly better (p={p_value:.4f})")
        
        # Overall recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        print("-" * 50)
        
        # Best overall model
        best_overall = min(successful_results.items(), 
                          key=lambda x: x[1]['aggregated_metrics']['mae_mean'])
        print(f"🏆 Best Overall Accuracy: {best_overall[1]['model_name']}")
        print(f"   Average MAE: {best_overall[1]['aggregated_metrics']['mae_mean']:.3f}")
        
        # Best speed
        if chronos_results:
            fastest_model = min(chronos_results, key=lambda x: x[1]['avg_inference_time'])
            print(f"🚀 Fastest ChronosX: {fastest_model[1]['model_name']}")
            print(f"   Inference Time: {fastest_model[1]['avg_inference_time']:.3f}s")
        
        # Best balanced
        # Score = 1/MAE + 1/Time (higher is better)
        balanced_scores = []
        for result_key, result in successful_results.items():
            if 'chronos' in result['model_name']:
                mae = result['aggregated_metrics']['mae_mean']
                time_taken = result['avg_inference_time']
                score = 1/(mae + 0.1) + 1/(time_taken + 0.1)
                balanced_scores.append((result_key, result, score))
        
        if balanced_scores:
            best_balanced = max(balanced_scores, key=lambda x: x[2])
            print(f"⚖️ Best Balanced: {best_balanced[1]['model_name']}")
            print(f"   MAE: {best_balanced[1]['aggregated_metrics']['mae_mean']:.3f}, "
                  f"Time: {best_balanced[1]['avg_inference_time']:.3f}s")
        
        print("\n🎉 Benchmark Complete!")
        print("="*80)
    
    def create_benchmark_visualizations(self):
        """Create comprehensive benchmark visualizations"""
        successful_results = {k: v for k, v in self.benchmark_results.items() if v.get('success', False)}
        
        if len(successful_results) < 2:
            logger.info("Need at least 2 successful results for visualization")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('ChronosX Comprehensive Benchmark Results', fontsize=16, fontweight='bold')
        
        # Plot 1: MAE comparison by dataset
        datasets = list(set([r['dataset_name'] for r in successful_results.values()]))
        models = list(set([r['model_name'] for r in successful_results.values()]))
        
        mae_data = []
        for dataset in datasets:
            for model in models:
                result_key = f"{model}_{dataset}"
                if result_key in successful_results:
                    mae = successful_results[result_key]['aggregated_metrics']['mae_mean']
                    mae_data.append({'Dataset': dataset, 'Model': model, 'MAE': mae})
        
        mae_df = pd.DataFrame(mae_data)
        if not mae_df.empty:
            mae_pivot = mae_df.pivot(index='Dataset', columns='Model', values='MAE')
            sns.heatmap(mae_pivot, annot=True, fmt='.3f', cmap='viridis_r', ax=axes[0, 0])
            axes[0, 0].set_title('MAE by Model and Dataset')
        
        # Plot 2: Speed comparison
        speed_data = []
        for result_key, result in successful_results.items():
            speed_data.append({
                'Model': result['model_name'],
                'Time': result['avg_inference_time'],
                'Type': 'ChronosX' if 'chronos' in result['model_name'] else 'Baseline'
            })
        
        speed_df = pd.DataFrame(speed_data)
        if not speed_df.empty:
            chronos_data = speed_df[speed_df['Type'] == 'ChronosX']
            baseline_data = speed_df[speed_df['Type'] == 'Baseline']
            
            if not chronos_data.empty:
                axes[0, 1].bar(chronos_data['Model'], chronos_data['Time'], 
                              alpha=0.7, label='ChronosX', color='skyblue')
            if not baseline_data.empty:
                axes[0, 1].bar(baseline_data['Model'], baseline_data['Time'], 
                              alpha=0.7, label='Baseline', color='lightcoral')
            
            axes[0, 1].set_ylabel('Inference Time (s)')
            axes[0, 1].set_title('Inference Speed Comparison')
            axes[0, 1].legend()
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Plot 3: Accuracy vs Speed scatter
        accuracy_speed_data = []
        for result_key, result in successful_results.items():
            if 'chronos' in result['model_name']:
                mae = result['aggregated_metrics']['mae_mean']
                time_taken = result['avg_inference_time']
                accuracy_speed_data.append({
                    'Model': result['model_name'],
                    'MAE': mae,
                    'Time': time_taken
                })
        
        if accuracy_speed_data:
            acc_speed_df = pd.DataFrame(accuracy_speed_data)
            axes[0, 2].scatter(acc_speed_df['Time'], acc_speed_df['MAE'], s=100, alpha=0.7)
            
            for _, row in acc_speed_df.iterrows():
                axes[0, 2].annotate(row['Model'].replace('chronos_', ''), 
                                  (row['Time'], row['MAE']),
                                  xytext=(5, 5), textcoords='offset points')
            
            axes[0, 2].set_xlabel('Inference Time (s)')
            axes[0, 2].set_ylabel('MAE (lower is better)')
            axes[0, 2].set_title('Accuracy vs Speed Tradeoff')
            axes[0, 2].grid(True, alpha=0.3)
        
        # Plot 4: Model performance distribution
        all_maes = []
        all_models = []
        for result_key, result in successful_results.items():
            mae_values = [w['mae'] for w in result['window_results']]
            all_maes.extend(mae_values)
            all_models.extend([result['model_name']] * len(mae_values))
        
        if all_maes:
            performance_df = pd.DataFrame({'Model': all_models, 'MAE': all_maes})
            chronos_models = [m for m in performance_df['Model'].unique() if 'chronos' in m]
            
            if chronos_models:
                chronos_performance = performance_df[performance_df['Model'].isin(chronos_models)]
                axes[1, 0].boxplot([chronos_performance[chronos_performance['Model'] == m]['MAE'].values 
                                  for m in chronos_models], 
                                 labels=[m.replace('chronos_', '') for m in chronos_models])
                axes[1, 0].set_ylabel('MAE')
                axes[1, 0].set_title('ChronosX Performance Distribution')
                axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Plot 5: Dataset difficulty ranking
        dataset_difficulty = {}
        for dataset in datasets:
            dataset_maes = []
            for result_key, result in successful_results.items():
                if result['dataset_name'] == dataset and 'chronos' in result['model_name']:
                    dataset_maes.append(result['aggregated_metrics']['mae_mean'])
            
            if dataset_maes:
                dataset_difficulty[dataset] = np.mean(dataset_maes)
        
        if dataset_difficulty:
            sorted_datasets = sorted(dataset_difficulty.items(), key=lambda x: x[1])
            dataset_names = [d[0] for d in sorted_datasets]
            dataset_scores = [d[1] for d in sorted_datasets]
            
            bars = axes[1, 1].bar(dataset_names, dataset_scores, alpha=0.7, color='lightgreen')
            axes[1, 1].set_ylabel('Average MAE')
            axes[1, 1].set_title('Dataset Difficulty Ranking')
            axes[1, 1].tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, value in zip(bars, dataset_scores):
                axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{value:.3f}', ha='center', va='bottom')
        
        # Plot 6: Model improvement over baseline
        improvement_data = []
        for dataset in datasets:
            # Find best baseline for this dataset
            baseline_results = [(k, v) for k, v in successful_results.items() 
                              if v['dataset_name'] == dataset and 'chronos' not in v['model_name']]
            
            if baseline_results:
                best_baseline_mae = min(r[1]['aggregated_metrics']['mae_mean'] for r in baseline_results)
                
                # Compare ChronosX models
                chronos_results = [(k, v) for k, v in successful_results.items() 
                                 if v['dataset_name'] == dataset and 'chronos' in v['model_name']]
                
                for result_key, result in chronos_results:
                    chronos_mae = result['aggregated_metrics']['mae_mean']
                    improvement = (best_baseline_mae - chronos_mae) / best_baseline_mae * 100
                    
                    improvement_data.append({
                        'Model': result['model_name'],
                        'Dataset': dataset,
                        'Improvement_%': improvement
                    })
        
        if improvement_data:
            improvement_df = pd.DataFrame(improvement_data)
            improvement_pivot = improvement_df.pivot(index='Dataset', columns='Model', values='Improvement_%')
            sns.heatmap(improvement_pivot, annot=True, fmt='.1f', cmap='RdYlGn', center=0, ax=axes[1, 2])
            axes[1, 2].set_title('Improvement over Best Baseline (%)')
        
        plt.tight_layout()
        plt.savefig('chronos_x_comprehensive_benchmark.png', dpi=300, bbox_inches='tight')
        logger.info("📊 Saved comprehensive benchmark visualization")
        plt.show()
    
    def save_benchmark_results(self):
        """Save detailed benchmark results to JSON"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Prepare results for JSON serialization
        json_results = {}
        for key, result in self.benchmark_results.items():
            json_result = result.copy()
            
            # Convert numpy arrays to lists
            if 'window_results' in json_result:
                for window in json_result['window_results']:
                    for k, v in window.items():
                        if isinstance(v, np.ndarray):
                            window[k] = v.tolist()
            
            json_results[key] = json_result
        
        # Save results
        results_file = f'chronos_x_benchmark_results_{timestamp}.json'
        with open(results_file, 'w') as f:
            json.dump({
                'benchmark_results': json_results,
                'statistical_tests': self.statistical_tests,
                'timestamp': timestamp,
                'summary': self.generate_summary_stats()
            }, f, indent=2)
        
        logger.info(f"💾 Saved detailed results to {results_file}")
    
    def generate_summary_stats(self) -> Dict:
        """Generate summary statistics"""
        successful_results = {k: v for k, v in self.benchmark_results.items() if v.get('success', False)}
        
        if not successful_results:
            return {}
        
        # Overall statistics
        all_maes = [r['aggregated_metrics']['mae_mean'] for r in successful_results.values()]
        all_times = [r['avg_inference_time'] for r in successful_results.values()]
        
        chronos_results = [r for r in successful_results.values() if 'chronos' in r['model_name']]
        baseline_results = [r for r in successful_results.values() if 'chronos' not in r['model_name']]
        
        summary = {
            'total_experiments': len(successful_results),
            'chronos_experiments': len(chronos_results),
            'baseline_experiments': len(baseline_results),
            'overall_mae_stats': {
                'mean': np.mean(all_maes),
                'std': np.std(all_maes),
                'min': np.min(all_maes),
                'max': np.max(all_maes)
            },
            'overall_time_stats': {
                'mean': np.mean(all_times),
                'std': np.std(all_times),
                'min': np.min(all_times),
                'max': np.max(all_times)
            }
        }
        
        if chronos_results:
            chronos_maes = [r['aggregated_metrics']['mae_mean'] for r in chronos_results]
            chronos_times = [r['avg_inference_time'] for r in chronos_results]
            
            summary['chronos_stats'] = {
                'mae_mean': np.mean(chronos_maes),
                'mae_std': np.std(chronos_maes),
                'time_mean': np.mean(chronos_times),
                'time_std': np.std(chronos_times)
            }
        
        if baseline_results:
            baseline_maes = [r['aggregated_metrics']['mae_mean'] for r in baseline_results]
            baseline_times = [r['avg_inference_time'] for r in baseline_results]
            
            summary['baseline_stats'] = {
                'mae_mean': np.mean(baseline_maes),
                'mae_std': np.std(baseline_maes),
                'time_mean': np.mean(baseline_times),
                'time_std': np.std(baseline_times)
            }
        
        return summary

def main():
    """Main benchmark function"""
    benchmark_suite = ChronosXBenchmarkSuite()
    results = benchmark_suite.run_comprehensive_benchmark()
    return results

if __name__ == "__main__":
    main()
