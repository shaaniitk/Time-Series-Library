"""
ChronosX Model Size Comparison and Testing

This script comprehensively tests different ChronosX model sizes
(tiny, small, base, large) with real data to find optimal performance/speed tradeoffs.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from chronos import ChronosPipeline
import time
import psutil
import logging
from typing import Dict, List, Tuple
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChronosXModelComparator:
    """Comprehensive comparison of ChronosX model sizes"""
    
    def __init__(self):
        self.model_configs = {
            'tiny': {
                'name': 'amazon/chronos-t5-tiny',
                'expected_params': '8M',
                'description': 'Fastest inference, lowest memory'
            },
            'small': {
                'name': 'amazon/chronos-t5-small', 
                'expected_params': '60M',
                'description': 'Balanced performance and speed'
            },
            'base': {
                'name': 'amazon/chronos-t5-base',
                'expected_params': '200M', 
                'description': 'High accuracy, moderate speed'
            },
            'large': {
                'name': 'amazon/chronos-t5-large',
                'expected_params': '700M',
                'description': 'Maximum accuracy, slower inference'
            }
        }
        
        self.results = {}
        
    def load_test_datasets(self) -> Dict[str, np.ndarray]:
        """Load multiple test datasets for comprehensive evaluation"""
        datasets = {}
        
        # Dataset 1: Synthetic data
        try:
            df = pd.read_csv('synthetic_timeseries.csv')
            datasets['synthetic'] = df.iloc[:, 1].values
            logger.info(f"Loaded synthetic dataset: {datasets['synthetic'].shape}")
        except:
            logger.warning("Could not load synthetic_timeseries.csv")
        
        # Dataset 2: Training data sample
        try:
            df = pd.read_csv('synthetic_timeseries_train.csv')
            datasets['training_sample'] = df.iloc[:1000, 1].values
            logger.info(f"Loaded training sample: {datasets['training_sample'].shape}")
        except:
            logger.warning("Could not load synthetic_timeseries_train.csv")
        
        # Dataset 3: Generated financial data
        np.random.seed(42)
        t = np.arange(1000)
        financial = 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, 1000)))
        datasets['financial'] = financial
        logger.info(f"Generated financial dataset: {financial.shape}")
        
        # Dataset 4: Generated seasonal data
        seasonal = 50 + 10 * np.sin(2 * np.pi * t / 50) + 5 * np.sin(2 * np.pi * t / 12) + np.random.normal(0, 2, 1000)
        datasets['seasonal'] = seasonal
        logger.info(f"Generated seasonal dataset: {seasonal.shape}")
        
        return datasets
    
    def measure_system_resources(self) -> Dict:
        """Measure current system resource usage"""
        return {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_available_gb': psutil.virtual_memory().available / (1024**3)
        }
    
    def test_model_size(self, model_size: str, test_data: np.ndarray, 
                       context_length: int = 96, forecast_length: int = 24) -> Dict:
        """Test a specific ChronosX model size"""
        
        model_config = self.model_configs[model_size]
        logger.info(f"Testing {model_size} model: {model_config['description']}")
        
        # Pre-test resource measurement
        pre_resources = self.measure_system_resources()
        
        try:
            # Model loading
            logger.info(f"Loading {model_size} model...")
            load_start = time.time()
            
            pipeline = ChronosPipeline.from_pretrained(
                model_config['name'],
                device_map="cpu",  # Use CPU for fair comparison
                torch_dtype=torch.float32,
            )
            
            load_time = time.time() - load_start
            
            # Post-load resource measurement
            post_load_resources = self.measure_system_resources()
            memory_increase = post_load_resources['memory_percent'] - pre_resources['memory_percent']
            
            # Prepare test data
            context = torch.tensor(test_data[-context_length:], dtype=torch.float32).unsqueeze(0)
            
            # Single inference timing
            logger.info(f"Running single inference...")
            inference_times = []
            
            for i in range(5):  # Multiple runs for average
                inference_start = time.time()
                
                forecast = pipeline.predict(
                    context=context,
                    prediction_length=forecast_length,
                    num_samples=20  # Standard uncertainty samples
                )
                
                inference_time = time.time() - inference_start
                inference_times.append(inference_time)
            
            avg_inference_time = np.mean(inference_times)
            std_inference_time = np.std(inference_times)
            
            # Process forecast results
            forecast_np = forecast.cpu().numpy()
            mean_forecast = np.mean(forecast_np, axis=0).flatten()
            std_forecast = np.std(forecast_np, axis=0).flatten()
            
            # Calculate forecast quality metrics (if we have ground truth)
            quality_metrics = {}
            if len(test_data) > context_length + forecast_length:
                true_values = test_data[-forecast_length:]
                
                # Ensure shapes match
                min_len = min(len(mean_forecast), len(true_values))
                pred_values = mean_forecast[:min_len]
                true_values = true_values[:min_len]
                
                quality_metrics = {
                    'mae': np.mean(np.abs(pred_values - true_values)),
                    'rmse': np.sqrt(np.mean((pred_values - true_values) ** 2)),
                    'mape': np.mean(np.abs((pred_values - true_values) / (true_values + 1e-8))) * 100,
                    'correlation': np.corrcoef(pred_values, true_values)[0, 1]
                }
            
            # Batch processing test
            logger.info(f"Testing batch processing...")
            batch_sizes = [1, 4, 8]
            batch_times = {}
            
            for batch_size in batch_sizes:
                if batch_size <= len(test_data) // context_length:
                    batch_context = torch.stack([
                        torch.tensor(test_data[i:i+context_length], dtype=torch.float32)
                        for i in range(0, batch_size * context_length, context_length)
                    ])
                    
                    batch_start = time.time()
                    batch_forecast = pipeline.predict(
                        context=batch_context,
                        prediction_length=forecast_length,
                        num_samples=10  # Fewer samples for batch
                    )
                    batch_time = time.time() - batch_start
                    batch_times[batch_size] = batch_time
            
            # Final resource measurement
            final_resources = self.measure_system_resources()
            
            results = {
                'model_size': model_size,
                'model_name': model_config['name'],
                'expected_params': model_config['expected_params'],
                'description': model_config['description'],
                'success': True,
                
                # Performance metrics
                'load_time': load_time,
                'avg_inference_time': avg_inference_time,
                'std_inference_time': std_inference_time,
                'batch_times': batch_times,
                
                # Resource usage
                'memory_increase_percent': memory_increase,
                'peak_memory_percent': final_resources['memory_percent'],
                
                # Forecast quality
                'forecast_shape': forecast_np.shape,
                'mean_uncertainty': np.mean(std_forecast),
                'quality_metrics': quality_metrics,
                
                # Additional info
                'context_length': context_length,
                'forecast_length': forecast_length,
                'num_uncertainty_samples': 20
            }
            
            logger.info(f"PASS {model_size} completed successfully")
            if quality_metrics:
                logger.info(f"   Quality: MAE={quality_metrics['mae']:.3f}, "
                          f"RMSE={quality_metrics['rmse']:.3f}")
            logger.info(f"   Performance: Load={load_time:.1f}s, "
                      f"Inference={avg_inference_time:.2f}{std_inference_time:.2f}s")
            logger.info(f"   Resources: Memory+{memory_increase:.1f}%")
            
            return results
            
        except Exception as e:
            logger.error(f"FAIL {model_size} model failed: {e}")
            return {
                'model_size': model_size,
                'success': False,
                'error': str(e),
                'description': model_config['description']
            }
    
    def run_comprehensive_comparison(self):
        """Run comprehensive comparison across all model sizes"""
        logger.info("ROCKET Starting comprehensive ChronosX model size comparison...")
        
        # Load test datasets
        datasets = self.load_test_datasets()
        
        # Use the first available dataset
        dataset_name, test_data = next(iter(datasets.items()))
        logger.info(f"Using dataset: {dataset_name} with {len(test_data)} points")
        
        # Test all model sizes
        model_sizes = ['tiny', 'small']  # Start with smaller models
        
        # Add larger models if system has enough memory
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        if available_memory_gb > 8:
            model_sizes.append('base')
        if available_memory_gb > 16:
            model_sizes.append('large')
        
        logger.info(f"Testing models: {model_sizes} (Available memory: {available_memory_gb:.1f}GB)")
        
        comparison_results = {}
        
        for model_size in model_sizes:
            logger.info(f"\nCHART Testing {model_size.upper()} model...")
            result = self.test_model_size(model_size, test_data)
            comparison_results[model_size] = result
            
            # Clear memory between models
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        # Test with multiple datasets if available
        if len(datasets) > 1:
            logger.info(f"\nREFRESH Testing best model on multiple datasets...")
            
            # Find best performing model
            best_model = self.find_best_model(comparison_results)
            if best_model:
                multi_dataset_results = {}
                for dataset_name, data in datasets.items():
                    if dataset_name != next(iter(datasets.keys())):  # Skip first dataset
                        logger.info(f"Testing {best_model} on {dataset_name}...")
                        result = self.test_model_size(best_model, data)
                        multi_dataset_results[dataset_name] = result
                
                comparison_results['multi_dataset'] = multi_dataset_results
        
        self.results = comparison_results
        self.generate_comparison_report()
        self.create_visualizations()
        
        return comparison_results
    
    def find_best_model(self, results: Dict) -> str:
        """Find the best performing model based on multiple criteria"""
        successful_models = {k: v for k, v in results.items() if v.get('success', False)}
        
        if not successful_models:
            return None
        
        # Score based on speed vs accuracy tradeoff
        best_score = -1
        best_model = None
        
        for model_size, result in successful_models.items():
            # Normalize metrics (lower is better for time, higher for accuracy)
            speed_score = 1.0 / (result['avg_inference_time'] + 0.1)  # Inverse time
            
            accuracy_score = 1.0  # Default if no quality metrics
            if result['quality_metrics']:
                # Lower MAE is better
                accuracy_score = 1.0 / (result['quality_metrics']['mae'] + 0.1)
            
            # Combined score (can be weighted differently)
            combined_score = 0.6 * speed_score + 0.4 * accuracy_score
            
            if combined_score > best_score:
                best_score = combined_score
                best_model = model_size
        
        return best_model
    
    def generate_comparison_report(self):
        """Generate comprehensive comparison report"""
        print("\n" + "="*70)
        print("TARGET CHRONOS-X MODEL SIZE COMPARISON REPORT")
        print("="*70)
        
        successful_results = {k: v for k, v in self.results.items() 
                            if k != 'multi_dataset' and v.get('success', False)}
        
        if not successful_results:
            print("FAIL No successful model tests to report")
            return
        
        # Performance comparison table
        print("\nCHART PERFORMANCE COMPARISON:")
        print("-" * 70)
        print(f"{'Model':<8} {'Load(s)':<8} {'Inference(s)':<12} {'Memory(+%)':<10} {'MAE':<8} {'RMSE':<8}")
        print("-" * 70)
        
        for model_size, result in successful_results.items():
            load_time = result['load_time']
            inference_time = result['avg_inference_time']
            memory_increase = result['memory_increase_percent']
            
            mae = result['quality_metrics'].get('mae', 0) if result['quality_metrics'] else 0
            rmse = result['quality_metrics'].get('rmse', 0) if result['quality_metrics'] else 0
            
            print(f"{model_size:<8} {load_time:<8.1f} {inference_time:<12.2f} "
                  f"{memory_increase:<10.1f} {mae:<8.3f} {rmse:<8.3f}")
        
        # Resource usage comparison
        print(f"\n RESOURCE USAGE:")
        print("-" * 40)
        for model_size, result in successful_results.items():
            print(f"{model_size.upper()}:")
            print(f"  Expected Parameters: {result['expected_params']}")
            print(f"  Memory Increase: +{result['memory_increase_percent']:.1f}%")
            print(f"  Peak Memory Usage: {result['peak_memory_percent']:.1f}%")
            print()
        
        # Speed comparison
        print(f"LIGHTNING SPEED ANALYSIS:")
        print("-" * 40)
        fastest_model = min(successful_results.items(), 
                           key=lambda x: x[1]['avg_inference_time'])
        
        print(f"Fastest Model: {fastest_model[0].upper()} "
              f"({fastest_model[1]['avg_inference_time']:.2f}s)")
        
        for model_size, result in successful_results.items():
            if model_size != fastest_model[0]:
                slowdown = result['avg_inference_time'] / fastest_model[1]['avg_inference_time']
                print(f"{model_size.upper()}: {slowdown:.1f}x slower than fastest")
        
        # Accuracy comparison (if available)
        quality_results = {k: v for k, v in successful_results.items() 
                         if v['quality_metrics']}
        
        if quality_results:
            print(f"\nTARGET ACCURACY ANALYSIS:")
            print("-" * 40)
            best_mae_model = min(quality_results.items(),
                               key=lambda x: x[1]['quality_metrics']['mae'])
            
            print(f"Most Accurate (MAE): {best_mae_model[0].upper()} "
                  f"(MAE: {best_mae_model[1]['quality_metrics']['mae']:.3f})")
            
            for model_size, result in quality_results.items():
                if model_size != best_mae_model[0]:
                    mae_ratio = result['quality_metrics']['mae'] / best_mae_model[1]['quality_metrics']['mae']
                    print(f"{model_size.upper()}: {mae_ratio:.2f}x MAE of best")
        
        # Recommendations
        print(f"\nIDEA RECOMMENDATIONS:")
        print("-" * 40)
        
        best_overall = self.find_best_model(successful_results)
        if best_overall:
            print(f"TROPHY Best Overall: {best_overall.upper()}")
            print(f"   {self.model_configs[best_overall]['description']}")
        
        if 'tiny' in successful_results:
            print(f"ROCKET For Speed: TINY")
            print(f"   {self.model_configs['tiny']['description']}")
        
        if quality_results:
            best_accuracy = min(quality_results.items(),
                              key=lambda x: x[1]['quality_metrics']['mae'])[0]
            print(f"TARGET For Accuracy: {best_accuracy.upper()}")
            print(f"   {self.model_configs[best_accuracy]['description']}")
        
        print("\nPARTY Comparison Complete!")
        print("="*70)
    
    def create_visualizations(self):
        """Create comparison visualizations"""
        successful_results = {k: v for k, v in self.results.items() 
                            if k != 'multi_dataset' and v.get('success', False)}
        
        if len(successful_results) < 2:
            logger.info("Need at least 2 successful models for visualization")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('ChronosX Model Size Comparison', fontsize=16, fontweight='bold')
        
        models = list(successful_results.keys())
        
        # Plot 1: Load time vs Inference time
        load_times = [successful_results[m]['load_time'] for m in models]
        inference_times = [successful_results[m]['avg_inference_time'] for m in models]
        
        axes[0, 0].scatter(load_times, inference_times, s=100, alpha=0.7)
        for i, model in enumerate(models):
            axes[0, 0].annotate(model.upper(), (load_times[i], inference_times[i]),
                              xytext=(5, 5), textcoords='offset points')
        axes[0, 0].set_xlabel('Load Time (s)')
        axes[0, 0].set_ylabel('Inference Time (s)')
        axes[0, 0].set_title('Performance: Load vs Inference Time')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Memory usage
        memory_usage = [successful_results[m]['memory_increase_percent'] for m in models]
        
        bars = axes[0, 1].bar(models, memory_usage, alpha=0.7, color='skyblue')
        axes[0, 1].set_ylabel('Memory Increase (%)')
        axes[0, 1].set_title('Memory Usage by Model Size')
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars, memory_usage):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                           f'{value:.1f}%', ha='center', va='bottom')
        
        # Plot 3: Accuracy comparison (if available)
        quality_models = [m for m in models if successful_results[m]['quality_metrics']]
        if quality_models:
            mae_values = [successful_results[m]['quality_metrics']['mae'] for m in quality_models]
            rmse_values = [successful_results[m]['quality_metrics']['rmse'] for m in quality_models]
            
            x = np.arange(len(quality_models))
            width = 0.35
            
            axes[1, 0].bar(x - width/2, mae_values, width, label='MAE', alpha=0.7)
            axes[1, 0].bar(x + width/2, rmse_values, width, label='RMSE', alpha=0.7)
            
            axes[1, 0].set_xlabel('Model')
            axes[1, 0].set_ylabel('Error')
            axes[1, 0].set_title('Accuracy Comparison')
            axes[1, 0].set_xticks(x)
            axes[1, 0].set_xticklabels([m.upper() for m in quality_models])
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Speed vs Accuracy tradeoff
        if quality_models:
            speed_scores = [1.0 / successful_results[m]['avg_inference_time'] for m in quality_models]
            accuracy_scores = [1.0 / successful_results[m]['quality_metrics']['mae'] for m in quality_models]
            
            axes[1, 1].scatter(speed_scores, accuracy_scores, s=100, alpha=0.7)
            for i, model in enumerate(quality_models):
                axes[1, 1].annotate(model.upper(), (speed_scores[i], accuracy_scores[i]),
                                  xytext=(5, 5), textcoords='offset points')
            
            axes[1, 1].set_xlabel('Speed Score (1/inference_time)')
            axes[1, 1].set_ylabel('Accuracy Score (1/MAE)')
            axes[1, 1].set_title('Speed vs Accuracy Tradeoff')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('chronos_x_model_comparison.png', dpi=300, bbox_inches='tight')
        logger.info("CHART Saved comparison visualization: chronos_x_model_comparison.png")
        plt.show()

def main():
    """Main comparison function"""
    comparator = ChronosXModelComparator()
    results = comparator.run_comprehensive_comparison()
    return results

if __name__ == "__main__":
    main()
