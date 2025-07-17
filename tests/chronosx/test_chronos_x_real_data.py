"""
ChronosX Real Time Series Data Testing

This script demonstrates ChronosX integration with real time series data,
testing various model sizes and uncertainty quantification capabilities.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import logging
from typing import Dict, List, Tuple, Optional
import time

# Add project path
sys.path.append(str(Path(__file__).parent))

from chronos import ChronosPipeline
from utils.modular_components.chronos_backbone import (
    ChronosXBackbone, ChronosXTinyBackbone, 
    ChronosXLargeBackbone, ChronosXUncertaintyBackbone
)
from utils.modular_components.config_schemas import BackboneConfig
from utils.modular_components.registry import create_global_registry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChronosXRealDataTester:
    """Comprehensive tester for ChronosX with real time series data"""
    
    def __init__(self):
        self.results = {}
        self.registry = create_global_registry()
        
    def load_real_data(self) -> Dict[str, np.ndarray]:
        """Load real time series data from available sources"""
        logger.info("Loading real time series data...")
        
        data_sources = {}
        
        # Try to load synthetic data if available
        synthetic_files = [
            'synthetic_timeseries.csv',
            'synthetic_timeseries_train.csv',
            'synthetic_timeseries_val.csv'
        ]
        
        for file_name in synthetic_files:
            if Path(file_name).exists():
                try:
                    df = pd.read_csv(file_name)
                    # Extract numeric columns
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        data = df[numeric_cols].values
                        data_sources[file_name] = data
                        logger.info(f"Loaded {file_name}: shape {data.shape}")
                except Exception as e:
                    logger.warning(f"Failed to load {file_name}: {e}")
        
        # Generate sample data if no files found
        if not data_sources:
            logger.info("No data files found, generating sample time series data...")
            
            # Generate various types of time series
            t = np.linspace(0, 100, 1000)
            
            # Trend + Seasonality + Noise
            trend_seasonal = (
                2 * t +  # Linear trend
                10 * np.sin(0.5 * t) +  # Seasonal component
                5 * np.sin(2 * t) +     # Higher frequency
                np.random.normal(0, 2, len(t))  # Noise
            )
            
            # Financial-like data (log returns)
            returns = np.random.normal(0.001, 0.02, 1000)
            financial = 100 * np.exp(np.cumsum(returns))
            
            # Non-stationary data
            non_stationary = np.cumsum(np.random.normal(0, 1, 1000)) + 0.1 * t**1.5
            
            data_sources = {
                'trend_seasonal': trend_seasonal.reshape(-1, 1),
                'financial': financial.reshape(-1, 1),
                'non_stationary': non_stationary.reshape(-1, 1)
            }
            
            logger.info("Generated synthetic datasets:")
            for name, data in data_sources.items():
                logger.info(f"  {name}: shape {data.shape}")
        
        return data_sources
    
    def test_chronos_pipeline_direct(self, data: np.ndarray, model_size: str = "tiny") -> Dict:
        """Test ChronosPipeline directly without modular architecture"""
        logger.info(f"Testing ChronosPipeline directly with {model_size} model...")
        
        model_names = {
            'tiny': 'amazon/chronos-t5-tiny',
            'small': 'amazon/chronos-t5-small',
            'base': 'amazon/chronos-t5-base',
            'large': 'amazon/chronos-t5-large'
        }
        
        try:
            # Load ChronosX model
            start_time = time.time()
            pipeline = ChronosPipeline.from_pretrained(
                model_names[model_size],
                device_map="cpu",  # Use CPU for compatibility
                torch_dtype=torch.float32,
            )
            load_time = time.time() - start_time
            
            # Prepare data
            if len(data.shape) == 1:
                data = data.reshape(-1, 1)
            
            # Use last 96 points as context, predict next 24
            context_length = min(96, len(data) - 24)
            context = torch.tensor(data[-context_length-24:-24], dtype=torch.float32)
            true_future = data[-24:]
            
            # Generate forecast
            start_time = time.time()
            forecast = pipeline.predict(
                context=context,
                prediction_length=24,
                num_samples=20  # For uncertainty
            )
            inference_time = time.time() - start_time
            
            # Convert to numpy for analysis
            if isinstance(forecast, torch.Tensor):
                forecast_np = forecast.cpu().numpy()
            else:
                forecast_np = np.array(forecast)
            
            # Compute metrics
            mean_forecast = np.mean(forecast_np, axis=0) if len(forecast_np.shape) > 2 else forecast_np
            
            # Handle different forecast shapes
            if len(mean_forecast.shape) == 3:  # [samples, time, features]
                mean_forecast = mean_forecast[0]  # Take first sample
            elif len(mean_forecast.shape) == 1:  # [time]
                mean_forecast = mean_forecast.reshape(-1, 1)
            
            # Ensure shapes match
            pred_len = min(len(mean_forecast), len(true_future))
            mean_forecast = mean_forecast[:pred_len]
            true_future = true_future[:pred_len]
            
            # Calculate errors
            mae = np.mean(np.abs(mean_forecast - true_future))
            mse = np.mean((mean_forecast - true_future) ** 2)
            rmse = np.sqrt(mse)
            
            # Uncertainty metrics
            if len(forecast_np.shape) > 2:
                std_forecast = np.std(forecast_np, axis=0)
                if len(std_forecast.shape) == 3:
                    std_forecast = std_forecast[0]
                uncertainty_mean = np.mean(std_forecast)
            else:
                uncertainty_mean = 0.0
            
            results = {
                'model_size': model_size,
                'load_time': load_time,
                'inference_time': inference_time,
                'forecast_shape': forecast_np.shape,
                'context_length': context_length,
                'prediction_length': pred_len,
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'uncertainty_mean': uncertainty_mean,
                'success': True,
                'forecast': mean_forecast,
                'true_values': true_future,
                'uncertainty': std_forecast if 'std_forecast' in locals() else None
            }
            
            logger.info(f"✅ Direct ChronosX {model_size}: MAE={mae:.4f}, RMSE={rmse:.4f}, Load={load_time:.2f}s")
            return results
            
        except Exception as e:
            logger.error(f"❌ Direct ChronosX {model_size} failed: {e}")
            return {
                'model_size': model_size,
                'success': False,
                'error': str(e)
            }
    
    def test_modular_chronos_backbone(self, data: np.ndarray, backbone_type: str = "chronos_x_tiny") -> Dict:
        """Test ChronosX through modular architecture"""
        logger.info(f"Testing modular ChronosX backbone: {backbone_type}...")
        
        try:
            # Create backbone config
            config = BackboneConfig(
                component_name=backbone_type,
                backbone_type=backbone_type,
                d_model=64,
                model_name='amazon/chronos-t5-tiny'
            )
            
            # Add additional parameters
            config.seq_len = 96
            config.pred_len = 24
            config.device = 'cpu'
            config.uncertainty_enabled = True
            config.model_size = 'tiny'
            
            # Create backbone
            if backbone_type == "chronos_x_tiny":
                backbone = ChronosXTinyBackbone(config)
            elif backbone_type == "chronos_x_large":
                backbone = ChronosXLargeBackbone(config)
            elif backbone_type == "chronos_x_uncertainty":
                backbone = ChronosXUncertaintyBackbone(config)
            else:
                backbone = ChronosXBackbone(config)
            
            # Prepare data
            if len(data.shape) == 1:
                data = data.reshape(-1, 1)
            
            context_length = min(96, len(data) - 24)
            batch_size = 1
            
            # Create input tensors
            x = torch.tensor(
                data[-context_length-24:-24].T.reshape(batch_size, context_length, -1), 
                dtype=torch.float32
            )
            x_mark = torch.randn(batch_size, context_length, 4)  # Temporal features
            
            # Forward pass
            start_time = time.time()
            output = backbone(x, x_mark)
            inference_time = time.time() - start_time
            
            # Get uncertainty if available
            uncertainty = None
            if hasattr(backbone, 'get_uncertainty'):
                uncertainty = backbone.get_uncertainty()
            
            # Extract results
            if isinstance(output, torch.Tensor):
                prediction = output.detach().cpu().numpy()
            else:
                prediction = np.array(output)
            
            results = {
                'backbone_type': backbone_type,
                'inference_time': inference_time,
                'output_shape': prediction.shape,
                'has_uncertainty': uncertainty is not None,
                'uncertainty_shape': uncertainty.shape if uncertainty is not None else None,
                'success': True,
                'prediction': prediction
            }
            
            logger.info(f"✅ Modular {backbone_type}: shape={prediction.shape}, time={inference_time:.2f}s")
            return results
            
        except Exception as e:
            logger.error(f"❌ Modular {backbone_type} failed: {e}")
            return {
                'backbone_type': backbone_type,
                'success': False,
                'error': str(e)
            }
    
    def compare_model_sizes(self, data: np.ndarray) -> Dict:
        """Compare different ChronosX model sizes"""
        logger.info("Comparing ChronosX model sizes...")
        
        model_sizes = ['tiny', 'small']  # Start with smaller models
        
        comparison_results = {}
        
        for model_size in model_sizes:
            logger.info(f"Testing {model_size} model...")
            result = self.test_chronos_pipeline_direct(data, model_size)
            comparison_results[model_size] = result
            
            if result['success']:
                logger.info(f"  ✅ {model_size}: MAE={result['mae']:.4f}, "
                          f"Load={result['load_time']:.2f}s, "
                          f"Inference={result['inference_time']:.2f}s")
            else:
                logger.info(f"  ❌ {model_size}: {result.get('error', 'Unknown error')}")
        
        return comparison_results
    
    def test_uncertainty_quantification(self, data: np.ndarray) -> Dict:
        """Test uncertainty quantification capabilities"""
        logger.info("Testing uncertainty quantification...")
        
        try:
            # Use uncertainty-optimized backbone
            config = BackboneConfig(
                component_name='chronos_x_uncertainty',
                backbone_type='chronos_x_uncertainty',
                d_model=64
            )
            
            config.seq_len = 96
            config.pred_len = 24
            config.device = 'cpu'
            config.uncertainty_enabled = True
            config.num_samples = 50  # More samples for better uncertainty
            
            backbone = ChronosXUncertaintyBackbone(config)
            
            # Test uncertainty prediction
            if len(data.shape) == 1:
                data = data.reshape(-1, 1)
            
            context_length = min(96, len(data) - 24)
            context = torch.tensor(data[-context_length-24:-24], dtype=torch.float32)
            
            uncertainty_result = backbone.predict_with_uncertainty(
                context, prediction_length=24
            )
            
            results = {
                'success': True,
                'prediction_mean': uncertainty_result['prediction'].cpu().numpy(),
                'prediction_std': uncertainty_result['uncertainty'].cpu().numpy(),
                'confidence_intervals': {
                    k: {
                        'lower': v['lower'].cpu().numpy(),
                        'upper': v['upper'].cpu().numpy()
                    }
                    for k, v in uncertainty_result['confidence_intervals'].items()
                },
                'num_samples': config.num_samples
            }
            
            logger.info("✅ Uncertainty quantification successful")
            return results
            
        except Exception as e:
            logger.error(f"❌ Uncertainty quantification failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def visualize_results(self, data: np.ndarray, results: Dict, save_plots: bool = True):
        """Create visualizations of forecasting results"""
        logger.info("Creating visualizations...")
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('ChronosX Real Data Forecasting Results', fontsize=16)
            
            # Plot 1: Original data
            axes[0, 0].plot(data[-200:], label='Time Series Data', color='blue')
            axes[0, 0].set_title('Original Time Series Data (Last 200 points)')
            axes[0, 0].set_ylabel('Value')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Plot 2: Model comparison
            if 'model_comparison' in results:
                for model_size, result in results['model_comparison'].items():
                    if result['success']:
                        axes[0, 1].plot(
                            result['forecast'].flatten(), 
                            label=f'{model_size} (MAE: {result["mae"]:.3f})',
                            marker='o', markersize=3
                        )
                        axes[0, 1].plot(
                            result['true_values'].flatten(),
                            label='True Values',
                            linestyle='--', color='black', alpha=0.7
                        )
                
                axes[0, 1].set_title('Model Size Comparison')
                axes[0, 1].set_ylabel('Predicted Value')
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: Uncertainty quantification
            if 'uncertainty' in results and results['uncertainty']['success']:
                unc_data = results['uncertainty']
                pred_mean = unc_data['prediction_mean'].flatten()
                pred_std = unc_data['prediction_std'].flatten()
                
                x_axis = np.arange(len(pred_mean))
                axes[1, 0].plot(x_axis, pred_mean, label='Prediction', color='red', linewidth=2)
                axes[1, 0].fill_between(
                    x_axis, 
                    pred_mean - pred_std, 
                    pred_mean + pred_std,
                    alpha=0.3, color='red', label='±1σ'
                )
                axes[1, 0].fill_between(
                    x_axis,
                    pred_mean - 2*pred_std,
                    pred_mean + 2*pred_std,
                    alpha=0.2, color='red', label='±2σ'
                )
                
                axes[1, 0].set_title('Uncertainty Quantification')
                axes[1, 0].set_ylabel('Predicted Value')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
            
            # Plot 4: Performance metrics
            performance_data = []
            labels = []
            
            if 'model_comparison' in results:
                for model_size, result in results['model_comparison'].items():
                    if result['success']:
                        performance_data.append([
                            result['mae'],
                            result['rmse'],
                            result['load_time'],
                            result['inference_time']
                        ])
                        labels.append(model_size)
            
            if performance_data:
                metrics = ['MAE', 'RMSE', 'Load Time (s)', 'Inference Time (s)']
                x = np.arange(len(metrics))
                width = 0.35
                
                for i, (data_points, label) in enumerate(zip(performance_data, labels)):
                    axes[1, 1].bar(x + i*width, data_points, width, label=label)
                
                axes[1, 1].set_title('Performance Metrics')
                axes[1, 1].set_ylabel('Value')
                axes[1, 1].set_xticks(x + width/2)
                axes[1, 1].set_xticklabels(metrics, rotation=45)
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_plots:
                plt.savefig('chronos_x_real_data_results.png', dpi=300, bbox_inches='tight')
                logger.info("📊 Saved visualization to chronos_x_real_data_results.png")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Visualization failed: {e}")
    
    def run_comprehensive_test(self):
        """Run comprehensive ChronosX testing with real data"""
        logger.info("🚀 Starting comprehensive ChronosX real data testing...")
        
        # Load real data
        data_sources = self.load_real_data()
        
        # Use the first available dataset
        data_name, data = next(iter(data_sources.items()))
        logger.info(f"Using dataset: {data_name} with shape {data.shape}")
        
        # Take first column if multivariate
        if len(data.shape) > 1 and data.shape[1] > 1:
            data = data[:, 0]
        
        all_results = {}
        
        # Test 1: Model size comparison
        logger.info("\n📊 Testing different model sizes...")
        model_comparison = self.compare_model_sizes(data)
        all_results['model_comparison'] = model_comparison
        
        # Test 2: Modular architecture testing
        logger.info("\n🏗️ Testing modular architecture...")
        modular_results = {}
        for backbone_type in ['chronos_x_tiny', 'chronos_x_uncertainty']:
            result = self.test_modular_chronos_backbone(data, backbone_type)
            modular_results[backbone_type] = result
        all_results['modular'] = modular_results
        
        # Test 3: Uncertainty quantification
        logger.info("\n🔮 Testing uncertainty quantification...")
        uncertainty_result = self.test_uncertainty_quantification(data)
        all_results['uncertainty'] = uncertainty_result
        
        # Generate report
        self.generate_report(all_results)
        
        # Create visualizations
        self.visualize_results(data, all_results)
        
        return all_results
    
    def generate_report(self, results: Dict):
        """Generate comprehensive test report"""
        print("\n" + "="*60)
        print("🎯 CHRONOS-X REAL DATA TESTING REPORT")
        print("="*60)
        
        # Model comparison results
        if 'model_comparison' in results:
            print("\n📊 MODEL SIZE COMPARISON:")
            print("-" * 40)
            for model_size, result in results['model_comparison'].items():
                if result['success']:
                    print(f"✅ {model_size.upper()}:")
                    print(f"   MAE: {result['mae']:.4f}")
                    print(f"   RMSE: {result['rmse']:.4f}")
                    print(f"   Load Time: {result['load_time']:.2f}s")
                    print(f"   Inference Time: {result['inference_time']:.2f}s")
                    print(f"   Uncertainty Mean: {result['uncertainty_mean']:.4f}")
                else:
                    print(f"❌ {model_size.upper()}: {result.get('error', 'Failed')}")
                print()
        
        # Modular architecture results
        if 'modular' in results:
            print("🏗️ MODULAR ARCHITECTURE:")
            print("-" * 40)
            for backbone_type, result in results['modular'].items():
                if result['success']:
                    print(f"✅ {backbone_type}:")
                    print(f"   Output Shape: {result['output_shape']}")
                    print(f"   Inference Time: {result['inference_time']:.2f}s")
                    print(f"   Has Uncertainty: {result['has_uncertainty']}")
                else:
                    print(f"❌ {backbone_type}: {result.get('error', 'Failed')}")
                print()
        
        # Uncertainty quantification
        if 'uncertainty' in results:
            print("🔮 UNCERTAINTY QUANTIFICATION:")
            print("-" * 40)
            if results['uncertainty']['success']:
                unc = results['uncertainty']
                print("✅ Uncertainty Analysis Complete:")
                print(f"   Prediction Shape: {unc['prediction_mean'].shape}")
                print(f"   Uncertainty Shape: {unc['prediction_std'].shape}")
                print(f"   Samples Used: {unc['num_samples']}")
                print(f"   Confidence Intervals: {list(unc['confidence_intervals'].keys())}")
            else:
                print(f"❌ Uncertainty: {results['uncertainty'].get('error', 'Failed')}")
        
        print("\n🎉 TESTING COMPLETE!")
        print("="*60)

def main():
    """Main testing function"""
    tester = ChronosXRealDataTester()
    results = tester.run_comprehensive_test()
    return results

if __name__ == "__main__":
    main()
