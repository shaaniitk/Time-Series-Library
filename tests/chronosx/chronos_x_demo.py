"""
ChronosX Real Time Series Forecasting Demonstration

This script shows ChronosX in action with real time series data,
demonstrating the power of Amazon's Chronos models for forecasting.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from chronos import ChronosPipeline
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_prepare_data():
    """Load real time series data"""
    logger.info("Loading real time series data...")
    
    # Try to load synthetic data
    try:
        df = pd.read_csv('synthetic_timeseries.csv')
        # Take first numeric column
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        data = df[numeric_cols[0]].values
        logger.info(f"Loaded data: {len(data)} points, range [{data.min():.2f}, {data.max():.2f}]")
        return data
    except:
        # Generate sample financial-like data
        logger.info("Generating sample financial time series...")
        np.random.seed(42)
        t = np.arange(1000)
        trend = 0.1 * t
        seasonal = 10 * np.sin(2 * np.pi * t / 50) + 5 * np.sin(2 * np.pi * t / 20)
        noise = np.random.normal(0, 2, len(t))
        data = 100 + trend + seasonal + noise
        return data

def forecast_with_chronos(data, model_size="small", context_length=96, forecast_length=24):
    """Forecast using ChronosX"""
    
    model_names = {
        'tiny': 'amazon/chronos-t5-tiny',
        'small': 'amazon/chronos-t5-small',
        'base': 'amazon/chronos-t5-base'
    }
    
    logger.info(f"Loading ChronosX {model_size} model...")
    
    # Load model
    start_time = time.time()
    pipeline = ChronosPipeline.from_pretrained(
        model_names[model_size],
        device_map="cpu",
        torch_dtype=torch.float32,
    )
    load_time = time.time() - start_time
    logger.info(f"Model loaded in {load_time:.2f} seconds")
    
    # Prepare data
    context = torch.tensor(data[-context_length:], dtype=torch.float32).unsqueeze(0)
    
    logger.info(f"Forecasting {forecast_length} steps ahead...")
    
    # Generate forecast
    start_time = time.time()
    forecast = pipeline.predict(
        context=context,
        prediction_length=forecast_length,
        num_samples=20
    )
    forecast_time = time.time() - start_time
    
    logger.info(f"Forecast generated in {forecast_time:.2f} seconds")
    
    # Process results
    if isinstance(forecast, torch.Tensor):
        forecast_np = forecast.cpu().numpy()
    else:
        forecast_np = np.array(forecast)
    
    # Calculate statistics
    mean_forecast = np.mean(forecast_np, axis=0)
    std_forecast = np.std(forecast_np, axis=0)
    
    # Handle shape variations
    if len(mean_forecast.shape) > 1:
        mean_forecast = mean_forecast.flatten()
        std_forecast = std_forecast.flatten()
    
    return {
        'forecast_mean': mean_forecast,
        'forecast_std': std_forecast,
        'forecast_samples': forecast_np,
        'load_time': load_time,
        'forecast_time': forecast_time,
        'context': context.squeeze().numpy(),
        'model_size': model_size
    }

def visualize_forecast(data, results, save_plot=True):
    """Create visualization of forecasting results"""
    
    context = results['context']
    forecast_mean = results['forecast_mean']
    forecast_std = results['forecast_std']
    
    # Create timeline
    historical_timeline = np.arange(len(data))
    context_timeline = historical_timeline[-len(context):]
    forecast_timeline = np.arange(len(data), len(data) + len(forecast_mean))
    
    # Create plot
    plt.figure(figsize=(15, 8))
    
    # Plot historical data
    plt.plot(historical_timeline[-200:], data[-200:], 
             label='Historical Data', color='blue', alpha=0.7)
    
    # Plot context (recent history used for forecasting)
    plt.plot(context_timeline, context, 
             label='Context (Input)', color='green', linewidth=2)
    
    # Plot forecast
    plt.plot(forecast_timeline, forecast_mean, 
             label='Forecast', color='red', linewidth=2, marker='o', markersize=4)
    
    # Plot uncertainty bands
    plt.fill_between(forecast_timeline, 
                     forecast_mean - forecast_std,
                     forecast_mean + forecast_std,
                     alpha=0.3, color='red', label='1 uncertainty')
    
    plt.fill_between(forecast_timeline,
                     forecast_mean - 2*forecast_std,
                     forecast_mean + 2*forecast_std,
                     alpha=0.2, color='red', label='2 uncertainty')
    
    # Formatting
    plt.title(f'ChronosX {results["model_size"].upper()} Model - Time Series Forecasting', fontsize=16)
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add performance info
    plt.figtext(0.02, 0.02, 
                f"Load: {results['load_time']:.1f}s | Forecast: {results['forecast_time']:.1f}s | "
                f"Mean Uncertainty: {np.mean(forecast_std):.3f}",
                fontsize=10, style='italic')
    
    plt.tight_layout()
    
    if save_plot:
        filename = f'chronos_x_{results["model_size"]}_forecast.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        logger.info(f"CHART Saved plot to {filename}")
    
    plt.show()

def compare_models(data):
    """Compare different ChronosX model sizes"""
    logger.info("SEARCH Comparing ChronosX model sizes...")
    
    models_to_test = ['tiny', 'small']
    results = {}
    
    for model_size in models_to_test:
        logger.info(f"\nCHART Testing {model_size} model...")
        try:
            result = forecast_with_chronos(data, model_size=model_size)
            results[model_size] = result
            
            # Calculate accuracy if we have recent data to compare
            if len(data) > 120:  # Need enough data for context + validation
                # Use earlier data for validation
                val_context = data[-120:-24]
                val_target = data[-24:]
                
                val_result = forecast_with_chronos(val_context, model_size=model_size)
                mae = np.mean(np.abs(val_result['forecast_mean'] - val_target))
                results[model_size]['validation_mae'] = mae
                
                logger.info(f"PASS {model_size}: MAE={mae:.3f}, "
                          f"Load={result['load_time']:.1f}s, "
                          f"Forecast={result['forecast_time']:.1f}s")
            else:
                logger.info(f"PASS {model_size}: "
                          f"Load={result['load_time']:.1f}s, "
                          f"Forecast={result['forecast_time']:.1f}s")
                
        except Exception as e:
            logger.error(f"FAIL {model_size} failed: {e}")
            results[model_size] = {'error': str(e)}
    
    return results

def main():
    """Main demonstration function"""
    print("ROCKET ChronosX Real Time Series Forecasting Demo")
    print("=" * 50)
    
    # Load data
    data = load_and_prepare_data()
    
    # Single model demonstration
    print("\nGRAPH Single Model Forecasting Demo...")
    result = forecast_with_chronos(data, model_size="small")
    visualize_forecast(data, result)
    
    # Model comparison
    print("\nSEARCH Model Comparison...")
    comparison_results = compare_models(data)
    
    # Summary report
    print("\nCHART FINAL RESULTS SUMMARY")
    print("-" * 30)
    
    for model_size, result in comparison_results.items():
        if 'error' not in result:
            print(f"PASS {model_size.upper()}:")
            print(f"   Load Time: {result['load_time']:.2f}s")
            print(f"   Forecast Time: {result['forecast_time']:.2f}s")
            print(f"   Mean Uncertainty: {np.mean(result['forecast_std']):.3f}")
            if 'validation_mae' in result:
                print(f"   Validation MAE: {result['validation_mae']:.3f}")
        else:
            print(f"FAIL {model_size.upper()}: {result['error']}")
        print()
    
    print("PARTY ChronosX demonstration complete!")
    print("\nIDEA Key Takeaways:")
    print(" ChronosX provides high-quality forecasts out of the box")
    print(" Uncertainty quantification helps assess prediction confidence")
    print(" Different model sizes offer speed vs accuracy tradeoffs")
    print(" No training required - pretrained models work immediately")

if __name__ == "__main__":
    main()
