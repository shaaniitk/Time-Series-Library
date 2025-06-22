import os
import sys
import argparse
import numpy as np
import pandas as pd

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from backtesting.strategy import QuantileStrategy
from backtesting.backtester import Backtester
from backtesting.pso_optimizer import PSOOptimizer
from utils.logger import logger

def load_data_and_predictions(model_id: str, data_path: str):
    """Loads historical data and the corresponding model predictions."""
    # Load historical data
    try:
        historical_df = pd.read_csv(data_path, parse_dates=['date'], index_col='date')
        logger.info(f"Loaded historical data: {historical_df.shape}")
    except Exception as e:
        logger.error(f"Failed to load historical data from {data_path}: {e}")
        return None, None

    # Load predictions
    # Assumes predictions are saved from the test() function of the experiment
    pred_path = os.path.join('./results', model_id, 'pred_original.npy')
    try:
        predictions = np.load(pred_path)
        logger.info(f"Loaded predictions from {pred_path}: {predictions.shape}")
    except FileNotFoundError:
        logger.error(f"Prediction file not found: {pred_path}")
        logger.error("Please run the test phase of the experiment first to generate predictions.")
        return None, None

    # Align data: ensure predictions match the historical data timeframe
    # The test set is the last `test_length` part of the data
    test_set_size = predictions.shape[0]
    if test_set_size > len(historical_df):
        logger.error("Prediction array is larger than historical data. Check data alignment.")
        return None, None
        
    aligned_historical_data = historical_df.iloc[-test_set_size:]
    logger.info(f"Aligned historical data for backtesting: {aligned_historical_data.shape}")

    return aligned_historical_data, predictions

def objective_function(params: dict, historical_data: pd.DataFrame, predictions: np.ndarray) -> float:
    """
    The function that PSO will try to maximize.
    It runs a backtest with the given strategy parameters and returns the Sharpe Ratio.
    """
    try:
        # 1. Create the strategy with the parameters from the particle
        strategy = QuantileStrategy(
            buy_quantile_idx=params['buy_quantile_idx'],
            sell_quantile_idx=params['sell_quantile_idx'],
            entry_threshold=params['entry_threshold'],
            exit_threshold=params['exit_threshold']
        )

        # 2. Run the backtester
        backtester = Backtester(historical_data, predictions, strategy)
        performance = backtester.run()

        # 3. Return the score to maximize (Sharpe Ratio)
        # Return a large negative number if Sharpe is NaN or Inf
        sharpe = performance.get('sharpe_ratio', -100.0)
        return sharpe if np.isfinite(sharpe) else -100.0
    except Exception as e:
        logger.error(f"Error in objective function with params {params}: {e}")
        return -100.0 # Penalize failures

def main(args):
    historical_data, predictions = load_data_and_predictions(args.model_id, args.data_path)
    if historical_data is None:
        return

    # Define the bounds for the parameters we want to optimize
    param_bounds = {
        'buy_quantile_idx': (0, 2),      # Assuming 5 quantiles, indices 0, 1, 2 are lower half
        'sell_quantile_idx': (2, 4),     # Indices 2, 3, 4 are upper half
        'entry_threshold': (0.0005, 0.01), # 0.05% to 1% entry signal
        'exit_threshold': (0.0, 0.005)      # 0% to 0.5% exit signal
    }

    # Create the objective function with fixed data
    obj_func = lambda params: objective_function(params, historical_data, predictions)

    # Initialize and run the optimizer
    pso = PSOOptimizer(obj_func, param_bounds, n_particles=args.particles, max_iter=args.iterations)
    best_params, best_score = pso.optimize()

    logger.info("\n" + "="*50)
    logger.info("🏆 PSO Optimization Finished 🏆")
    logger.info(f"Best Sharpe Ratio: {best_score:.4f}")
    logger.info("Best Strategy Parameters:")
    for name, val in best_params.items():
        logger.info(f"  - {name}: {val:.4f}")
    logger.info("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Backtesting Framework with PSO')
    parser.add_argument('--model_id', type=str, required=True, help='Model ID of the experiment to backtest (e.g., BayesianEnhancedAutoformer_MS_light_sl100_pl20)')
    parser.add_argument('--data_path', type=str, default='data/prepared_financial_data.csv', help='Path to the historical data CSV file')
    parser.add_argument('--particles', type=int, default=20, help='Number of particles for PSO')
    parser.add_argument('--iterations', type=int, default=30, help='Number of iterations for PSO')
    args = parser.parse_args()
    main(args)