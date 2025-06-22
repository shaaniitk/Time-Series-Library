from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

class TradingStrategy(ABC):
    """Abstract base class for all trading strategies."""
    
    @abstractmethod
    def generate_signals(self, predictions: np.ndarray, historical_data: pd.DataFrame) -> pd.Series:
        """
        Generates trading signals based on model predictions.
        
        Args:
            predictions (np.ndarray): Model's forecast output. Shape [num_samples, pred_len, features].
            historical_data (pd.DataFrame): Historical price data for context.
            
        Returns:
            pd.Series: A series of signals (1 for Buy, -1 for Sell, 0 for Hold).
        """
        pass

class QuantileStrategy(TradingStrategy):
    """
    A strategy that uses quantile predictions to make trading decisions.
    The parameters of this strategy are what the PSO will optimize.
    """
    def __init__(self, 
                 buy_quantile_idx: int = 0, 
                 sell_quantile_idx: int = 4,
                 close_price_idx: int = 3,
                 entry_threshold: float = 0.001,
                 exit_threshold: float = 0.0005):
        """
        Args:
            buy_quantile_idx (int): Index of the lower quantile for buy signals (e.g., 0 for 10th percentile).
            sell_quantile_idx (int): Index of the upper quantile for sell signals (e.g., 4 for 90th percentile).
            close_price_idx (int): Index of the 'close' price in the feature dimension.
            entry_threshold (float): % price change required for a buy/sell signal.
            exit_threshold (float): % price change required to close a position.
        """
        self.buy_quantile_idx = int(buy_quantile_idx)
        self.sell_quantile_idx = int(sell_quantile_idx)
        self.close_price_idx = int(close_price_idx)
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold

    def generate_signals(self, predictions: np.ndarray, historical_data: pd.DataFrame) -> pd.Series:
        """
        Generates signals based on quantile forecasts.
        - BUY if the lower quantile forecast is significantly above the current close.
        - SELL if the upper quantile forecast is significantly below the current close.
        """
        signals = pd.Series(index=historical_data.index, data=0)
        
        # Predictions are for the *next* step. We compare with the *current* close price.
        # The predictions array aligns with the historical data from the start of the test set.
        current_close = historical_data.iloc[:, self.close_price_idx]
        
        # Predicted lower bound for the next step (we only look 1 step ahead for this simple strategy)
        predicted_low = predictions[:, 0, self.buy_quantile_idx]
        # Predicted upper bound for the next step
        predicted_high = predictions[:, 0, self.sell_quantile_idx]

        signals[predicted_low > current_close * (1 + self.entry_threshold)] = 1  # Buy signal
        signals[predicted_high < current_close * (1 - self.entry_threshold)] = -1 # Sell signal
        
        return signals