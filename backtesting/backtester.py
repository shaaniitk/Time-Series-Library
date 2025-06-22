import pandas as pd
import numpy as np
from .strategy import TradingStrategy

class Backtester:
    """
    A simple event-driven backtester for evaluating trading strategies.
    """
    def __init__(self, 
                 historical_data: pd.DataFrame, 
                 predictions: np.ndarray,
                 strategy: TradingStrategy,
                 initial_cash: float = 100000.0,
                 commission: float = 0.001):
        """
        Args:
            historical_data (pd.DataFrame): DataFrame with historical prices (e.g., OHLC).
            predictions (np.ndarray): Model's predictions aligned with historical_data.
            strategy (TradingStrategy): The strategy instance to use.
            initial_cash (float): Starting cash for the simulation.
            commission (float): Trading commission fee.
        """
        self.historical_data = historical_data
        self.predictions = predictions
        self.strategy = strategy
        self.initial_cash = initial_cash
        self.commission = commission
        self.results = None

    def run(self):
        """Runs the backtest simulation."""
        signals = self.strategy.generate_signals(self.predictions, self.historical_data)
        
        positions = pd.Series(index=signals.index, data=0.0)
        portfolio = pd.DataFrame(index=signals.index)
        portfolio['cash'] = self.initial_cash
        portfolio['holdings'] = 0.0
        portfolio['total'] = self.initial_cash

        # Use 'log_Close' for trading price, assuming it's the 4th column (index 3)
        close_prices = self.historical_data.iloc[:, 3]

        for i in range(1, len(signals)):
            # Carry over portfolio values
            portfolio.loc[signals.index[i], 'cash'] = portfolio.loc[signals.index[i-1], 'cash']
            positions.loc[signals.index[i]] = positions.loc[signals.index[i-1]]

            # Trading logic
            if signals.iloc[i-1] == 1 and positions.iloc[i-1] == 0: # Buy signal and no position
                # Buy one unit
                positions.iloc[i] = 1.0
                cost = close_prices.iloc[i] * (1 + self.commission)
                portfolio.loc[signals.index[i], 'cash'] -= cost
            elif signals.iloc[i-1] == -1 and positions.iloc[i-1] == 1: # Sell signal and in position
                # Sell one unit
                positions.iloc[i] = 0.0
                revenue = close_prices.iloc[i] * (1 - self.commission)
                portfolio.loc[signals.index[i], 'cash'] += revenue

            # Update holdings and total portfolio value
            portfolio.loc[signals.index[i], 'holdings'] = positions.iloc[i] * close_prices.iloc[i]
            portfolio.loc[signals.index[i], 'total'] = portfolio.loc[signals.index[i], 'cash'] + portfolio.loc[signals.index[i], 'holdings']

        portfolio['returns'] = portfolio['total'].pct_change()
        self.results = portfolio
        return self.calculate_metrics()

    def calculate_metrics(self):
        """Calculates performance metrics from the backtest results."""
        if self.results is None:
            return {}

        total_return = (self.results['total'].iloc[-1] / self.initial_cash) - 1
        
        # Sharpe Ratio (assuming risk-free rate is 0)
        daily_returns = self.results['returns'].dropna()
        if daily_returns.std() == 0:
            sharpe_ratio = 0.0
        else:
            sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252) # Annualized

        # Max Drawdown
        cumulative_max = self.results['total'].cummax()
        drawdown = (self.results['total'] - cumulative_max) / cumulative_max
        max_drawdown = drawdown.min()

        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'final_portfolio_value': self.results['total'].iloc[-1]
        }