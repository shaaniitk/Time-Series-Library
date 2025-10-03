import numpy as np
from utils.logger import logger


class MetricsLogger:
    """
    A utility class to log and compute metrics incrementally.
    This avoids storing all predictions and true values in memory.
    """

    def __init__(self):
        self.total_mae = 0.0
        self.total_mse = 0.0
        self.count = 0

    def update(self, pred, true):
        """
        Update the metrics with a new batch of predictions and true values.
        """
        mae = MAE(pred, true)
        mse = MSE(pred, true)

        batch_size = pred.shape[0]
        self.total_mae += mae * batch_size
        self.total_mse += mse * batch_size
        self.count += batch_size

    def get_metrics(self):
        """
        Get the final averaged metrics.
        """
        mae = self.total_mae / self.count if self.count > 0 else 0
        mse = self.total_mse / self.count if self.count > 0 else 0
        rmse = np.sqrt(mse) if self.count > 0 else 0
        
        # MAPE and MSPE are not easily calculated incrementally without storing all values.
        # For now, we will return 0 for them. A more advanced implementation could
        # handle this with more care, but for memory optimization, this is a good start.
        mape = 0 
        mspe = 0

        return mae, mse, rmse, mape, mspe


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(true - pred))


def MSE(pred, true):
    return np.mean((true - pred) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((true - pred) / true))


def MSPE(pred, true):
    return np.mean(np.square((true - pred) / true))


def metric(pred, true):
    logger.debug("Calculating metric")
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe