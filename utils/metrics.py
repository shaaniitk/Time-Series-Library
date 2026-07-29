import numpy as np


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
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe


def quantile_pinball_loss(pred_quantiles, true, quantiles):
    pred_quantiles = np.asarray(pred_quantiles, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    quantiles = np.asarray(quantiles, dtype=np.float64)

    if pred_quantiles.ndim != 4:
        raise ValueError(f"pred_quantiles must have shape [N,T,Q,C], got {pred_quantiles.shape}.")
    if true.ndim != 3:
        raise ValueError(f"true must have shape [N,T,C], got {true.shape}.")
    if pred_quantiles.shape[:2] != true.shape[:2] or pred_quantiles.shape[3] != true.shape[2]:
        raise ValueError("pred_quantiles and true must agree on N/T/C dimensions.")
    if pred_quantiles.shape[2] != quantiles.shape[0]:
        raise ValueError("Q dimension must match the number of quantiles.")

    errors = true[:, :, None, :] - pred_quantiles
    q = quantiles.reshape(1, 1, -1, 1)
    loss = np.maximum(q * errors, (q - 1.0) * errors)
    return float(loss.mean())


def quantile_crossing_rate(pred_quantiles):
    pred_quantiles = np.asarray(pred_quantiles, dtype=np.float64)
    if pred_quantiles.ndim != 4:
        raise ValueError(f"pred_quantiles must have shape [N,T,Q,C], got {pred_quantiles.shape}.")
    if pred_quantiles.shape[2] <= 1:
        return 0.0
    crossings = np.diff(pred_quantiles, axis=2) < 0.0
    return float(crossings.mean())


def prediction_interval_coverage(pred_quantiles, true):
    pred_quantiles = np.asarray(pred_quantiles, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    if pred_quantiles.ndim != 4 or true.ndim != 3:
        raise ValueError("prediction_interval_coverage expects pred_quantiles [N,T,Q,C] and true [N,T,C].")
    lower = pred_quantiles[:, :, 0, :]
    upper = pred_quantiles[:, :, -1, :]
    covered = (true >= lower) & (true <= upper)
    return float(covered.mean())


def prediction_interval_width(pred_quantiles):
    pred_quantiles = np.asarray(pred_quantiles, dtype=np.float64)
    if pred_quantiles.ndim != 4:
        raise ValueError(f"pred_quantiles must have shape [N,T,Q,C], got {pred_quantiles.shape}.")
    return float((pred_quantiles[:, :, -1, :] - pred_quantiles[:, :, 0, :]).mean())


def quantile_metric(pred_quantiles, true, quantiles):
    return {
        "pinball": quantile_pinball_loss(pred_quantiles, true, quantiles),
        "coverage": prediction_interval_coverage(pred_quantiles, true),
        "interval_width": prediction_interval_width(pred_quantiles),
        "crossing_rate": quantile_crossing_rate(pred_quantiles),
    }
