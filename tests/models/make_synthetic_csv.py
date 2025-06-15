import pandas as pd
import numpy as np
import os

def create_synthetic_csv(filepath, n_points=1000):
    # Angles
    X = np.arange(n_points) * 5 * np.pi / 180
    X1 = np.arange(n_points) * 10 * np.pi / 180
    X2 = np.arange(n_points) * 15 * np.pi / 180
    # Covariates
    cov1 = np.sin(X)
    cov2 = np.sin(X1)
    cov3 = np.sin(X2)
    # Targets
    t1 = np.sin(X - X1)
    t2 = np.sin(X1 - X2)
    t3 = np.sin(X2 - X)
    # Date range
    date = pd.date_range(start='2020-01-01', periods=n_points, freq='H')
    df = pd.DataFrame({
        'date': date,
        'cov1': cov1,
        'cov2': cov2,
        'cov3': cov3,
        't1': t1,
        't2': t2,
        't3': t3
    })
    df.to_csv(filepath, index=False)
