import os
import numpy as np
import pandas as pd


def generate_csv(path: str, rows: int = 600, seed: int = 42):
    rng = np.random.default_rng(seed)

    dates = pd.date_range('2020-01-01', periods=rows, freq='D')
    df = pd.DataFrame({'date': dates})

    # Generate OHLC with a smooth random-walk backbone
    prices = np.cumsum(rng.normal(0, 0.1, rows)) + 100.0
    open_ = prices + rng.normal(0, 0.05, rows)
    high = open_ + np.abs(rng.normal(0, 0.1, rows))
    low = open_ - np.abs(rng.normal(0, 0.1, rows))
    close = open_ + rng.normal(0, 0.05, rows)

    df['Open'] = open_
    df['High'] = high
    df['Low'] = low
    df['Close'] = close

    # Add remaining features to reach 118 total (excluding 'date')
    # First 4 are OHLC, so add feat_4 .. feat_117 (114 features)
    for i in range(4, 118):
        df[f'feat_{i}'] = rng.normal(0, 1.0, rows)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    return df


if __name__ == '__main__':
    out_path = os.path.join('data', 'synthetic_multi_wave.csv')
    df = generate_csv(out_path)
    print(f"Wrote {out_path} with shape {df.shape} and columns: {list(df.columns)[:8]} ...")