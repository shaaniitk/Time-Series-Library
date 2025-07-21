import pandas as pd
import numpy as np

def generate_data(n_points=500, n_targets=2, n_covariates=4):
    """
    Generates a dummy time series dataset and saves it to a CSV file.
    """
    print("Generating dummy time series data...")
    
    # Create a date range
    dates = pd.to_datetime(pd.date_range(start='2023-01-01', periods=n_points, freq='h'))
    
    # Generate target variables with some seasonality and trend
    time = np.arange(n_points)
    targets = {}
    for i in range(n_targets):
        trend = 0.001 * time * (i + 1)
        seasonality1 = np.sin(2 * np.pi * time / (24 * (i + 1)))
        seasonality2 = np.cos(2 * np.pi * time / (168 * (i + 1))) # Weekly
        noise = np.random.randn(n_points) * 0.1
        targets[f'target_{i+1}'] = trend + seasonality1 + seasonality2 + noise

    # Generate covariates
    covariates = {}
    for i in range(n_covariates):
        seasonality = np.sin(2 * np.pi * time / (12 * (i + 1)))
        noise = np.random.randn(n_points) * 0.2
        covariates[f'covariate_{i+1}'] = seasonality + noise
        
    # Combine into a DataFrame
    df = pd.DataFrame(targets)
    df_cov = pd.DataFrame(covariates)
    df = pd.concat([df, df_cov], axis=1)
    df['date'] = dates
    
    # Set date as index
    df = df.set_index('date')
    
    # Save to CSV
    output_path = 'examples/dummy_timeseries.csv'
    df.to_csv(output_path)
    
    print(f"Successfully generated and saved data to {output_path}")
    print("\nData Head:")
    print(df.head())
    
if __name__ == "__main__":
    generate_data()
