from utils.logger import logger
import pandas as pd
import numpy as np

def make_synthetic_csv(file_path, num_rows=1000):
    logger.info("Creating synthetic CSV file")
    
    # Generate random data
    data = {
        'id': range(1, num_rows + 1),
        'name': [f'Name{i}' for i in range(1, num_rows + 1)],
        'age': np.random.randint(18, 70, size=num_rows),
        'email': [f'user{i}@example.com' for i in range(1, num_rows + 1)],
        'signup_date': pd.date_range(start='1/1/2020', periods=num_rows, freq='D')
    }

    # Create a DataFrame
    df = pd.DataFrame(data)

    # Write to CSV
    df.to_csv(file_path, index=False)
    logger.info(f"Synthetic CSV file created at {file_path}")