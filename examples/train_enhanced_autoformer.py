import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from argparse import Namespace
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.EnhancedAutoformer import Model as EnhancedAutoformer
from utils.logger import logger

def prepare_data(df, seq_len, label_len, pred_len, n_targets, n_covariates):
    """Prepares sliding window samples from the dataframe."""
    data = df.values
    n_features = n_targets + n_covariates
    
    x_enc_list, x_mark_enc_list, x_dec_list, x_mark_dec_list, y_list = [], [], [], [], []
    
    for i in range(len(data) - seq_len - pred_len + 1):
        # Encoder inputs
        x_enc = data[i : i + seq_len, :n_features]
        
        # Decoder inputs
        dec_input_start = i + seq_len - label_len
        dec_input_end = i + seq_len + pred_len
        x_dec = data[dec_input_start:dec_input_end, :n_features]
        
        # Ground truth
        y = data[i + seq_len : i + seq_len + pred_len, :n_targets]
        
        # Dummy time features
        x_mark_enc = np.zeros((seq_len, 1))
        x_mark_dec = np.zeros((label_len + pred_len, 1))
        
        x_enc_list.append(x_enc)
        x_mark_enc_list.append(x_mark_enc)
        x_dec_list.append(x_dec)
        x_mark_dec_list.append(x_mark_dec)
        y_list.append(y)
        
    return (
        torch.FloatTensor(np.array(x_enc_list)),
        torch.FloatTensor(np.array(x_mark_enc_list)),
        torch.FloatTensor(np.array(x_dec_list)),
        torch.FloatTensor(np.array(x_mark_dec_list)),
        torch.FloatTensor(np.array(y_list))
    )

def main():
    logger.info("--- Training EnhancedAutoformer Example ---")
    
    # 1. Generate or ensure data exists
    data_path = 'examples/dummy_timeseries.csv'
    if not os.path.exists(data_path):
        from generate_dummy_data import generate_data
        generate_data()
        
    # 2. Load Data
    df = pd.read_csv(data_path, index_col='date', parse_dates=True)
    
    # 3. Model Configuration
    n_targets = 2
    n_covariates = 4
    
    configs = Namespace(
        task_name='long_term_forecast',
        seq_len=96,
        label_len=48,
        pred_len=24,
        enc_in=n_targets + n_covariates,
        dec_in=n_targets + n_covariates,
        c_out=n_targets,
        d_model=64,
        d_ff=128,
        e_layers=2,
        d_layers=1,
        n_heads=4,
        dropout=0.1,
        moving_avg=25,
        embed='timeF',
        freq='h',
        activation='gelu',
        norm_type='LayerNorm',
        # EnhancedAutoformer specific params (can be tuned)
        attention_type='adaptive_autocorrelation' 
    )
    
    # 4. Prepare Data
    x_enc, x_mark_enc, x_dec, x_mark_dec, y = prepare_data(
        df, configs.seq_len, configs.label_len, configs.pred_len, n_targets, n_covariates
    )
    
    logger.info(f"Data prepared. Shapes: x_enc={x_enc.shape}, y={y.shape}")
    
    # 5. Initialize Model, Loss, and Optimizer
    model = EnhancedAutoformer(configs)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 6. Training Loop
    epochs = 5
    batch_size = 16
    
    for epoch in range(epochs):
        epoch_loss = 0
        num_batches = 0
        
        for i in range(0, len(x_enc), batch_size):
            # Get batch
            batch_x_enc = x_enc[i : i + batch_size]
            batch_x_mark_enc = x_mark_enc[i : i + batch_size]
            batch_x_dec = x_dec[i : i + batch_size]
            batch_x_mark_dec = x_mark_dec[i : i + batch_size]
            batch_y = y[i : i + batch_size]
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(batch_x_enc, batch_x_mark_enc, batch_x_dec, batch_x_mark_dec)
            
            loss = criterion(outputs, batch_y)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
        avg_loss = epoch_loss / num_batches
        logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
        
    logger.info("--- Training Complete ---")

if __name__ == "__main__":
    main()
