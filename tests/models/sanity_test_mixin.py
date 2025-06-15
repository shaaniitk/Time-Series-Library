import unittest
import torch
import numpy as np
import pandas as pd
import os
from data_provider.data_factory import data_provider
from models.TimesNet import Model as TimesNet
import matplotlib.pyplot as plt

class SanityTestMixin:
    def run_sanity_test(self, ModelClass, device='cpu', epochs=10):
        # Parameters
        seq_len = 100
        pred_len = 50
        n_points = 1000
        d_model = 16
        e_layers = 2
        d_ff = 32
        enc_in = 3
        c_out = 3
        num_kernels = 6
        label_len = 50
        top_k = 2
        batch_size = 16
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
        # Save to a temp csv
        csv_path = 'synthetic_timeseries.csv'
        df.to_csv(csv_path, index=False)
        # Only cov1, cov2, cov3 as features, t1, t2, t3 as targets
        target_cols = ['t1', 't2', 't3']
        # Args for data provider
        class Args:
            pass
        args = Args()
        args.seq_len = seq_len
        args.label_len = label_len
        args.pred_len = pred_len
        args.d_model = d_model
        args.e_layers = e_layers
        args.d_ff = d_ff
        args.enc_in = 3  # Only cov1, cov2, cov3
        args.c_out = 3  # Predict all three targets
        args.embed = 'timeF'
        args.freq = 'h'
        args.dropout = 0.1
        args.task_name = 'long_term_forecast'
        args.features = 'M'
        args.data = 'custom'
        args.target = ','.join(target_cols)  # use all three as targets
        args.root_path = '.'
        args.data_path = csv_path
        args.num_workers = 0
        args.batch_size = batch_size
        args.augmentation_ratio = 0
        args.seasonal_patterns = None
        args.top_k = top_k
        args.num_kernels = num_kernels
        args.use_amp = False  # Disable AMP for simplicity
        args.use_multi_gpu = False  # Disable multi-GPU for simplicity
        # Use data provider
        dataset, loader = data_provider(args, flag='train')
        model = ModelClass(args).to(device)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.MSELoss()
        for epoch in range(epochs):
            for batch in loader:
                batch_x, batch_y, batch_x_mark, batch_y_mark = batch
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                batch_x_mark = batch_x_mark.float().to(device)
                batch_y_mark = batch_y_mark.float().to(device)
                optimizer.zero_grad()
                y_pred = model(batch_x, batch_x_mark, batch_y[:, :label_len, :], batch_y_mark)
                # Only compare prediction part
                y_true = batch_y[:, -pred_len:, :]
                y_pred = y_pred[:, -pred_len:, :]
                loss = criterion(y_pred, y_true)
                loss.backward()
                optimizer.step()
        # Evaluate on one batch
        model.eval()
        with torch.no_grad():
            for batch in loader:
                batch_x, batch_y, batch_x_mark, batch_y_mark = batch
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                batch_x_mark = batch_x_mark.float().to(device)
                batch_y_mark = batch_y_mark.float().to(device)
                y_pred = model(batch_x, batch_x_mark, batch_y[:, :label_len, :], batch_y_mark)
                y_true = batch_y[:, -pred_len:, :]
                y_pred = y_pred[:, -pred_len:, :]
                mse = torch.mean((y_pred - y_true) ** 2).item()
                # Plot first sample, first channel
                plt.figure(figsize=(8,4))
                plt.plot(y_true[0,:,0].cpu().numpy(), label='Ground Truth')
                plt.plot(y_pred[0,:,0].cpu().numpy(), label='Prediction')
                plt.title('Sanity Forecast: First Channel')
                plt.xlabel('Prediction Step')
                plt.ylabel('Value')
                plt.legend()
                plt.tight_layout()
                plt.show()
                break
        # Clean up temp file
        if os.path.exists(csv_path):
            os.remove(csv_path)
        return mse, y_pred.cpu().numpy(), y_true.cpu().numpy()
