#!/usr/bin/env python3
"""
Simple Training Script with Dummy Data

This script creates dummy time series data and trains a basic Autoformer model
to validate the training pipeline works correctly.
"""

import sys
import os
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import argparse
import json
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import models and utilities
try:
    from models.modular_autoformer import ModularAutoformer as Autoformer
    print("✅ Successfully imported Autoformer")
except ImportError as e:
    print(f"❌ Could not import Autoformer: {e}")
    Autoformer = None

try:
    from models.TimesNet import Model as TimesNet
    print("✅ Successfully imported TimesNet")
except ImportError as e:
    print(f"❌ Could not import TimesNet: {e}")
    TimesNet = None

try:
    from models.EnhancedAutoformer import Model as EnhancedAutoformer
    print("✅ Successfully imported EnhancedAutoformer")
except ImportError as e:
    print(f"❌ Could not import EnhancedAutoformer: {e}")
    EnhancedAutoformer = None

class Config:
    """Configuration class for model training"""
    def __init__(self, **kwargs):
        # Basic data parameters
        self.seq_len = 96          # Input sequence length
        self.label_len = 48        # Label length for encoder-decoder
        self.pred_len = 24         # Prediction horizon
        self.enc_in = 7            # Number of input features
        self.dec_in = 7            # Number of decoder input features
        self.c_out = 7             # Number of output features
        self.features = 'M'        # Forecasting mode: M-multivariate, S-univariate
        
        # Model parameters
        self.d_model = 512         # Model dimension
        self.n_heads = 8           # Number of attention heads
        self.e_layers = 2          # Number of encoder layers
        self.d_layers = 1          # Number of decoder layers
        self.d_ff = 2048          # Feed forward dimension
        self.moving_avg = 25       # Moving average window
        self.factor = 1            # Attention factor
        self.distil = True         # Whether to use distilling in encoder
        self.dropout = 0.1         # Dropout rate
        self.embed = 'timeF'       # Time features encoding
        self.activation = 'gelu'   # Activation function
        self.output_attention = False
        
        # Additional required parameters
        self.task_name = 'long_term_forecast'  # Task type
        self.is_training = 1       # Training flag
        self.model_id = 'dummy'    # Model identifier
        self.model = 'Autoformer'  # Model name
        self.data = 'dummy'        # Dataset name
        self.root_path = './'      # Root path
        self.data_path = 'dummy.csv'  # Data path
        self.freq = 'h'            # Frequency
        self.checkpoints = './checkpoints/'  # Checkpoint path
        self.norm_type = 'LayerNorm'  # Normalization type
        
        # Training parameters
        self.num_workers = 0       # Number of data loading workers
        self.train_epochs = 10     # Number of training epochs
        self.batch_size = 32       # Batch size
        self.patience = 3          # Early stopping patience
        self.learning_rate = 0.0001
        
        # TimesNet specific parameters
        self.top_k = 5
        self.num_kernels = 6
        
        # Override with provided kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

def create_dummy_time_series_data(batch_size, seq_len, pred_len, num_features, signal_type='mixed'):
    """Create dummy time series data with realistic patterns"""
    
    total_len = seq_len + pred_len
    t = torch.linspace(0, 4*np.pi, total_len)
    
    if signal_type == 'trend':
        # Linear trend with some noise
        data = []
        for i in range(batch_size):
            trend = 0.01 * t + np.random.normal(0, 0.1)
            series = trend.unsqueeze(-1).expand(-1, num_features)
            series = series + 0.1 * torch.randn(total_len, num_features)
            data.append(series)
        
    elif signal_type == 'seasonal':
        # Seasonal patterns
        data = []
        for i in range(batch_size):
            seasonal = torch.sin(2 * t) + 0.5 * torch.cos(4 * t)
            series = seasonal.unsqueeze(-1).expand(-1, num_features)
            series = series + 0.1 * torch.randn(total_len, num_features)
            data.append(series)
            
    elif signal_type == 'mixed':
        # Mixed signal: trend + seasonal + noise
        data = []
        for i in range(batch_size):
            trend = 0.005 * t
            seasonal = 0.5 * torch.sin(2 * t) + 0.3 * torch.cos(4 * t)
            noise = 0.1 * torch.randn(total_len)
            combined = trend + seasonal + noise
            series = combined.unsqueeze(-1).expand(-1, num_features)
            data.append(series)
    
    else:
        # Random data
        data = [torch.randn(total_len, num_features) for _ in range(batch_size)]
    
    return torch.stack(data)

def create_time_features(data, freq='h'):
    """Create time features for the data"""
    batch_size, seq_len, _ = data.shape
    
    # Simple time features (hour, day of week, month, etc.)
    # For dummy data, we'll create simple cyclic features
    time_features = []
    
    for i in range(seq_len):
        # Hour of day (0-23) -> normalized cyclic features
        hour_sin = np.sin(2 * np.pi * (i % 24) / 24)
        hour_cos = np.cos(2 * np.pi * (i % 24) / 24)
        
        # Day of week (0-6) -> normalized cyclic features  
        day_sin = np.sin(2 * np.pi * (i % (24*7)) / (24*7))
        day_cos = np.cos(2 * np.pi * (i % (24*7)) / (24*7))
        
        time_features.append([hour_sin, hour_cos, day_sin, day_cos])
    
    # Convert to tensor and expand for batch
    time_feat = torch.tensor(time_features, dtype=torch.float32)
    time_feat = time_feat.unsqueeze(0).expand(batch_size, -1, -1)
    
    return time_feat

def create_data_batch(config, signal_type='mixed'):
    """Create a complete data batch for training"""
    
    # Create full time series
    full_data = create_dummy_time_series_data(
        config.batch_size, 
        config.seq_len + config.pred_len, 
        0,  # No additional pred_len since we include it in seq_len
        config.enc_in, 
        signal_type
    )
    
    # Split into encoder and decoder inputs
    x_enc = full_data[:, :config.seq_len, :]
    x_dec_start = full_data[:, config.seq_len-config.label_len:config.seq_len, :]
    x_dec_zeros = torch.zeros(config.batch_size, config.pred_len, config.dec_in)
    x_dec = torch.cat([x_dec_start, x_dec_zeros], dim=1)
    
    # Create time features
    x_mark_enc = create_time_features(x_enc)
    x_mark_dec = create_time_features(x_dec)
    
    # True values for prediction
    y = full_data[:, config.seq_len:config.seq_len+config.pred_len, :]
    
    return x_enc, x_mark_enc, x_dec, x_mark_dec, y

def train_model(model, config, num_batches=10):
    """Train the model with dummy data"""
    
    print(f"🚀 Starting training with {num_batches} batches...")
    
    # Setup optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()
    
    # Training loop
    model.train()
    total_loss = 0
    
    for batch_idx in range(num_batches):
        # Create data batch
        x_enc, x_mark_enc, x_dec, x_mark_dec, y_true = create_data_batch(config)
        
        # Forward pass
        optimizer.zero_grad()
        
        try:
            # Try standard forward pass
            y_pred = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        except Exception as e:
            print(f"❌ Forward pass failed: {e}")
            try:
                # Try alternative forward pass (some models have different signatures)
                y_pred = model(x_enc)
            except Exception as e2:
                print(f"❌ Alternative forward pass also failed: {e2}")
                return False
        
        # Calculate loss
        loss = criterion(y_pred, y_true)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 5 == 0:
            print(f"Batch {batch_idx}: Loss = {loss.item():.6f}")
    
    avg_loss = total_loss / num_batches
    print(f"✅ Training completed! Average Loss: {avg_loss:.6f}")
    
    return True

def test_model(model, config):
    """Test the model with dummy data"""
    
    print("🧪 Testing model inference...")
    
    model.eval()
    with torch.no_grad():
        # Create test batch
        x_enc, x_mark_enc, x_dec, x_mark_dec, y_true = create_data_batch(config)
        
        try:
            # Forward pass
            y_pred = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            
            print(f"✅ Inference successful!")
            print(f"   Input shape: {x_enc.shape}")
            print(f"   Output shape: {y_pred.shape}")
            print(f"   Target shape: {y_true.shape}")
            
            # Calculate test loss
            test_loss = nn.MSELoss()(y_pred, y_true)
            print(f"   Test Loss: {test_loss.item():.6f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Inference failed: {e}")
            return False

def main():
    """Main training function"""
    
    parser = argparse.ArgumentParser(description='Train time series model with dummy data')
    parser.add_argument('--model', type=str, default='autoformer', 
                       choices=['autoformer', 'timesnet', 'enhanced_autoformer'],
                       help='Model to train')
    parser.add_argument('--seq_len', type=int, default=96, help='Input sequence length')
    parser.add_argument('--pred_len', type=int, default=24, help='Prediction length')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    
    args = parser.parse_args()
    
    print("🎯 Time Series Model Training with Dummy Data")
    print("=" * 60)
    
    # Create configuration
    config = Config(
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        batch_size=args.batch_size,
        train_epochs=args.epochs,
        learning_rate=args.lr
    )
    
    print(f"📊 Configuration:")
    print(f"   Model: {args.model}")
    print(f"   Sequence Length: {config.seq_len}")
    print(f"   Prediction Length: {config.pred_len}")
    print(f"   Batch Size: {config.batch_size}")
    print(f"   Features: {config.enc_in}")
    print(f"   Learning Rate: {config.learning_rate}")
    
    # Initialize model
    print(f"\n🔧 Initializing {args.model} model...")
    
    try:
        if args.model == 'autoformer':
            if Autoformer is None:
                print("❌ Autoformer not available")
                return
            model = Autoformer(config)
            
        elif args.model == 'timesnet':
            if TimesNet is None:
                print("❌ TimesNet not available")
                return
            model = TimesNet(config)
            
        elif args.model == 'enhanced_autoformer':
            if EnhancedAutoformer is None:
                print("❌ EnhancedAutoformer not available")
                return
            model = EnhancedAutoformer(config)
            
        else:
            print(f"❌ Unknown model: {args.model}")
            return
            
        print(f"✅ {args.model} model initialized successfully")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        
    except Exception as e:
        print(f"❌ Failed to initialize model: {e}")
        return
    
    # Test model first
    print(f"\n🧪 Testing model inference...")
    if not test_model(model, config):
        print("❌ Model testing failed, aborting training")
        return
    
    # Train model
    print(f"\n🚀 Training model...")
    if train_model(model, config, num_batches=20):
        print("✅ Training completed successfully!")
        
        # Test again after training
        print(f"\n🧪 Testing trained model...")
        test_model(model, config)
        
        # Save model if desired
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"trained/dummy_{args.model}_{timestamp}.pth"
        
        # Create directory if it doesn't exist
        os.makedirs("trained", exist_ok=True)
        
        try:
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': config.__dict__,
                'timestamp': timestamp
            }, save_path)
            print(f"💾 Model saved to: {save_path}")
        except Exception as e:
            print(f"⚠️ Could not save model: {e}")
        
    else:
        print("❌ Training failed")

if __name__ == "__main__":
    main()
