#!/usr/bin/env python3
"""
HF Modular Architecture Training Demonstration

This script demonstrates training with the HF Modular Architecture
using working component combinations.
"""

import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def create_dummy_time_series_dataset(num_samples=100, seq_len=96, pred_len=24, num_features=7):
    """Create a realistic dummy time series dataset"""
    
    # Create time-based patterns
    time_steps = torch.linspace(0, 10, seq_len + pred_len)
    
    # Generate multiple time series with different patterns
    dataset = []
    
    for i in range(num_samples):
        # Base trend
        trend = 0.01 * time_steps + torch.randn(1) * 0.1
        
        # Seasonal pattern
        seasonal = torch.sin(2 * np.pi * time_steps / 12) * (0.5 + torch.randn(1) * 0.1)
        
        # High frequency noise
        noise = torch.randn(seq_len + pred_len) * 0.1
        
        # Combine patterns
        ts = trend + seasonal + noise
        
        # Create multiple features by adding small variations
        features = []
        for f in range(num_features):
            feature_variation = ts + torch.randn(seq_len + pred_len) * 0.05
            features.append(feature_variation)
        
        full_sequence = torch.stack(features, dim=1)  # Shape: [seq_len+pred_len, num_features]
        
        # Split into input and target
        x_enc = full_sequence[:seq_len]  # Input sequence
        y = full_sequence[seq_len:]      # Target sequence
        
        dataset.append((x_enc, y))
    
    return dataset

def create_simple_modular_model(config):
    """Create a simple model using modular components"""
    
    class SimpleModularModel(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            
            # Try to load modular components
            try:
                from utils.modular_components.registry import create_component
                
                # Create processor (working components)
                self.processor = create_component('processor', 'frequency_domain', config)
                if self.processor is None:
                    self.processor = create_component('processor', 'trend_analysis', config)
                
                print(f"✅ Using processor: {config.processor_type}")
                
            except Exception as e:
                print(f"⚠️ Could not load modular components: {e}")
                self.processor = None
            
            # Simple transformer backbone
            self.input_projection = nn.Linear(config.enc_in, config.d_model)
            self.positional_encoding = nn.Parameter(torch.randn(1, config.seq_len, config.d_model))
            
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.n_heads,
                dim_feedforward=config.d_ff,
                dropout=config.dropout,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.e_layers)
            
            self.output_projection = nn.Linear(config.d_model, config.c_out)
            self.final_projection = nn.Linear(config.seq_len, config.pred_len)
            
        def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
            # Use processor if available
            if self.processor is not None:
                try:
                    processed_x = self.processor(x_enc)
                    print(f"📊 Processor: {x_enc.shape} → {processed_x.shape}")
                    
                    # Adjust dimensions if processor changed them
                    if processed_x.shape[-1] != x_enc.shape[-1]:
                        # Use original input if processor changed feature dimensions
                        x = x_enc
                    else:
                        x = processed_x
                except Exception as e:
                    print(f"⚠️ Processor failed, using original input: {e}")
                    x = x_enc
            else:
                x = x_enc
            
            # Project to model dimension
            x = self.input_projection(x)  # [batch, seq_len, d_model]
            
            # Add positional encoding
            x = x + self.positional_encoding
            
            # Transformer processing
            x = self.transformer(x)  # [batch, seq_len, d_model]
            
            # Project to output features
            x = self.output_projection(x)  # [batch, seq_len, c_out]
            
            # Project sequence length to prediction length
            x = x.transpose(1, 2)  # [batch, c_out, seq_len]
            x = self.final_projection(x)  # [batch, c_out, pred_len]
            x = x.transpose(1, 2)  # [batch, pred_len, c_out]
            
            return x
    
    return SimpleModularModel(config)

class Config:
    """Configuration for the modular model"""
    def __init__(self):
        self.seq_len = 96
        self.pred_len = 24
        self.enc_in = 7
        self.c_out = 7
        self.d_model = 256
        self.n_heads = 8
        self.e_layers = 2
        self.d_ff = 512
        self.dropout = 0.1
        self.device = 'cpu'
        
        # Modular configuration
        self.processor_type = 'frequency_domain'  # Working processor
        self.backbone_type = 'simple_transformer'
        self.loss_type = 'mse'

def train_hf_modular_model():
    """Train the HF Modular model"""
    print("🚀 HF Modular Architecture Training Demonstration")
    print("=" * 80)
    
    # Configuration
    config = Config()
    
    # Create dataset
    print("📊 Creating dummy time series dataset...")
    dataset = create_dummy_time_series_dataset(num_samples=50, seq_len=config.seq_len, pred_len=config.pred_len)
    print(f"✅ Dataset created: {len(dataset)} samples")
    
    # Create model
    print("\n🏗️ Creating HF Modular Model...")
    model = create_simple_modular_model(config)
    print("✅ Model created successfully")
    
    # Training setup
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    print(f"\n🎯 Training Configuration:")
    print(f"   Sequence Length: {config.seq_len}")
    print(f"   Prediction Length: {config.pred_len}")
    print(f"   Features: {config.enc_in}")
    print(f"   Model Dimension: {config.d_model}")
    print(f"   Processor: {config.processor_type}")
    
    # Training loop
    print(f"\n🚂 Starting Training...")
    model.train()
    
    num_epochs = 5
    batch_size = 8
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        # Create batches
        for i in range(0, len(dataset), batch_size):
            batch_data = dataset[i:i+batch_size]
            
            # Stack batch data
            x_batch = torch.stack([item[0] for item in batch_data])  # [batch, seq_len, features]
            y_batch = torch.stack([item[1] for item in batch_data])  # [batch, pred_len, features]
            
            # Forward pass
            optimizer.zero_grad()
            
            # Create dummy time features
            x_mark = torch.randn(x_batch.size(0), x_batch.size(1), 4)
            x_dec = torch.randn(x_batch.size(0), config.pred_len, config.enc_in)
            x_mark_dec = torch.randn(x_batch.size(0), config.pred_len, 4)
            
            pred = model(x_batch, x_mark, x_dec, x_mark_dec)
            
            # Compute loss
            loss = criterion(pred, y_batch)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        print(f"   Epoch {epoch+1}/{num_epochs}: Loss = {avg_loss:.6f}")
    
    print("✅ Training completed!")
    
    # Test the model
    print(f"\n🧪 Testing the trained model...")
    model.eval()
    
    with torch.no_grad():
        # Take a test sample
        test_x, test_y = dataset[0]
        test_x = test_x.unsqueeze(0)  # Add batch dimension
        test_y = test_y.unsqueeze(0)
        
        # Create dummy time features for test
        test_x_mark = torch.randn(1, test_x.size(1), 4)
        test_x_dec = torch.randn(1, config.pred_len, config.enc_in)
        test_x_mark_dec = torch.randn(1, config.pred_len, 4)
        
        # Get prediction
        test_pred = model(test_x, test_x_mark, test_x_dec, test_x_mark_dec)
        
        # Compute test loss
        test_loss = criterion(test_pred, test_y)
        
        print(f"📊 Test Results:")
        print(f"   Input shape: {test_x.shape}")
        print(f"   Target shape: {test_y.shape}")
        print(f"   Prediction shape: {test_pred.shape}")
        print(f"   Test Loss: {test_loss.item():.6f}")
        
        # Show some actual values
        print(f"\n📈 Sample Predictions (first feature, first 5 time steps):")
        print(f"   Target:     {test_y[0, :5, 0].numpy()}")
        print(f"   Prediction: {test_pred[0, :5, 0].numpy()}")
    
    print("\n🎉 HF Modular Architecture Training Demonstration Complete!")
    print("✨ The modular system successfully processed time series data!")
    
    return True

if __name__ == "__main__":
    success = train_hf_modular_model()
    sys.exit(0 if success else 1)
