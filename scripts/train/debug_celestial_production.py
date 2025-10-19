#!/usr/bin/env python3
"""
DEBUG Version of Celestial Enhanced PGAT Production Training
Simplified for easier debugging and issue identification
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
import time
from datetime import datetime
import json
import traceback

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from data_provider.data_factory import data_provider
from models.Celestial_Enhanced_PGAT import Model
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric
import warnings
warnings.filterwarnings('ignore')

class SimpleConfig:
    """Simple configuration class"""
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)

def debug_memory_usage():
    """Debug memory usage"""
    if torch.cuda.is_available():
        print(f"🔍 GPU Memory: {torch.cuda.memory_allocated()/1024**2:.1f}MB allocated, {torch.cuda.memory_reserved()/1024**2:.1f}MB reserved")
        print(f"🔍 GPU Memory Free: {torch.cuda.memory_reserved()/1024**2 - torch.cuda.memory_allocated()/1024**2:.1f}MB")

def debug_tensor_shapes(batch_x, batch_y, batch_x_mark, batch_y_mark, outputs, args):
    """Debug tensor shapes"""
    print(f"🔍 Tensor Shapes Debug:")
    print(f"   - batch_x: {batch_x.shape}")
    print(f"   - batch_y: {batch_y.shape}")
    print(f"   - batch_x_mark: {batch_x_mark.shape}")
    print(f"   - batch_y_mark: {batch_y_mark.shape}")
    if isinstance(outputs, torch.Tensor):
        print(f"   - outputs: {outputs.shape}")
    elif isinstance(outputs, (tuple, list)):
        print(f"   - outputs: tuple/list with {len(outputs)} elements")
        for i, out in enumerate(outputs):
            if isinstance(out, torch.Tensor):
                print(f"     - outputs[{i}]: {out.shape}")
            else:
                print(f"     - outputs[{i}]: {type(out)}")
    print(f"   - Expected pred_len: {args.pred_len}")
    print(f"   - Expected c_out: {args.c_out}")

def main():
    """Main debug function"""
    print("🐛 Starting DEBUG Celestial Enhanced PGAT Training")
    print("=" * 60)
    
    try:
        # Load configuration
        config_path = "configs/celestial_enhanced_pgat_production.yaml"
        print(f"📁 Loading config from: {config_path}")
        
        if not os.path.exists(config_path):
            print(f"❌ Config file not found: {config_path}")
            return 1
            
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Create config object
        args = SimpleConfig(config_dict)
        
        # Add missing required attributes
        args.task_name = 'long_term_forecast'
        args.model_name = 'Celestial_Enhanced_PGAT'
        args.data_name = 'custom'
        args.checkpoints = './checkpoints/'
        args.inverse = False
        args.cols = None
        args.num_workers = 0
        args.itr = 1
        args.train_only = False
        args.do_predict = False
        args.model_id = f"debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Wave aggregation settings
        args.aggregate_waves_to_celestial = getattr(args, 'aggregate_waves_to_celestial', True)
        args.wave_to_celestial_mapping = getattr(args, 'wave_to_celestial_mapping', False)
        args.celestial_node_features = getattr(args, 'celestial_node_features', 13)
        args.target_wave_indices = getattr(args, 'target_wave_indices', [0, 1, 2, 3])
        
        print(f"✅ Configuration loaded successfully")
        print(f"   - Model: {args.model_name}")
        print(f"   - Sequence length: {args.seq_len}")
        print(f"   - Prediction length: {args.pred_len}")
        print(f"   - Model dimension: {args.d_model}")
        print(f"   - Batch size: {args.batch_size}")
        
        # Setup device
        if getattr(args, 'use_gpu', True) and torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"🚀 Using device: {device}")
            debug_memory_usage()
        else:
            device = torch.device('cpu')
            print(f"🚀 Using device: {device}")
        
        # Load data
        print("📂 Loading data...")
        try:
            train_data, train_loader = data_provider(args, flag='train')
            print(f"✅ Training data loaded: {len(train_loader)} batches")
            
            # Get first batch for debugging
            first_batch = next(iter(train_loader))
            batch_x, batch_y, batch_x_mark, batch_y_mark = first_batch
            
            print(f"🔍 First batch shapes:")
            print(f"   - batch_x: {batch_x.shape}")
            print(f"   - batch_y: {batch_y.shape}")
            print(f"   - batch_x_mark: {batch_x_mark.shape}")
            print(f"   - batch_y_mark: {batch_y_mark.shape}")
            
        except Exception as e:
            print(f"❌ Data loading failed: {e}")
            traceback.print_exc()
            return 1
        
        # Initialize model
        print("🏗️  Initializing model...")
        try:
            model = Model(args).to(device)
            print(f"✅ Model initialized successfully")
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"   - Total parameters: {total_params:,}")
            print(f"   - Trainable parameters: {trainable_params:,}")
            
            debug_memory_usage()
            
        except Exception as e:
            print(f"❌ Model initialization failed: {e}")
            traceback.print_exc()
            return 1
        
        # Test forward pass
        print("🧪 Testing forward pass...")
        try:
            model.eval()
            
            # Move batch to device
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)
            
            print(f"🔍 Input shapes on device:")
            print(f"   - batch_x: {batch_x.shape}")
            print(f"   - batch_y: {batch_y.shape}")
            
            # Prepare decoder input
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)
            print(f"   - dec_inp: {dec_inp.shape}")
            
            debug_memory_usage()
            
            # Forward pass
            with torch.no_grad():
                print("🔄 Running forward pass...")
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                print(f"✅ Forward pass successful!")
                
                debug_tensor_shapes(batch_x, batch_y, batch_x_mark, batch_y_mark, outputs, args)
                debug_memory_usage()
                
        except Exception as e:
            print(f"❌ Forward pass failed: {e}")
            traceback.print_exc()
            debug_memory_usage()
            return 1
        
        # Test loss computation
        print("🎯 Testing loss computation...")
        try:
            criterion = nn.MSELoss()
            
            # Handle model output
            if isinstance(outputs, (tuple, list)):
                outputs_tensor = outputs[0]
            else:
                outputs_tensor = outputs
            
            # Extract predictions and targets
            y_pred = outputs_tensor[:, -args.pred_len:, :args.c_out]
            y_true = batch_y[:, -args.pred_len:, :args.c_out]
            
            print(f"🔍 Loss computation shapes:")
            print(f"   - y_pred: {y_pred.shape}")
            print(f"   - y_true: {y_true.shape}")
            
            loss = criterion(y_pred, y_true)
            print(f"✅ Loss computation successful: {loss.item():.6f}")
            
        except Exception as e:
            print(f"❌ Loss computation failed: {e}")
            traceback.print_exc()
            return 1
        
        # Test training step
        print("🏋️  Testing training step...")
        try:
            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            
            # Handle model output
            if isinstance(outputs, (tuple, list)):
                outputs_tensor = outputs[0]
            else:
                outputs_tensor = outputs
            
            # Loss computation
            y_pred = outputs_tensor[:, -args.pred_len:, :args.c_out]
            y_true = batch_y[:, -args.pred_len:, :args.c_out]
            loss = criterion(y_pred, y_true)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            print(f"✅ Training step successful: {loss.item():.6f}")
            debug_memory_usage()
            
        except Exception as e:
            print(f"❌ Training step failed: {e}")
            traceback.print_exc()
            debug_memory_usage()
            return 1
        
        print("\n" + "=" * 60)
        print("🎉 DEBUG Test Complete - All components working!")
        print("✅ Model can be trained successfully")
        return 0
        
    except Exception as e:
        print(f"❌ DEBUG failed with error: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())