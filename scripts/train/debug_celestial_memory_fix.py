#!/usr/bin/env python3
"""
DEBUG Script to Test Celestial Memory Fix
Specifically targets the memory explosion in celestial graph processing
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import yaml
from pathlib import Path
import traceback

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from data_provider.data_factory import data_provider
from models.Celestial_Enhanced_PGAT import Model
import warnings
warnings.filterwarnings('ignore')

class SimpleConfig:
    """Simple configuration class"""
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)

def debug_celestial_graph_processing():
    """Debug the celestial graph processing specifically"""
    print("🐛 DEBUG: Celestial Graph Memory Issue")
    print("=" * 50)
    
    # Load debug config
    config_path = "configs/celestial_enhanced_pgat_production_debug.yaml"
    if os.path.exists(config_path):
        print(f"📁 Loading debug config from: {config_path}")
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
    else:
        print("⚠️  Debug config not found, using minimal config")
        # Load minimal config for testing
        config_dict = {
        'model': 'Celestial_Enhanced_PGAT',
        'data': 'custom',
        'root_path': './data',
        'data_path': 'prepared_financial_data.csv',
        'features': 'MS',
        'target': 'log_Open,log_High,log_Low,log_Close',
        'embed': 'timeF',
        'freq': 'd',
        'use_gpu': True,
        'seq_len': 96,  # Smaller for debugging
        'label_len': 48,
        'pred_len': 24,
        'batch_size': 4,  # Very small batch
        'd_model': 64,   # Smaller model
        'n_heads': 4,
        'e_layers': 2,
        'd_layers': 1,
        'd_ff': 256,
        'dropout': 0.1,
        'enc_in': 118,
        'dec_in': 118,
        'c_out': 4,
        'use_celestial_graph': True,
        'aggregate_waves_to_celestial': True,
        'celestial_fusion_layers': 2,
        'num_input_waves': 118,
        'target_wave_indices': [0, 1, 2, 3],
        'use_mixture_decoder': False,
        'use_stochastic_learner': False,
        'use_hierarchical_mapping': False,
        'use_efficient_covariate_interaction': True
    }
    
    args = SimpleConfig(config_dict)
    
    # Add required attributes
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
    args.model_id = "debug_memory"
    args.aggregate_waves_to_celestial = True
    args.wave_to_celestial_mapping = False
    args.celestial_node_features = 13
    args.target_wave_indices = [0, 1, 2, 3]
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"🔍 Initial GPU Memory: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
    
    try:
        # Load data
        print("📂 Loading minimal data...")
        train_data, train_loader = data_provider(args, flag='train')
        print(f"✅ Data loaded: {len(train_loader)} batches")
        
        # Get first batch
        batch_x, batch_y, batch_x_mark, batch_y_mark = next(iter(train_loader))
        print(f"🔍 Batch shapes:")
        print(f"   - batch_x: {batch_x.shape}")
        print(f"   - batch_y: {batch_y.shape}")
        
        # Move to device
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        batch_x_mark = batch_x_mark.float().to(device)
        batch_y_mark = batch_y_mark.float().to(device)
        
        if torch.cuda.is_available():
            print(f"🔍 After data loading: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
        
        # Initialize model
        print("🏗️  Initializing model...")
        model = Model(args).to(device)
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✅ Model initialized: {total_params:,} parameters")
        
        if torch.cuda.is_available():
            print(f"🔍 After model init: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
        
        # Prepare decoder input
        dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)
        
        print(f"🔍 Decoder input shape: {dec_inp.shape}")
        
        # Test forward pass with memory monitoring
        print("🧪 Testing forward pass with memory monitoring...")
        model.eval()
        
        with torch.no_grad():
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                start_memory = torch.cuda.memory_allocated()
                print(f"🔍 Start forward pass: {start_memory/1024**2:.1f}MB")
            
            try:
                # Forward pass
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                if torch.cuda.is_available():
                    end_memory = torch.cuda.memory_allocated()
                    peak_memory = torch.cuda.max_memory_allocated()
                    print(f"🔍 End forward pass: {end_memory/1024**2:.1f}MB")
                    print(f"🔍 Peak memory: {peak_memory/1024**2:.1f}MB")
                    print(f"🔍 Memory increase: {(end_memory - start_memory)/1024**2:.1f}MB")
                
                print(f"✅ Forward pass successful!")
                
                if isinstance(outputs, (tuple, list)):
                    print(f"🔍 Output type: tuple/list with {len(outputs)} elements")
                    for i, out in enumerate(outputs):
                        if isinstance(out, torch.Tensor):
                            print(f"   - outputs[{i}]: {out.shape}")
                else:
                    print(f"🔍 Output shape: {outputs.shape}")
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"❌ CUDA OOM Error: {e}")
                    if torch.cuda.is_available():
                        print(f"🔍 Memory at failure: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
                        print(f"🔍 Peak memory: {torch.cuda.max_memory_allocated()/1024**2:.1f}MB")
                    return False
                else:
                    raise e
        
        # Test with even smaller batch if successful
        print("🧪 Testing with batch size 1...")
        batch_x_small = batch_x[:1]
        batch_y_small = batch_y[:1]
        batch_x_mark_small = batch_x_mark[:1]
        batch_y_mark_small = batch_y_mark[:1]
        dec_inp_small = dec_inp[:1]
        
        with torch.no_grad():
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                start_memory = torch.cuda.memory_allocated()
            
            outputs_small = model(batch_x_small, batch_x_mark_small, dec_inp_small, batch_y_mark_small)
            
            if torch.cuda.is_available():
                end_memory = torch.cuda.memory_allocated()
                peak_memory = torch.cuda.max_memory_allocated()
                print(f"🔍 Batch=1 memory increase: {(end_memory - start_memory)/1024**2:.1f}MB")
                print(f"🔍 Batch=1 peak memory: {peak_memory/1024**2:.1f}MB")
        
        print("✅ Memory debugging complete - model works with small batches")
        return True
        
    except Exception as e:
        print(f"❌ Error during debugging: {e}")
        traceback.print_exc()
        if torch.cuda.is_available():
            print(f"🔍 Memory at error: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
        return False

def main():
    """Main debug function"""
    try:
        success = debug_celestial_graph_processing()
        if success:
            print("\n🎉 DEBUG SUCCESSFUL - Model can run with proper memory management")
            print("💡 Recommendations:")
            print("   - Use smaller batch sizes (4-8)")
            print("   - Monitor memory usage during training")
            print("   - Consider gradient checkpointing for larger models")
        else:
            print("\n❌ DEBUG FAILED - Memory issues detected")
            print("💡 Try:")
            print("   - Reducing batch_size further")
            print("   - Reducing seq_len or d_model")
            print("   - Using CPU for debugging")
        return 0 if success else 1
    except Exception as e:
        print(f"❌ Debug script failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())