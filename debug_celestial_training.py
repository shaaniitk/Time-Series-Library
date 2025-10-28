#!/usr/bin/env python3
"""
Debug script to test Celestial model training directly
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

class SimpleConfig:
    def __init__(self):
        # Data configuration
        self.data = 'custom'
        self.root_path = './data'
        self.data_path = 'prepared_financial_data.csv'
        self.features = 'MS'
        self.target = 'log_Open,log_High,log_Low,log_Close'
        self.freq = 'd'
        self.embed = 'timeF'
        
        # Sequence settings - Very short for debugging
        self.seq_len = 32
        self.label_len = 16
        self.pred_len = 8
        
        # Data split settings
        self.validation_length = 50
        self.test_length = 50
        
        # Model configuration - Small for debugging
        self.d_model = 64
        self.n_heads = 4
        self.e_layers = 2
        self.d_layers = 1
        self.d_ff = 128
        self.dropout = 0.1
        
        # Input/Output dimensions - Fixed for Celestial model
        self.enc_in = 113
        self.dec_in = 113
        self.c_out = 4
        
        # Training configuration - MINIMAL for debugging
        self.train_epochs = 1
        self.batch_size = 1
        self.learning_rate = 0.001
        self.patience = 5
        self.lradj = 'warmup_cosine'
        self.warmup_epochs = 1
        self.min_lr = 1e-6
        self.weight_decay = 0.0001
        self.clip_grad_norm = 1.0
        
        # Production workflow settings
        self.use_gpu = True
        self.mixed_precision = False
        self.gradient_accumulation_steps = 1
        
        # Component flags - All OFF for baseline test
        self.use_multi_scale_patching = False
        self.use_hierarchical_mapping = False
        self.use_stochastic_learner = False
        self.use_gated_graph_combiner = False
        self.use_mixture_decoder = False

try:
    from scripts.train.train_celestial_production import train_celestial_pgat_production
    
    print("🔍 Testing Celestial model training directly...")
    
    config_path = "debug_config.yaml"
    
    print(f"📊 Using config file: {config_path}")
    
    print("🚀 Starting Celestial training test...")
    result = train_celestial_pgat_production(config_path)
    
    print(f"✅ Training result: {result}")
    
except Exception as e:
    print(f"💥 ERROR: {e}")
    import traceback
    traceback.print_exc()