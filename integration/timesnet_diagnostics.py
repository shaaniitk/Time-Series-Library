#!/usr/bin/env python3
"""
TimesNet Training Diagnostics Script

This script runs comprehensive diagnostics to identify training bottlenecks.
Simply run: python timesnet_diagnostics.py

It will automatically:
1. Check system capabilities (CUDA, PyTorch, etc.)
2. Test data loading performance
3. Test model creation and forward pass
4. Run micro-benchmark training test
5. Provide specific recommendations

Author: AI Assistant
Date: June 2025
"""

import os
import sys
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

try:
    from models.TimesNet import Model as TimesNet
    from data_provider.data_loader import Dataset_Custom
    from torch.utils.data import DataLoader
    from utils.metrics import metric
    from utils.tools import EarlyStopping, adjust_learning_rate
except ImportError as e:
    print(f"❌ Import error: {e}")
    print(f"💡 Make sure you're running this from the Time-Series-Library directory")
    print(f"🔧 Current directory: {os.getcwd()}")
    sys.exit(1)


class DiagnosticConfig:
    """Ultra-minimal configuration for diagnostics"""
    # === DATA CONFIGURATION ===
    data = 'custom'
    root_path = './data/'
    data_path = 'prepared_financial_data.csv'
    features = 'M'
    target = 'log_Close'
    freq = 'b'
    
    # === SEQUENCE PARAMETERS ===
    seq_len = 12                       # MINIMAL: Very short sequences
    label_len = 2                      # MINIMAL: Short label
    pred_len = 1                       # MINIMAL: Single prediction step
    
    # === MODEL CONFIGURATION ===
    enc_in = 118
    dec_in = 118
    c_out = 118
    d_model = 8                        # MINIMAL: Tiny model
    d_ff = 16                          # MINIMAL: Tiny feed-forward
    
    # === ATTENTION ===
    n_heads = 1                        # MINIMAL: Single head
    e_layers = 1                       # MINIMAL: Single layer
    d_layers = 1                       # MINIMAL: Single layer
    
    # === TIMESNET ===
    top_k = 1                          # MINIMAL: Single frequency
    num_kernels = 1                    # MINIMAL: Single kernel
    
    # === REGULARIZATION ===
    dropout = 0.0
    
    # === ADDITIONAL SETTINGS ===
    embed = 'timeF'
    activation = 'gelu'
    factor = 1
    distil = False
    moving_avg = 3
    output_attention = False
    
    # === TRAINING ===
    train_epochs = 1                   # MINIMAL: Single epoch
    batch_size = 8                     # MINIMAL: Small batch
    learning_rate = 0.01
    patience = 1
    lradj = 'type1'
    
    # === OPTIMIZATION ===
    loss = 'MSE'
    use_amp = False                    # Disable for debugging
    
    # === SYSTEM ===
    num_workers = 0                    # No multiprocessing
    seed = 2024
    task_name = 'short_term_forecast'
    des = 'diagnostic_test'
    checkpoints = f'./checkpoints/TimesNet_diagnostic_{datetime.now().strftime("%Y%m%d_%H%M")}'
    
    # Additional required parameters
    validation_length = 10
    test_length = 10
    val_len = 10
    test_len = 10


def print_header(title):
    """Print a formatted header"""
    print("\n" + "=" * 80)
    print(f"🔍 {title}")
    print("=" * 80)


def print_section(title):
    """Print a formatted section header"""
    print(f"\n{title}")
    print("-" * 60)


def check_system():
    """Check system capabilities"""
    print_header("SYSTEM DIAGNOSTICS")
    
    # Python version
    print(f"🐍 Python: {sys.version.split()[0]}")
    
    # PyTorch version and CUDA
    print(f"🔥 PyTorch: {torch.__version__}")
    
    # CUDA availability
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        print(f"🚀 CUDA: ✅ Available")
        print(f"   📱 GPU: {torch.cuda.get_device_name(0)}")
        print(f"   💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"   ⚡ CUDA Version: {torch.version.cuda}")
        device = torch.device('cuda')
    else:
        print("🚀 CUDA: ❌ Not available (CPU only)")
        print("   💡 Training will be slower but should still work")
        device = torch.device('cpu')
    
    # Memory test
    print_section("💾 Memory Test")
    try:
        test_tensor = torch.randn(1000, 1000)
        if cuda_available:
            test_tensor = test_tensor.cuda()
        print("   ✅ Basic tensor operations work")
        del test_tensor
        if cuda_available:
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"   ❌ Memory issue: {e}")
        return False, device
    
    # Data file check
    print_section("📊 Data File Check")
    data_path = "./data/prepared_financial_data.csv"
    if os.path.exists(data_path):
        file_size = os.path.getsize(data_path) / 1024**2  # MB
        print(f"   ✅ Data file found: {file_size:.1f} MB")
        data_ok = True
    else:
        print(f"   ❌ Data file not found: {data_path}")
        print("   💡 Run the data preparation notebook first")
        data_ok = False
    
    return data_ok, device


def test_data_loading(args, device):
    """Test data loading performance"""
    print_header("DATA LOADING DIAGNOSTICS")
    
    try:
        # Test dataset creation
        print_section("1️⃣ Dataset Creation")
        ds_start = time.time()
        
        dataset = Dataset_Custom(
            args=args,
            root_path=args.root_path,
            data_path=args.data_path,
            flag='train',
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            scale=True,
            timeenc=1 if args.embed == 'timeF' else 0,
            freq=args.freq
        )
        
        ds_time = time.time() - ds_start
        print(f"   ⏱️ Dataset creation: {ds_time:.3f}s")
        print(f"   📈 Train samples: {len(dataset)}")
        
        # Test dataloader creation
        print_section("2️⃣ DataLoader Creation")
        dl_start = time.time()
        
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,  # No shuffle for speed
            num_workers=args.num_workers,
            drop_last=True
        )
        
        dl_time = time.time() - dl_start
        print(f"   ⏱️ DataLoader creation: {dl_time:.3f}s")
        print(f"   📊 Number of batches: {len(dataloader)}")
        
        # Test batch loading
        print_section("3️⃣ Batch Loading Speed")
        batch_times = []
        
        data_iter = iter(dataloader)
        for i in range(min(3, len(dataloader))):
            batch_start = time.time()
            batch_x, batch_y, batch_x_mark, batch_y_mark = next(data_iter)
            batch_time = time.time() - batch_start
            batch_times.append(batch_time)
            print(f"   Batch {i+1}: {batch_time:.3f}s - Shape: {batch_x.shape}")
        
        avg_batch_time = sum(batch_times) / len(batch_times)
        print(f"\n📊 Data Loading Results:")
        print(f"   ⏱️ Average batch time: {avg_batch_time:.3f}s")
        
        if avg_batch_time > 0.1:
            print("   ⚠️ Data loading is slow!")
            return False, None, None
        else:
            print("   ✅ Data loading speed looks good!")
            return True, dataset, dataloader
        
    except Exception as e:
        print(f"   ❌ Data loading failed: {e}")
        return False, None, None


def test_model_creation(args, device):
    """Test model creation and forward pass"""
    print_header("MODEL DIAGNOSTICS")
    
    try:
        # Test model creation
        print_section("1️⃣ Model Creation")
        model_start = time.time()
        
        model = TimesNet(args).float().to(device)
        
        model_time = time.time() - model_start
        total_params = sum(p.numel() for p in model.parameters())
        
        print(f"   ⏱️ Model creation: {model_time:.3f}s")
        print(f"   🔢 Parameters: {total_params:,}")
        print(f"   💾 Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB")
        
        # Test forward pass
        print_section("2️⃣ Forward Pass Test")
        
        # Create dummy inputs
        batch_size = args.batch_size
        batch_x = torch.randn(batch_size, args.seq_len, args.enc_in).to(device)
        batch_x_mark = torch.randn(batch_size, args.seq_len, 4).to(device)
        batch_y = torch.randn(batch_size, args.label_len + args.pred_len, args.dec_in).to(device)
        batch_y_mark = torch.randn(batch_size, args.label_len + args.pred_len, 4).to(device)
        
        # Prepare decoder input
        dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).to(device)
        dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).to(device)
        
        # Forward pass
        model.eval()
        forward_start = time.time()
        
        with torch.no_grad():
            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        
        forward_time = time.time() - forward_start
        
        print(f"   ⏱️ Forward pass: {forward_time:.3f}s")
        print(f"   📊 Output shape: {outputs.shape}")
        
        if forward_time > 1.0:
            print("   ⚠️ Forward pass is slow!")
            return False, None
        else:
            print("   ✅ Forward pass speed looks good!")
            return True, model
        
    except Exception as e:
        print(f"   ❌ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_training_loop(args, device, model, dataloader):
    """Test actual training loop performance"""
    print_header("TRAINING LOOP DIAGNOSTICS")
    
    try:
        # Setup training components
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        
        print_section("🏃 Mini Training Test")
        print("Testing 3 batches to measure training speed...")
        
        model.train()
        batch_times = []
        data_iter = iter(dataloader)
        
        for i in range(min(3, len(dataloader))):
            batch_start = time.time()
            
            # Get batch
            batch_x, batch_y, batch_x_mark, batch_y_mark = next(data_iter)
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)
            
            # Prepare decoder input
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float().to(device)
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            
            # Calculate loss (only on target columns)
            target_outputs = outputs[:, -args.pred_len:, :4]
            target_y = batch_y[:, -args.pred_len:, :4]
            loss = criterion(target_outputs, target_y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            batch_time = time.time() - batch_start
            batch_times.append(batch_time)
            
            print(f"   Batch {i+1}: {batch_time:.3f}s - Loss: {loss.item():.6f}")
        
        avg_batch_time = sum(batch_times) / len(batch_times)
        estimated_epoch_time = avg_batch_time * len(dataloader)
        
        print(f"\n📊 Training Performance:")
        print(f"   ⏱️ Average batch time: {avg_batch_time:.3f}s")
        print(f"   🚀 Estimated epoch time: {estimated_epoch_time:.1f}s ({estimated_epoch_time/60:.1f} min)")
        print(f"   🎯 Full training estimate: {estimated_epoch_time * 10:.1f}s ({estimated_epoch_time * 10/60:.1f} min)")
        
        # Performance classification
        if avg_batch_time < 0.1:
            print("   ✅ EXCELLENT: Training speed is very fast!")
            performance = "excellent"
        elif avg_batch_time < 0.5:
            print("   🟢 GOOD: Training speed is acceptable")
            performance = "good"
        elif avg_batch_time < 2.0:
            print("   🟡 MODERATE: Training is slower than expected")
            performance = "moderate"
        else:
            print("   🔴 SLOW: Major performance issues detected")
            performance = "slow"
        
        return True, avg_batch_time, performance
        
    except Exception as e:
        print(f"   ❌ Training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None, "failed"


def detailed_bottleneck_analysis(args, device, model, dataloader):
    """Detailed analysis of where bottlenecks occur"""
    print_header("DETAILED BOTTLENECK ANALYSIS")
    
    try:
        model.train()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        
        data_iter = iter(dataloader)
        batch_x, batch_y, batch_x_mark, batch_y_mark = next(data_iter)
        
        # Time each component
        components = {}
        
        # 1. Data transfer
        transfer_start = time.time()
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        batch_x_mark = batch_x_mark.float().to(device)
        batch_y_mark = batch_y_mark.float().to(device)
        components['Data Transfer'] = time.time() - transfer_start
        
        # 2. Decoder preparation
        prep_start = time.time()
        dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float().to(device)
        dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)
        components['Decoder Prep'] = time.time() - prep_start
        
        # 3. Forward pass
        forward_start = time.time()
        optimizer.zero_grad()
        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        components['Forward Pass'] = time.time() - forward_start
        
        # 4. Loss calculation
        loss_start = time.time()
        target_outputs = outputs[:, -args.pred_len:, :4]
        target_y = batch_y[:, -args.pred_len:, :4]
        loss = criterion(target_outputs, target_y)
        components['Loss Calculation'] = time.time() - loss_start
        
        # 5. Backward pass
        backward_start = time.time()
        loss.backward()
        components['Backward Pass'] = time.time() - backward_start
        
        # 6. Optimizer step
        step_start = time.time()
        optimizer.step()
        components['Optimizer Step'] = time.time() - step_start
        
        # Analysis
        total_time = sum(components.values())
        
        print_section("⏱️ Component Timing Breakdown")
        sorted_components = sorted(components.items(), key=lambda x: x[1], reverse=True)
        
        for i, (component, comp_time) in enumerate(sorted_components):
            percentage = (comp_time / total_time) * 100
            symbol = "🔴" if i == 0 else "🟡" if i == 1 else "🟢"
            print(f"   {symbol} {component}: {comp_time:.3f}s ({percentage:.1f}%)")
        
        print(f"\n📊 Total time per batch: {total_time:.3f}s")
        
        # Memory usage
        if torch.cuda.is_available():
            print_section("💾 GPU Memory Usage")
            allocated = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2
            print(f"   📊 Allocated: {allocated:.1f} MB")
            print(f"   📈 Reserved: {reserved:.1f} MB")
        
        return sorted_components[0]  # Return biggest bottleneck
        
    except Exception as e:
        print(f"   ❌ Bottleneck analysis failed: {e}")
        return None


def provide_recommendations(system_ok, data_ok, model_ok, training_ok, performance, bottleneck):
    """Provide specific recommendations based on test results"""
    print_header("🎯 RECOMMENDATIONS & NEXT STEPS")
    
    if not system_ok:
        print("🚨 CRITICAL: System issues detected")
        print("   💡 Check PyTorch installation")
        print("   💡 Check CUDA installation if using GPU")
        print("   💡 Verify data file exists")
        return
    
    if not data_ok:
        print("🚨 CRITICAL: Data loading issues")
        print("   💡 Run data preparation notebook first")
        print("   💡 Check file paths and permissions")
        return
    
    if not model_ok:
        print("🚨 CRITICAL: Model creation failed")
        print("   💡 Check model imports and dependencies")
        print("   💡 Try reducing model size")
        return
    
    if not training_ok:
        print("🚨 CRITICAL: Training loop failed")
        print("   💡 Check for memory issues")
        print("   💡 Try smaller batch size")
        return
    
    # Performance-based recommendations
    print("✅ All systems functional! Performance recommendations:")
    
    if performance == "excellent":
        print("\n🎉 EXCELLENT PERFORMANCE!")
        print("   🚀 Your system is ready for fast training")
        print("   💡 You can use larger models and longer sequences")
        print("   🎯 Try the Ultra-Fast or Light configurations")
        
    elif performance == "good":
        print("\n🟢 GOOD PERFORMANCE")
        print("   ✅ Training should work well")
        print("   💡 Start with Light configuration")
        print("   🎯 You can experiment with medium configurations")
        
    elif performance == "moderate":
        print("\n🟡 MODERATE PERFORMANCE")
        print("   ⚠️ Training will be slower than optimal")
        print("   💡 Use Ultra-Fast or Debugging configurations")
        print("   🔧 Consider these optimizations:")
        
        if bottleneck and bottleneck[0] == 'Forward Pass':
            print("     - Reduce model size (d_model, layers)")
            print("     - Use shorter sequences")
        elif bottleneck and bottleneck[0] == 'Data Transfer':
            print("     - Use smaller batch sizes")
            print("     - Check if using appropriate device")
        elif bottleneck and bottleneck[0] == 'Backward Pass':
            print("     - Reduce model complexity")
            print("     - Use gradient clipping")
            
    else:  # slow
        print("\n🔴 SLOW PERFORMANCE")
        print("   ⚠️ Significant performance issues detected")
        print("   🚨 Use only Debugging configuration")
        print("   🔧 Required optimizations:")
        print("     - Set num_workers = 0")
        print("     - Use batch_size = 4 or 8")
        print("     - Use d_model = 8")
        print("     - Use single layer models")
        
    # Configuration recommendations
    print(f"\n🎯 RECOMMENDED CONFIGURATIONS:")
    
    if performance in ["excellent", "good"]:
        print("   ⚡ Ultra-Fast: 2 minutes (quick experiments)")
        print("   💡 Light: 5-10 minutes (standard training)")
        print("   ⚖️ Medium: 15-25 minutes (better accuracy)")
        
    elif performance == "moderate":
        print("   ⚡ Ultra-Fast: 5-10 minutes")
        print("   💡 Light: 15-30 minutes")
        print("   🚨 Debugging: 2-5 minutes (if issues persist)")
        
    else:  # slow
        print("   🚨 Debugging ONLY: 5-15 minutes")
        print("   ⚠️ Other configurations may be too slow")
    
    print(f"\n📋 FILES TO USE:")
    print(f"   📓 TimesNet_Light_Config.ipynb - Main training notebook")
    print(f"   📓 TimesNet_Medium_Config.ipynb - Medium complexity")
    print(f"   📓 TimesNet_Mid_Heavy_Config.ipynb - High complexity")


def main():
    """Main diagnostic function"""
    print("🧪 TIMESNET TRAINING DIAGNOSTICS")
    print("🎯 Comprehensive performance analysis for TimesNet training")
    print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize
    args = DiagnosticConfig()
    os.makedirs(args.checkpoints, exist_ok=True)
    
    # Run diagnostics
    system_ok, device = check_system()
    
    if system_ok:
        data_ok, dataset, dataloader = test_data_loading(args, device)
        
        if data_ok:
            model_ok, model = test_model_creation(args, device)
            
            if model_ok:
                training_ok, avg_batch_time, performance = test_training_loop(args, device, model, dataloader)
                
                if training_ok:
                    bottleneck = detailed_bottleneck_analysis(args, device, model, dataloader)
                else:
                    bottleneck = None
            else:
                training_ok, avg_batch_time, performance, bottleneck = False, None, "failed", None
        else:
            model_ok, training_ok, avg_batch_time, performance, bottleneck = False, False, None, "failed", None
    else:
        data_ok, model_ok, training_ok, avg_batch_time, performance, bottleneck = False, False, False, None, "failed", None
    
    # Provide recommendations
    provide_recommendations(system_ok, data_ok, model_ok, training_ok, performance, bottleneck)
    
    # Final summary
    print_header("📋 DIAGNOSTIC SUMMARY")
    
    status_symbols = {
        True: "✅", 
        False: "❌"
    }
    
    print(f"🔧 System Check: {status_symbols[system_ok]}")
    print(f"📊 Data Loading: {status_symbols[data_ok]}")
    print(f"🧠 Model Creation: {status_symbols[model_ok]}")
    print(f"🏃 Training Loop: {status_symbols[training_ok]}")
    
    if training_ok:
        print(f"⚡ Performance Level: {performance.upper()}")
        print(f"⏱️ Batch Time: {avg_batch_time:.3f}s")
    
    print(f"\n🎯 NEXT STEP: Open the appropriate Jupyter notebook and start training!")
    print(f"📍 All notebooks are in the same directory as this script")
    
    return system_ok and data_ok and model_ok and training_ok


if __name__ == "__main__":
    try:
        success = main()
        if success:
            print(f"\n🎉 Diagnostics completed successfully!")
            sys.exit(0)
        else:
            print(f"\n⚠️ Issues detected - follow the recommendations above")
            sys.exit(1)
    except KeyboardInterrupt:
        print(f"\n⏹️ Diagnostics interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error during diagnostics: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
