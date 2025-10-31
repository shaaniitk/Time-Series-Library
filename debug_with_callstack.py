#!/usr/bin/env python3
"""
Debug training with full call stack tracing
This will show exactly where the crash occurs
"""

import sys
import traceback
import signal
import threading
import time
from pathlib import Path

def timeout_handler():
    """Handle timeout to prevent infinite hangs"""
    print("\n" + "="*60)
    print("⏰ TIMEOUT DETECTED - TRAINING APPEARS TO BE HANGING")
    print("="*60)
    sys.exit(1)

def trace_calls(frame, event, arg):
    """Trace function calls to see where we get stuck"""
    if event == 'call':
        filename = frame.f_code.co_filename
        function_name = frame.f_code.co_name
        line_number = frame.f_lineno
        
        # Only trace our project files, not system libraries
        if any(path in filename for path in ['models/', 'scripts/', 'layers/', 'data_provider/']):
            print(f"📍 CALL: {Path(filename).name}:{line_number} -> {function_name}()")
    
    return trace_calls

def debug_training_with_callstack():
    """Debug training with full call stack and timeout protection"""
    
    print("🔍 DEBUGGING TRAINING WITH CALL STACK TRACING")
    print("=" * 60)
    
    # Set up timeout protection using threading (Windows compatible)
    timeout_timer = threading.Timer(30.0, timeout_handler)
    timeout_timer.start()
    
    try:
        # First, let's restore the original config and test it
        print("1. RESTORING ORIGINAL CONFIG...")
        
        # Re-enable celestial features in the ultimate config
        config_path = "configs/celestial_production_deep_ultimate.yaml"
        with open(config_path, 'r', encoding='utf-8') as f:
            config_content = f.read()
        
        # Restore celestial features
        config_content = config_content.replace(
            'use_celestial_graph: false        # Temporarily disabled for debugging',
            'use_celestial_graph: true'
        )
        config_content = config_content.replace(
            'aggregate_waves_to_celestial: false  # Temporarily disabled for debugging',
            'aggregate_waves_to_celestial: true'
        )
        
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(config_content)
        
        print("✅ Original config restored")
        
        # Now test with call tracing
        print("\n2. STARTING TRAINING WITH CALL TRACING...")
        print("   (Will timeout after 30 seconds if hanging)")
        
        # Enable call tracing
        sys.settrace(trace_calls)
        
        # Import and run training
        import yaml
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        # Add timeout protection to config
        config_dict['train_epochs'] = 1  # Just 1 epoch for testing
        config_dict['batch_size'] = 2    # Small batch
        
        print("📊 Config loaded, starting training function...")
        
        # Import training function
        from scripts.train.train_celestial_production import train_celestial_pgat_production
        
        print("🚀 Calling training function...")
        
        # This is where we expect the crash/hang to occur
        result = train_celestial_pgat_production(config_path)
        
        print(f"✅ Training completed successfully: {result}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        sys.settrace(None)
        signal.alarm(0)
        return False
        
    except Exception as e:
        print(f"\n❌ TRAINING CRASHED: {e}")
        print("\n📋 FULL CALL STACK:")
        print("=" * 60)
        traceback.print_exc()
        print("=" * 60)
        
        sys.settrace(None)
        signal.alarm(0)
        return False
    
    finally:
        # Clean up
        sys.settrace(None)
        signal.alarm(0)
    
    return True

def debug_zero_loss_issue():
    """Debug the zero loss issue specifically"""
    
    print("\n🔍 DEBUGGING ZERO LOSS ISSUE")
    print("=" * 40)
    
    try:
        import torch
        import yaml
        
        # Test with minimal config
        config_path = "configs/celestial_minimal_debug.yaml"
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        class SimpleConfig:
            def __init__(self, config_dict):
                for key, value in config_dict.items():
                    setattr(self, key, value)
        
        args = SimpleConfig(config)
        
        # Load data
        from data_provider.data_factory import data_provider
        train_data, train_loader = data_provider(args, flag='train')
        
        # Get first batch
        batch_iter = iter(train_loader)
        batch_x, batch_y, batch_x_mark, batch_y_mark = next(batch_iter)
        
        # Convert to float32
        batch_x = batch_x.float()
        batch_y = batch_y.float()
        batch_x_mark = batch_x_mark.float()
        batch_y_mark = batch_y_mark.float()
        
        print(f"📊 Batch shapes: x={batch_x.shape}, y={batch_y.shape}")
        print(f"📊 Data ranges: x=[{batch_x.min():.3f}, {batch_x.max():.3f}], y=[{batch_y.min():.3f}, {batch_y.max():.3f}]")
        
        # Load model
        from models.Celestial_Enhanced_PGAT_Modular import Model
        model = Model(args)
        
        # Create decoder input
        dec_inp = torch.zeros_like(batch_y[:, -args.pred_len-args.label_len:, :]).float()
        dec_inp[:, :args.label_len, :] = batch_x[:, -args.label_len:, :]
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        
        predictions = outputs[0] if isinstance(outputs, tuple) else outputs
        
        print(f"📊 Predictions shape: {predictions.shape}")
        print(f"📊 Predictions range: [{predictions.min():.6f}, {predictions.max():.6f}]")
        print(f"📊 Predictions std: {predictions.std():.6f}")
        
        # Check if predictions are constant
        if predictions.std() < 1e-6:
            print("⚠️  PREDICTIONS ARE CONSTANT - This is the zero loss issue!")
            print("🔍 Analyzing model components...")
            
            # Check model parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            print(f"   Total parameters: {total_params:,}")
            print(f"   Trainable parameters: {trainable_params:,}")
            
            # Check if parameters are initialized
            param_stats = []
            for name, param in model.named_parameters():
                if param.requires_grad:
                    param_stats.append({
                        'name': name,
                        'shape': param.shape,
                        'mean': param.data.mean().item(),
                        'std': param.data.std().item(),
                        'min': param.data.min().item(),
                        'max': param.data.max().item()
                    })
            
            # Show first few parameter stats
            print("🔍 Parameter statistics (first 5):")
            for i, stat in enumerate(param_stats[:5]):
                print(f"   {stat['name']}: mean={stat['mean']:.6f}, std={stat['std']:.6f}")
                if stat['std'] < 1e-6:
                    print(f"      ⚠️  Parameter {stat['name']} appears to be constant!")
        
        # Extract targets and compute loss
        targets = batch_y[:, -args.pred_len:, :args.c_out]
        loss = torch.nn.MSELoss()(predictions, targets)
        
        print(f"📊 Targets shape: {targets.shape}")
        print(f"📊 Targets range: [{targets.min():.6f}, {targets.max():.6f}]")
        print(f"📊 Loss: {loss.item():.6f}")
        
        if loss.item() < 1e-6:
            print("⚠️  LOSS IS ZERO - Predictions match targets exactly!")
            diff = (predictions - targets).abs()
            print(f"   Max difference: {diff.max():.6f}")
            print(f"   Mean difference: {diff.mean():.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Zero loss debugging failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 COMPREHENSIVE TRAINING DEBUG")
    print("=" * 60)
    
    # First debug the zero loss issue
    zero_loss_success = debug_zero_loss_issue()
    
    # Then debug the crash with call stack
    if zero_loss_success:
        print("\n" + "="*60)
        crash_debug_success = debug_training_with_callstack()
    else:
        print("❌ Skipping crash debug due to zero loss issue")
        crash_debug_success = False
    
    if zero_loss_success and crash_debug_success:
        print("\n✅ ALL DEBUGGING COMPLETED SUCCESSFULLY")
    else:
        print("\n❌ DEBUGGING REVEALED ISSUES - SEE OUTPUT ABOVE")
    
    sys.exit(0 if (zero_loss_success and crash_debug_success) else 1)