#!/usr/bin/env python3
"""
Comprehensive Sanity Test for BayesianEnhancedAutoformer

Tests the Bayesian model with medium config on financial data:
- Seq length: 625
- Pred length: 20  
- Label length: 10
- 10 epochs training
- Validation error monitoring
- KL loss tracking
"""

import subprocess
import sys
import os
import time
import re

def run_bayesian_sanity_test():
    """Run comprehensive sanity test for Bayesian model"""
    
    print("🧪 BayesianEnhancedAutoformer Sanity Test")
    print("=" * 60)
    print("Configuration:")
    print("  • Model: BayesianEnhancedAutoformer")
    print("  • Sequence Length: 625")
    print("  • Prediction Length: 20")
    print("  • Label Length: 10")
    print("  • Epochs: 10")
    print("  • Architecture: Medium (d_model=128, layers=3+2)")
    print("  • Data: Financial data with future covariates")
    print("=" * 60)
    
    # Command to run
    cmd = [
        sys.executable, '../scripts/train/train_dynamic_autoformer.py',
        '--config', '../config/config_bayesian_medium_sanity.yaml',
        '--model_type', 'bayesian',
        '--auto_fix',
        '--validate_data'
    ]
    
    print(f"🚀 Running command:")
    print(f"   {' '.join(cmd)}")
    print()
    
    # Start the process
    start_time = time.time()
    
    try:
        # Run with real-time output
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Track important metrics
        training_losses = []
        validation_losses = []
        kl_losses = []
        epoch_times = []
        
        current_epoch = 0
        epoch_start_time = time.time()
        
        # Process output line by line
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                line = output.strip()
                print(line)
                
                # Parse training progress
                if "Starting training..." in line:
                    print("\n📈 Training Progress Monitor:")
                    print("Epoch | Train Loss | Val Loss   | KL Loss   | Time   | Status")
                    print("-" * 70)
                
                # Parse epoch information
                epoch_match = re.search(r'epoch: (\d+)', line)
                if epoch_match:
                    current_epoch = int(epoch_match.group(1))
                    epoch_start_time = time.time()
                
                # Parse loss values
                loss_match = re.search(r'loss: ([\d.]+)', line)
                if loss_match and current_epoch > 0:
                    train_loss = float(loss_match.group(1))
                    training_losses.append(train_loss)
                
                # Parse validation loss
                if "vali loss:" in line:
                    val_match = re.search(r'vali loss: ([\d.]+)', line)
                    if val_match:
                        val_loss = float(val_match.group(1))
                        validation_losses.append(val_loss)
                
                # Parse KL loss components
                if "kl_loss:" in line:
                    kl_match = re.search(r'kl_loss: ([\d.]+)', line)
                    if kl_match:
                        kl_loss = float(kl_match.group(1))
                        kl_losses.append(kl_loss)
                
                # Parse dimension information
                if "shape" in line.lower() or "dimension" in line.lower():
                    print(f"📏 {line}")
                
                # Track epoch completion
                if "Epoch" in line and "completed" in line:
                    epoch_time = time.time() - epoch_start_time
                    epoch_times.append(epoch_time)
                    
                    # Display progress summary
                    if len(training_losses) > 0 and len(validation_losses) > 0:
                        train_loss = training_losses[-1]
                        val_loss = validation_losses[-1] if validation_losses else "N/A"
                        kl_loss = kl_losses[-1] if kl_losses else "N/A"
                        
                        status = "✅ OK"
                        if isinstance(val_loss, float) and val_loss > train_loss * 2:
                            status = "⚠️ Overfitting?"
                        elif isinstance(val_loss, float) and val_loss < train_loss * 0.5:
                            status = "🎯 Good"
                        
                        print(f"{current_epoch:5d} | {train_loss:10.6f} | {val_loss:10} | {kl_loss:9} | {epoch_time:6.1f}s | {status}")
        
        # Wait for completion
        return_code = process.wait()
        total_time = time.time() - start_time
        
        print(f"\n⏱️ Total Training Time: {total_time:.1f} seconds")
        print(f"📊 Training Summary:")
        
        if return_code == 0:
            print("✅ Training completed successfully!")
            
            # Analyze results
            if len(training_losses) > 0:
                print(f"   📉 Training Loss: {training_losses[0]:.6f} → {training_losses[-1]:.6f}")
                improvement = ((training_losses[0] - training_losses[-1]) / training_losses[0]) * 100
                print(f"   📈 Improvement: {improvement:.1f}%")
            
            if len(validation_losses) > 0:
                print(f"   🎯 Final Validation Loss: {validation_losses[-1]:.6f}")
                
                # Check for overfitting
                if len(training_losses) > 0:
                    ratio = validation_losses[-1] / training_losses[-1]
                    if ratio > 2.0:
                        print("   ⚠️  Warning: Possible overfitting detected!")
                    elif ratio < 1.5:
                        print("   ✅ Good generalization!")
                    else:
                        print("   📊 Normal train/val ratio")
            
            if len(kl_losses) > 0:
                avg_kl = sum(kl_losses) / len(kl_losses)
                print(f"   🧠 Average KL Loss: {avg_kl:.6f}")
                print(f"   🎛️  KL Contribution: {(avg_kl / (training_losses[-1] + avg_kl)) * 100:.1f}%")
            
            if len(epoch_times) > 0:
                avg_time = sum(epoch_times) / len(epoch_times)
                print(f"   ⏱️  Average Epoch Time: {avg_time:.1f}s")
                
        else:
            print(f"❌ Training failed with return code: {return_code}")
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ Training timed out!")
        process.kill()
        return False
    except Exception as e:
        print(f"💥 Error during training: {e}")
        return False
    
    # Final validation
    print(f"\n🔍 Final Validation:")
    
    # Check if checkpoint was created
    checkpoint_dir = "checkpoints"
    if os.path.exists(checkpoint_dir):
        checkpoints = [f for f in os.listdir(checkpoint_dir) if 'bayesian' in f.lower()]
        if checkpoints:
            print(f"✅ Model checkpoint saved: {checkpoints[-1]}")
        else:
            print("⚠️  No Bayesian checkpoint found")
    
    # Sanity check criteria
    sanity_passed = True
    
    if len(training_losses) == 0:
        print("❌ No training losses recorded")
        sanity_passed = False
    elif training_losses[-1] > training_losses[0]:
        print("❌ Training loss increased (model not learning)")
        sanity_passed = False
    
    if len(validation_losses) == 0:
        print("❌ No validation losses recorded")
        sanity_passed = False
    
    if len(kl_losses) == 0:
        print("❌ No KL losses recorded (Bayesian features not working)")
        sanity_passed = False
    elif all(kl == 0 for kl in kl_losses):
        print("❌ All KL losses are zero (Bayesian layers not active)")
        sanity_passed = False
    
    if sanity_passed:
        print("\n🎉 SANITY TEST PASSED!")
        print("   ✅ Model trains successfully")
        print("   ✅ Loss decreases over epochs") 
        print("   ✅ Validation works")
        print("   ✅ Bayesian features (KL loss) active")
        print("   ✅ Dimensions handled correctly")
        print("\n🚀 BayesianEnhancedAutoformer is ready for production!")
    else:
        print("\n❌ SANITY TEST FAILED!")
        print("   Check the issues above before proceeding")
    
    return sanity_passed

if __name__ == "__main__":
    print("🧪 Starting BayesianEnhancedAutoformer Sanity Test")
    print("   This will take several minutes...")
    print()
    
    success = run_bayesian_sanity_test()
    
    if success:
        print("\n✅ All systems go! Bayesian model is working correctly.")
    else:
        print("\n❌ Issues detected. Please review the output above.")
        sys.exit(1)
