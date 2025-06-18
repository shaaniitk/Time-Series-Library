#!/usr/bin/env python3
"""
Convenience script to train Enhanced Autoformers in all three modes (M, MS, S)
Medium complexity configurations
"""

import os
import sys
import subprocess
import time
from datetime import datetime

def run_command(cmd, description):
    """Run a command and log the output"""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"Command: {cmd}")
    print(f"{'='*60}")
    
    start_time = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=False)
    end_time = time.time()
    
    duration = end_time - start_time
    print(f"\n⏱️ Duration: {duration:.1f} seconds")
    
    if result.returncode == 0:
        print(f"✅ {description} completed successfully!")
    else:
        print(f"❌ {description} failed with return code {result.returncode}")
    
    return result.returncode == 0

def main():
    """Train all three Enhanced Autoformer variants in all three modes"""
    
    print("🔧 Enhanced Autoformer Training Suite")
    print("=" * 60)
    print("Training all three model variants in all three forecasting modes")
    print("Variants: Enhanced, Bayesian, Hierarchical") 
    print("Modes: M (All→All), MS (All→Targets), S (Targets→Targets)")
    print("Complexity: Medium")
    print("=" * 60)
    
    # Configuration files
    configs = {
        'M': '../config/config_enhanced_autoformer_M_medium.yaml',
        'MS': '../config/config_enhanced_autoformer_MS_medium.yaml', 
        'S': '../config/config_enhanced_autoformer_S_medium.yaml'
    }
    
    # Model types
    model_types = ['enhanced', 'bayesian', 'hierarchical']
    
    # Results tracking
    results = {}
    start_time = datetime.now()
    
    # Check if config files exist
    for mode, config_file in configs.items():
        if not os.path.exists(config_file):
            print(f"❌ Config file not found: {config_file}")
            return
    
    print("✅ All config files found")
    
    # Train each combination
    for model_type in model_types:
        results[model_type] = {}
        
        for mode, config_file in configs.items():
            
            description = f"Training {model_type.title()} Autoformer in {mode} mode"
            cmd = f"python ../scripts/train/train_configurable_autoformer.py --config {config_file} --model_type {model_type}"
            
            success = run_command(cmd, description)
            results[model_type][mode] = 'Success' if success else 'Failed'
            
            if not success:
                print(f"⚠️ Training failed for {model_type} in {mode} mode. Continuing with next...")
    
    # Summary
    end_time = datetime.now()
    total_duration = end_time - start_time
    
    print(f"\n🏁 TRAINING SUITE COMPLETED")
    print(f"{'='*60}")
    print(f"Total duration: {total_duration}")
    print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Finished: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    print(f"\n📊 RESULTS SUMMARY:")
    print(f"{'Model':<15} {'M Mode':<10} {'MS Mode':<10} {'S Mode':<10}")
    print("-" * 50)
    
    for model_type in model_types:
        m_result = results[model_type].get('M', 'Not run')
        ms_result = results[model_type].get('MS', 'Not run')
        s_result = results[model_type].get('S', 'Not run')
        
        print(f"{model_type.title():<15} {m_result:<10} {ms_result:<10} {s_result:<10}")
    
    # Check for any failures
    failures = []
    for model_type in model_types:
        for mode in ['M', 'MS', 'S']:
            if results[model_type].get(mode) == 'Failed':
                failures.append(f"{model_type}-{mode}")
    
    if failures:
        print(f"\n⚠️ Some training runs failed: {', '.join(failures)}")
        print("Check the logs above for error details.")
    else:
        print(f"\n🎉 All training runs completed successfully!")
    
    print(f"\n💡 Next steps:")
    print(f"   - Check model checkpoints in ./checkpoints/")
    print(f"   - Review training logs for performance comparison")
    print(f"   - Run inference/evaluation scripts to compare modes")


if __name__ == '__main__':
    main()
