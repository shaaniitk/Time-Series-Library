#!/usr/bin/env python3
"""
Validate compatibility between train_celestial_production.py and celestial_production_deep_ultimate.yaml
Check for potential issues before running full training
"""

import yaml
import sys
from pathlib import Path

def validate_compatibility():
    """Check for compatibility issues between script and config"""
    
    print("🔍 VALIDATING SCRIPT-CONFIG COMPATIBILITY")
    print("=" * 60)
    
    # Load the ultimate config
    config_path = "configs/celestial_production_deep_ultimate.yaml"
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"✅ Config loaded: {config_path}")
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        return False
    
    issues_found = []
    warnings = []
    
    # Check 1: Training script function name
    print("\n1. CHECKING TRAINING FUNCTION...")
    script_path = "scripts/train/train_celestial_production.py"
    try:
        with open(script_path, 'r', encoding='utf-8') as f:
            script_content = f.read()
        
        if "def train_celestial_pgat_production(" in script_content:
            print("✅ Training function found")
        else:
            issues_found.append("Training function 'train_celestial_pgat_production' not found in script")
    except Exception as e:
        issues_found.append(f"Could not read training script: {e}")
    
    # Check 2: Required config attributes
    print("\n2. CHECKING REQUIRED CONFIG ATTRIBUTES...")
    required_attrs = [
        'seq_len', 'pred_len', 'label_len', 'train_epochs', 'batch_size', 
        'learning_rate', 'd_model', 'n_heads', 'e_layers', 'd_layers',
        'enc_in', 'dec_in', 'c_out', 'data', 'root_path', 'data_path'
    ]
    
    for attr in required_attrs:
        if attr in config:
            print(f"  ✅ {attr}: {config[attr]}")
        else:
            issues_found.append(f"Missing required config attribute: {attr}")
    
    # Check 3: Dimension compatibility
    print("\n3. CHECKING DIMENSION COMPATIBILITY...")
    d_model = config.get('d_model', 0)
    n_heads = config.get('n_heads', 0)
    num_celestial = config.get('num_celestial_bodies', 13)
    
    if d_model > 0 and n_heads > 0:
        if d_model % n_heads == 0:
            print(f"  ✅ d_model ({d_model}) divisible by n_heads ({n_heads})")
        else:
            issues_found.append(f"d_model ({d_model}) not divisible by n_heads ({n_heads})")
    
    if d_model > 0 and num_celestial > 0:
        if d_model % num_celestial == 0:
            print(f"  ✅ d_model ({d_model}) divisible by num_celestial_bodies ({num_celestial})")
        else:
            warnings.append(f"d_model ({d_model}) not divisible by num_celestial_bodies ({num_celestial}) - may cause issues")
    
    # Check 4: Loss configuration
    print("\n4. CHECKING LOSS CONFIGURATION...")
    loss_config = config.get('loss', {})
    if isinstance(loss_config, dict):
        loss_type = loss_config.get('type', 'mse')
        print(f"  ✅ Loss type: {loss_type}")
        
        if loss_type == 'hybrid_mdn_directional':
            # Check if required loss components exist
            required_loss_files = [
                'layers/modular/losses/directional_trend_loss.py'
            ]
            for loss_file in required_loss_files:
                if Path(loss_file).exists():
                    print(f"    ✅ Loss file exists: {loss_file}")
                else:
                    warnings.append(f"Loss file missing: {loss_file} - will use fallback")
    
    # Check 5: MDN Decoder configuration
    print("\n5. CHECKING MDN DECODER CONFIGURATION...")
    enable_mdn = config.get('enable_mdn_decoder', False)
    if enable_mdn:
        mdn_components = config.get('mdn_components', 3)
        print(f"  ✅ MDN enabled with {mdn_components} components")
        
        # Check if MDN decoder file exists
        mdn_file = "layers/modular/decoder/mdn_decoder.py"
        if Path(mdn_file).exists():
            print(f"    ✅ MDN decoder file exists")
        else:
            issues_found.append(f"MDN decoder file missing: {mdn_file}")
    
    # Check 6: Data configuration
    print("\n6. CHECKING DATA CONFIGURATION...")
    data_path = config.get('data_path', '')
    root_path = config.get('root_path', './data')
    full_data_path = Path(root_path) / data_path
    
    if full_data_path.exists():
        print(f"  ✅ Data file exists: {full_data_path}")
    else:
        warnings.append(f"Data file not found: {full_data_path} - ensure data is available")
    
    # Check 7: Target configuration
    print("\n7. CHECKING TARGET CONFIGURATION...")
    target_indices = config.get('target_wave_indices', [])
    c_out = config.get('c_out', 4)
    
    if target_indices:
        if len(target_indices) == c_out:
            print(f"  ✅ Target indices ({len(target_indices)}) match c_out ({c_out})")
        else:
            issues_found.append(f"Target indices length ({len(target_indices)}) != c_out ({c_out})")
    else:
        warnings.append("No target_wave_indices specified - will use first c_out features")
    
    # Check 8: Hardware configuration
    print("\n8. CHECKING HARDWARE CONFIGURATION...")
    use_gpu = config.get('use_gpu', True)
    use_amp = config.get('use_amp', False)
    
    if not use_gpu:
        print("  ⚠️  GPU disabled - training will be slow")
        warnings.append("GPU training disabled - expect very slow training")
    
    if use_amp and not use_gpu:
        issues_found.append("AMP enabled but GPU disabled - incompatible configuration")
    
    # Check 9: Memory configuration
    print("\n9. CHECKING MEMORY CONFIGURATION...")
    batch_size = config.get('batch_size', 4)
    seq_len = config.get('seq_len', 500)
    
    estimated_memory_gb = (batch_size * seq_len * d_model * 4) / (1024**3)  # Rough estimate
    print(f"  📊 Estimated memory usage: ~{estimated_memory_gb:.1f} GB")
    
    if estimated_memory_gb > 16:
        warnings.append(f"High memory usage estimated ({estimated_memory_gb:.1f} GB) - may cause OOM")
    
    # Summary
    print("\n" + "=" * 60)
    print("COMPATIBILITY VALIDATION SUMMARY")
    print("=" * 60)
    
    if not issues_found and not warnings:
        print("🎉 ALL CHECKS PASSED - SCRIPT AND CONFIG ARE COMPATIBLE!")
        print("✅ Ready for production training")
        return True
    
    if issues_found:
        print("❌ CRITICAL ISSUES FOUND:")
        for issue in issues_found:
            print(f"  - {issue}")
    
    if warnings:
        print("⚠️  WARNINGS:")
        for warning in warnings:
            print(f"  - {warning}")
    
    if not issues_found:
        print("\n✅ NO CRITICAL ISSUES - TRAINING SHOULD WORK")
        print("⚠️  Address warnings for optimal performance")
        return True
    else:
        print("\n❌ CRITICAL ISSUES MUST BE FIXED BEFORE TRAINING")
        return False

if __name__ == "__main__":
    success = validate_compatibility()
    sys.exit(0 if success else 1)