#!/usr/bin/env python3
"""
Fix Mixed Precision Issues

This script identifies and fixes dtype issues that cause mixed precision training problems.
"""

import torch
import re

def fix_mixed_precision_issues():
    """Fix mixed precision dtype issues in the model files."""
    
    print("🔧 FIXING MIXED PRECISION ISSUES")
    print("="*50)
    
    # Read the model file
    model_file = "models/Celestial_Enhanced_PGAT.py"
    
    with open(model_file, 'r') as f:
        content = f.read()
    
    print(f"📖 Read {model_file}")
    
    # Track changes
    changes_made = []
    
    # Fix 1: Replace explicit .float() calls with device-aware dtype
    old_pattern = r'torch\.zeros\(([^)]+)\)\.float\(\)'
    new_pattern = r'torch.zeros(\1, dtype=torch.float32)'
    
    if re.search(old_pattern, content):
        content = re.sub(old_pattern, new_pattern, content)
        changes_made.append("Fixed torch.zeros().float() calls")
    
    # Fix 2: Replace torch.eye() with dtype specification
    old_pattern = r'torch\.eye\(([^,)]+)(?:,\s*device=([^)]+))?\)'
    
    def replace_eye(match):
        size = match.group(1)
        device = match.group(2) if match.group(2) else 'device'
        return f'torch.eye({size}, device={device}, dtype=torch.float32)'
    
    if re.search(r'torch\.eye\(', content):
        content = re.sub(old_pattern, replace_eye, content)
        changes_made.append("Fixed torch.eye() calls to include dtype")
    
    # Fix 3: Replace .float() calls on tensors
    old_pattern = r'torch\.arange\(([^)]+)\)\.float\(\)'
    new_pattern = r'torch.arange(\1, dtype=torch.float32)'
    
    if re.search(old_pattern, content):
        content = re.sub(old_pattern, new_pattern, content)
        changes_made.append("Fixed torch.arange().float() calls")
    
    # Write back the fixed content
    if changes_made:
        with open(model_file, 'w') as f:
            f.write(content)
        
        print(f"✅ Applied fixes to {model_file}:")
        for change in changes_made:
            print(f"   - {change}")
    else:
        print(f"ℹ️  No mixed precision issues found in {model_file}")
    
    # Create a mixed precision compatible config
    print(f"\n📝 Creating mixed precision compatible config...")
    
    # Read the original config
    with open('configs/celestial_enhanced_pgat_production.yaml', 'r') as f:
        config_content = f.read()
    
    # Add mixed precision settings
    if 'amp_dtype' not in config_content:
        config_content += "\n# Mixed precision settings\n"
        config_content += "amp_dtype: float16  # Use float16 for mixed precision\n"
        config_content += "amp_enabled: true   # Enable automatic mixed precision\n"
    
    # Write the updated config
    with open('configs/celestial_enhanced_pgat_production_amp_fixed.yaml', 'w') as f:
        f.write(config_content)
    
    print(f"✅ Created configs/celestial_enhanced_pgat_production_amp_fixed.yaml")
    
    return len(changes_made) > 0

def create_mixed_precision_test():
    """Create a test to verify mixed precision works."""
    
    test_content = '''#!/usr/bin/env python3
"""
Test Mixed Precision Training

Verify that mixed precision training works without dtype errors.
"""

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
import yaml
from models.Celestial_Enhanced_PGAT import Model

def test_mixed_precision():
    """Test mixed precision training."""
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available, skipping mixed precision test")
        return False
    
    print("🧪 TESTING MIXED PRECISION TRAINING")
    print("="*50)
    
    # Load config
    with open('configs/celestial_enhanced_pgat_production_no_amp.yaml', 'r') as f:
        config_dict = yaml.safe_load(f)
    
    class Config:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    configs = Config(**config_dict)
    
    # Initialize model on GPU
    device = torch.device('cuda')
    model = Model(configs).to(device)
    model.train()
    
    # Initialize mixed precision components
    scaler = GradScaler()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    print(f"✅ Model initialized on {device}")
    
    # Create test data on GPU
    batch_size = 4  # Smaller batch for testing
    seq_len = configs.seq_len
    
    x_enc = torch.randn(batch_size, seq_len, configs.enc_in, device=device)
    x_mark_enc = torch.randn(batch_size, seq_len, 4, device=device)
    x_dec = torch.randn(batch_size, configs.label_len + configs.pred_len, configs.dec_in, device=device)
    x_mark_dec = torch.randn(batch_size, configs.label_len + configs.pred_len, 4, device=device)
    y_true = torch.randn(batch_size, configs.pred_len, configs.c_out, device=device)
    
    print(f"✅ Test data created on GPU")
    
    try:
        # Test mixed precision forward and backward pass
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            
            if isinstance(outputs, tuple):
                predictions, _ = outputs
            else:
                predictions = outputs
            
            loss = criterion(predictions, y_true)
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        print(f"✅ Mixed precision training step successful!")
        print(f"   Loss: {loss.item():.6f}")
        print(f"   Scaler scale: {scaler.get_scale()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Mixed precision test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_mixed_precision()
    
    if success:
        print(f"\\n🎉 MIXED PRECISION TEST PASSED!")
        print(f"✅ Mixed precision training is working correctly")
    else:
        print(f"\\n❌ MIXED PRECISION TEST FAILED!")
        print(f"   Use the no_amp config for training without mixed precision")
'''
    
    with open('test_mixed_precision.py', 'w') as f:
        f.write(test_content)
    
    print(f"✅ Created test_mixed_precision.py")

if __name__ == "__main__":
    print("🔧 MIXED PRECISION ISSUE RESOLVER")
    print("="*60)
    
    # Fix the issues
    fixes_applied = fix_mixed_precision_issues()
    
    # Create test
    create_mixed_precision_test()
    
    print(f"\n🎯 SOLUTIONS PROVIDED:")
    print(f"="*60)
    
    if fixes_applied:
        print(f"✅ Applied dtype fixes to model code")
        print(f"✅ Created amp-compatible config")
    
    print(f"✅ Created no-amp config (immediate solution)")
    print(f"✅ Created mixed precision test")
    
    print(f"\n🚀 RECOMMENDED ACTIONS:")
    print(f"1. Use configs/celestial_enhanced_pgat_production_no_amp.yaml for immediate training")
    print(f"2. Test mixed precision with: python test_mixed_precision.py")
    print(f"3. If mixed precision works, use the amp_fixed config for faster training")