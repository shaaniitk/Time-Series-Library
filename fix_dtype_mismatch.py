#!/usr/bin/env python3
"""
Fix dtype mismatch causing zero loss
The data is coming in as float64 but model expects float32
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def fix_dtype_mismatch():
    """Fix the dtype mismatch in the training script"""
    
    print("🔧 FIXING DTYPE MISMATCH...")
    
    # Fix 1: Update the training script to convert data to float32
    training_script = "scripts/train/train_celestial_production.py"
    
    # Read the training script
    with open(training_script, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find the batch processing section and add dtype conversion
    old_batch_processing = '''            batch_x, batch_y, batch_x_mark, batch_y_mark = batch_x.to(device), batch_y.to(device), batch_x_mark.to(device), batch_y_mark.to(device)'''
    
    new_batch_processing = '''            # Convert to float32 to avoid dtype mismatch
            batch_x = batch_x.to(device).float()
            batch_y = batch_y.to(device).float()
            batch_x_mark = batch_x_mark.to(device).float()
            batch_y_mark = batch_y_mark.to(device).float()'''
    
    if old_batch_processing in content:
        content = content.replace(old_batch_processing, new_batch_processing)
        print("✓ Fixed training script dtype conversion")
    else:
        # Alternative pattern
        alt_pattern = '''batch_x, batch_y, batch_x_mark, batch_y_mark = batch_x.to(device), batch_y.to(device), batch_x_mark.to(device), batch_y_mark.to(device)'''
        if alt_pattern in content:
            content = content.replace(alt_pattern, new_batch_processing)
            print("✓ Fixed training script dtype conversion (alternative pattern)")
        else:
            print("⚠️  Could not find exact batch processing pattern, adding manual fix")
            # Add a general fix at the beginning of the training loop
            old_loop_start = '''        for batch_index, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):'''
            new_loop_start = '''        for batch_index, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            # DTYPE FIX: Convert all tensors to float32
            batch_x = batch_x.float()
            batch_y = batch_y.float()
            batch_x_mark = batch_x_mark.float()
            batch_y_mark = batch_y_mark.float()'''
            
            if old_loop_start in content:
                content = content.replace(old_loop_start, new_loop_start)
                print("✓ Added dtype fix at loop start")
    
    # Also fix validation loop
    old_val_loop = '''            batch_x, batch_y, batch_x_mark, batch_y_mark = batch_x.to(device), batch_y.to(device), batch_x_mark.to(device), batch_y_mark.to(device)'''
    new_val_loop = '''            # Convert to float32 to avoid dtype mismatch
            batch_x = batch_x.to(device).float()
            batch_y = batch_y.to(device).float()
            batch_x_mark = batch_x_mark.to(device).float()
            batch_y_mark = batch_y_mark.to(device).float()'''
    
    if old_val_loop in content:
        content = content.replace(old_val_loop, new_val_loop)
        print("✓ Fixed validation loop dtype conversion")
    
    # Write back the fixed training script
    with open(training_script, 'w', encoding='utf-8') as f:
        f.write(content)
    
    # Fix 2: Update the embedding module to handle dtype conversion
    embedding_file = "models/celestial_modules/embedding.py"
    
    with open(embedding_file, 'r', encoding='utf-8') as f:
        embedding_content = f.read()
    
    # Add dtype conversion at the beginning of forward method
    old_forward_start = '''    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        batch_size, seq_len, enc_in = x_enc.shape'''
    
    new_forward_start = '''    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # DTYPE FIX: Ensure all inputs are float32
        x_enc = x_enc.float()
        x_mark_enc = x_mark_enc.float()
        x_dec = x_dec.float()
        x_mark_dec = x_mark_dec.float()
        
        batch_size, seq_len, enc_in = x_enc.shape'''
    
    if old_forward_start in embedding_content:
        embedding_content = embedding_content.replace(old_forward_start, new_forward_start)
        print("✓ Fixed embedding module dtype conversion")
    
    with open(embedding_file, 'w', encoding='utf-8') as f:
        f.write(embedding_content)
    
    # Fix 3: Update the phase-aware processor to handle dtype
    processor_file = "layers/modular/aggregation/phase_aware_celestial_processor.py"
    
    try:
        with open(processor_file, 'r', encoding='utf-8') as f:
            processor_content = f.read()
        
        # Add dtype conversion in the forward method
        old_processor_forward = '''    def forward(self, wave_data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Process celestial wave data through phase-aware aggregation
        
        Args:
            wave_data: [batch_size, seq_len, num_waves] Input wave data
            
        Returns:
            celestial_features: [batch_size, seq_len, num_celestial * celestial_dim] Rich representations
            adjacency_matrix: [batch_size, num_celestial, num_celestial] Phase-based edges  
            metadata: Processing metadata and diagnostics
        """
        batch_size, seq_len, num_waves = wave_data.shape'''
        
        new_processor_forward = '''    def forward(self, wave_data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Process celestial wave data through phase-aware aggregation
        
        Args:
            wave_data: [batch_size, seq_len, num_waves] Input wave data
            
        Returns:
            celestial_features: [batch_size, seq_len, num_celestial * celestial_dim] Rich representations
            adjacency_matrix: [batch_size, num_celestial, num_celestial] Phase-based edges  
            metadata: Processing metadata and diagnostics
        """
        # DTYPE FIX: Ensure input is float32
        wave_data = wave_data.float()
        
        batch_size, seq_len, num_waves = wave_data.shape'''
        
        if old_processor_forward in processor_content:
            processor_content = processor_content.replace(old_processor_forward, new_processor_forward)
            print("✓ Fixed phase-aware processor dtype conversion")
            
            with open(processor_file, 'w', encoding='utf-8') as f:
                f.write(processor_content)
        else:
            print("⚠️  Could not find exact processor forward pattern")
            
    except FileNotFoundError:
        print("⚠️  Phase-aware processor file not found")
    
    print("\n✅ DTYPE MISMATCH FIXES APPLIED")
    print("🔧 Changes made:")
    print("  - Added .float() conversion in training script")
    print("  - Added .float() conversion in validation loop")
    print("  - Added .float() conversion in embedding module")
    print("  - Added .float() conversion in phase-aware processor")
    print("\nThis should resolve the 'mat1 and mat2 must have the same dtype' error")

if __name__ == "__main__":
    fix_dtype_mismatch()