#!/usr/bin/env python3
'''
Quick OOM Debug Wrapper
Wraps existing training with memory debugging
'''

import sys
import torch
from pathlib import Path

# Import original training function
from scripts.train import train_celestial_production

# Patch torch.Tensor.backward to add debugging
original_backward = torch.Tensor.backward

def debug_backward(self, gradient=None, retain_graph=None, create_graph=False, inputs=None):
    '''Wrapped backward with debugging.'''
    device = self.device
    
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        pre = torch.cuda.memory_allocated(device) / (1024**2)
        
        # Call original
        original_backward(self, gradient, retain_graph, create_graph, inputs)
        
        torch.cuda.synchronize(device)
        post = torch.cuda.memory_allocated(device) / (1024**2)
        growth = post - pre
        
        if growth > 100:  # Alert on >100MB growth
            print(f"  [BACKWARD DEBUG] Memory growth: {growth:.1f}MB (Pre: {pre:.1f}MB, Post: {post:.1f}MB)")
    else:
        original_backward(self, gradient, retain_graph, create_graph, inputs)

# Apply patch
torch.Tensor.backward = debug_backward

# Run original training
if __name__ == "__main__":
    train_celestial_production.train_celestial_pgat_production()
