#!/usr/bin/env python3
"""
Monitor file changes to detect what's modifying the layer files
"""

import os
import time
import hashlib
import json
from datetime import datetime

def get_file_hash(filepath):
    """Get MD5 hash of file content"""
    try:
        with open(filepath, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except:
        return None

def get_file_size(filepath):
    """Get file size"""
    try:
        return os.path.getsize(filepath)
    except:
        return None

def get_file_lines(filepath):
    """Get number of lines in file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return len(f.readlines())
    except:
        return None

def monitor_files():
    """Monitor specific files for changes"""
    
    # Files to monitor
    files_to_monitor = [
        'layers/Embed.py',
        'layers/Attention.py', 
        'layers/AutoCorrelation.py',
        'layers/Autoformer_EncDec.py',
        'layers/Normalization.py'
    ]
    
    print("Starting file monitoring...")
    print("Monitoring files:")
    for f in files_to_monitor:
        print(f"  - {f}")
    
    # Initial state
    file_states = {}
    for filepath in files_to_monitor:
        if os.path.exists(filepath):
            file_states[filepath] = {
                'hash': get_file_hash(filepath),
                'size': get_file_size(filepath),
                'lines': get_file_lines(filepath),
                'last_modified': os.path.getmtime(filepath),
                'last_check': datetime.now().isoformat()
            }
            print(f"Initial state - {filepath}: {file_states[filepath]['lines']} lines, {file_states[filepath]['size']} bytes")
    
    print(f"\nMonitoring started at {datetime.now()}")
    print("Press Ctrl+C to stop monitoring")
    print("-" * 60)
    
    try:
        while True:
            time.sleep(2)  # Check every 2 seconds
            
            changes_detected = False
            
            for filepath in files_to_monitor:
                if not os.path.exists(filepath):
                    continue
                    
                current_hash = get_file_hash(filepath)
                current_size = get_file_size(filepath)
                current_lines = get_file_lines(filepath)
                current_mtime = os.path.getmtime(filepath)
                
                if filepath in file_states:
                    old_state = file_states[filepath]
                    
                    # Check for changes
                    if (current_hash != old_state['hash'] or 
                        current_size != old_state['size'] or
                        current_lines != old_state['lines']):
                        
                        changes_detected = True
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        
                        print(f"\n[{timestamp}] CHANGE DETECTED in {filepath}:")
                        print(f"  Lines: {old_state['lines']} -> {current_lines} (diff: {current_lines - old_state['lines']})")
                        print(f"  Size:  {old_state['size']} -> {current_size} bytes (diff: {current_size - old_state['size']})")
                        print(f"  Hash:  {old_state['hash'][:8]}... -> {current_hash[:8]}...")
                        
                        # Update state
                        file_states[filepath] = {
                            'hash': current_hash,
                            'size': current_size,
                            'lines': current_lines,
                            'last_modified': current_mtime,
                            'last_check': datetime.now().isoformat()
                        }
                        
                        # Show first few lines to see what changed
                        try:
                            with open(filepath, 'r', encoding='utf-8') as f:
                                first_lines = [f.readline().strip() for _ in range(3)]
                            print(f"  First 3 lines: {first_lines}")
                        except:
                            pass
            
            if changes_detected:
                print("-" * 60)
                
    except KeyboardInterrupt:
        print(f"\nMonitoring stopped at {datetime.now()}")
        
        # Final report
        print("\nFinal file states:")
        for filepath, state in file_states.items():
            print(f"  {filepath}: {state['lines']} lines, {state['size']} bytes")

if __name__ == "__main__":
    monitor_files()
