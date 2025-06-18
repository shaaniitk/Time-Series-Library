#!/usr/bin/env python3
"""
Monitor training progress by checking checkpoint directories and log outputs
"""
import os
import time
import glob
from datetime import datetime

def monitor_training():
    checkpoint_dir = "./checkpoints/"
    
    print("🔍 Training Monitor Started")
    print("=" * 50)
    
    last_checkpoint_count = 0
    
    while True:
        # Check for new checkpoints
        if os.path.exists(checkpoint_dir):
            checkpoints = glob.glob(os.path.join(checkpoint_dir, "**/*.pth"), recursive=True)
            
            if len(checkpoints) > last_checkpoint_count:
                print(f"✅ New checkpoint detected! Total: {len(checkpoints)}")
                for cp in checkpoints[-3:]:  # Show last 3
                    stat = os.stat(cp)
                    mod_time = datetime.fromtimestamp(stat.st_mtime)
                    print(f"   📁 {cp} - Modified: {mod_time}")
                last_checkpoint_count = len(checkpoints)
        
        # Check for KL plots
        kl_plots = glob.glob("./kl_*.png")
        if kl_plots:
            print(f"📊 KL plots available: {len(kl_plots)}")
            for plot in kl_plots[-2:]:  # Show last 2
                stat = os.stat(plot)
                mod_time = datetime.fromtimestamp(stat.st_mtime)
                print(f"   📈 {plot} - Modified: {mod_time}")
        
        print(f"🕐 {datetime.now().strftime('%H:%M:%S')} - Monitoring...")
        time.sleep(30)  # Check every 30 seconds

if __name__ == "__main__":
    try:
        monitor_training()
    except KeyboardInterrupt:
        print("\n⏹️  Monitor stopped by user")
