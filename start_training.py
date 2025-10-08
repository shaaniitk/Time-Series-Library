#!/usr/bin/env python3

import sys
import os
sys.path.append('.')

from scripts.train.train_celestial_direct import train_celestial_pgat

if __name__ == "__main__":
    print("🚀 Starting Enhanced Celestial PGAT Training...")
    success = train_celestial_pgat()
    if success:
        print("✅ Training completed successfully!")
    else:
        print("❌ Training failed!")