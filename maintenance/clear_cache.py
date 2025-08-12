#!/usr/bin/env python3
"""
Cache clearing utility script for Time Series Library project.
Run this script to clear all types of cache.
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from utils.tools import (
    clear_all_cache, 
    clear_pytorch_cache, 
    clear_python_cache,
    clear_model_cache,
    clear_vscode_cache,
    clear_workspace_cache
)
from utils.logger import logger

def main():
    print("="*60)
    print("TIME SERIES LIBRARY - CACHE CLEARING UTILITY")
    print("="*60)
    
    try:
        # Clear all programming caches
        print("\n1. Clearing all programming caches...")
        clear_all_cache()
        
        # Clear workspace cache
        print("\n2. Clearing workspace cache...")
        clear_workspace_cache(project_root)
        
        # Show VS Code cache clearing instructions
        print("\n3. VS Code cache clearing instructions...")
        clear_vscode_cache()
        
        print("\n" + "="*60)
        print("CACHE CLEARING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nNext steps:")
        print("1. Close VS Code completely")
        print("2. Follow the VS Code cache clearing instructions above")
        print("3. Restart VS Code")
        print("4. Reopen your workspace")
        
    except Exception as e:
        logger.error(f"Error during cache clearing: {e}")
        print(f"\nERROR: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
