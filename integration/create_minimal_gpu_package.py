#!/usr/bin/env python3
"""
Quick GPU Package Creator - Minimal Version

Creates a lightweight ZIP with only essential TimesNet files for GPU deployment.
"""

import os
import zipfile
from datetime import datetime

def create_minimal_gpu_package():
    """Create minimal GPU package with only essential files"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    package_name = f"timesnet_minimal_{timestamp}.zip"
    
    # Essential items only
    essential_items = [
        # Core files
        'requirements.txt',
        'scripts/train/train_financial_timesnet.py',
        
        # Notebooks
        'TimesNet_Light_Config.ipynb',
        'TimesNet_Medium_Config.ipynb', 
        'TimesNet_Mid_Heavy_Config.ipynb',
        
        # Essential directories (entire folders)
        'models/',
        'layers/',
        'exp/',
        'utils/',
        'data_provider/',
        
        # Data (if exists)
        'data/',
    ]
    
    print(f"📦 Creating minimal package: {package_name}")
    
    with zipfile.ZipFile(package_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for item in essential_items:
            if item.endswith('/'):
                # Directory
                folder = item.rstrip('/')
                if os.path.exists(folder):
                    for root, dirs, files in os.walk(folder):
                        # Skip cache and unnecessary dirs
                        dirs[:] = [d for d in dirs if not d.startswith('__pycache__')]
                        for file in files:
                            if not file.endswith('.pyc'):
                                file_path = os.path.join(root, file)
                                zipf.write(file_path, file_path)
                    print(f"   ✅ {folder}/")
            else:
                # File
                if os.path.exists(item):
                    zipf.write(item, item)
                    print(f"   ✅ {item}")
    
    size_mb = os.path.getsize(package_name) / 1024 / 1024
    print(f"🎉 Package created: {package_name} ({size_mb:.1f}MB)")
    
    return package_name

if __name__ == "__main__":
    create_minimal_gpu_package()
