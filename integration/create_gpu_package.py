#!/usr/bin/env python3
"""
GPU Deployment Package Creator

Creates a minimal ZIP package containing only the essential files needed
for GPU computation and TimesNet training, excluding unnecessary files
like virtual environments, git history, and temporary files.

Usage:
    python create_gpu_package.py
    
Output:
    timesnet_gpu_package_YYYYMMDD_HHMM.zip
"""

import os
import zipfile
import shutil
import json
from datetime import datetime
from pathlib import Path


def convert_notebook_to_python(notebook_path, output_path):
    """Convert Jupyter notebook to Python script with main function"""
    
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
    except Exception as e:
        print(f"   ❌ Error reading {notebook_path}: {e}")
        return False
    
    # Extract notebook name for configuration
    notebook_name = Path(notebook_path).stem
    config_type = notebook_name.replace('TimesNet_', '').replace('_Config', '').lower()
    
    python_code = f'''#!/usr/bin/env python3
"""
{notebook_name} - Converted from Jupyter Notebook
Generated automatically for GPU deployment

Run with: python {Path(output_path).name}
"""

import os
import sys
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.TimesNet import Model as TimesNet
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric
from utils.logger import logger
from data_provider.data_loader import Dataset_Custom
from torch.utils.data import DataLoader


'''
    
    # Extract Python code cells
    code_cells = []
    for cell in notebook.get('cells', []):
        if cell.get('cell_type') == 'python':
            source = cell.get('source', [])
            if isinstance(source, list):
                cell_code = ''.join(source)
            else:
                cell_code = source
            
            # Skip import cells (already included above)
            if any(skip_pattern in cell_code for skip_pattern in [
                'import os', 'import sys', 'import time', 'import torch',
                'from models.TimesNet', 'from utils.tools', 'from data_provider'
            ]):
                continue
                
            # Skip cells that just print configuration info
            if cell_code.strip().startswith('print("✅ All imports successful")'):
                continue
                
            code_cells.append(cell_code)
    
    # Create main function
    python_code += f'''
def main():
    """Main training function for {config_type} configuration"""
    print("🚀 Starting {config_type.title()} TimesNet Training")
    print("=" * 60)
    
    # Enhanced GPU information
    if torch.cuda.is_available():
        print(f"🚀 GPU Name: {{torch.cuda.get_device_name(0)}}")
        print(f"💾 GPU Memory: {{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}}GB")
        print(f"⚡ CUDA Version: {{torch.version.cuda}}")
        print("🎯 GPU acceleration will be used automatically!")
    else:
        print("⚠️  No GPU detected - will use CPU (training will be slower)")
    print()

'''
    
    # Add all code cells with proper indentation
    for i, cell_code in enumerate(code_cells):
        # Add cell as part of main function with proper indentation
        indented_code = '\n'.join(['    ' + line if line.strip() else line 
                                 for line in cell_code.split('\n')])
        python_code += f"    # === Cell {i+1} ===\n"
        python_code += indented_code + '\n\n'
    
    # Add the function call
    python_code += '''
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\\n⏹️  Training interrupted by user")
    except Exception as e:
        print(f"\\n❌ Error during training: {e}")
        raise
'''
    
    # Write Python script
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(python_code)
        return True
    except Exception as e:
        print(f"   ❌ Error writing {output_path}: {e}")
        return False


def create_gpu_package():
    """Create a minimal package for GPU deployment"""
    
    # Get current directory (should be Time-Series-Library root)
    root_dir = Path.cwd()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    package_name = f"timesnet_gpu_package_{timestamp}.zip"
    
    print(f"🚀 Creating GPU deployment package: {package_name}")
    print(f"📁 Source directory: {root_dir}")
    
    # Define essential files and folders to include
    essential_items = {
        # Core Python files
        'files': [
            'run.py',
            'scripts/train/train_financial_timesnet.py',
            'example_data_preparation.py',
            'requirements.txt',
            'README.md',
            'LICENSE',
        ],
        
        # Essential directories (include entire folder)
        'directories': [
            'models',           # TimesNet and other model implementations
            'layers',           # Model layers and components
            'exp',              # Experiment framework
            'utils',            # Utilities and tools
            'data_provider',    # Data loading and processing
        ],
        
        # Jupyter notebooks
        'notebooks': [
            'TimesNet_Light_Config.ipynb',
            'TimesNet_Medium_Config.ipynb',
            'TimesNet_Mid_Heavy_Config.ipynb',
        ],
        
        # Data files (if they exist)
        'data_files': [
            'data/prepared_financial_data.csv',
            'data/nifty50_returns.csv',
            'data/nifty50_returns.parquet',
            'data/comprehensive_dynamic_features_nifty.csv',
            'data/india_static_features.csv',
        ]
    }
    
    # Files and directories to exclude
    exclude_patterns = {
        'directories': [
            'tsl-env',          # Virtual environment
            '.venv',            # Alternative venv name
            'venv',             # Alternative venv name
            '__pycache__',      # Python cache
            '.git',             # Git repository
            '.pytest_cache',    # Pytest cache
            'checkpoints',      # Model checkpoints (too large)
            'logs',             # Log files
            'temp',             # Temporary files
            'tmp',              # Temporary files
        ],
        'extensions': [
            '.pyc',             # Compiled Python
            '.pyo',             # Optimized Python
            '.log',             # Log files
            '.tmp',             # Temporary files
            '.DS_Store',        # macOS files
            'Thumbs.db',        # Windows files
        ]
    }
    
    def should_exclude(path):
        """Check if a path should be excluded"""
        path_str = str(path)
        
        # Check for excluded directories
        for exclude_dir in exclude_patterns['directories']:
            if f"/{exclude_dir}/" in path_str or path_str.endswith(f"/{exclude_dir}") or path_str.startswith(f"{exclude_dir}/"):
                return True
        
        # Check for excluded extensions
        for ext in exclude_patterns['extensions']:
            if path_str.endswith(ext):
                return True
                
        return False
    
    # Create the ZIP package
    with zipfile.ZipFile(package_name, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zipf:
        
        # Add essential files
        print("\n📄 Adding essential files:")
        for file_path in essential_items['files']:
            if os.path.exists(file_path):
                zipf.write(file_path, file_path)
                print(f"   ✅ {file_path}")
            else:
                print(f"   ⚠️  {file_path} (not found)")        
        # Convert Jupyter notebooks to Python scripts
        print("\n📓 Converting Jupyter notebooks to Python scripts:")
        python_scripts = []
        for notebook in essential_items['notebooks']:
            if os.path.exists(notebook):
                # Create Python script name
                script_name = notebook.replace('.ipynb', '.py')
                python_scripts.append(script_name)
                
                # Convert notebook to Python
                success = convert_notebook_to_python(notebook, script_name)
                if success:
                    # Add Python script to ZIP
                    zipf.write(script_name, script_name)
                    print(f"   ✅ {notebook} → {script_name}")
                    
                    # Also add original notebook for reference
                    zipf.write(notebook, f"notebooks_original/{notebook}")
                else:
                    print(f"   ❌ Failed to convert {notebook}")
            else:
                print(f"   ⚠️  {notebook} (not found)")
        
        # Create a runner script for all configurations
        runner_script = "run_timesnet_training.py"
        runner_content = f'''#!/usr/bin/env python3
"""
TimesNet Training Runner
Automatically generated script to run different TimesNet configurations

Usage:
    python run_timesnet_training.py [light|medium|mid_heavy]
    
If no argument provided, will prompt for selection.
"""

import sys
import os

def main():
    configurations = {{
        'light': 'TimesNet_Light_Config.py',
        'medium': 'TimesNet_Medium_Config.py', 
        'mid_heavy': 'TimesNet_Mid_Heavy_Config.py'
    }}
    
    # Check command line argument
    if len(sys.argv) > 1:
        config = sys.argv[1].lower()
    else:
        print("🎯 Available TimesNet Configurations:")
        print("   1. light     - Fast training, small model")
        print("   2. medium    - Balanced performance")
        print("   3. mid_heavy - High capacity, longer training")
        print()
        config = input("Select configuration (light/medium/mid_heavy): ").lower()
    
    if config not in configurations:
        print(f"❌ Invalid configuration: {{config}}")
        print(f"Available: {{', '.join(configurations.keys())}}")
        return
    
    script_name = configurations[config]
    
    if not os.path.exists(script_name):
        print(f"❌ Script not found: {{script_name}}")
        return
    
    print(f"🚀 Running {{config}} configuration...")
    print(f"📜 Executing: {{script_name}}")
    print("=" * 60)
    
    # Import and run the selected configuration
    try:
        if config == 'light':
            import TimesNet_Light_Config
            TimesNet_Light_Config.main()
        elif config == 'medium':
            import TimesNet_Medium_Config
            TimesNet_Medium_Config.main()
        elif config == 'mid_heavy':
            import TimesNet_Mid_Heavy_Config
            TimesNet_Mid_Heavy_Config.main()
    except Exception as e:
        print(f"❌ Error running configuration: {{e}}")
        raise

if __name__ == "__main__":
    main()
'''
        
        # Add runner script to ZIP
        zipf.writestr(runner_script, runner_content)
        print(f"   🎯 Created runner script: {runner_script}")
        
        # Add essential directories
        print("\n📂 Adding essential directories:")
        for dir_name in essential_items['directories']:
            if os.path.exists(dir_name):
                print(f"   📁 {dir_name}/")
                file_count = 0
                for root, dirs, files in os.walk(dir_name):
                    # Filter out excluded directories
                    dirs[:] = [d for d in dirs if not should_exclude(Path(root) / d)]
                    
                    for file in files:
                        file_path = Path(root) / file
                        if not should_exclude(file_path):
                            arcname = str(file_path).replace('\\', '/')
                            zipf.write(file_path, arcname)
                            file_count += 1
                print(f"      → {file_count} files added")
            else:
                print(f"   ⚠️  {dir_name}/ (not found)")
        
        # Add data files
        print("\n💾 Adding data files:")
        for data_file in essential_items['data_files']:
            if os.path.exists(data_file):
                # Create data directory in zip if it doesn't exist
                zipf.write(data_file, data_file)
                file_size = os.path.getsize(data_file) / 1024 / 1024  # MB
                print(f"   ✅ {data_file} ({file_size:.1f}MB)")
            else:
                print(f"   ⚠️  {data_file} (not found)")
        
        # Add any scripts directory if it exists
        if os.path.exists('scripts'):
            print("\n🔧 Adding training scripts:")
            script_count = 0
            for root, dirs, files in os.walk('scripts'):
                # Only include .sh and .py files from scripts
                for file in files:
                    if file.endswith(('.sh', '.py')):
                        file_path = Path(root) / file
                        arcname = str(file_path).replace('\\', '/')
                        zipf.write(file_path, arcname)
                        script_count += 1
            print(f"   📜 {script_count} script files added")
    
    # Get package information
    package_size = os.path.getsize(package_name) / 1024 / 1024  # MB
    
    # Clean up temporary Python files
    print("\n🧹 Cleaning up temporary files:")
    for notebook in essential_items['notebooks']:
        script_name = notebook.replace('.ipynb', '.py')
        if os.path.exists(script_name):
            os.remove(script_name)
            print(f"   🗑️  Removed {script_name}")
    
    print(f"\n🎉 GPU package created successfully!")
    print(f"📦 Package: {package_name}")
    print(f"📏 Size: {package_size:.1f}MB")
      # Create a deployment guide
    guide_name = f"deployment_guide_{timestamp}.txt"
    with open(guide_name, 'w', encoding='utf-8') as f:
        f.write("TimesNet GPU Deployment Guide\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Package: {package_name}\n")
        f.write(f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("📋 DEPLOYMENT STEPS:\n\n")
        f.write("1. Extract the ZIP package on your GPU machine:\n")
        f.write(f"   unzip {package_name}\n\n")
        
        f.write("2. Install dependencies:\n")
        f.write("   # For GPU (CUDA 11.8):\n")
        f.write("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118\n")
        f.write("   pip install -r requirements.txt\n\n")
        
        f.write("3. Verify GPU is available:\n")
        f.write("   python -c \"import torch; print(f'CUDA available: {torch.cuda.is_available()}')\"\n\n")
          f.write("4. Run TimesNet training:\n")
        f.write("   # Option 1: Use the automated runner (RECOMMENDED)\n")
        f.write("   python run_timesnet_training.py\n")
        f.write("   # Or specify configuration directly:\n")
        f.write("   python run_timesnet_training.py light\n")
        f.write("   python run_timesnet_training.py medium\n")
        f.write("   python run_timesnet_training.py mid_heavy\n\n")
        
        f.write("   # Option 2: Run individual Python scripts\n")
        f.write("   python TimesNet_Light_Config.py\n")
        f.write("   python TimesNet_Medium_Config.py\n")
        f.write("   python TimesNet_Mid_Heavy_Config.py\n\n")
        
        f.write("   # Option 3: Run original Jupyter notebooks\n")
        f.write("   jupyter notebook\n")
        f.write("   # Then open files in notebooks_original/ folder\n\n")
        
        f.write("   # Option 4: Run training script directly\n")
        f.write("   python scripts/train/train_financial_timesnet.py\n\n")
        
        f.write("📁 PACKAGE CONTENTS:\n")
        f.write("   ✅ Core models and layers\n")
        f.write("   ✅ Experiment framework\n")
        f.write("   ✅ Data providers and utilities\n")
        f.write("   ✅ Python scripts (converted from notebooks)\n")
        f.write("   ✅ Original Jupyter notebooks (in notebooks_original/)\n")
        f.write("   ✅ Automated runner script\n")
        f.write("   ✅ Training scripts\n")
        f.write("   ✅ Financial data (if available)\n")
        f.write("   ✅ Requirements and documentation\n\n")
        
        f.write("🚫 EXCLUDED (not needed for GPU training):\n")
        f.write("   ❌ Virtual environments (tsl-env, .venv)\n")
        f.write("   ❌ Git history and cache files\n")
        f.write("   ❌ Python cache files (__pycache__)\n")
        f.write("   ❌ Log files and checkpoints\n")
        f.write("   ❌ Temporary files\n\n")
        
        f.write("💡 TIPS:\n")
        f.write("   - Check GPU memory before running mid-heavy config\n")
        f.write("   - Start with light config to verify setup\n")
        f.write("   - Monitor GPU usage: nvidia-smi\n")
        f.write("   - Adjust batch_size if you get out-of-memory errors\n")
    
    print(f"📋 Deployment guide: {guide_name}")
    
    return package_name, guide_name


if __name__ == "__main__":
    print("🚀 TimesNet GPU Package Creator")
    print("=" * 50)
    
    try:
        package_name, guide_name = create_gpu_package()
        
        print("\n✅ Package creation completed!")
        print(f"📦 Files created:")
        print(f"   - {package_name}")
        print(f"   - {guide_name}")
        print("\n🚀 Ready for GPU deployment!")
        
    except Exception as e:
        print(f"\n❌ Error creating package: {e}")
        raise
