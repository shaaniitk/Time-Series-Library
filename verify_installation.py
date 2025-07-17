#!/usr/bin/env python3
"""
Installation Verification Script for Time Series Library with ChronosX

This script verifies that all required dependencies are properly installed
and working correctly.
"""

import sys
import importlib
import subprocess
from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')

class Color:
    """ANSI color codes for terminal output"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    print(f"{Color.BLUE}Python Version:{Color.END} {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 8:
        print(f"{Color.GREEN}✅ Python version is compatible{Color.END}")
        return True
    else:
        print(f"{Color.RED}❌ Python 3.8+ required, found {version.major}.{version.minor}{Color.END}")
        return False

def check_package(package_name: str, import_name: str = None, min_version: str = None) -> bool:
    """Check if a package is installed and optionally check version"""
    if import_name is None:
        import_name = package_name
    
    try:
        module = importlib.import_module(import_name)
        
        # Check version if specified
        version_check = True
        version_str = "unknown"
        
        if min_version and hasattr(module, '__version__'):
            version_str = module.__version__
            # Simple version comparison (works for most cases)
            try:
                from packaging import version
                version_check = version.parse(version_str) >= version.parse(min_version)
            except:
                # Fallback to string comparison if packaging not available
                version_check = True
        
        if version_check:
            print(f"{Color.GREEN}✅ {package_name}{Color.END} ({version_str})")
            return True
        else:
            print(f"{Color.YELLOW}⚠️ {package_name}{Color.END} ({version_str}) - version {min_version}+ recommended")
            return True
            
    except ImportError:
        print(f"{Color.RED}❌ {package_name} not found{Color.END}")
        return False
    except Exception as e:
        print(f"{Color.YELLOW}⚠️ {package_name} - error: {e}{Color.END}")
        return False

def check_core_dependencies() -> Tuple[int, int]:
    """Check core machine learning dependencies"""
    print(f"\n{Color.BOLD}Core Dependencies:{Color.END}")
    
    core_packages = [
        ("torch", "torch", "2.1.0"),
        ("numpy", "numpy", "1.20.0"),
        ("pandas", "pandas", "1.3.0"),
        ("scikit-learn", "sklearn"),
        ("matplotlib", "matplotlib"),
        ("scipy", "scipy"),
    ]
    
    passed = 0
    total = len(core_packages)
    
    for package_info in core_packages:
        if check_package(*package_info):
            passed += 1
    
    return passed, total

def check_time_series_dependencies() -> Tuple[int, int]:
    """Check time series specific dependencies"""
    print(f"\n{Color.BOLD}Time Series Dependencies:{Color.END}")
    
    ts_packages = [
        ("einops", "einops"),
        ("reformer-pytorch", "reformer_pytorch"),
        ("local-attention", "local_attention"),
        ("PyWavelets", "pywt"),
        ("statsmodels", "statsmodels"),
        ("sympy", "sympy"),
    ]
    
    passed = 0
    total = len(ts_packages)
    
    for package_info in ts_packages:
        if check_package(*package_info):
            passed += 1
    
    return passed, total

def check_chronosx_dependencies() -> Tuple[int, int]:
    """Check ChronosX specific dependencies"""
    print(f"\n{Color.BOLD}ChronosX Dependencies:{Color.END}")
    
    chronos_packages = [
        ("chronos-forecasting", "chronos", "1.5.2"),
        ("transformers", "transformers", "4.30.0"),
        ("accelerate", "accelerate"),
        ("datasets", "datasets"),
        ("tokenizers", "tokenizers"),
    ]
    
    passed = 0
    total = len(chronos_packages)
    
    for package_info in chronos_packages:
        if check_package(*package_info):
            passed += 1
    
    return passed, total

def check_visualization_dependencies() -> Tuple[int, int]:
    """Check visualization and analysis dependencies"""
    print(f"\n{Color.BOLD}Visualization & Analysis:{Color.END}")
    
    viz_packages = [
        ("seaborn", "seaborn"),
        ("plotly", "plotly"),
        ("psutil", "psutil"),
    ]
    
    passed = 0
    total = len(viz_packages)
    
    for package_info in viz_packages:
        if check_package(*package_info):
            passed += 1
    
    return passed, total

def check_development_dependencies() -> Tuple[int, int]:
    """Check development dependencies (optional)"""
    print(f"\n{Color.BOLD}Development Tools (Optional):{Color.END}")
    
    dev_packages = [
        ("pytest", "pytest"),
        ("black", "black"),
        ("jupyter", "jupyter"),
        ("hydra-core", "hydra"),
    ]
    
    passed = 0
    total = len(dev_packages)
    
    for package_info in dev_packages:
        if check_package(*package_info):
            passed += 1
    
    return passed, total

def test_chronosx_functionality():
    """Test basic ChronosX functionality"""
    print(f"\n{Color.BOLD}ChronosX Functionality Test:{Color.END}")
    
    try:
        from chronos import ChronosPipeline
        import torch
        
        # Test model loading (tiny model for speed)
        print("Loading ChronosX tiny model...")
        pipeline = ChronosPipeline.from_pretrained(
            "amazon/chronos-t5-tiny",
            device_map="cpu",
            torch_dtype=torch.float32,
        )
        
        # Test prediction
        print("Testing prediction...")
        context = torch.randn(1, 50, dtype=torch.float32)
        forecast = pipeline.predict(
            context=context,
            prediction_length=12,
            num_samples=5
        )
        
        print(f"{Color.GREEN}✅ ChronosX functionality test passed{Color.END}")
        print(f"   Model loaded successfully")
        print(f"   Forecast shape: {forecast.shape}")
        return True
        
    except ImportError:
        print(f"{Color.YELLOW}⚠️ ChronosX not installed - skipping functionality test{Color.END}")
        return False
    except Exception as e:
        print(f"{Color.RED}❌ ChronosX functionality test failed: {e}{Color.END}")
        return False

def test_modular_components():
    """Test modular component system"""
    print(f"\n{Color.BOLD}Modular Components Test:{Color.END}")
    
    try:
        from utils.modular_components.registry import create_global_registry
        from utils.modular_components.example_components import register_example_components
        
        # Test registry creation
        registry = create_global_registry()
        print(f"{Color.GREEN}✅ Component registry created{Color.END}")
        
        # Test component registration
        register_example_components(registry)
        print(f"{Color.GREEN}✅ Example components registered{Color.END}")
        
        return True
        
    except ImportError as e:
        print(f"{Color.YELLOW}⚠️ Modular components not found: {e}{Color.END}")
        return False
    except Exception as e:
        print(f"{Color.RED}❌ Modular components test failed: {e}{Color.END}")
        return False

def check_gpu_availability():
    """Check GPU availability and CUDA"""
    print(f"\n{Color.BOLD}GPU & CUDA Check:{Color.END}")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
            cuda_version = torch.version.cuda
            
            print(f"{Color.GREEN}✅ CUDA available{Color.END}")
            print(f"   CUDA version: {cuda_version}")
            print(f"   GPU count: {gpu_count}")
            print(f"   GPU 0: {gpu_name}")
            return True
        else:
            print(f"{Color.YELLOW}⚠️ CUDA not available - using CPU{Color.END}")
            return False
            
    except Exception as e:
        print(f"{Color.RED}❌ GPU check failed: {e}{Color.END}")
        return False

def generate_installation_report(results: dict):
    """Generate installation report"""
    print(f"\n{Color.BOLD}{'='*60}{Color.END}")
    print(f"{Color.BOLD}INSTALLATION VERIFICATION REPORT{Color.END}")
    print(f"{Color.BOLD}{'='*60}{Color.END}")
    
    total_passed = 0
    total_packages = 0
    
    for category, (passed, total) in results.items():
        total_passed += passed
        total_packages += total
        percentage = (passed / total * 100) if total > 0 else 0
        
        if percentage == 100:
            status = f"{Color.GREEN}✅ Complete{Color.END}"
        elif percentage >= 80:
            status = f"{Color.YELLOW}⚠️ Mostly Complete{Color.END}"
        else:
            status = f"{Color.RED}❌ Incomplete{Color.END}"
        
        print(f"{category}: {passed}/{total} ({percentage:.0f}%) {status}")
    
    overall_percentage = (total_passed / total_packages * 100) if total_packages > 0 else 0
    
    print(f"\n{Color.BOLD}Overall Status:{Color.END} {total_passed}/{total_packages} ({overall_percentage:.0f}%)")
    
    if overall_percentage >= 90:
        print(f"{Color.GREEN}🎉 Excellent! Your installation is ready for production use.{Color.END}")
    elif overall_percentage >= 75:
        print(f"{Color.YELLOW}✅ Good! Most features are available. Some optional components missing.{Color.END}")
    elif overall_percentage >= 50:
        print(f"{Color.YELLOW}⚠️ Partial installation. Core features available but missing important components.{Color.END}")
    else:
        print(f"{Color.RED}❌ Incomplete installation. Please install missing dependencies.{Color.END}")
    
    # Recommendations
    print(f"\n{Color.BOLD}Recommendations:{Color.END}")
    
    if results.get("ChronosX", (0, 1))[0] == 0:
        print(f"📦 Install ChronosX: pip install chronos-forecasting transformers accelerate")
    
    if results.get("Visualization", (0, 1))[0] < results.get("Visualization", (0, 1))[1]:
        print(f"📊 Install visualization: pip install seaborn plotly psutil")
    
    if results.get("Development", (0, 1))[0] < results.get("Development", (0, 1))[1] // 2:
        print(f"🛠️ Install development tools: pip install -r requirements_dev.txt")
    
    print(f"\n📚 For detailed installation guide: see INSTALLATION_GUIDE.md")

def main():
    """Main verification function"""
    print(f"{Color.BOLD}Time Series Library Installation Verification{Color.END}")
    print("="*50)
    
    # Check Python version
    if not check_python_version():
        print(f"\n{Color.RED}❌ Python version incompatible. Please upgrade to Python 3.8+{Color.END}")
        sys.exit(1)
    
    # Check dependencies
    results = {}
    
    results["Core"] = check_core_dependencies()
    results["Time Series"] = check_time_series_dependencies()
    results["ChronosX"] = check_chronosx_dependencies()
    results["Visualization"] = check_visualization_dependencies()
    results["Development"] = check_development_dependencies()
    
    # Test functionality
    chronosx_works = test_chronosx_functionality()
    modular_works = test_modular_components()
    gpu_available = check_gpu_availability()
    
    # Generate report
    generate_installation_report(results)
    
    # Final recommendations
    print(f"\n{Color.BOLD}Quick Start:{Color.END}")
    if chronosx_works:
        print(f"🚀 Try: python chronos_x_simple_demo.py")
    else:
        print(f"🚀 Try: python scripts/TimesNet_ETTh1.sh")
    
    if modular_works:
        print(f"🔧 Modular components: Available")
    
    if gpu_available:
        print(f"⚡ GPU acceleration: Enabled")
    else:
        print(f"💻 Using CPU mode")

if __name__ == "__main__":
    main()
