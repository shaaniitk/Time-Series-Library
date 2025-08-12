#!/usr/bin/env python3
"""
Enhanced training script with comprehensive debug logging and progress tracking.
"""

import sys
import os
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def setup_debug_logging():
    """Setup comprehensive debug logging."""
    from utils.logger import logger
    
    # Set to DEBUG level
    logger.setLevel(logging.DEBUG)
    
    # Create detailed formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    logger.addHandler(console_handler)
    
    print("🔍 DEBUG LOGGING ENABLED")
    print("=" * 60)
    print("You will now see detailed logs including:")
    print("- Input/output tensor shapes at each stage")
    print("- Hierarchical fusion processing details")
    print("- Encoder/decoder component analysis")
    print("- Progress tracking every 10 iterations")
    print("- Component-level debugging information")
    print("=" * 60)

def main():
    """Main training function with debug support."""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Config file path')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--progress', action='store_true', default=True, help='Show detailed progress')
    
    args = parser.parse_args()
    
    if args.debug:
        setup_debug_logging()
    
    # Import training script
    from scripts.train.train_dynamic_autoformer import main as train_main
    
    # Set config in sys.argv for the training script
    sys.argv = ['train_dynamic_autoformer.py', '--config', args.config]
    
    print(f"🚀 Starting training with config: {args.config}")
    print(f"📊 Debug mode: {'ON' if args.debug else 'OFF'}")
    print(f"📈 Progress tracking: {'ON' if args.progress else 'OFF'}")
    print("=" * 60)
    
    # Run training
    train_main()

if __name__ == "__main__":
    main()
