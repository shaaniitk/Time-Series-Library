#!/usr/bin/env python3
"""
Enable debug logging for detailed model analysis.
Run this before your training script to see detailed debug output.
"""

import logging
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.logger import logger

def enable_debug():
    """Enable DEBUG level logging."""
    logger.setLevel(logging.DEBUG)
    
    # Create console handler with debug level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    
    # Create detailed formatter for debug
    debug_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    console_handler.setFormatter(debug_formatter)
    
    # Remove existing handlers and add debug handler
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    logger.addHandler(console_handler)
    
    print("DEBUG logging enabled - you will see detailed model execution logs")
    print("This includes input/output shapes, fusion details, and component progress")
    print("=" * 70)

if __name__ == "__main__":
    enable_debug()
