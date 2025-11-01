#!/usr/bin/env python3
"""Test the config validation for the ultimate deep config."""

import sys
import yaml
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from models.Celestial_Enhanced_PGAT_Modular import Model

class SimpleConfig:
    """Dictionary-backed configuration with attribute access support."""
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)

def test_config_validation():
    """Test the ultimate deep config validation."""
    
    # Load the FIXED config
    config_path = "configs/celestial_production_deep_ultimate_fixed.yaml"
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    config = SimpleConfig(config_dict)
    
    print("🔍 TESTING CONFIG VALIDATION")
    print("=" * 50)
    print(f"Config: {config_path}")
    print(f"d_model: {config.d_model}")
    print(f"n_heads: {config.n_heads}")
    print(f"celestial_dim: {config.celestial_dim}")
    print(f"calendar_embedding_dim: {config.calendar_embedding_dim}")
    print(f"use_efficient_covariate_interaction: {config.use_efficient_covariate_interaction}")
    print(f"num_celestial_bodies: {config.num_celestial_bodies}")
    print()
    
    try:
        Model.validate_configuration(config)
        print("✅ Configuration validation PASSED!")
        return True
    except ValueError as e:
        print("❌ Configuration validation FAILED:")
        print(str(e))
        return False

if __name__ == "__main__":
    test_config_validation()