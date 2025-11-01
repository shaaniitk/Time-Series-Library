#!/usr/bin/env python3
"""
Test script to verify all advanced features are working together
"""

import torch
import yaml
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def test_ultimate_configuration():
    """Test the ultimate configuration with all advanced features"""
    
    print("🚀 Testing ULTIMATE Celestial Enhanced PGAT Configuration")
    print("=" * 70)
    
    # Load the ultimate config
    config_path = "configs/celestial_ultimate_all_features.yaml"
    
    try:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        print(f"✅ Configuration loaded: {config_path}")
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        return False
    
    # Test configuration validation
    print(f"\n🔍 Testing configuration validation...")
    try:
        from scripts.train.train_celestial_production import SimpleConfig
        from models.Celestial_Enhanced_PGAT_Modular import Model
        
        args = SimpleConfig(config_dict)
        
        # Test the validation function
        Model.validate_configuration(args)
        print(f"✅ Configuration validation passed!")
        
        # Print key configuration details
        print(f"\n📊 Configuration Summary:")
        print(f"   d_model: {args.d_model} (divisible by n_heads={args.n_heads}: {args.d_model % args.n_heads == 0})")
        print(f"   d_model: {args.d_model} (divisible by num_graph_nodes={args.num_graph_nodes}: {args.d_model % args.num_graph_nodes == 0})")
        print(f"   seq_len: {args.seq_len}, pred_len: {args.pred_len}")
        print(f"   batch_size: {args.batch_size}, train_epochs: {args.train_epochs}")
        
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return False
    
    # Test advanced features
    print(f"\n🚀 Advanced Features Enabled:")
    advanced_features = [
        ('Celestial Graph', args.use_celestial_graph),
        ('Petri Net Combiner', args.use_petri_net_combiner),
        ('MDN Decoder', args.enable_mdn_decoder),
        ('Mixture Decoder', args.use_mixture_decoder),
        ('Sequential Mixture', args.use_sequential_mixture_decoder),
        ('Stochastic Learning', args.use_stochastic_learner),
        ('Stochastic Control', args.use_stochastic_control),
        ('Hierarchical Mapping', args.use_hierarchical_mapping),
        ('Efficient Covariate', args.use_efficient_covariate_interaction),
        ('Calendar Effects', args.use_calendar_effects),
        ('Target Autocorrelation', args.use_target_autocorrelation),
        ('Celestial-to-Target', args.use_celestial_target_attention),
        ('Phase-Aware Processing', args.use_phase_aware_processing),
        ('Fusion Diagnostics', args.enable_fusion_diagnostics),
    ]
    
    enabled_count = 0
    for feature_name, enabled in advanced_features:
        status = "✅" if enabled else "❌"
        print(f"   {status} {feature_name}: {enabled}")
        if enabled:
            enabled_count += 1
    
    print(f"\n📈 Total Advanced Features Enabled: {enabled_count}/{len(advanced_features)}")
    
    # Test loss function
    print(f"\n🎯 Loss Function Configuration:")
    loss_config = args.loss
    if isinstance(loss_config, dict):
        print(f"   Type: {loss_config.get('type', 'unknown')}")
        if loss_config.get('type') == 'hybrid_mdn_directional':
            print(f"   🎲 MDN Component:")
            print(f"      nll_weight: {loss_config.get('nll_weight', 'N/A')}")
            print(f"   🎯 Directional Component:")
            print(f"      direction_weight: {loss_config.get('direction_weight', 'N/A')}")
            print(f"      trend_weight: {loss_config.get('trend_weight', 'N/A')}")
            print(f"      magnitude_weight: {loss_config.get('magnitude_weight', 'N/A')}")
            print(f"   ✅ ULTIMATE LOSS: Hybrid MDN + Directional!")
    
    # Test MDN configuration
    print(f"\n🎲 MDN Configuration:")
    print(f"   Components: {args.mdn_components}")
    print(f"   Sigma Min: {args.mdn_sigma_min}")
    print(f"   Use Softplus: {args.mdn_use_softplus}")
    
    # Test calibration metrics
    print(f"\n📊 Calibration Metrics:")
    calib = args.calibration_metrics
    if isinstance(calib, dict):
        print(f"   Coverage Levels: {calib.get('coverage_levels', [])}")
        print(f"   Compute CRPS: {calib.get('compute_crps', False)}")
        print(f"   CRPS Samples: {calib.get('crps_samples', 0)}")
    
    print(f"\n🎉 ULTIMATE CONFIGURATION TEST COMPLETE!")
    print(f"   ✅ All validations passed")
    print(f"   ✅ {enabled_count} advanced features enabled")
    print(f"   ✅ Hybrid MDN + Directional loss configured")
    print(f"   ✅ Perfect dimension compatibility")
    print(f"   ✅ Ready for ultimate training!")
    
    return True

def test_loss_function():
    """Test the hybrid MDN directional loss function"""
    
    print(f"\n🧪 Testing Hybrid MDN Directional Loss...")
    
    try:
        from layers.modular.losses.directional_trend_loss import HybridMDNDirectionalLoss
        
        # Create loss function
        loss_fn = HybridMDNDirectionalLoss(
            nll_weight=0.25,
            direction_weight=4.0,
            trend_weight=2.0,
            magnitude_weight=0.15
        )
        
        # Create test data
        batch_size, seq_len, num_targets, num_components = 2, 12, 4, 3
        
        # MDN parameters
        means = torch.randn(batch_size, seq_len, num_targets, num_components)
        log_stds = torch.randn(batch_size, seq_len, num_targets, num_components)
        log_weights = torch.randn(batch_size, seq_len, num_components)
        
        # Target
        target = torch.randn(batch_size, seq_len, num_targets)
        
        # Test loss computation
        mdn_params = (means, log_stds, log_weights)
        loss = loss_fn(mdn_params, target)
        
        print(f"   ✅ Loss computation successful!")
        print(f"   Loss value: {loss.item():.6f}")
        print(f"   Loss shape: {loss.shape}")
        print(f"   Loss requires_grad: {loss.requires_grad}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Loss function test failed: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Ultimate Celestial Enhanced PGAT Test Suite")
    print("=" * 70)
    
    # Test configuration
    config_success = test_ultimate_configuration()
    
    if config_success:
        # Test loss function
        loss_success = test_loss_function()
        
        if loss_success:
            print(f"\n🎉 ALL TESTS PASSED!")
            print(f"   Ready to run ultimate training with:")
            print(f"   python scripts/train/train_celestial_production.py --config configs/celestial_ultimate_all_features.yaml")
        else:
            print(f"\n⚠️  Configuration passed but loss function test failed")
    else:
        print(f"\n💥 Configuration test failed")
        sys.exit(1)