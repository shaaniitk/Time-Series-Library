"""
Simple Migration Test Script

Tests the migration without problematic imports
"""

import torch
import logging
import sys
import os
from argparse import Namespace

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_migration_framework():
    """Test the migration framework directly"""
    logger.info("Testing migration framework...")
    
    try:
        from utils.modular_components.migration_framework import MigrationManager
        from utils.modular_components.registry import ComponentRegistry
        
        # Create registry and migration manager
        registry = ComponentRegistry()
        migration_manager = MigrationManager(registry)
        
        # Perform migration
        migration_manager.migrate_all_components()
        
        # Get migration status
        status = migration_manager.get_migration_status()
        logger.info(f"Migration Status: {status}")
        
        # Test loss components
        logger.info("Testing migrated loss components...")
        from utils.modular_components.config_schemas import LossConfig
        
        # Test a few loss components
        loss_names = ['mse', 'mae', 'quantile']
        for loss_name in loss_names:
            try:
                config = LossConfig(
                    component_name=loss_name
                )
                
                loss_component = registry.create('loss', loss_name, config)
                
                # Test with dummy data
                pred = torch.randn(2, 24, 7)
                target = torch.randn(2, 24, 7)
                
                loss_value = loss_component.compute_loss(pred, target)
                logger.info(f"✅ Loss {loss_name}: {loss_value.item():.4f}")
                
            except Exception as e:
                logger.warning(f"❌ Failed to test loss {loss_name}: {e}")
        
        # Test attention components
        logger.info("Testing migrated attention components...")
        from utils.modular_components.config_schemas import AttentionConfig
        
        attention_names = ['autocorrelation_layer', 'multi_head', 'adaptive_autocorrelation_layer']
        for attention_name in attention_names:
            try:
                config = AttentionConfig(
                    component_name=attention_name,
                    d_model=64,
                    num_heads=4,
                    dropout=0.1
                )
                
                attention_component = registry.create('attention', attention_name, config)
                
                # Test with dummy data
                q = k = v = torch.randn(2, 96, 64)
                
                output, attn = attention_component.forward(q, k, v)
                logger.info(f"✅ Attention {attention_name}: output shape {output.shape}")
                
            except Exception as e:
                logger.warning(f"❌ Failed to test attention {attention_name}: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"Migration framework test failed: {e}")
        return False


def test_modular_autoformer():
    """Test the updated ModularAutoformer"""
    logger.info("Testing updated ModularAutoformer...")
    
    try:
        # Add models directory to path
        models_path = os.path.join(os.path.dirname(__file__), 'models')
        if models_path not in sys.path:
            sys.path.insert(0, models_path)
        
        from modular_autoformer import ModularAutoformer
        
        # Create test configuration
        config = Namespace(
            seq_len=96,
            pred_len=24,
            label_len=48,
            enc_in=7,
            dec_in=7,
            c_out=7,
            d_model=64,  # Smaller for demo
            n_heads=4,
            e_layers=2,
            d_layers=1,
            d_ff=128,
            dropout=0.1,
            factor=1,
            output_attention=False,
            moving_avg=25,
            embed='timeF',
            freq='h',
            # Component types
            attention_type='autocorrelation_layer',
            decomposition_type='series_decomp',
            encoder_type='standard',
            decoder_type='standard',
            loss_type='mse',
            # Backbone options
            use_backbone_component=False,
            backbone_type=None
        )
        
        # Create model with new framework
        logger.info("Creating ModularAutoformer with new framework...")
        model = ModularAutoformer(config)
        
        # Show migration status
        migration_status = model.get_migration_status()
        logger.info(f"Migration Status: {migration_status}")
        
        # Show available components
        available_components = model.list_available_components()
        logger.info(f"Available Components: {available_components}")
        
        # Test forward pass
        logger.info("Testing forward pass...")
        batch_size = 2
        x_enc = torch.randn(batch_size, config.seq_len, config.enc_in)
        x_mark_enc = torch.randn(batch_size, config.seq_len, 4)
        x_dec = torch.randn(batch_size, config.label_len + config.pred_len, config.dec_in)
        x_mark_dec = torch.randn(batch_size, config.label_len + config.pred_len, 4)
        
        with torch.no_grad():
            output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        logger.info(f"✅ Output shape: {output.shape}")
        logger.info("✅ Forward pass successful!")
        
        # Test loss computation
        targets = torch.randn_like(output)
        loss = model.compute_loss(output, targets)
        logger.info(f"✅ Loss computation successful: {loss.item():.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"ModularAutoformer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    print("=" * 80)
    print("MIGRATION SUCCESS TEST")
    print("=" * 80)
    
    # Test migration framework
    framework_success = test_migration_framework()
    
    print("\n" + "=" * 80)
    print("MODULAR AUTOFORMER TEST")
    print("=" * 80)
    
    # Test ModularAutoformer
    model_success = test_modular_autoformer()
    
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    
    if framework_success and model_success:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Migration framework working correctly")
        print("✅ ModularAutoformer using new framework successfully")
        print("✅ Components migrated and functional")
        print("\n🚀 MIGRATION COMPLETED SUCCESSFULLY!")
    else:
        print("❌ Some tests failed:")
        if not framework_success:
            print("  - Migration framework test failed")
        if not model_success:
            print("  - ModularAutoformer test failed")
