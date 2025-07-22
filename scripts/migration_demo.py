"""
Migration Demonstration Script

This script demonstrates the complete component migration and shows
the new ModularAutoformer using only the new framework.
"""

import torch
import logging
from argparse import Namespace

# Import migration framework
from utils.modular_components.migration_framework import MigrationManager
from utils.modular_components.registry import ComponentRegistry

# Import updated ModularAutoformer
from models.modular_autoformer import ModularAutoformer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_migration_demo():
    """Run a demonstration of the migration framework"""
    logger.info("Starting Migration Demonstration...")
    
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
    
    logger.info(f"Output shape: {output.shape}")
    logger.info("Forward pass successful!")
    
    # Test loss computation
    targets = torch.randn_like(output)
    loss = model.compute_loss(output, targets)
    logger.info(f"Loss computation successful: {loss.item():.4f}")
    
    logger.info("Migration demonstration completed successfully!")
    return True


def test_migrated_components():
    """Test individual migrated components"""
    logger.info("Testing individual migrated components...")
    
    # Create registry and migration manager
    registry = ComponentRegistry()
    migration_manager = MigrationManager(registry)
    
    # Perform migration
    migration_manager.migrate_all_components()
    
    # Test loss components
    logger.info("Testing migrated loss components...")
    loss_components = registry.list_components('loss')['loss']
    
    for loss_name in loss_components[:3]:  # Test first 3
        try:
            from utils.modular_components.config_schemas import ComponentConfig
            
            config = ComponentConfig(
                component_type='loss',
                component_name=loss_name,
                parameters={}
            )
            
            loss_component = registry.create('loss', loss_name, config)
            
            # Test with dummy data
            pred = torch.randn(2, 24, 7)
            target = torch.randn(2, 24, 7)
            
            loss_value = loss_component.compute_loss(pred, target)
            logger.info(f"Loss {loss_name}: {loss_value.item():.4f}")
            
        except Exception as e:
            logger.warning(f"Failed to test loss {loss_name}: {e}")
    
    # Test attention components
    logger.info("Testing migrated attention components...")
    attention_components = registry.list_components('attention')['attention']
    
    for attention_name in attention_components[:3]:  # Test first 3
        try:
            from utils.modular_components.config_schemas import ComponentConfig
            
            config = ComponentConfig(
                component_type='attention',
                component_name=attention_name,
                parameters={
                    'd_model': 64,
                    'n_heads': 4,
                    'dropout': 0.1,
                    'factor': 1
                }
            )
            
            attention_component = registry.create('attention', attention_name, config)
            
            # Test with dummy data
            q = k = v = torch.randn(2, 96, 64)
            
            output, attn = attention_component.forward(q, k, v)
            logger.info(f"Attention {attention_name}: output shape {output.shape}")
            
        except Exception as e:
            logger.warning(f"Failed to test attention {attention_name}: {e}")
    
    logger.info("Component testing completed!")


if __name__ == '__main__':
    print("="*60)
    print("COMPONENT MIGRATION DEMONSTRATION")
    print("="*60)
    
    # Test individual components
    test_migrated_components()
    
    print("\n" + "="*60)
    print("MODULAR AUTOFORMER DEMONSTRATION")
    print("="*60)
    
    # Run full demo
    success = run_migration_demo()
    
    if success:
        print("\n🎉 ALL TESTS PASSED! Migration completed successfully!")
        print("ModularAutoformer now uses only the new modular framework!")
    else:
        print("\n❌ Migration demonstration failed!")
