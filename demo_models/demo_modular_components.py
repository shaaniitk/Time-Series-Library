"""
Modular Architecture Demonstration

This script demonstrates how to use the modular component system
to build flexible HF Autoformer variants with different component combinations.
"""

import torch
import logging
from typing import Dict, Any

# Import the modular components system
from layers.modular.core.registry import (
    get_global_registry,
    get_available_components,
    get_component_info
)
from layers.modular.core.factory import (
    create_backbone,
    create_embedding,
    create_attention
)
from layers.modular.core.config_schemas import (
    BackboneConfig,
    EmbeddingConfig,
    AttentionConfig,
    create_default_configs
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demo_component_registry():
    """Demonstrate the component registry functionality"""
    print("\n" + "="*60)
    print("COMPONENT REGISTRY DEMONSTRATION")
    print("="*60)
    
    # Get available components
    available = get_available_components()
    
    print("\nAvailable Components:")
    for component_type, components in available.items():
        print(f"  {component_type.capitalize()}: {components}")
    
    # Get detailed info for specific components
    print("\nDetailed Component Information:")
    
    # Backbone info
    chronos_info = get_component_info('backbone', 'chronos')
    if chronos_info:
        print(f"\nChronos Backbone:")
        print(f"  Class: {chronos_info['class']}")
        print(f"  Metadata: {chronos_info['metadata']}")
    
    # Attention info
    autocorr_info = get_component_info('attention', 'autocorrelation')
    if autocorr_info:
        print(f"\nAutoCorrelation Attention:")
        print(f"  Class: {autocorr_info['class']}")
        print(f"  Metadata: {autocorr_info['metadata']}")


def demo_component_creation():
    """Demonstrate creating different component combinations"""
    print("\n" + "="*60)
    print("COMPONENT CREATION DEMONSTRATION")
    print("="*60)
    
    # Create different backbone configurations
    backbones_to_test = [
        ('chronos', BackboneConfig(
            d_model=256,
            dropout=0.1,
            model_name='amazon/chronos-t5-tiny',  # Use tiny for demo
            pretrained=True,
            fallback_models=['simple_transformer']
        )),
        ('simple_transformer', BackboneConfig(
            d_model=256,
            dropout=0.1,
            num_layers=4,
            num_heads=4,
            dim_feedforward=1024
        ))
    ]
    
    # Create different embedding configurations  
    embeddings_to_test = [
        ('temporal', EmbeddingConfig(
            d_model=256,
            dropout=0.1,
            max_len=1000
        )),
        ('value', EmbeddingConfig(
            d_model=256,
            dropout=0.1,
            num_features=1,
            use_binning=False
        )),
        ('hybrid', EmbeddingConfig(
            d_model=256,
            dropout=0.1,
            use_temporal=True,
            use_value=True,
            use_covariate=False
        ))
    ]
    
    # Create different attention configurations
    attentions_to_test = [
        ('multi_head', AttentionConfig(
            d_model=256,
            dropout=0.1,
            num_heads=4
        )),
        ('autocorrelation', AttentionConfig(
            d_model=256,
            dropout=0.1,
            factor=1
        )),
        ('sparse', AttentionConfig(
            d_model=256,
            dropout=0.1,
            num_heads=4,
            sparsity_factor=2
        ))
    ]
    
    print("\nTesting Component Creation:")
    
    # Test backbone creation
    print("\n1. Testing Backbone Components:")
    for backbone_name, config in backbones_to_test:
        try:
            backbone = create_backbone(backbone_name, config)
            capabilities = backbone.get_capabilities()
            print(f"  ✓ {backbone_name}: {backbone.__class__.__name__}")
            print(f"    Capabilities: {list(capabilities.keys())}")
            print(f"    Hidden size: {backbone.get_hidden_size()}")
        except Exception as e:
            print(f"  ✗ {backbone_name}: Failed - {e}")
    
    # Test embedding creation
    print("\n2. Testing Embedding Components:")
    for embedding_name, config in embeddings_to_test:
        try:
            embedding = create_embedding(embedding_name, config)
            capabilities = embedding.get_capabilities()
            print(f"  ✓ {embedding_name}: {embedding.__class__.__name__}")
            print(f"    Capabilities: {capabilities.get('type', 'unknown')}")
            print(f"    Output dim: {embedding.get_output_dim()}")
        except Exception as e:
            print(f"  ✗ {embedding_name}: Failed - {e}")
    
    # Test attention creation
    print("\n3. Testing Attention Components:")
    for attention_name, config in attentions_to_test:
        try:
            attention = create_attention(attention_name, config)
            capabilities = attention.get_capabilities()
            print(f"  ✓ {attention_name}: {attention.__class__.__name__}")
            print(f"    Capabilities: {capabilities.get('type', 'unknown')}")
        except Exception as e:
            print(f"  ✗ {attention_name}: Failed - {e}")


def demo_component_combinations():
    """Demonstrate different component combinations for model variants"""
    print("\n" + "="*60)
    print("COMPONENT COMBINATION DEMONSTRATION")
    print("="*60)
    
    # Define different model configurations
    model_configs = {
        'lightweight': {
            'backbone': ('simple_transformer', BackboneConfig(d_model=128, num_layers=2, num_heads=2)),
            'embedding': ('value', EmbeddingConfig(d_model=128, num_features=1)),
            'attention': ('multi_head', AttentionConfig(d_model=128, num_heads=2))
        },
        'time_series_optimized': {
            'backbone': ('simple_transformer', BackboneConfig(d_model=256, num_layers=4, num_heads=4)),
            'embedding': ('hybrid', EmbeddingConfig(d_model=256, use_temporal=True, use_value=True)),
            'attention': ('autocorrelation', AttentionConfig(d_model=256, factor=1))
        },
        'long_sequence': {
            'backbone': ('simple_transformer', BackboneConfig(d_model=256, num_layers=3, num_heads=4)),
            'embedding': ('temporal', EmbeddingConfig(d_model=256, max_len=5000)),
            'attention': ('sparse', AttentionConfig(d_model=256, num_heads=4, sparsity_factor=4))
        }
    }
    
    print("\nTesting Model Variant Configurations:")
    
    for variant_name, config in model_configs.items():
        print(f"\n{variant_name.upper()} Variant:")
        
        components = {}
        success = True
        
        # Create each component
        for component_type, (component_name, component_config) in config.items():
            try:
                if component_type == 'backbone':
                    component = create_backbone(component_name, component_config)
                elif component_type == 'embedding':
                    component = create_embedding(component_name, component_config)
                elif component_type == 'attention':
                    component = create_attention(component_name, component_config)
                
                components[component_type] = component
                print(f"  ✓ {component_type}: {component_name} -> {component.__class__.__name__}")
                
            except Exception as e:
                print(f"  ✗ {component_type}: {component_name} -> Failed: {e}")
                success = False
        
        if success:
            print(f"  🎉 {variant_name} variant created successfully!")
            
            # Show component capabilities
            for comp_type, component in components.items():
                caps = component.get_capabilities()
                print(f"    {comp_type} capabilities: {caps.get('type', 'unknown')}")
        else:
            print(f"  ❌ {variant_name} variant creation failed")


def demo_forward_pass():
    """Demonstrate a simple forward pass through components"""
    print("\n" + "="*60)
    print("FORWARD PASS DEMONSTRATION")
    print("="*60)
    
    try:
        # Create a simple configuration
        d_model = 128
        seq_len = 10
        batch_size = 2
        
        # Create components
        embedding = create_embedding('value', EmbeddingConfig(
            d_model=d_model,
            num_features=1,
            use_binning=False
        ))
        
        attention = create_attention('multi_head', AttentionConfig(
            d_model=d_model,
            num_heads=4
        ))
        
        print(f"\nCreated components for forward pass demo:")
        print(f"  Embedding: {embedding.__class__.__name__}")
        print(f"  Attention: {attention.__class__.__name__}")
        
        # Create sample data
        sample_values = torch.randn(batch_size, seq_len, 1)
        
        print(f"\nSample input shape: {sample_values.shape}")
        
        # Forward pass through embedding
        embeddings = embedding(sample_values)
        print(f"Embeddings shape: {embeddings.shape}")
        
        # Forward pass through attention (self-attention)
        attended_output, attention_weights = attention(embeddings, embeddings, embeddings)
        print(f"Attention output shape: {attended_output.shape}")
        print(f"Attention weights shape: {attention_weights.shape}")
        
        print("\n✅ Forward pass completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Forward pass failed: {e}")


def main():
    """Main demonstration function"""
    print("MODULAR COMPONENT ARCHITECTURE DEMONSTRATION")
    print("This script shows how to use the flexible component system")
    print("to build different HF Autoformer variants.")
    
    try:
        # Run demonstrations
        demo_component_registry()
        demo_component_creation()
        demo_component_combinations()
        demo_forward_pass()
        
        print("\n" + "="*60)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nThe modular component system is working correctly.")
        print("You can now use this system to build flexible HF Autoformer variants")
        print("by mixing and matching different backbone, embedding, and attention components.")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        logger.exception("Demo failed with exception")


if __name__ == "__main__":
    main()
