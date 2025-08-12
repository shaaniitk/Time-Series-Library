#!/usr/bin/env python3
"""
Test script to validate all integrated advanced components
"""

from configs.concrete_components import component_registry
from configs.schemas import ComponentType

def test_component_registry():
    """Test that all new components are properly registered"""
    
    print("🔍 Testing Advanced Components Integration...")
    print("=" * 60)
    
    # New advanced components to test
    new_components = [
        ComponentType.FOURIER_ATTENTION,
        ComponentType.ADAPTIVE_AUTOCORRELATION, 
        ComponentType.FOURIER_BLOCK,
        ComponentType.ADVANCED_WAVELET,
        ComponentType.TEMPORAL_CONV_ENCODER,
        ComponentType.META_LEARNING_ADAPTER,
        ComponentType.ADAPTIVE_MIXTURE,
        ComponentType.BAYESIAN_HEAD,
        ComponentType.FOCAL_LOSS,
        ComponentType.ADAPTIVE_AUTOFORMER_LOSS,
        ComponentType.ADAPTIVE_LOSS_WEIGHTING
    ]
    
    success_count = 0
    total_count = len(new_components)
    
    for component_type in new_components:
        try:
            component_class = component_registry.get_component(component_type)
            metadata = component_registry.get_metadata(component_type)
            
            print(f"✅ {component_type.value}")
            print(f"   Class: {component_class.__name__}")
            print(f"   Description: {metadata.description}")
            print(f"   Required params: {metadata.required_params}")
            print()
            
            success_count += 1
            
        except Exception as e:
            print(f"❌ {component_type.value}: {e}")
            print()
    
    print("=" * 60)
    print(f"📊 Results: {success_count}/{total_count} components successfully integrated")
    
    if success_count == total_count:
        print("🎉 All advanced components integrated successfully!")
        return True
    else:
        print("⚠️  Some components failed integration")
        return False

def test_component_counts():
    """Test total component counts"""
    
    print("\n📈 Component Count Analysis...")
    print("=" * 40)
    
    component_types = {
        "Attention": [
            ComponentType.AUTOCORRELATION,
            ComponentType.ADAPTIVE_AUTOCORRELATION,
            ComponentType.FOURIER_ATTENTION,
            ComponentType.FOURIER_BLOCK,
            ComponentType.CROSS_RESOLUTION,
            ComponentType.MULTI_HEAD,
            ComponentType.SPARSE,
            ComponentType.LOG_SPARSE,
            ComponentType.PROB_SPARSE
        ],
        "Decomposition": [
            ComponentType.MOVING_AVG,
            ComponentType.LEARNABLE_DECOMP,
            ComponentType.WAVELET_DECOMP,
            ComponentType.ADVANCED_WAVELET
        ],
        "Encoder": [
            ComponentType.STANDARD_ENCODER,
            ComponentType.ENHANCED_ENCODER,
            ComponentType.HIERARCHICAL_ENCODER,
            ComponentType.TEMPORAL_CONV_ENCODER,
            ComponentType.META_LEARNING_ADAPTER
        ],
        "Decoder": [
            ComponentType.STANDARD_DECODER,
            ComponentType.ENHANCED_DECODER,
            ComponentType.HIERARCHICAL_DECODER
        ],
        "Sampling": [
            ComponentType.DETERMINISTIC,
            ComponentType.BAYESIAN,
            ComponentType.MONTE_CARLO,
            ComponentType.ADAPTIVE_MIXTURE
        ],
        "Output Head": [
            ComponentType.STANDARD_HEAD,
            ComponentType.QUANTILE,
            ComponentType.BAYESIAN_HEAD
        ],
        "Loss": [
            ComponentType.MSE,
            ComponentType.MAE,
            ComponentType.QUANTILE_LOSS,
            ComponentType.BAYESIAN_MSE,
            ComponentType.BAYESIAN_QUANTILE,
            ComponentType.FOCAL_LOSS,
            ComponentType.ADAPTIVE_AUTOFORMER_LOSS,
            ComponentType.ADAPTIVE_LOSS_WEIGHTING
        ]
    }
    
    total_components = 0
    
    for category, components in component_types.items():
        registered_count = 0
        for comp_type in components:
            try:
                component_registry.get_component(comp_type)
                registered_count += 1
            except:
                pass
        
        print(f"{category:15}: {registered_count:2d} components")
        total_components += registered_count
    
    print("-" * 40)
    print(f"{'Total':15}: {total_components:2d} components")
    
    return total_components

if __name__ == "__main__":
    print("🚀 Advanced Components Integration Test")
    print("=" * 60)
    
    # Test component registration
    registry_success = test_component_registry()
    
    # Test component counts
    total_count = test_component_counts()
    
    print(f"\n📋 Summary:")
    print(f"   • Registry test: {'✅ PASSED' if registry_success else '❌ FAILED'}")
    print(f"   • Total components: {total_count}")
    print(f"   • Target components: 38")
    print(f"   • Integration: {'✅ COMPLETE' if total_count >= 38 else '⚠️  INCOMPLETE'}")
