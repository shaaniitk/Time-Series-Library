"""
Unified Component Registry Facade

This provides a single interface for accessing all components while maintaining
the broader context of the Time-Series-Library modularization project.

CONTEXT:
- Part of comprehensive layers library analysis
- Implements modular architecture consolidation
- Preserves sophisticated algorithmic implementations
"""

import warnings
from typing import Dict, List, Any, Type, Optional
from utils.modular_components.registry import _global_registry, register_component
from utils.modular_components.factory import ComponentFactory
from utils.modular_components.base_interfaces import BaseComponent
import torch

# Register wrapped legacy attentions from layers folder
try:
    from utils.implementations.attention.layers_wrapped_attentions import register_layers_attentions
    _HAS_LAYER_ATTENTION_WRAPPERS = True
except Exception:
    _HAS_LAYER_ATTENTION_WRAPPERS = False

# Register wrapped legacy decomposition processors
try:
    from utils.implementations.decomposition.wrapped_decompositions import register_layers_decompositions
    _HAS_LAYER_DECOMP_WRAPPERS = True
except Exception:
    _HAS_LAYER_DECOMP_WRAPPERS = False

# Register wrapped legacy encoder processors
try:
    from utils.implementations.encoder.wrapped_encoders import register_layers_encoders
    _HAS_LAYER_ENCODER_WRAPPERS = True
except Exception:
    _HAS_LAYER_ENCODER_WRAPPERS = False

# Register wrapped legacy decoder processors
try:
    from utils.implementations.decoder.wrapped_decoders import register_layers_decoders
    _HAS_LAYER_DECODER_WRAPPERS = True
except Exception:
    _HAS_LAYER_DECODER_WRAPPERS = False

# Register wrapped legacy fusion processors
try:
    from utils.implementations.fusion.wrapped_fusions import register_layers_fusions
    _HAS_LAYER_FUSION_WRAPPERS = True
except Exception:
    _HAS_LAYER_FUSION_WRAPPERS = False

# Register utils loss implementations
try:
    from utils.implementations.loss.wrapped_losses import register_utils_losses
    _HAS_UTILS_LOSSES = True
except Exception:
    _HAS_UTILS_LOSSES = False

# Register utils output implementations
try:
    from utils.implementations.output.wrapped_outputs import register_utils_outputs
    _HAS_UTILS_OUTPUTS = True
except Exception:
    _HAS_UTILS_OUTPUTS = False

# Register legacy layers output head wrappers
try:
    from utils.implementations.output.layers_wrapped_outputs import register_layers_output_heads
    _HAS_LAYER_OUTPUT_WRAPPERS = True
except Exception:
    _HAS_LAYER_OUTPUT_WRAPPERS = False

# Register utils embedding implementations
try:
    from utils.implementations.embedding.wrapped_embeddings import register_utils_embeddings
    _HAS_UTILS_EMBEDDINGS = True
except Exception:
    _HAS_UTILS_EMBEDDINGS = False

# Register utils feedforward implementations
try:
    from utils.implementations.feedforward.wrapped_feedforward import register_utils_feedforwards
    _HAS_UTILS_FFN = True
except Exception:
    _HAS_UTILS_FFN = False

# Register utils adapter implementations
try:
    from utils.implementations.adapter.wrapped_adapters import register_utils_adapters
    _HAS_UTILS_ADAPTERS = True
except Exception:
    _HAS_UTILS_ADAPTERS = False

# Import sophisticated algorithms
try:
    from utils.implementations.attention.restored_algorithms import (
        RestoredFourierAttention,
        RestoredAutoCorrelationAttention, 
        RestoredMetaLearningAttention,
        register_restored_algorithms
    )
    ALGORITHMS_AVAILABLE = True
except ImportError:
    # Fallback to direct import
    from utils_algorithm_adapters import (
        RestoredFourierAttention,
        RestoredAutoCorrelationAttention, 
        RestoredMetaLearningAttention,
        register_restored_algorithms
    )
    ALGORITHMS_AVAILABLE = True


class UnifiedComponentRegistry:
    """
    Unified registry facade providing single access point for all components
    
    This maintains compatibility during the migration from layers/modular/ 
    to utils/ system while preserving the broader layers analysis context.
    """
    
    def __init__(self):
        self.utils_registry = _global_registry
        self.factory = ComponentFactory()
        
        # Ensure sophisticated algorithms are registered
        if ALGORITHMS_AVAILABLE:
            register_restored_algorithms()

        # Also register adapters for legacy layers attention implementations
        if _HAS_LAYER_ATTENTION_WRAPPERS:
            try:
                register_layers_attentions()
            except Exception as e:
                print(f"Warning: could not register layers attentions: {e}")
        
        # Register decomposition processors
        if _HAS_LAYER_DECOMP_WRAPPERS:
            try:
                register_layers_decompositions()
            except Exception as e:
                print(f"Warning: could not register decomposition processors: {e}")
        
        # Register encoder processors
        if _HAS_LAYER_ENCODER_WRAPPERS:
            try:
                register_layers_encoders()
            except Exception as e:
                print(f"Warning: could not register encoder processors: {e}")
        
        # Register decoder processors
        if _HAS_LAYER_DECODER_WRAPPERS:
            try:
                register_layers_decoders()
            except Exception as e:
                print(f"Warning: could not register decoder processors: {e}")
        
        # Register fusion processors
        if _HAS_LAYER_FUSION_WRAPPERS:
            try:
                register_layers_fusions()
            except Exception as e:
                print(f"Warning: could not register fusion processors: {e}")
        
        # Register utils losses
        if _HAS_UTILS_LOSSES:
            try:
                register_utils_losses()
            except Exception as e:
                print(f"Warning: could not register utils losses: {e}")

        # Register utils outputs
        if _HAS_UTILS_OUTPUTS:
            try:
                register_utils_outputs()
            except Exception as e:
                print(f"Warning: could not register utils outputs: {e}")
        
        # Register legacy output heads
        if _HAS_LAYER_OUTPUT_WRAPPERS:
            try:
                register_layers_output_heads()
            except Exception as e:
                print(f"Warning: could not register legacy output heads: {e}")
        
        # Register utils embeddings
        if _HAS_UTILS_EMBEDDINGS:
            try:
                register_utils_embeddings()
            except Exception as e:
                print(f"Warning: could not register utils embeddings: {e}")
        
        # Register utils feedforwards
        if _HAS_UTILS_FFN:
            try:
                register_utils_feedforwards()
            except Exception as e:
                print(f"Warning: could not register utils feedforwards: {e}")
        
        # Register utils adapters
        if _HAS_UTILS_ADAPTERS:
            try:
                register_utils_adapters()
            except Exception as e:
                print(f"Warning: could not register utils adapters: {e}")
    
    def get_component(self, component_type: str, component_name: str) -> Type[BaseComponent]:
        """Get component class (unified interface)"""
        return self.utils_registry.get(component_type, component_name)
    
    def create_component(self, component_type: str, component_name: str, config: Any) -> BaseComponent:
        """Create component instance (unified interface)"""
        component_class = self.get_component(component_type, component_name)
        return component_class(config)
    
    def list_all_components(self) -> Dict[str, List[str]]:
        """List all available components"""
        return self.utils_registry.list_components()
    
    def get_sophisticated_algorithms(self) -> List[Dict[str, Any]]:
        """Get information about sophisticated algorithms"""
        algorithms = []
        restored_algos = [
            'restored_fourier_attention',
            'restored_autocorrelation_attention', 
            'restored_meta_learning_attention'
        ]
        
        for algo in restored_algos:
            if self.utils_registry.is_registered('attention', algo):
                metadata = self.utils_registry.get_metadata('attention', algo)
                algorithms.append({
                    'name': algo,
                    'type': 'attention',
                    'sophistication_level': metadata.get('sophistication_level'),
                    'features': metadata.get('features', []),
                    'algorithm_source': metadata.get('algorithm_source')
                })
                
        return algorithms
    
    def validate_migration_status(self) -> Dict[str, Any]:
        """Validate current migration status"""
        status = {
            'utils_components': 0,
            'sophisticated_algorithms': 0,
            'migration_complete': False,
            'issues': []
        }
        
        try:
            # Count utils components
            all_components = self.utils_registry.list_components()
            status['utils_components'] = sum(len(comps) for comps in all_components.values())
            
            # Count sophisticated algorithms
            sophisticated = self.get_sophisticated_algorithms()
            status['sophisticated_algorithms'] = len(sophisticated)
            
            # Check migration completeness
            if status['sophisticated_algorithms'] >= 3:
                status['migration_complete'] = True
            else:
                status['issues'].append("Missing sophisticated algorithms")
                
        except Exception as e:
            status['issues'].append(f"Validation error: {e}")
            
        return status
    
    def get_migration_summary(self) -> str:
        """Get human-readable migration summary"""
        status = self.validate_migration_status()
        sophisticated = self.get_sophisticated_algorithms()
        
        summary = "🎯 UNIFIED REGISTRY STATUS\n"
        summary += "=" * 30 + "\n"
        summary += f"Utils Components: {status['utils_components']}\n"
        summary += f"Sophisticated Algorithms: {status['sophisticated_algorithms']}/3\n"
        summary += f"Migration Complete: {'✅' if status['migration_complete'] else '❌'}\n\n"
        
        if sophisticated:
            summary += "🧠 SOPHISTICATED ALGORITHMS:\n"
            for algo in sophisticated:
                summary += f"   • {algo['name']}: {algo['sophistication_level']} sophistication\n"
                
        if status['issues']:
            summary += "\n⚠️  ISSUES:\n"
            for issue in status['issues']:
                summary += f"   • {issue}\n"
                
        return summary
    
    def test_component_functionality(self) -> bool:
        """Test that components actually work"""
        try:
            from utils_algorithm_adapters import RestoredFourierConfig
            config = RestoredFourierConfig(d_model=128, num_heads=4, dropout=0.1)
            component = self.create_component('attention', 'restored_fourier_attention', config)
            
            # Test the component
            x = torch.randn(2, 16, 128)
            output, _ = component.apply_attention(x, x, x)
            assert output.shape == x.shape
            
            return True
        except Exception as e:
            print(f"Component test failed: {e}")
            return False


# Global unified registry instance
unified_registry = UnifiedComponentRegistry()

# Backwards compatibility functions
def get_component(component_type: str, component_name: str) -> Type[BaseComponent]:
    """Backwards compatible component access"""
    warnings.warn("Direct get_component is deprecated, use unified_registry", DeprecationWarning)
    return unified_registry.get_component(component_type, component_name)

def create_component(component_type: str, component_name: str, config: Any) -> BaseComponent:
    """Backwards compatible component creation"""
    warnings.warn("Direct create_component is deprecated, use unified_registry", DeprecationWarning)
    return unified_registry.create_component(component_type, component_name, config)
