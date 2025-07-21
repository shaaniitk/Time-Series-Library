#!/usr/bin/env python3
"""
COMPONENT INVENTORY TOOL
Counts all actual components defined in the new modular architecture.
"""

import os
import inspect
from utils.modular_components.registry import ComponentRegistry
from utils.modular_components.base_interfaces import BaseComponent, BaseLoss, BaseAttention

def count_modular_components():
    print('🔍 ANALYZING NEW MODULAR ARCHITECTURE COMPONENTS...')
    print('=' * 60)
    
    # Import all implementation modules
    try:
        from utils.modular_components.implementations import (
            losses, attentions, advanced_losses, advanced_attentions,
            embeddings, backbones, simple_backbones, outputs, 
            feedforward, processors, specialized_processors
        )
        
        modules = [
            ('losses', losses),
            ('attentions', attentions), 
            ('advanced_losses', advanced_losses),
            ('advanced_attentions', advanced_attentions),
            ('embeddings', embeddings),
            ('backbones', backbones),
            ('simple_backbones', simple_backbones),
            ('outputs', outputs),
            ('feedforward', feedforward),
            ('processors', processors),
            ('specialized_processors', specialized_processors)
        ]
        
    except ImportError as e:
        print(f"⚠️  Some modules not available: {e}")
        modules = []
        
        # Try individual imports
        try:
            from utils.modular_components.implementations import losses
            modules.append(('losses', losses))
        except ImportError:
            pass
            
        try:
            from utils.modular_components.implementations import attentions
            modules.append(('attentions', attentions))
        except ImportError:
            pass
            
        try:
            from utils.modular_components.implementations import advanced_losses
            modules.append(('advanced_losses', advanced_losses))
        except ImportError:
            pass
    
    total_components = 0
    component_breakdown = {}
    
    for module_name, module in modules:
        print(f'\n📦 Module: {module_name}')
        components_in_module = []
        
        for name in dir(module):
            try:
                obj = getattr(module, name)
                if (inspect.isclass(obj) and 
                    issubclass(obj, (BaseComponent, BaseLoss, BaseAttention)) and
                    obj not in [BaseComponent, BaseLoss, BaseAttention]):
                    components_in_module.append(name)
                    total_components += 1
            except:
                pass
        
        print(f'   ✅ {len(components_in_module)} components: {components_in_module[:5]}{"..." if len(components_in_module) > 5 else ""}')
        component_breakdown[module_name] = len(components_in_module)
    
    print(f'\n📊 COMPONENT BREAKDOWN:')
    for module, count in component_breakdown.items():
        print(f'   {module}: {count} components')
    
    print(f'\n🎯 TOTAL NEW MODULAR COMPONENTS: {total_components}')
    
    # Also check layer components
    try:
        print(f'\n🔧 CHECKING MODULAR LAYERS...')
        
        # Check encoder/decoder components
        encoder_dir = 'layers/modular/encoder'
        decoder_dir = 'layers/modular/decoder'
        
        if os.path.exists(encoder_dir):
            encoder_files = [f for f in os.listdir(encoder_dir) if f.endswith('.py') and f != '__init__.py']
            print(f'   📁 Encoder components: {len(encoder_files)} files: {encoder_files}')
        
        if os.path.exists(decoder_dir):
            decoder_files = [f for f in os.listdir(decoder_dir) if f.endswith('.py') and f != '__init__.py']
            print(f'   📁 Decoder components: {len(decoder_files)} files: {decoder_files}')
        
    except Exception as e:
        print(f'   ⚠️  Layer check failed: {e}')
    
    return total_components, component_breakdown

if __name__ == "__main__":
    count_modular_components()
