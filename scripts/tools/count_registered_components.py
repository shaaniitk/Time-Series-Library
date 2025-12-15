#!/usr/bin/env python3
"""
Utility script to count and display registered components across all registries.

This script provides multiple ways to check component registration counts:
1. Total count across all component families
2. Count by individual component family
3. Detailed breakdown with component names
4. Registry health check
"""

import sys
import os
from typing import Dict, Any
from collections import defaultdict

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def count_components_from_files():
    """Count components by analyzing registry files directly."""
    registry_info = {
        'attention': {
            'file': 'layers/modular/attention/registry.py',
            'components': []
        },
        'backbone': {
            'file': 'layers/modular/backbone/registry.py', 
            'components': []
        },
        'feedforward': {
            'file': 'layers/modular/feedforward/registry.py',
            'components': []
        },
        'output': {
            'file': 'layers/modular/output/registry.py',
            'components': []
        },
        'output_heads': {
            'file': 'layers/modular/output_heads/registry.py',
            'components': []
        },
        'decomposition': {
            'file': 'layers/modular/decomposition/registry.py',
            'components': []
        },
        'normalization': {
            'file': 'layers/modular/normalization/registry.py',
            'components': []
        },
        'processor': {
            'file': 'layers/modular/processor/registry.py',
            'components': []
        },
        'tuning': {
            'file': 'layers/modular/tuning/registry.py',
            'components': []
        },
        'encoder': {
            'file': 'layers/modular/encoder/registry.py',
            'components': []
        },
        'decoder': {
            'file': 'layers/modular/decoder/registry.py',
            'components': []
        }
    }
    
    total_components = 0
    
    print("=" * 60)
    print("REGISTERED COMPONENTS ANALYSIS")
    print("=" * 60)
    
    for family, info in registry_info.items():
        file_path = info['file']
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Simple counting methods
                register_calls = content.count('register_component(')
                registry_entries = 0
                
                # Look for common registry patterns
                if '_registry = {' in content:
                    # Count entries in _registry dictionary
                    lines = content.split('\n')
                    in_registry = False
                    for line in lines:
                        if '_registry = {' in line:
                            in_registry = True
                            continue
                        if in_registry:
                            if '}' in line and not line.strip().startswith('#'):
                                break
                            if ':' in line and not line.strip().startswith('#'):
                                registry_entries += 1
                
                # Estimate component count
                component_count = max(register_calls, registry_entries)
                
                # Try to extract component names from registry patterns
                components = []
                
                # Method 1: Extract from _registry dictionary keys
                import re
                registry_pattern = r'_registry\s*=\s*{([^}]*)'
                registry_match = re.search(registry_pattern, content, re.DOTALL)
                if registry_match:
                    registry_content = registry_match.group(1)
                    # Extract quoted keys
                    key_pattern = r'["\']([a-zA-Z_][a-zA-Z0-9_]*)["\']\s*:'
                    keys = re.findall(key_pattern, registry_content)
                    components.extend(keys)
                
                # Method 2: Extract from register_component calls
                register_pattern = r'register_component\(["\']([a-zA-Z_][a-zA-Z0-9_]*)["\']'
                register_matches = re.findall(register_pattern, content)
                components.extend(register_matches)
                
                # Method 3: Extract from list_X_components function returns
                list_pattern = r'def\s+list_\w+_components\(\)[^{]*{[^}]*return\s*\[([^\]]*)]'
                list_match = re.search(list_pattern, content, re.DOTALL)
                if list_match:
                    list_content = list_match.group(1)
                    list_keys = re.findall(r'["\']([a-zA-Z_][a-zA-Z0-9_]*)["\']', list_content)
                    components.extend(list_keys)
                
                # Method 4: Fallback to manual detection for specific patterns
                if not components:
                    if 'attention' in family:
                        if 'multi_head' in content: components.append('multi_head')
                        if 'self_attention' in content: components.append('self_attention')
                        if 'cross_attention' in content: components.append('cross_attention')
                        if 'sparse_attention' in content: components.append('sparse_attention')
                    elif 'backbone' in family:
                        if 'chronos' in content: components.append('chronos')
                        if 'transformer' in content: components.append('transformer')
                        if 'bert' in content: components.append('bert')
                        if 't5' in content: components.append('t5')
                        if 'simple_transformer' in content: components.append('simple_transformer')
                    elif 'feedforward' in family:
                        if 'standard' in content: components.append('standard')
                        if 'gated' in content: components.append('gated')
                        if 'moe' in content: components.append('moe')
                        if 'conv' in content: components.append('conv')
                    elif 'output' in family:
                        if 'linear' in content: components.append('linear')
                        if 'forecasting' in content: components.append('forecasting')
                        if 'regression' in content: components.append('regression')
                    elif 'decomposition' in family:
                        if 'seasonal_trend' in content: components.append('seasonal_trend')
                        if 'moving_avg' in content: components.append('moving_avg')
                        if 'series_decomp' in content: components.append('series_decomp')
                    elif 'normalization' in family:
                        if 'layernorm' in content: components.append('layernorm')
                        if 'rmsnorm' in content: components.append('rmsnorm')
                        if 'ts_normalizer' in content: components.append('ts_normalizer')
                        if 'normalization_processor' in content: components.append('normalization_processor')
                    elif 'processor' in family:
                        if 'WaveletProcessor' in content: components.append('wavelet_processor')
                        if 'NormalizationProcessor' in content: components.append('normalization_processor')
                    elif 'tuning' in family:
                        if 'KLTuner' in content: components.append('kl_tuner')
                    elif 'encoder' in family:
                        if 'StandardEncoder' in content: components.append('standard')
                        if 'EnhancedEncoder' in content: components.append('enhanced')
                        if 'StableEncoder' in content: components.append('stable')
                        if 'HierarchicalEncoder' in content: components.append('hierarchical')
                        if 'GraphTimeSeriesEncoder' in content: components.append('graph')
                        if 'HybridGraphEncoder' in content: components.append('hybrid_graph')
                        if 'AdaptiveGraphEncoder' in content: components.append('adaptive_graph')
                    elif 'decoder' in family:
                        if 'StandardDecoder' in content: components.append('standard')
                        if 'EnhancedDecoder' in content: components.append('enhanced')
                        if 'StableDecoder' in content: components.append('stable')
                
                # Remove duplicates and clean up
                components = list(set(components))
                components = [comp.strip() for comp in components if comp.strip()]
                
                info['components'] = components
                if component_count == 0 and components:
                    component_count = len(components)
                    
                total_components += component_count
                
                print(f"\n{family.upper():15} : {component_count:3d} components")
                if components:
                    for comp in sorted(components):
                        print(f"  - {comp}")
                elif component_count > 0:
                    print(f"  ({component_count} components detected - see registry file for details)")
                else:
                    print("  (No components detected - may need manual inspection)")
                    
            except Exception as e:
                print(f"\n{family.upper():15} : Error reading file - {e}")
        else:
            print(f"\n{family.upper():15} : Registry file not found")
    
    print("\n" + "=" * 60)
    print(f"TOTAL ESTIMATED COMPONENTS: {total_components}")
    print("=" * 60)
    
    print("\nNOTE: This is an estimation based on file analysis.")
    print("For exact counts, you would need to:")
    print("1. Import the actual registry modules")
    print("2. Call registry.get_all_by_type() for each ComponentFamily")
    print("3. Sum the lengths of returned dictionaries")
    
    return total_components

def show_registry_usage_examples():
    """Show code examples for checking component counts programmatically."""
    print("\n" + "=" * 60)
    print("HOW TO CHECK COMPONENT COUNTS PROGRAMMATICALLY")
    print("=" * 60)
    
    print("\n1. Using the unified registry system:")
    print("""
# Import the registry
from layers.modular.core.registry import component_registry, ComponentFamily

# Count total components
total = 0
for family in ComponentFamily:
    components = component_registry.get_all_by_type(family)
    total += len(components)
    print(f"{family.value}: {len(components)} components")

print(f"Total: {total} components")
""")
    
    print("\n2. Using individual registries:")
    print("""
# Import specific registries
from layers.modular.attention.registry import list_attention_components
from layers.modular.output.registry import list_output_components
from layers.modular.processor.registry import list_processor_components
from layers.modular.tuning.registry import list_tuning_components
from layers.modular.encoder.registry import EncoderRegistry
from layers.modular.decoder.registry import DecoderRegistry
# ... import other registries

# Count components in each registry
attention_count = len(list_attention_components())
output_count = len(list_output_components())
processor_count = len(list_processor_components())
tuning_count = len(list_tuning_components())
# ... count other registries

encoder_count = len(EncoderRegistry.list_components())
decoder_count = len(DecoderRegistry.list_components())
# ... count other registries

total = attention_count + output_count + processor_count + tuning_count + encoder_count + decoder_count  # + other counts
print(f"Total: {total} components")
""")
    
    print("\n3. Using registry factory functions:")
    print("""
# Most registries provide list functions
from layers.modular.backbone.registry import list_backbone_components
from layers.modular.feedforward.registry import list_feedforward_components
from layers.modular.processor.registry import list_processor_components
from layers.modular.tuning.registry import list_tuning_components

backbone_components = list_backbone_components()
feedforward_components = list_feedforward_components()
processor_components = list_processor_components()
tuning_components = list_tuning_components()

print(f"Backbone: {len(backbone_components)} components")
print(f"Feedforward: {len(feedforward_components)} components")
print(f"Processor: {len(processor_components)} components")
print(f"Tuning: {len(tuning_components)} components")
""")

def main():
    """Main function to run the component counter."""
    if len(sys.argv) > 1 and sys.argv[1].lower() == 'examples':
        show_registry_usage_examples()
    else:
        count_components_from_files()
        show_registry_usage_examples()

if __name__ == "__main__":
    main()