#!/usr/bin/env python3
"""
COMPREHENSIVE CONSOLIDATION PLAN
Systematic approach to create unified component files and proper registries
"""

import os
from pathlib import Path

# =============================================================================
# PHASE 1: CONSOLIDATION PLAN
# =============================================================================

COMPONENT_TYPES = {
    'attentions': {
        'source_files': [
            'utils/modular_components/implementations/attentions.py',
            'utils/modular_components/implementations/advanced_attentions.py', 
            'utils/modular_components/implementations/attention_migrated.py'
        ],
        'target_file': 'utils/modular_components/implementations/attentions_unified.py',
        'registry_name': 'ATTENTION_REGISTRY',
        'base_class': 'BaseAttention',
        'priority': 1
    },
    'losses': {
        'source_files': [
            'utils/modular_components/implementations/losses.py',
            'utils/modular_components/implementations/losses_clean.py',
            'utils/modular_components/implementations/advanced_losses.py',
            'utils/modular_components/implementations/losses_migrated.py'
        ],
        'target_file': 'utils/modular_components/implementations/losses_unified.py',
        'registry_name': 'LOSS_REGISTRY', 
        'base_class': 'BaseLoss',
        'priority': 1
    },
    'decomposition': {
        'source_files': [
            'utils/modular_components/implementations/decomposition_migrated.py'
        ],
        'target_file': 'utils/modular_components/implementations/decomposition_unified.py',
        'registry_name': 'DECOMPOSITION_REGISTRY',
        'base_class': 'BaseComponent',
        'priority': 2
    },
    'encoders': {
        'source_files': [
            'utils/modular_components/implementations/encoder_migrated.py'
        ],
        'target_file': 'utils/modular_components/implementations/encoders_unified.py', 
        'registry_name': 'ENCODER_REGISTRY',
        'base_class': 'BaseComponent',
        'priority': 2
    },
    'decoders': {
        'source_files': [
            'utils/modular_components/implementations/decoder_migrated.py'
        ],
        'target_file': 'utils/modular_components/implementations/decoders_unified.py',
        'registry_name': 'DECODER_REGISTRY', 
        'base_class': 'BaseComponent',
        'priority': 2
    },
    'sampling': {
        'source_files': [
            'utils/modular_components/implementations/sampling_migrated.py'
        ],
        'target_file': 'utils/modular_components/implementations/sampling_unified.py',
        'registry_name': 'SAMPLING_REGISTRY',
        'base_class': 'BaseComponent', 
        'priority': 3
    },
    'output_heads': {
        'source_files': [
            'utils/modular_components/implementations/outputs.py',
            'utils/modular_components/implementations/output_heads_migrated.py'
        ],
        'target_file': 'utils/modular_components/implementations/output_heads_unified.py',
        'registry_name': 'OUTPUT_HEADS_REGISTRY',
        'base_class': 'BaseComponent',
        'priority': 3
    },
    'layers': {
        'source_files': [
            'utils/modular_components/implementations/layers_migrated.py'
        ],
        'target_file': 'utils/modular_components/implementations/layers_unified.py',
        'registry_name': 'LAYERS_REGISTRY',
        'base_class': 'BaseComponent',
        'priority': 3
    },
    'fusion': {
        'source_files': [
            'utils/modular_components/implementations/fusion_migrated.py'
        ],
        'target_file': 'utils/modular_components/implementations/fusion_unified.py',
        'registry_name': 'FUSION_REGISTRY',
        'base_class': 'BaseComponent',
        'priority': 3
    }
}

# =============================================================================
# PHASE 2: REGISTRY STRUCTURE
# =============================================================================

MASTER_REGISTRY_TEMPLATE = '''"""
UNIFIED COMPONENT REGISTRY
Master registry that imports all unified component types
"""

from typing import Dict, Any, Optional, Type
import logging

from ..registry import ComponentRegistry
from ..config_schemas import ComponentConfig
from ..base_interfaces import BaseComponent, BaseLoss, BaseAttention

{imports}

logger = logging.getLogger(__name__)


class UnifiedComponentRegistry:
    """Master registry for all unified components"""
    
    def __init__(self):
        self.registry = ComponentRegistry()
        self._register_all_unified_components()
    
    def _register_all_unified_components(self):
        """Register all unified components"""
        logger.info("Registering all unified components...")
        
{registrations}
        
        total_components = sum(len(cat) for cat in self.registry._components.values())
        logger.info(f"✅ Registered {{total_components}} unified components across {{len(self.registry._components)}} categories")
    
    def create(self, category: str, name: str, config: ComponentConfig = None, **kwargs):
        """Create a component by category and name"""
        return self.registry.create(category, name, config, **kwargs)
    
    def list_components(self):
        """List all available components"""
        return self.registry.list_components()
    
    def get_summary(self):
        """Get summary of all registered components"""
        components = self.list_components()
        return {{
            'total_categories': len(components),
            'total_components': sum(len(comp_list) for comp_list in components.values()),
            'categories': {{k: len(v) for k, v in components.items()}}
        }}

# Global unified registry instance
unified_registry = UnifiedComponentRegistry()
'''

# =============================================================================
# PHASE 3: DELETION LIST
# =============================================================================

FILES_TO_DELETE = [
    # Original scattered implementations
    'utils/modular_components/implementations/attentions.py',
    'utils/modular_components/implementations/advanced_attentions.py',
    'utils/modular_components/implementations/losses.py', 
    'utils/modular_components/implementations/advanced_losses.py',
    'utils/modular_components/implementations/outputs.py',
    
    # Migrated files (will be replaced by unified versions)
    'utils/modular_components/implementations/attention_migrated.py',
    'utils/modular_components/implementations/decomposition_migrated.py',
    'utils/modular_components/implementations/encoder_migrated.py',
    'utils/modular_components/implementations/decoder_migrated.py',
    'utils/modular_components/implementations/sampling_migrated.py',
    'utils/modular_components/implementations/output_heads_migrated.py',
    'utils/modular_components/implementations/layers_migrated.py',
    'utils/modular_components/implementations/fusion_migrated.py',
    'utils/modular_components/implementations/migrated_registry.py',
    
    # Migration artifacts
    'utils/modular_components/implementations/patch_losses.py',
    'utils/modular_components/implementations/register_advanced.py',
    
    # Eventually delete entire layers/modular directory
    'layers/modular/',
    
    # Clean up scripts and temporary files
    'migrate_layers_to_utils.py',
    'complete_migration_script.py',
    'migration_demo.py',
    'test_migration_success.py',
    'test_direct_migration.py',
    'final_migration_demo.py',
    'quick_migration_demo.py',
    'clean_modular_autoformer.py',
    'demo_clean_modular_autoformer.py'
]

# =============================================================================
# TESTING FRAMEWORK
# =============================================================================

COMPONENT_TESTS = {
    'test_unified_components.py': {
        'purpose': 'Test all unified component registries',
        'components_to_test': ['attentions', 'losses', 'decomposition', 'encoders', 'decoders'],
        'tests': [
            'registry_loading',
            'component_creation', 
            'component_functionality',
            'error_handling'
        ]
    },
    'test_modular_autoformer_integration.py': {
        'purpose': 'Test ModularAutoformer with unified components',
        'components_to_test': ['all'],
        'tests': [
            'model_creation',
            'forward_pass',
            'loss_computation',
            'component_swapping'
        ]
    }
}

def print_consolidation_plan():
    """Print the complete consolidation plan"""
    print("🏗️  COMPONENT CONSOLIDATION PLAN")
    print("=" * 60)
    
    print("\n📋 PHASE 1: Component Consolidation")
    print("-" * 40)
    
    priority_groups = {}
    for comp_type, info in COMPONENT_TYPES.items():
        priority = info['priority']
        if priority not in priority_groups:
            priority_groups[priority] = []
        priority_groups[priority].append((comp_type, info))
    
    for priority in sorted(priority_groups.keys()):
        print(f"\n🎯 Priority {priority} Components:")
        for comp_type, info in priority_groups[priority]:
            print(f"   📦 {comp_type}: {len(info['source_files'])} files → {info['target_file']}")
    
    print(f"\n📋 PHASE 2: Master Registry")
    print("-" * 40)
    print("   📄 Create: utils/modular_components/implementations/unified_registry.py")
    print("   🔧 Integrate all unified component registries")
    
    print(f"\n📋 PHASE 3: Testing")
    print("-" * 40)
    for test_file, test_info in COMPONENT_TESTS.items():
        print(f"   🧪 {test_file}: {test_info['purpose']}")
    
    print(f"\n📋 PHASE 4: Model Updates")
    print("-" * 40)
    model_files = [
        'models/modular_autoformer.py',
        'models/Autoformer.py',
        'models/Autoformer_Fixed.py', 
        'models/EnhancedAutoformer.py',
        'models/EnhancedAutoformer_Fixed.py',
        'models/BayesianEnhancedAutoformer.py',
        'models/HierarchicalEnhancedAutoformer.py',
        'models/QuantileBayesianAutoformer.py'
    ]
    for model_file in model_files:
        print(f"   🔄 Update: {model_file}")
    
    print(f"\n📋 PHASE 5: Cleanup")
    print("-" * 40)
    print(f"   🗑️  Delete {len(FILES_TO_DELETE)} files/directories")
    print("   🧹 Remove layers/modular/ entirely")
    
    print(f"\n✅ EXPECTED OUTCOMES:")
    print(f"   📦 {len(COMPONENT_TYPES)} unified component files")
    print(f"   🏗️  1 master registry")
    print(f"   🧪 {len(COMPONENT_TESTS)} test suites")
    print(f"   🔄 8 updated model files")
    print(f"   🗑️  {len(FILES_TO_DELETE)} files cleaned up")

if __name__ == "__main__":
    print_consolidation_plan()
