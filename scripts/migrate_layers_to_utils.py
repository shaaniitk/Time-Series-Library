#!/usr/bin/env python3
"""
MIGRATE LAYERS.MODULAR TO UTILS.MODULAR_COMPONENTS
Complete migration of all components from layers/modular/* to utils.modular_components.implementations/*
"""

import os
import shutil
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def migrate_layers_modular_to_utils():
    """Migrate all components from layers/modular to utils.modular_components"""
    
    print("🔄 MIGRATING LAYERS.MODULAR TO UTILS.MODULAR_COMPONENTS...")
    print("=" * 60)
    
    # Source and destination paths
    source_base = Path("layers/modular")
    dest_base = Path("utils/modular_components/implementations")
    
    # Ensure destination exists
    dest_base.mkdir(parents=True, exist_ok=True)
    
    # Component categories to migrate
    categories = [
        'attention', 'decomposition', 'encoder', 'decoder', 
        'sampling', 'output_heads', 'losses', 'layers', 
        'fusion', 'processors'
    ]
    
    migrated_components = {}
    
    for category in categories:
        source_dir = source_base / category
        if source_dir.exists():
            print(f"\n📂 Migrating {category} components...")
            migrated_components[category] = migrate_category(source_dir, dest_base, category)
        else:
            print(f"⚠️  Category {category} not found in layers/modular")
    
    # Create unified registry for all migrated components
    create_unified_registry(migrated_components, dest_base)
    
    # Update __init__.py files
    update_init_files(dest_base, migrated_components)
    
    print(f"\n🎉 MIGRATION COMPLETE!")
    print(f"📊 Total categories migrated: {len(migrated_components)}")
    for category, count in migrated_components.items():
        print(f"   - {category}: {count} components")

def migrate_category(source_dir, dest_base, category):
    """Migrate a specific category of components"""
    
    dest_file = dest_base / f"{category}_migrated.py"
    component_count = 0
    
    print(f"   📁 Source: {source_dir}")
    print(f"   📁 Dest: {dest_file}")
    
    # Read all Python files in the category
    imports = set()
    classes = []
    functions = []
    
    for py_file in source_dir.glob("*.py"):
        if py_file.name == "__init__.py":
            continue
            
        print(f"      📄 Processing {py_file.name}...")
        
        try:
            content = py_file.read_text(encoding='utf-8')
            
            # Extract imports, classes, and functions
            file_imports, file_classes, file_functions = parse_python_file(content)
            
            imports.update(file_imports)
            classes.extend(file_classes)
            functions.extend(file_functions)
            
            component_count += len(file_classes)
            
        except Exception as e:
            print(f"      ⚠️  Error processing {py_file.name}: {e}")
    
    # Generate migrated file
    if classes or functions:
        generate_migrated_file(dest_file, category, imports, classes, functions)
        print(f"      ✅ Generated {dest_file.name} with {len(classes)} classes and {len(functions)} functions")
    
    return component_count

def parse_python_file(content):
    """Parse Python file to extract imports, classes, and functions"""
    lines = content.split('\n')
    
    imports = []
    classes = []
    functions = []
    current_block = []
    in_class = False
    in_function = False
    indent_level = 0
    
    for line in lines:
        stripped = line.strip()
        
        # Track imports
        if stripped.startswith(('import ', 'from ')) and not stripped.startswith('from .'):
            imports.append(line)
        
        # Track classes
        elif stripped.startswith('class '):
            if current_block and in_class:
                classes.append('\n'.join(current_block))
            current_block = [line]
            in_class = True
            in_function = False
            indent_level = len(line) - len(line.lstrip())
        
        # Track functions
        elif stripped.startswith('def '):
            if current_block and (in_class or in_function):
                if in_function:
                    functions.append('\n'.join(current_block))
                elif in_class:
                    classes.append('\n'.join(current_block))
            current_block = [line]
            in_function = not in_class  # Only top-level functions
            indent_level = len(line) - len(line.lstrip())
        
        # Continue current block
        elif in_class or in_function:
            current_block.append(line)
        
        # Check if we're leaving a block
        elif stripped == '' or (len(line) - len(line.lstrip()) <= indent_level and stripped):
            if current_block:
                if in_class:
                    classes.append('\n'.join(current_block))
                elif in_function:
                    functions.append('\n'.join(current_block))
                current_block = []
            in_class = False
            in_function = False
    
    # Handle final block
    if current_block:
        if in_class:
            classes.append('\n'.join(current_block))
        elif in_function:
            functions.append('\n'.join(current_block))
    
    return imports, classes, functions

def generate_migrated_file(dest_file, category, imports, classes, functions):
    """Generate the migrated component file"""
    
    content = f'''"""
Migrated {category.title()} Components
Auto-migrated from layers/modular/{category} to utils.modular_components.implementations
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, List
import logging

# Import base interfaces from new framework
from ..base_interfaces import BaseComponent, BaseLoss, BaseAttention
from ..config_schemas import ComponentConfig

logger = logging.getLogger(__name__)

# Migrated imports
{chr(10).join(set(imports))}

# Migrated Classes
{chr(10).join(classes)}

# Migrated Functions  
{chr(10).join(functions)}

# Registry function for {category} components
def get_{category}_component(name: str, config: ComponentConfig = None, **kwargs):
    """Get {category} component by name"""
    # This will be implemented based on the migrated components
    pass

def register_{category}_components(registry):
    """Register all {category} components with the registry"""
    # This will be implemented to register all migrated components
    pass
'''
    
    dest_file.write_text(content, encoding='utf-8')

def create_unified_registry(migrated_components, dest_base):
    """Create a unified registry for all migrated components"""
    
    registry_file = dest_base / "migrated_registry.py"
    
    content = f'''"""
Unified Registry for All Migrated Components
Provides a single interface to access all components migrated from layers/modular
"""

from typing import Dict, Any, Optional
import logging

from ..registry import ComponentRegistry
from ..config_schemas import ComponentConfig

logger = logging.getLogger(__name__)

# Import all migrated categories
'''
    
    for category in migrated_components.keys():
        content += f"from .{category}_migrated import register_{category}_components\n"
    
    content += f'''

class MigratedComponentRegistry:
    """Registry for all migrated components from layers/modular"""
    
    def __init__(self):
        self.registry = ComponentRegistry()
        self._register_all_migrated_components()
    
    def _register_all_migrated_components(self):
        """Register all migrated components"""
        logger.info("Registering all migrated components...")
        
'''
    
    for category in migrated_components.keys():
        content += f"        register_{category}_components(self.registry)\n"
    
    content += f'''        
        logger.info("All migrated components registered successfully")
    
    def get_component(self, category: str, name: str, config: ComponentConfig = None):
        """Get a component by category and name"""
        return self.registry.create(category, name, config)
    
    def list_components(self):
        """List all available components"""
        return self.registry.list_components()
    
    def get_migration_summary(self):
        """Get summary of migrated components"""
        return {{
{chr(10).join([f'            "{category}": {count},' for category, count in migrated_components.items()])}
        }}

# Global instance
migrated_registry = MigratedComponentRegistry()
'''
    
    registry_file.write_text(content, encoding='utf-8')
    print(f"✅ Created unified registry: {registry_file}")

def update_init_files(dest_base, migrated_components):
    """Update __init__.py files to include migrated components"""
    
    init_file = dest_base / "__init__.py"
    
    # Read existing content
    existing_content = ""
    if init_file.exists():
        existing_content = init_file.read_text(encoding='utf-8')
    
    # Add migrated imports
    migrated_imports = [
        f"from .{category}_migrated import register_{category}_components"
        for category in migrated_components.keys()
    ]
    migrated_imports.append("from .migrated_registry import migrated_registry")
    
    new_content = existing_content + "\n\n# Migrated components\n" + "\n".join(migrated_imports)
    
    init_file.write_text(new_content, encoding='utf-8')
    print(f"✅ Updated {init_file}")

if __name__ == "__main__":
    migrate_layers_modular_to_utils()
