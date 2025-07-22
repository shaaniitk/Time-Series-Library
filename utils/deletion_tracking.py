"""
FILES TO DELETE - CLEANUP TRACKING
List of duplicate and migrated files that should be deleted after consolidation
"""

# =============================================================================
# LAYERS/MODULAR DIRECTORY - ENTIRE DIRECTORY TO DELETE
# =============================================================================
LAYERS_MODULAR_DIR = "layers/modular/"

# All files in layers/modular/ should be deleted as they've been migrated to utils/modular_components

# =============================================================================
# DUPLICATE COMPONENT FILES IN UTILS/MODULAR_COMPONENTS/IMPLEMENTATIONS
# =============================================================================

DUPLICATE_FILES_TO_DELETE = [
    # Attention duplicates (keeping attentions_unified.py)
    "utils/modular_components/implementations/attentions.py",  # Has mixed content
    "utils/modular_components/implementations/attention_clean.py",  # If exists
    "utils/modular_components/implementations/attention_migrated.py",  # Migrated file
    "utils/modular_components/implementations/advanced_attentions.py",  # If exists
    
    # Loss duplicates (keeping losses_unified.py) 
    "utils/modular_components/implementations/losses.py",  # Original
    "utils/modular_components/implementations/advanced_losses.py",  # Advanced version
    "utils/modular_components/implementations/losses_migrated.py",  # Migrated file
    
    # Decomposition duplicates
    "utils/modular_components/implementations/decomposition_migrated.py",  # Migrated
    
    # Encoder duplicates
    "utils/modular_components/implementations/encoder_migrated.py",  # Migrated
    
    # Decoder duplicates  
    "utils/modular_components/implementations/decoder_migrated.py",  # Migrated
    
    # Sampling duplicates
    "utils/modular_components/implementations/sampling_migrated.py",  # Migrated
    
    # Output heads duplicates
    "utils/modular_components/implementations/output_heads_migrated.py",  # Migrated
    
    # Layers duplicates
    "utils/modular_components/implementations/layers_migrated.py",  # Migrated
    
    # Fusion duplicates
    "utils/modular_components/implementations/fusion_migrated.py",  # Migrated
]

# =============================================================================
# UNIFIED FILES TO KEEP
# =============================================================================

UNIFIED_FILES_TO_KEEP = [
    "utils/modular_components/implementations/attentions_unified.py",
    "utils/modular_components/implementations/attentions_clean.py",  # Clean import wrapper
    "utils/modular_components/implementations/losses_unified.py", 
    "utils/modular_components/implementations/migrated_registry.py",  # Unified registry
    # Will add more unified files as we create them
]

# =============================================================================
# CLEANUP PLAN
# =============================================================================

CLEANUP_PLAN = """
Phase 1: Create all unified component files
- ✅ attentions_unified.py (DONE)
- ✅ losses_unified.py (DONE)
- ⏳ decomposition_unified.py (TODO)
- ⏳ encoder_unified.py (TODO)
- ⏳ decoder_unified.py (TODO)
- ⏳ sampling_unified.py (TODO)
- ⏳ output_heads_unified.py (TODO)
- ⏳ layers_unified.py (TODO)
- ⏳ fusion_unified.py (TODO)

Phase 2: Update all imports to use unified files
- ⏳ Update clean_modular_autoformer.py
- ⏳ Update demo files
- ⏳ Update test files

Phase 3: Delete duplicate files
- ⏳ Delete layers/modular/ directory
- ⏳ Delete all *_migrated.py files
- ⏳ Delete duplicate component files

Phase 4: Verify everything works
- ⏳ Run all tests
- ⏳ Run demo files
- ⏳ Verify no import errors
"""

def get_files_to_delete():
    """Get comprehensive list of files to delete"""
    return DUPLICATE_FILES_TO_DELETE

def get_cleanup_plan():
    """Get the cleanup plan"""
    return CLEANUP_PLAN

def log_deletion_progress():
    """Log current deletion progress"""
    print("=== DELETION TRACKING ===")
    print(f"Files marked for deletion: {len(DUPLICATE_FILES_TO_DELETE)}")
    print(f"Unified files created: {len(UNIFIED_FILES_TO_KEEP)}")
    print("\nNext steps:")
    print("1. Create remaining unified component files")
    print("2. Update imports in all consuming files") 
    print("3. Delete duplicate files")
    print("4. Delete layers/modular/ directory")

if __name__ == "__main__":
    log_deletion_progress()
