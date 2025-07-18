#!/usr/bin/env python3
"""
Fix Unicode issues in test files by replacing emoji with plain text
"""

import os
import re
import glob

def fix_unicode_in_file(filepath):
    """Remove Unicode characters and replace with plain ASCII"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Replace common Unicode symbols with plain text
        replacements = {
            '🚀': 'ROCKET',
            '🔍': 'SEARCH',
            '🧪': 'TEST',
            '📊': 'CHART',
            '✅': 'PASS',
            '❌': 'FAIL',
            '⚠️': 'WARN',
            '🔧': 'TOOL',
            '🔮': 'CRYSTAL',
            '🎲': 'DICE',
            '⚡': 'LIGHTNING',
            '🧠': 'BRAIN',
            '🎉': 'PARTY',
            '💡': 'IDEA',
            '🎯': 'TARGET',
            '🔬': 'MICROSCOPE',
            '📋': 'CLIPBOARD',
            '📄': 'PAGE',
            '⏱️': 'TIMER',
            '🔄': 'REFRESH',
            '📈': 'GRAPH',
            '🌊': 'WAVE',
            '💎': 'DIAMOND',
            '🎪': 'CIRCUS',
            '🔥': 'FIRE',
            '⭐': 'STAR',
            '🎭': 'MASK',
            '🎨': 'ART',
            '🏆': 'TROPHY',
            '🎪': 'CIRCUS',
            '🌟': 'SPARKLE',
        }
        
        # Replace all Unicode symbols
        for unicode_char, replacement in replacements.items():
            content = content.replace(unicode_char, replacement)
        
        # Remove any remaining Unicode characters by encoding/decoding
        content = content.encode('ascii', 'ignore').decode('ascii')
        
        # Write back the cleaned content
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"Fixed Unicode in: {filepath}")
        return True
        
    except Exception as e:
        print(f"Error fixing {filepath}: {e}")
        return False

def main():
    """Fix Unicode in all test files"""
    print("Fixing Unicode issues in test files...")
    
    # Find all Python test files
    test_patterns = [
        'tests/**/*.py',
        'test_*.py',
        '*_test.py'
    ]
    
    files_fixed = 0
    total_files = 0
    
    for pattern in test_patterns:
        for filepath in glob.glob(pattern, recursive=True):
            total_files += 1
            if fix_unicode_in_file(filepath):
                files_fixed += 1
    
    print(f"\nCompleted: {files_fixed}/{total_files} files processed")
    print("All Unicode symbols replaced with plain ASCII text")

if __name__ == "__main__":
    main()
