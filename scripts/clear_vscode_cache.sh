#!/bin/bash
# VS Code Cache Cleaner Script
# This script safely clears VS Code cache and temporary files

echo "🧹 VS Code Cache Cleaner"
echo "========================"

# Check if VS Code is running
if pgrep -f "Visual Studio Code" > /dev/null; then
    echo "⚠️  VS Code is currently running."
    echo "Please save your work and close VS Code completely before running this script."
    echo ""
    echo "To close VS Code:"
    echo "1. Save all files (Cmd+S)"
    echo "2. Close all windows (Cmd+W)"
    echo "3. Quit VS Code completely (Cmd+Q)"
    echo ""
    read -p "Press Enter when VS Code is closed, or Ctrl+C to cancel..."
    
    # Check again
    if pgrep -f "Visual Studio Code" > /dev/null; then
        echo "❌ VS Code is still running. Exiting."
        exit 1
    fi
fi

echo "✅ VS Code is not running. Proceeding with cache cleanup..."
echo ""

# Define cache directories
CODE_SUPPORT_DIR="$HOME/Library/Application Support/Code"
CACHE_DIRS=(
    "Cache"
    "CachedData" 
    "Code Cache"
    "GPUCache"
    "DawnGraphiteCache"
    "DawnWebGPUCache"
    "WebStorage"
    "Service Worker"
    "logs"
)

# Calculate total cache size before cleanup
echo "📊 Calculating cache sizes..."
total_size=0
for dir in "${CACHE_DIRS[@]}"; do
    cache_path="$CODE_SUPPORT_DIR/$dir"
    if [ -d "$cache_path" ]; then
        size=$(du -sk "$cache_path" 2>/dev/null | cut -f1)
        total_size=$((total_size + size))
        echo "   $dir: $(du -sh "$cache_path" 2>/dev/null | cut -f1)"
    fi
done

echo ""
echo "💾 Total cache size: $(echo $total_size | awk '{printf "%.1f MB", $1/1024}')"
echo ""

read -p "🗑️  Proceed with cache cleanup? (y/N): " confirm
if [[ $confirm != [yY] && $confirm != [yY][eE][sS] ]]; then
    echo "❌ Cache cleanup cancelled."
    exit 0
fi

echo ""
echo "🧹 Cleaning VS Code cache..."

# Clean each cache directory
cleaned_size=0
for dir in "${CACHE_DIRS[@]}"; do
    cache_path="$CODE_SUPPORT_DIR/$dir"
    if [ -d "$cache_path" ]; then
        echo "   Cleaning $dir..."
        size=$(du -sk "$cache_path" 2>/dev/null | cut -f1)
        rm -rf "$cache_path"
        cleaned_size=$((cleaned_size + size))
        echo "   ✅ Removed $dir ($(echo $size | awk '{printf "%.1f MB", $1/1024}'))"
    fi
done

echo ""
echo "✨ Cache cleanup completed!"
echo "🗑️  Total space freed: $(echo $cleaned_size | awk '{printf "%.1f MB", $1/1024}')"
echo ""
echo "📝 What was cleaned:"
echo "   • Browser cache and temporary files"
echo "   • Extension cache data"
echo "   • GPU cache files"
echo "   • Log files"
echo "   • Web storage data"
echo ""
echo "✅ VS Code will rebuild cache on next startup."
echo "🚀 Performance should be improved!"
