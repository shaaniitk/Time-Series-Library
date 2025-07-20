#!/usr/bin/env pwsh
# VS Code Performance Optimization Script for Time-Series-Library

Write-Host "🚀 VS Code Performance Optimization" -ForegroundColor Green
Write-Host "=" * 50

# 1. Clean up Python cache files
Write-Host "🧹 Cleaning Python cache files..." -ForegroundColor Yellow
Get-ChildItem -Path . -Recurse -Directory -Name "__pycache__" | ForEach-Object {
    $path = Join-Path -Path $pwd -ChildPath $_
    if (Test-Path $path) {
        Remove-Item -Path $path -Recurse -Force -ErrorAction SilentlyContinue
        Write-Host "   Removed: $_" -ForegroundColor Gray
    }
}

Get-ChildItem -Path . -Recurse -File -Name "*.pyc" | ForEach-Object {
    Remove-Item -Path $_ -Force -ErrorAction SilentlyContinue
}

# 2. Clean up pytest cache
Write-Host "🧹 Cleaning pytest cache..." -ForegroundColor Yellow
Get-ChildItem -Path . -Recurse -Directory -Name ".pytest_cache" | ForEach-Object {
    $path = Join-Path -Path $pwd -ChildPath $_
    if (Test-Path $path) {
        Remove-Item -Path $path -Recurse -Force -ErrorAction SilentlyContinue
        Write-Host "   Removed: $_" -ForegroundColor Gray
    }
}

# 3. Check sizes of major directories
Write-Host "📊 Directory size analysis..." -ForegroundColor Yellow
$dirs = @("checkpoints", "logs", "test_results", "trained", "tsl-env", "data")
foreach ($dir in $dirs) {
    if (Test-Path $dir) {
        $size = (Get-ChildItem -Path $dir -Recurse -File | Measure-Object -Property Length -Sum).Sum
        $sizeMB = [math]::Round($size / 1MB, 2)
        Write-Host "   $dir`: ${sizeMB} MB" -ForegroundColor Cyan
    }
}

# 4. Create .gitignore additions for performance
Write-Host "📝 Updating .gitignore for performance..." -ForegroundColor Yellow

$gitignoreAdditions = @"

# Performance optimization - exclude from VS Code indexing
**/__pycache__/
*.pyc
*.pyo
*.pyd
.Python
.pytest_cache/
.coverage
.tox/
.cache
.mypy_cache/
.dmypy.json
dmypy.json

# Large directories that slow down VS Code
checkpoints/
logs/
test_results/
trained/
tsl-env/
data/
*.zip
pic/

# Temporary files
*.tmp
*.temp
*~
.DS_Store
Thumbs.db
"@

Add-Content -Path ".gitignore" -Value $gitignoreAdditions -ErrorAction SilentlyContinue

# 5. Recommendations
Write-Host "💡 Additional Recommendations:" -ForegroundColor Green
Write-Host "   1. Restart VS Code to apply settings changes" -ForegroundColor White
Write-Host "   2. Consider disabling these extensions temporarily:" -ForegroundColor White
Write-Host "      - Amazon Q (if not actively using)" -ForegroundColor Gray
Write-Host "      - GitLens (reduce git operations)" -ForegroundColor Gray
Write-Host "      - Jupyter extensions (if not using notebooks now)" -ForegroundColor Gray
Write-Host "   3. Close unused tabs and workspace windows" -ForegroundColor White
Write-Host "   4. Use Ctrl+Shift+P > 'Developer: Reload Window' for quick restart" -ForegroundColor White

# 6. VS Code workspace settings summary
Write-Host "⚙️  Applied VS Code Optimizations:" -ForegroundColor Green
Write-Host "   ✓ Excluded large directories from file watching" -ForegroundColor Gray
Write-Host "   ✓ Disabled Python analysis indexing" -ForegroundColor Gray
Write-Host "   ✓ Reduced Jupyter completion extensions" -ForegroundColor Gray
Write-Host "   ✓ Disabled git auto-refresh" -ForegroundColor Gray
Write-Host "   ✓ Disabled semantic highlighting" -ForegroundColor Gray
Write-Host "   ✓ Disabled extension auto-updates" -ForegroundColor Gray

Write-Host "🎉 Optimization Complete!" -ForegroundColor Green
Write-Host "   VS Code should be significantly faster now." -ForegroundColor White
Write-Host "   Please restart VS Code to see full effects." -ForegroundColor Yellow
