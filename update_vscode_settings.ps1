# PowerShell script to update VS Code settings for MCP auto-approval

$settingsPath = "$env:APPDATA\Code\User\settings.json"

# Read current settings
if (Test-Path $settingsPath) {
    $content = Get-Content $settingsPath -Raw
    $settings = $content | ConvertFrom-Json
} else {
    $settings = @{}
}

# Add MCP auto-approval settings
$settings | Add-Member -MemberType NoteProperty -Name "mcp.autoApprove" -Value $true -Force
$settings | Add-Member -MemberType NoteProperty -Name "mcp.trustedServers" -Value @("desktop-commander", "zen", "serena", "memento", "sequentialthinking") -Force
$settings | Add-Member -MemberType NoteProperty -Name "mcp.permissions.autoGrant" -Value @("read", "write", "execute", "search", "files", "terminal") -Force
$settings | Add-Member -MemberType NoteProperty -Name "mcp.timeout" -Value 30000 -Force
$settings | Add-Member -MemberType NoteProperty -Name "mcp.enableLogging" -Value $true -Force

# Convert back to JSON and save
$jsonOutput = $settings | ConvertTo-Json -Depth 10
$jsonOutput | Set-Content $settingsPath -Encoding UTF8

Write-Host "VS Code settings updated successfully!"
Write-Host "MCP auto-approval enabled for trusted servers."
Write-Host "Settings saved to: $settingsPath"
