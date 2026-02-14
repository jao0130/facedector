# Check WSL2 vhdx
Write-Host "=== WSL2 vhdx ==="
Get-ChildItem "C:\Users\o991g\AppData\Local\Packages\" -Recurse -Filter "ext4.vhdx" -ErrorAction SilentlyContinue | ForEach-Object {
    $sizeGB = [math]::Round($_.Length / 1GB, 2)
    Write-Host "$($_.FullName) -> $sizeGB GB"
}

# Check .claude folder
Write-Host ""
Write-Host "=== .claude config ==="
if (Test-Path "C:\Users\o991g\.claude") {
    $size = (Get-ChildItem "C:\Users\o991g\.claude" -Recurse -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
    $sizeMB = [math]::Round($size / 1MB)
    Write-Host "C:\Users\o991g\.claude -> $sizeMB MB"
}

# Check Claude Code location
Write-Host ""
Write-Host "=== Claude Code ==="
$claude = Get-Command claude -ErrorAction SilentlyContinue
if ($claude) {
    Write-Host "Location: $($claude.Source)"
}

# Check npm global root
Write-Host ""
Write-Host "=== npm global ==="
$npmRoot = npm root -g 2>$null
Write-Host "npm root: $npmRoot"
if (Test-Path $npmRoot) {
    $size = (Get-ChildItem $npmRoot -Recurse -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
    $sizeMB = [math]::Round($size / 1MB)
    Write-Host "npm global size: $sizeMB MB"
}

# Check WSL distro list
Write-Host ""
Write-Host "=== WSL distros ==="
wsl --list --verbose
