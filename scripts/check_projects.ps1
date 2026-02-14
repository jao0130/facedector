$exclude = @(
    'AppData', 'anaconda3', 'OneDrive', 'Links', 'Searches',
    'Favorites', 'Contacts', 'Saved Games', '3D Objects', 'Music', 'Pictures',
    'Videos', 'Documents', 'Desktop', 'Downloads', 'Templates', 'SendTo',
    'Recent', 'PrintHood', 'NetHood', 'IntelGraphicsProfiles',
    'MicrosoftEdgeBackups', 'source', 'facedector-env', 'pip_cache', 'databases'
)

Write-Host "=== Project folders under C:\Users\o991g ===" -ForegroundColor Cyan
Write-Host ""

$total = 0
$results = @()

Get-ChildItem "C:\Users\o991g" -Directory -Force -ErrorAction SilentlyContinue |
    Where-Object { $_.Name[0] -ne '.' -and $_.Name -notin $exclude } |
    ForEach-Object {
        $size = (Get-ChildItem $_.FullName -Recurse -Force -ErrorAction SilentlyContinue |
            Measure-Object -Property Length -Sum).Sum
        $mb = [math]::Round($size / 1MB)
        $total += $mb
        Write-Host ("{0,-30} {1,8} MB" -f $_.Name, $mb)
    }

Write-Host ""
Write-Host ("Total: {0} MB" -f $total) -ForegroundColor Green
