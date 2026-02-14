# ============================================================
# 搬移 Claude Code + WSL2 至 D 槽
# 請以「系統管理員」身份在 PowerShell 中執行
#
# 執行前:
#   1. 關閉 Claude Code (Ctrl+C 退出)
#   2. 右鍵 Windows Terminal → 以系統管理員身份執行
#   3. 執行: powershell -ExecutionPolicy Bypass -File "D:\Projects\facedector\scripts\migrate_to_d.ps1"
#
# 搬移內容:
#   .claude (~231 MB) + .local (~695 MB) → D:\ClaudeCode
#   WSL2 Ubuntu-22.04 (~30 GB)           → D:\WSL
# ============================================================

$ErrorActionPreference = "Stop"

# ========================================
#  Step 1: 搬移 Claude Code (~926 MB)
#  先搬這個，釋放空間給 WSL 匯出用
# ========================================

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Step 1/2: 搬移 Claude Code"            -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# 檢查 Claude Code 是否仍在執行
$claudeProc = Get-Process -Name "claude" -ErrorAction SilentlyContinue
if ($claudeProc) {
    Write-Host "[ERROR] Claude Code 仍在執行，請先關閉再執行此腳本" -ForegroundColor Red
    Write-Host "        按 Ctrl+C 退出 Claude Code 後重試" -ForegroundColor Red
    exit 1
}

$claudeTargetDir = "D:\ClaudeCode"
if (!(Test-Path $claudeTargetDir)) {
    New-Item -ItemType Directory -Path $claudeTargetDir -Force | Out-Null
}

# --- .claude ---
$claudeSrc = "C:\Users\o991g\.claude"
$claudeDst = "$claudeTargetDir\.claude"

$item = Get-Item $claudeSrc -Force -ErrorAction SilentlyContinue
if ($item -and ($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint)) {
    Write-Host "[SKIP] .claude 已經是 symlink" -ForegroundColor Yellow
} elseif ($item) {
    Write-Host "[..] 搬移 .claude → $claudeDst" -ForegroundColor Yellow
    # 用 robocopy 搬移 (比 Move-Item 更穩定處理鎖定檔案)
    robocopy $claudeSrc $claudeDst /E /MOVE /NFL /NDL /NJH /NJS /R:3 /W:2 | Out-Null
    # 刪除殘留空目錄
    if (Test-Path $claudeSrc) { Remove-Item $claudeSrc -Recurse -Force -ErrorAction SilentlyContinue }
    # 建立 symlink
    New-Item -ItemType SymbolicLink -Path $claudeSrc -Target $claudeDst -Force | Out-Null
    Write-Host "[OK] .claude 搬移完成 + symlink 建立" -ForegroundColor Green
} else {
    Write-Host "[SKIP] .claude 不存在" -ForegroundColor Yellow
}

# --- .local ---
$localSrc = "C:\Users\o991g\.local"
$localDst = "$claudeTargetDir\.local"

$item = Get-Item $localSrc -Force -ErrorAction SilentlyContinue
if ($item -and ($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint)) {
    Write-Host "[SKIP] .local 已經是 symlink" -ForegroundColor Yellow
} elseif ($item) {
    Write-Host "[..] 搬移 .local → $localDst" -ForegroundColor Yellow
    robocopy $localSrc $localDst /E /MOVE /NFL /NDL /NJH /NJS /R:3 /W:2 | Out-Null
    if (Test-Path $localSrc) { Remove-Item $localSrc -Recurse -Force -ErrorAction SilentlyContinue }
    New-Item -ItemType SymbolicLink -Path $localSrc -Target $localDst -Force | Out-Null
    Write-Host "[OK] .local 搬移完成 + symlink 建立" -ForegroundColor Green
} else {
    Write-Host "[SKIP] .local 不存在" -ForegroundColor Yellow
}

# 驗證 symlink
Write-Host ""
Write-Host "[驗證] Symlink 狀態:" -ForegroundColor Yellow
Get-Item "C:\Users\o991g\.claude" -Force | Format-List Name, LinkTarget
Get-Item "C:\Users\o991g\.local" -Force | Format-List Name, LinkTarget

Write-Host "[OK] Claude Code 搬移完成，釋放 ~926 MB" -ForegroundColor Green

# ========================================
#  Step 2: 搬移 WSL2 (~30 GB)
# ========================================

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Step 2/2: 搬移 WSL2 Ubuntu-22.04"      -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

$wslDir = "D:\WSL"
$wslInstallDir = "$wslDir\Ubuntu-22.04"
$tarFile = "$wslDir\ubuntu-backup.tar"

if (!(Test-Path $wslDir)) {
    New-Item -ItemType Directory -Path $wslDir -Force | Out-Null
}

# 檢查 D 槽剩餘空間 (匯出需要 ~30 GB 暫存)
$dDrive = Get-PSDrive D -ErrorAction SilentlyContinue
if ($dDrive) {
    $freeGB = [math]::Round($dDrive.Free / 1GB, 1)
    Write-Host "[INFO] D 槽剩餘空間: $freeGB GB" -ForegroundColor Yellow
    if ($freeGB -lt 35) {
        Write-Host "[WARNING] D 槽空間可能不足 (需要 ~35 GB: 30 GB tar 暫存 + 30 GB 匯入後自動刪 tar)" -ForegroundColor Red
        Write-Host "          建議至少保留 65 GB 以確保安全" -ForegroundColor Red
        $confirm = Read-Host "是否繼續? (y/n)"
        if ($confirm -ne "y") { exit 0 }
    }
}

# 2-1. 關閉 WSL
Write-Host "[..] 關閉 WSL..." -ForegroundColor Yellow
wsl --shutdown
Start-Sleep -Seconds 3
Write-Host "[OK] WSL 已關閉" -ForegroundColor Green

# 2-2. 匯出 (寫到 D 槽，不佔 C 槽空間)
Write-Host "[..] 匯出 Ubuntu-22.04 → $tarFile" -ForegroundColor Yellow
Write-Host "     這需要幾分鐘，請耐心等候..." -ForegroundColor Yellow
wsl --export Ubuntu-22.04 $tarFile
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] WSL 匯出失敗" -ForegroundColor Red
    exit 1
}
$tarSizeGB = [math]::Round((Get-Item $tarFile).Length / 1GB, 2)
Write-Host "[OK] 匯出完成 ($tarSizeGB GB)" -ForegroundColor Green

# 2-3. 移除舊註冊 (釋放 C 槽 30 GB)
Write-Host "[..] 移除 C 槽的 WSL 註冊..." -ForegroundColor Yellow
wsl --unregister Ubuntu-22.04
Write-Host "[OK] C 槽已釋放 ~30 GB" -ForegroundColor Green

# 2-4. 匯入到 D 槽
Write-Host "[..] 匯入至 $wslInstallDir ..." -ForegroundColor Yellow
Write-Host "     這需要幾分鐘，請耐心等候..." -ForegroundColor Yellow
wsl --import Ubuntu-22.04 $wslInstallDir $tarFile
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] WSL 匯入失敗，tar 檔保留在: $tarFile" -ForegroundColor Red
    exit 1
}
Write-Host "[OK] 匯入完成" -ForegroundColor Green

# 2-5. 設定預設使用者
Write-Host "[..] 設定預設使用者 jao0130..." -ForegroundColor Yellow
wsl -d Ubuntu-22.04 -- sh -c "echo '[user]' > /etc/wsl.conf && echo 'default=jao0130' >> /etc/wsl.conf"
wsl --shutdown
Start-Sleep -Seconds 2

# 2-6. 驗證
Write-Host "[..] 驗證..." -ForegroundColor Yellow
$whoami = wsl -d Ubuntu-22.04 -- whoami
Write-Host "     whoami = $whoami" -ForegroundColor Yellow

if ($whoami.Trim() -eq "jao0130") {
    Write-Host "[OK] 使用者驗證通過" -ForegroundColor Green

    # 2-7. 刪除 tar 暫存檔
    Write-Host "[..] 刪除暫存 tar 檔..." -ForegroundColor Yellow
    Remove-Item $tarFile -Force
    Write-Host "[OK] 暫存檔已刪除" -ForegroundColor Green
} else {
    Write-Host "[WARNING] 使用者不是 jao0130，tar 檔保留: $tarFile" -ForegroundColor Red
    Write-Host "          請手動確認後刪除 tar 檔" -ForegroundColor Red
}

wsl --shutdown

# ========================================
#  完成摘要
# ========================================

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  搬移完成！"                              -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "  D:\ClaudeCode\.claude  ← symlink C:\Users\o991g\.claude"
Write-Host "  D:\ClaudeCode\.local   ← symlink C:\Users\o991g\.local"
Write-Host "  D:\WSL\Ubuntu-22.04   ← WSL2 虛擬磁碟"
Write-Host ""
Write-Host "  C 槽釋放: ~31 GB" -ForegroundColor Green
Write-Host ""
Write-Host "使用方式 (與搬移前完全相同):" -ForegroundColor Yellow
Write-Host "  wsl -d Ubuntu-22.04     # 啟動 WSL"
Write-Host "  claude                   # 啟動 Claude Code"
Write-Host ""
