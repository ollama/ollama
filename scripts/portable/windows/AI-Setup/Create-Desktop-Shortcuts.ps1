Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$usbRoot = $PSScriptRoot
$desktop = [Environment]::GetFolderPath('Desktop')
$wsh = New-Object -ComObject WScript.Shell

$shortcutPath = Join-Path $desktop 'Launch-AI.lnk'
$shortcut = $wsh.CreateShortcut($shortcutPath)
$shortcut.TargetPath = Join-Path $usbRoot 'Launch-AI.cmd'
$shortcut.WorkingDirectory = $usbRoot
# shell32.dll,220: general app icon
$shortcut.IconLocation = "$env:SystemRoot\System32\shell32.dll,220"
$shortcut.Description = '從隨身碟啟動 AI 工作流'
$shortcut.Save()

$firstTimePath = Join-Path $desktop 'Launch-AI-FirstTime.lnk'
$firstTime = $wsh.CreateShortcut($firstTimePath)
$firstTime.TargetPath = Join-Path $usbRoot 'Launch-AI-FirstTime.cmd'
$firstTime.WorkingDirectory = $usbRoot
# shell32.dll,221: setup-oriented icon for first-time initialization
$firstTime.IconLocation = "$env:SystemRoot\System32\shell32.dll,221"
$firstTime.Description = '首次初始化並啟動 AI 工作流'
$firstTime.Save()

Write-Host '桌面捷徑建立完成。'
