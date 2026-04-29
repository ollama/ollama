<#
.SYNOPSIS
    Uninstall Ollama from Windows.

.DESCRIPTION
    Safely removes Ollama from a Windows system by:
    1. Stopping running Ollama processes
    2. Invoking the official Inno Setup uninstaller
    3. Cleaning up orphaned application folders
    4. Optionally removing downloaded model data

    Requires Administrator privileges.

.PARAMETER DryRun
    Preview what would be deleted without actually deleting anything.

.EXAMPLE
    .\uninstall-windows.ps1

.EXAMPLE
    .\uninstall-windows.ps1 -DryRun
#>

param(
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

function Write-Info    { param([string]$Message) Write-Host "[INFO] $Message" -ForegroundColor Green }
function Write-Warn    { param([string]$Message) Write-Host "[WARN] $Message" -ForegroundColor Yellow }
function Write-Err     { param([string]$Message) Write-Host "[ERROR] $Message" -ForegroundColor Red }
function Write-Dry     { param([string]$Message) Write-Host "[DRY-RUN] Would: $Message" -ForegroundColor Cyan }

function Remove-Item-Safe {
    param(
        [string]$Path,
        [string]$Label
    )
    if (Test-Path $Path) {
        if ($DryRun) {
            Write-Dry "Remove $Label`: $Path"
        } else {
            Write-Info "Removing $Label`: $Path"
            Remove-Item -Recurse -Force -Path $Path -ErrorAction SilentlyContinue
        }
    } else {
        Write-Info "Not found (skipping): $Path"
    }
}

# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

Write-Host "========================================="
Write-Host "  Ollama Uninstaller for Windows" -ForegroundColor Cyan
Write-Host "========================================="
Write-Host ""

if ($DryRun) {
    Write-Warn "DRY RUN MODE - No changes will be made"
    Write-Host ""
}

# Check for Administrator privileges
if (-not ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Err "This script requires Administrator privileges. Please run PowerShell as Administrator."
    exit 1
}

# Step 1: Stop running Ollama processes
Write-Info "Step 1: Stopping Ollama processes..."
$ollamaProcesses = Get-Process -Name "ollama", "ollama app" -ErrorAction SilentlyContinue
if ($ollamaProcesses) {
    if ($DryRun) {
        Write-Dry "Stop processes: $($ollamaProcesses.Name -join ', ')"
    } else {
        $ollamaProcesses | Stop-Process -Force -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 1
        Write-Info "Ollama processes stopped."
    }
} else {
    Write-Info "No running Ollama processes found."
}
Write-Host ""

# Step 2: Find and run official uninstaller
Write-Info "Step 2: Running official uninstaller..."

$InnoSetupUninstallGuid = "{44E83376-CE68-45EB-8FC1-393500EB558C}_is1"
$possibleKeys = @(
    "HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\$InnoSetupUninstallGuid",
    "HKLM:\Software\Microsoft\Windows\CurrentVersion\Uninstall\$InnoSetupUninstallGuid",
    "HKLM:\Software\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\$InnoSetupUninstallGuid"
)

$regKey = $null
foreach ($key in $possibleKeys) {
    if (Test-Path $key) {
        $regKey = $key
        break
    }
}

if ($regKey) {
    $uninstallString = (Get-ItemProperty -Path $regKey).UninstallString
    if ($uninstallString) {
        $uninstallExe = $uninstallString -replace '"', ''
        Write-Info "Found uninstaller: $uninstallExe"
        if ($DryRun) {
            Write-Dry "Run uninstaller: $uninstallExe /VERYSILENT /SUPPRESSMSGBOXES /NORESTART"
        } else {
            if (Test-Path $uninstallExe) {
                Start-Process -FilePath $uninstallExe -ArgumentList "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART" -Wait -ErrorAction SilentlyContinue
                Write-Info "Official uninstaller completed."
            } else {
                Write-Warn "Uninstaller executable not found at: $uninstallExe"
            }
        }
    } else {
        Write-Warn "No uninstall string found in registry."
    }
} else {
    # Fallback: try default location
    $defaultUninstaller = "$env:LOCALAPPDATA\Programs\Ollama\unins000.exe"
    if (Test-Path $defaultUninstaller) {
        Write-Info "Found uninstaller at default location: $defaultUninstaller"
        if ($DryRun) {
            Write-Dry "Run uninstaller: $defaultUninstaller /VERYSILENT /SUPPRESSMSGBOXES /NORESTART"
        } else {
            Start-Process -FilePath $defaultUninstaller -ArgumentList "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART" -Wait -ErrorAction SilentlyContinue
            Write-Info "Official uninstaller completed."
        }
    } else {
        Write-Warn "Ollama uninstaller not found. Proceeding with manual cleanup."
    }
}
Write-Host ""

# Step 3: Remove orphaned application folders
Write-Info "Step 3: Cleaning up application folders..."
Remove-Item-Safe -Path "$env:LOCALAPPDATA\Programs\Ollama" -Label "Program folder"
Remove-Item-Safe -Path "$env:LOCALAPPDATA\Ollama" -Label "AppData folder"
Write-Host ""

# Step 4: Ask about model data
Write-Info "Step 4: Model data..."
$modelPath = Join-Path $env:USERPROFILE ".ollama"

if (Test-Path $modelPath) {
    Write-Info "Model directory found at: $modelPath"
    if ($DryRun) {
        Write-Dry "Remove model directory: $modelPath"
    } else {
        $wipeModels = Read-Host "Do you want to remove downloaded models and cache? [y/N]"
        if ($wipeModels -match "^[Yy]") {
            Remove-Item-Safe -Path $modelPath -Label "Model directory"
            Write-Info "Model data removed."
        } else {
            Write-Info "Model data retained at $modelPath"
        }
    }
} else {
    Write-Info "No model directory found at $modelPath"
}
Write-Host ""

# Summary
Write-Host "========================================="
if ($DryRun) {
    Write-Info "Dry run complete. No changes were made."
    Write-Info "Run without -DryRun to perform the actual uninstall."
} else {
    Write-Info "Ollama has been uninstalled from your system."
    if (Test-Path $modelPath) {
        Write-Warn "Model data still exists at $modelPath"
        Write-Info "To remove it manually: Remove-Item -Recurse -Force `"$modelPath`""
    }
}
Write-Host "========================================="
