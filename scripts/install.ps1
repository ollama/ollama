<#
.SYNOPSIS
    Install, upgrade, or uninstall Ollama on Windows.

.DESCRIPTION
    Downloads and installs Ollama.

    Quick install:

        irm https://ollama.com/install.ps1 | iex

    Specific version:

        $env:OLLAMA_VERSION="0.5.7"; irm https://ollama.com/install.ps1 | iex

    Custom install directory:

        $env:OLLAMA_INSTALL_DIR="D:\Ollama"; irm https://ollama.com/install.ps1 | iex

    Uninstall:

        $env:OLLAMA_UNINSTALL=1; irm https://ollama.com/install.ps1 | iex

    Environment variables:

        OLLAMA_VERSION       Target version (default: latest stable)
        OLLAMA_INSTALL_DIR   Custom install directory
        OLLAMA_UNINSTALL     Set to 1 to uninstall Ollama
        OLLAMA_DEBUG         Enable verbose output

.EXAMPLE
    irm https://ollama.com/install.ps1 | iex

.EXAMPLE
    $env:OLLAMA_VERSION = "0.5.7"; irm https://ollama.com/install.ps1 | iex

.LINK
    https://ollama.com
#>

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

# --------------------------------------------------------------------------
# Configuration from environment variables
# --------------------------------------------------------------------------

$Version      = if ($env:OLLAMA_VERSION) { $env:OLLAMA_VERSION } else { "" }
$InstallDir   = if ($env:OLLAMA_INSTALL_DIR) { $env:OLLAMA_INSTALL_DIR } else { "" }
$Uninstall    = $env:OLLAMA_UNINSTALL -eq "1"
$DebugInstall = [bool]$env:OLLAMA_DEBUG

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

# OLLAMA_DOWNLOAD_URL for developer testing only
$DownloadBaseURL = if ($env:OLLAMA_DOWNLOAD_URL) { $env:OLLAMA_DOWNLOAD_URL.TrimEnd('/') } else { "https://ollama.com/download" }
$InnoSetupUninstallGuid = "{44E83376-CE68-45EB-8FC1-393500EB558C}_is1"

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

function Write-Status {
    param([string]$Message)
    if ($DebugInstall) { Write-Host $Message }
}

function Write-Step {
    param([string]$Message)
    if ($DebugInstall) { Write-Host ">>> $Message" -ForegroundColor Cyan }
}

function Test-Signature {
    param([string]$FilePath)

    $sig = Get-AuthenticodeSignature -FilePath $FilePath
    if ($sig.Status -ne "Valid") {
        Write-Status "  Signature status: $($sig.Status)"
        return $false
    }

    # Verify it's signed by Ollama Inc. (check exact organization name)
    # Anchor with comma/boundary to prevent "O=Not Ollama Inc." from matching
    $subject = $sig.SignerCertificate.Subject
    if ($subject -notmatch "(^|, )O=Ollama Inc\.(,|$)") {
        Write-Status "  Unexpected signer: $subject"
        return $false
    }

    Write-Status "  Signature valid: $subject"
    return $true
}

function Find-InnoSetupInstall {
    # Check both HKCU (per-user) and HKLM (per-machine) locations
    $possibleKeys = @(
        "HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\$InnoSetupUninstallGuid",
        "HKLM:\Software\Microsoft\Windows\CurrentVersion\Uninstall\$InnoSetupUninstallGuid",
        "HKLM:\Software\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\$InnoSetupUninstallGuid"
    )

    foreach ($key in $possibleKeys) {
        if (Test-Path $key) {
            Write-Status "  Found install at: $key"
            return $key
        }
    }
    return $null
}

function Update-SessionPath {
    # Update PATH in current session so 'ollama' works immediately
    if ($InstallDir) {
        $ollamaDir = $InstallDir
    } else {
        $ollamaDir = Join-Path $env:LOCALAPPDATA "Programs\Ollama"
    }

    # Add to PATH if not already present
    if (Test-Path $ollamaDir) {
        $currentPath = $env:PATH -split ';'
        if ($ollamaDir -notin $currentPath) {
            $env:PATH = "$ollamaDir;$env:PATH"
            Write-Status "  Added $ollamaDir to session PATH"
        }
    }
}

function Invoke-Download {
    param(
        [string]$Url,
        [string]$OutFile
    )

    Write-Status "  Downloading: $Url"
    try {
        Invoke-WebRequest -Uri $Url -OutFile $OutFile -UseBasicParsing
        $size = (Get-Item $OutFile).Length
        Write-Status "  Downloaded: $([math]::Round($size / 1MB, 1)) MB"
    } catch {
        if ($_.Exception.Response.StatusCode -eq 404) {
            throw "Download failed: not found at $Url"
        }
        throw "Download failed for ${Url}: $($_.Exception.Message)"
    }
}

# --------------------------------------------------------------------------
# Uninstall
# --------------------------------------------------------------------------

function Invoke-Uninstall {
    Write-Step "Uninstalling Ollama"

    $regKey = Find-InnoSetupInstall
    if (-not $regKey) {
        Write-Host "Ollama is not installed."
        return
    }

    $uninstallString = (Get-ItemProperty -Path $regKey).UninstallString
    if (-not $uninstallString) {
        Write-Warning "No uninstall string found in registry"
        return
    }

    # Strip quotes if present
    $uninstallExe = $uninstallString -replace '"', ''
    Write-Status "  Uninstaller: $uninstallExe"

    if (-not (Test-Path $uninstallExe)) {
        Write-Warning "Uninstaller not found at: $uninstallExe"
        return
    }

    Write-Host "Launching uninstaller..."
    # Run with GUI so user can choose whether to keep models
    Start-Process -FilePath $uninstallExe -Wait

    # Verify removal
    if (Find-InnoSetupInstall) {
        Write-Warning "Uninstall may not have completed"
    } else {
        Write-Host "Ollama has been uninstalled."
    }
}

# --------------------------------------------------------------------------
# Install
# --------------------------------------------------------------------------

function Invoke-Install {
    # Determine installer URL
    if ($Version) {
        $installerUrl = "$DownloadBaseURL/OllamaSetup.exe?version=$Version"
    } else {
        $installerUrl = "$DownloadBaseURL/OllamaSetup.exe"
    }

    # Download installer
    Write-Step "Downloading Ollama"
    if (-not $DebugInstall) {
        Write-Host "Downloading Ollama..."
    }

    $tempInstaller = Join-Path $env:TEMP "OllamaSetup.exe"
    Invoke-Download -Url $installerUrl -OutFile $tempInstaller

    # Verify signature
    Write-Step "Verifying signature"
    if (-not (Test-Signature -FilePath $tempInstaller)) {
        Remove-Item $tempInstaller -Force -ErrorAction SilentlyContinue
        throw "Installer signature verification failed"
    }

    # Build installer arguments
    $installerArgs = "/VERYSILENT /NORESTART /SUPPRESSMSGBOXES"
    if ($InstallDir) {
        $installerArgs += " /DIR=`"$InstallDir`""
    }
    Write-Status "  Installer args: $installerArgs"

    # Run installer
    Write-Step "Installing Ollama"
    if (-not $DebugInstall) {
        Write-Host "Installing..."
    }

    # Create upgrade marker so the app starts hidden
    # The app checks for this file on startup and removes it after
    $markerDir = Join-Path $env:LOCALAPPDATA "Ollama"
    $markerFile = Join-Path $markerDir "upgraded"
    if (-not (Test-Path $markerDir)) {
        New-Item -ItemType Directory -Path $markerDir -Force | Out-Null
    }
    New-Item -ItemType File -Path $markerFile -Force | Out-Null
    Write-Status "  Created upgrade marker: $markerFile"

    # Start installer and wait for just the installer process (not children)
    # Using -Wait would wait for Ollama to exit too, which we don't want
    $proc = Start-Process -FilePath $tempInstaller `
        -ArgumentList $installerArgs `
        -PassThru
    $proc.WaitForExit()

    if ($proc.ExitCode -ne 0) {
        Remove-Item $tempInstaller -Force -ErrorAction SilentlyContinue
        throw "Installation failed with exit code $($proc.ExitCode)"
    }

    # Cleanup
    Remove-Item $tempInstaller -Force -ErrorAction SilentlyContinue

    # Update PATH in current session so 'ollama' works immediately
    Write-Step "Updating session PATH"
    Update-SessionPath

    Write-Host "Install complete. You can now run 'ollama'."
}

# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

if ($Uninstall) {
    Invoke-Uninstall
} else {
    Invoke-Install
}
