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

        OLLAMA_VERSION         Target version (default: latest stable)
        OLLAMA_INSTALL_DIR     Custom install directory
        OLLAMA_UNINSTALL       Set to 1 to uninstall Ollama
        OLLAMA_CACHE_ONLY      Set to 1 to download installer payloads without installing
        OLLAMA_INSTALL_CACHED  Set to 1 to install from the Ollama installer cache without downloading
        OLLAMA_DEBUG           Enable verbose output

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
$CacheOnly    = $env:OLLAMA_CACHE_ONLY -eq "1"
$InstallCached = $env:OLLAMA_INSTALL_CACHED -eq "1"
$DebugInstall = [bool]$env:OLLAMA_DEBUG

if ($CacheOnly -and $InstallCached) {
    throw "OLLAMA_CACHE_ONLY and OLLAMA_INSTALL_CACHED cannot both be set"
}
if ($Uninstall -and ($CacheOnly -or $InstallCached)) {
    throw "OLLAMA_UNINSTALL cannot be combined with OLLAMA_CACHE_ONLY or OLLAMA_INSTALL_CACHED"
}

<#
Returns a stable filesystem-safe cache key for an installer ETag.
#>
function Get-InstallerCacheKey {
    param([string]$ETag)

    $normalizedETag = $ETag.Trim().Trim('"')
    if (-not $normalizedETag) {
        throw "Installer ETag is required for installer cache"
    }

    $sha256 = [System.Security.Cryptography.SHA256]::Create()
    try {
        $bytes = [System.Text.Encoding]::UTF8.GetBytes($normalizedETag)
        $hash = $sha256.ComputeHash($bytes)
        return -join ($hash | ForEach-Object { $_.ToString("x2") })
    } finally {
        $sha256.Dispose()
    }
}

function Get-InstallerCacheRoot {
    return Join-Path $env:LOCALAPPDATA "Ollama\install_cache"
}

function Get-TemporaryInstallerCacheRoot {
    return Join-Path $env:TEMP "Ollama\install_cache"
}

function New-InstallerTarget {
    param(
        [string]$CacheRoot,
        [string]$CacheDir,
        [string]$ETag = "",
        [bool]$ReplaceCacheRoot = $false
    )

    return [PSCustomObject]@{
        Path             = (Join-Path $CacheDir "OllamaSetup.exe")
        StagingPath      = (Join-Path "${CacheDir}.download" "OllamaSetup.exe")
        CacheDir         = $CacheDir
        StagingCacheDir  = "${CacheDir}.download"
        CacheRoot        = $CacheRoot
        ETag             = $ETag
        ReplaceCacheRoot = $ReplaceCacheRoot
    }
}

<#
Removes a resolved installer cache entry after signature, ETag, or install failure.
#>
function Remove-InstallerCacheEntry {
    param($Installer)

    Remove-Item -LiteralPath $Installer.CacheDir -Recurse -Force -ErrorAction SilentlyContinue
    if ($Installer.StagingCacheDir) {
        Remove-Item -LiteralPath $Installer.StagingCacheDir -Recurse -Force -ErrorAction SilentlyContinue
    }
}

<#
Resolves the installer URL into the exact local path this run should use.
Cache-only mode uses the persistent cache. If ETags are unavailable, it refreshes the cache with a one-shot GUID entry.
Normal installs fall back to a throwaway temp directory when ETags are unavailable.
#>
function Get-InstallerTarget {
    param(
        [string]$InstallerUrl,
        [bool]$CacheOnlyMode = $false
    )

    $installerETag = Get-RemoteETag -Url $InstallerUrl
    if ($installerETag) {
        $cacheRoot = Get-InstallerCacheRoot
        $cacheDir = Join-Path $cacheRoot (Get-InstallerCacheKey -ETag $installerETag)
        $replaceCacheRoot = $true
    } elseif ($CacheOnlyMode) {
        Write-Status "  Installer ETag unavailable; refreshing installer cache without cache reuse."
        $cacheRoot = Get-InstallerCacheRoot
        $cacheDir = Join-Path $cacheRoot ([guid]::NewGuid().ToString("N"))
        $replaceCacheRoot = $true
    } else {
        $cacheRoot = Get-TemporaryInstallerCacheRoot
        $cacheDir = Join-Path $cacheRoot ([guid]::NewGuid().ToString("N"))
        $replaceCacheRoot = $false
    }

    return New-InstallerTarget -CacheRoot $cacheRoot -CacheDir $cacheDir -ETag $installerETag -ReplaceCacheRoot $replaceCacheRoot
}

<#
Finds the single completed installer cache entry for install-cached mode.
#>
function Get-CachedInstallerTarget {
    $cacheRoot = Get-InstallerCacheRoot
    if (-not (Test-Path -LiteralPath $cacheRoot)) {
        throw "Cached installer not found in $cacheRoot"
    }

    $installers = @()
    foreach ($entry in @(Get-ChildItem -LiteralPath $cacheRoot -Directory -ErrorAction SilentlyContinue)) {
        if ($entry.Name.EndsWith(".download", [System.StringComparison]::OrdinalIgnoreCase)) {
            continue
        }
        $candidate = Join-Path $entry.FullName "OllamaSetup.exe"
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            $installers += $candidate
        }
    }

    if ($installers.Count -eq 0) {
        throw "Cached installer not found in $cacheRoot"
    }
    if ($installers.Count -gt 1) {
        # Cache-only replaces the cache root before staging a new installer, so
        # multiple completed installers means the cache is stale or corrupt.
        throw "Multiple cached installers found in $cacheRoot"
    }

    $cacheDir = Split-Path -Parent $installers[0]
    return New-InstallerTarget -CacheRoot $cacheRoot -CacheDir $cacheDir
}

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

$DownloadBaseURL = "https://ollama.com/download"
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

function Quote-ProcessArgument {
    param([string]$Argument)

    if ($null -eq $Argument -or $Argument.Length -eq 0) {
        return '""'
    }

    # Quote args containing whitespace or a double quote: space, tab, LF, VT, FF, CR, or ".
    $charsRequiringQuotes = @([char]32, [char]9, [char]10, [char]11, [char]12, [char]13, [char]34)
    if ($Argument.IndexOfAny($charsRequiringQuotes) -lt 0) {
        return $Argument
    }

    $quoted = [System.Text.StringBuilder]::new()
    [void]$quoted.Append('"')
    $backslashes = 0
    foreach ($char in $Argument.ToCharArray()) {
        if ($char -eq [char]92) {
            $backslashes++
            continue
        }

        if ($char -eq [char]34) {
            if ($backslashes -gt 0) {
                [void]$quoted.Append(('\' * ($backslashes * 2)))
                $backslashes = 0
            }
            [void]$quoted.Append('\"')
            continue
        }

        if ($backslashes -gt 0) {
            [void]$quoted.Append(('\' * $backslashes))
            $backslashes = 0
        }
        [void]$quoted.Append($char)
    }

    if ($backslashes -gt 0) {
        [void]$quoted.Append(('\' * ($backslashes * 2)))
    }
    [void]$quoted.Append('"')
    return $quoted.ToString()
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
        $request = [System.Net.HttpWebRequest]::Create($Url)
        $request.AllowAutoRedirect = $true
        $response = $request.GetResponse()
        $responseETag = $response.Headers["ETag"]
        $totalBytes = $response.ContentLength
        $stream = $response.GetResponseStream()
        $fileStream = [System.IO.FileStream]::new($OutFile, [System.IO.FileMode]::Create)
        $buffer = [byte[]]::new(65536)
        $totalRead = 0
        $lastUpdate = [DateTime]::MinValue
        $barWidth = 40

        try {
            while (($read = $stream.Read($buffer, 0, $buffer.Length)) -gt 0) {
                $fileStream.Write($buffer, 0, $read)
                $totalRead += $read

                $now = [DateTime]::UtcNow
                if (($now - $lastUpdate).TotalMilliseconds -ge 250) {
                    if ($totalBytes -gt 0) {
                        $pct = [math]::Min(100.0, ($totalRead / $totalBytes) * 100)
                        $filled = [math]::Floor($barWidth * $pct / 100)
                        $empty = $barWidth - $filled
                        $bar = ('#' * $filled) + (' ' * $empty)
                        $pctFmt = $pct.ToString("0.0")
                        Write-Host -NoNewline "`r$bar ${pctFmt}%"
                    } else {
                        $sizeMB = [math]::Round($totalRead / 1MB, 1)
                        Write-Host -NoNewline "`r${sizeMB} MB downloaded..."
                    }
                    $lastUpdate = $now
                }
            }

            # Final progress update
            if ($totalBytes -gt 0) {
                $bar = '#' * $barWidth
                Write-Host "`r$bar 100.0%"
            } else {
                $sizeMB = [math]::Round($totalRead / 1MB, 1)
                Write-Host "`r${sizeMB} MB downloaded.          "
            }
        } finally {
            $fileStream.Close()
            $stream.Close()
            $response.Close()
        }
        return $responseETag
    } catch {
        if ($_.Exception -is [System.Net.WebException]) {
            $webEx = [System.Net.WebException]$_.Exception
            if ($webEx.Response -and ([System.Net.HttpWebResponse]$webEx.Response).StatusCode -eq [System.Net.HttpStatusCode]::NotFound) {
                throw "Download failed: not found at $Url"
            }
        }
        if ($_.Exception.InnerException -is [System.Net.WebException]) {
            $webEx = [System.Net.WebException]$_.Exception.InnerException
            if ($webEx.Response -and ([System.Net.HttpWebResponse]$webEx.Response).StatusCode -eq [System.Net.HttpStatusCode]::NotFound) {
                throw "Download failed: not found at $Url"
            }
        }
        throw "Download failed for ${Url}: $($_.Exception.Message)"
    }
}

function Get-RemoteETag {
    param([string]$Url)

    try {
        $request = [System.Net.HttpWebRequest]::Create($Url)
        $request.AllowAutoRedirect = $true
        $request.Method = "HEAD"
        $response = $request.GetResponse()
        try {
            return $response.Headers["ETag"]
        } finally {
            $response.Close()
        }
    } catch {
        Write-Status "  Unable to read remote ETag for ${Url}: $($_.Exception.Message)"
        return ""
    }
}

# --------------------------------------------------------------------------
# Uninstall
# --------------------------------------------------------------------------

function Invoke-Uninstall {
    Write-Step "Uninstalling Ollama"

    $regKey = Find-InnoSetupInstall
    if (-not $regKey) {
        Write-Host ">>> Ollama is not installed."
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

    Write-Host ">>> Launching uninstaller..."
    # Run with GUI so user can choose whether to keep models
    Start-Process -FilePath $uninstallExe -Wait

    # Verify removal
    if (Find-InnoSetupInstall) {
        Write-Warning "Uninstall may not have completed"
    } else {
        Write-Host ">>> Ollama has been uninstalled."
    }
}

# --------------------------------------------------------------------------
# Install
# --------------------------------------------------------------------------

<#
Entry point for install behavior.
Resolves the target, optionally populates the cache, then either returns for cache-only mode or runs the installer.
#>
function Invoke-Install {
    $downloadedInstaller = $false
    if ($InstallCached) {
        $installer = Get-CachedInstallerTarget
    } else {
        # Determine installer URL
        if ($Version) {
            $installerUrl = "$DownloadBaseURL/OllamaSetup.exe?version=$Version"
        } else {
            $installerUrl = "$DownloadBaseURL/OllamaSetup.exe"
        }

        $installer = Get-InstallerTarget -InstallerUrl $installerUrl -CacheOnlyMode $CacheOnly
        $downloadedInstaller = Prepare-InstallerPayload -Installer $installer -InstallerUrl $installerUrl
    }

    if ($CacheOnly) {
        if (-not $downloadedInstaller) {
            Write-Step "Verifying signature"
            if (-not $DebugInstall) {
                Write-Host ">>> Verifying signature..."
            }
            if (-not (Test-Signature -FilePath $installer.Path)) {
                Remove-InstallerCacheEntry -Installer $installer
                throw "Installer signature verification failed"
            }
        }

        if ($downloadedInstaller) {
            Write-Host ""
            if ($DebugInstall) {
                Write-Host "Downloads complete. Installer cached in $($installer.CacheDir)"
            } else {
                Write-Host "Downloads complete."
            }
        } else {
            if ($DebugInstall) {
                Write-Host "Installer cache is current: $($installer.CacheDir)"
            } else {
                Write-Host "Installer cache is current."
            }
        }
        return
    }

    Start-Installer -Installer $installer -SignatureVerifiedThisRun $downloadedInstaller
}

<#
Ensures the resolved installer payload exists and is trusted.
This may download the installer, reuse a warm cache, and validates the ETag/signature for new downloads before promotion.
#>
function Prepare-InstallerPayload {
    param(
        $Installer,
        [string]$InstallerUrl
    )

    $cacheHasInstaller = Test-Path -LiteralPath $Installer.Path
    $downloadedInstaller = $false
    $downloadedInstallerETag = ""
    $needsDownload = -not $cacheHasInstaller
    if (-not $DebugInstall -and $needsDownload) {
        Write-Host ">>> Downloading Ollama for Windows..."
    }
    if ($needsDownload) {
        if ($Installer.ReplaceCacheRoot) {
            Remove-Item -LiteralPath $Installer.CacheRoot -Recurse -Force -ErrorAction SilentlyContinue
        } else {
            Remove-InstallerCacheEntry -Installer $Installer
        }
        if (-not (Test-Path -LiteralPath $Installer.StagingCacheDir)) {
            [System.IO.Directory]::CreateDirectory($Installer.StagingCacheDir) | Out-Null
        }

        Write-Step "Downloading Ollama"
        try {
            $downloadedInstallerETag = Invoke-Download -Url $InstallerUrl -OutFile $Installer.StagingPath
        } catch {
            Remove-InstallerCacheEntry -Installer $Installer
            throw
        }
        $downloadedInstallerETag = if ($downloadedInstallerETag) { $downloadedInstallerETag.Trim() } else { "" }
        $downloadedInstaller = $true
    } else {
        Write-Status "  Using cached installer: $($Installer.Path)"
    }

    if ($downloadedInstaller) {
        if ($Installer.ETag -and $downloadedInstallerETag -and ($downloadedInstallerETag -ne $Installer.ETag)) {
            Remove-InstallerCacheEntry -Installer $Installer
            throw "Downloaded installer ETag mismatch: expected $($Installer.ETag), found $downloadedInstallerETag"
        }
        Write-Step "Verifying signature"
        if (-not $DebugInstall) {
            Write-Host ">>> Verifying signature..."
        }
        if (-not (Test-Signature -FilePath $Installer.StagingPath)) {
            Remove-InstallerCacheEntry -Installer $Installer
            throw "Installer signature verification failed"
        }
        try {
            Move-Item -LiteralPath $Installer.StagingCacheDir -Destination $Installer.CacheDir -ErrorAction Stop
        } catch {
            Remove-InstallerCacheEntry -Installer $Installer
            throw "Failed to stage installer cache: $($_.Exception.Message)"
        }
    }
    return $downloadedInstaller
}

<#
Runs an already resolved installer payload.
Cached installs recheck the signature immediately before launch; freshly downloaded installers verified by this same run do not need a second Authenticode pass.
#>
function Start-Installer {
    param(
        $Installer,
        [bool]$SignatureVerifiedThisRun = $false
    )

    if (-not (Test-Path -LiteralPath $Installer.Path)) {
        throw "Cached installer not found: $($Installer.Path)"
    }

    $markerDir = Join-Path $env:LOCALAPPDATA "Ollama"
    $installerLog = Join-Path $markerDir "OllamaSetup.log"

    # Build installer arguments
    $installerArgs = @("/VERYSILENT", "/NORESTART", "/SUPPRESSMSGBOXES", "/LOG=$installerLog")
    if ($InstallDir) {
        $installerArgs += "/DIR=$InstallDir"
    }
    $installerArgumentList = (($installerArgs | ForEach-Object { Quote-ProcessArgument $_ }) -join " ")
    Write-Status "  Installer args: $installerArgumentList"
    Write-Status "  Installer log: $installerLog"

    if (-not $SignatureVerifiedThisRun) {
        Write-Step "Verifying signature"
        if (-not $DebugInstall) {
            Write-Host ">>> Verifying signature..."
        }
        if (-not (Test-Signature -FilePath $Installer.Path)) {
            Remove-InstallerCacheEntry -Installer $Installer
            throw "Installer signature verification failed before launch"
        }
    }

    # Run installer
    Write-Step "Installing Ollama"
    if (-not $DebugInstall) {
        Write-Host ">>> Installing Ollama..."
    }

    # Create upgrade marker so the app starts hidden
    # The app checks for this file on startup and removes it after
    $markerFile = Join-Path $markerDir "upgraded"
    if (-not (Test-Path $markerDir)) {
        New-Item -ItemType Directory -Path $markerDir -Force | Out-Null
    }
    New-Item -ItemType File -Path $markerFile -Force | Out-Null
    Write-Status "  Created upgrade marker: $markerFile"

    # Start installer and wait for just the installer process (not children)
    # Using -Wait would wait for Ollama to exit too, which we don't want
    $proc = Start-Process -FilePath $Installer.Path `
        -ArgumentList $installerArgumentList `
        -PassThru
    $proc.WaitForExit()

    if ($proc.ExitCode -ne 0) {
        Remove-InstallerCacheEntry -Installer $Installer
        throw "Installation failed with exit code $($proc.ExitCode). Installer log: $installerLog"
    }

    # Cleanup
    Remove-InstallerCacheEntry -Installer $Installer

    # Update PATH in current session so 'ollama' works immediately
    Write-Step "Updating session PATH"
    Update-SessionPath

    Write-Host ">>> Install complete. Run 'ollama' from the command line."
}

# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

if ($MyInvocation.InvocationName -ne ".") {
    if ($Uninstall) {
        Invoke-Uninstall
    } else {
        Invoke-Install
    }
}
