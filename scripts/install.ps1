<#
.SYNOPSIS
    Install, upgrade, or uninstall Ollama on Windows.

.DESCRIPTION
    Downloads and installs Ollama, optimized for your GPU hardware.

    Quick install with defaults:

        irm https://ollama.com/install.ps1 | iex

    All GPU backends:

        $env:OLLAMA_INSTALL_ALL=1; irm https://ollama.com/install.ps1 | iex

    CPU-only (no GPU backends):

        $env:OLLAMA_INSTALL_MINIMAL=1; irm https://ollama.com/install.ps1 | iex

    Specific version:

        $env:OLLAMA_VERSION="0.15.0"; irm https://ollama.com/install.ps1 | iex

    Custom install directory:

        $env:OLLAMA_INSTALL_DIR="D:\Ollama"; irm https://ollama.com/install.ps1 | iex

    Uninstall:

        $env:OLLAMA_UNINSTALL=1; irm https://ollama.com/install.ps1 | iex

    If you download the script, you can set environment variables before running:

        .\install.ps1                                            # defaults
        $env:OLLAMA_INSTALL_ALL=1; .\install.ps1                 # all backends
        $env:OLLAMA_INSTALL_DIR="D:\Ollama"; .\install.ps1       # custom dir

    Environment variables:

        OLLAMA_VERSION            Target version (default: latest stable)
        OLLAMA_INSTALL_ALL        Set to 1 to install all GPU backends
        OLLAMA_INSTALL_MINIMAL    Set to 1 for CPU-only (no GPU backends)
        OLLAMA_INSTALL_BACKENDS   Comma-separated backend list (e.g. cuda_v12,rocm)
        OLLAMA_INSTALL_DIR        Custom install directory
        OLLAMA_UNINSTALL          Set to 1 to uninstall all Ollama packages
        OLLAMA_CACHE_ONLY         Set to 1 to download installer payloads without installing
        OLLAMA_INSTALL_CACHED     Set to 1 to install from the Ollama installer cache without downloading
        OLLAMA_REMOVE_MODELS      Set to 1 to remove models on uninstall, 0 to keep (skips prompt)
        OLLAMA_DEBUG              Enable verbose output (any non-empty value)

.EXAMPLE
    irm https://ollama.com/install.ps1 | iex

.EXAMPLE
    $env:OLLAMA_INSTALL_ALL = "1"; irm https://ollama.com/install.ps1 | iex

.EXAMPLE
    $env:OLLAMA_INSTALL_MINIMAL = "1"; irm https://ollama.com/install.ps1 | iex

.EXAMPLE
    $env:OLLAMA_VERSION = "0.15.0"; irm https://ollama.com/install.ps1 | iex

.LINK
    https://ollama.com
#>

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"  # Speed up Invoke-WebRequest

# --------------------------------------------------------------------------
# Read configuration from environment variables
# --------------------------------------------------------------------------

$Version      = if ($env:OLLAMA_VERSION) { $env:OLLAMA_VERSION } else { "" }
$All          = $env:OLLAMA_INSTALL_ALL -eq "1"
$Minimal      = $env:OLLAMA_INSTALL_MINIMAL -eq "1"
$Backends     = if ($env:OLLAMA_INSTALL_BACKENDS) {
    ($env:OLLAMA_INSTALL_BACKENDS -split ',') | ForEach-Object { $_.Trim() } | Where-Object { $_ }
} else { @() }
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

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

$DownloadBaseURL = "https://ollama.com/download"
$InnoSetupUninstallKey = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\{44E83376-CE68-45EB-8FC1-393500EB558C}_is1"
$OllamaRegistryKey = "HKCU:\Software\Ollama"

# UpgradeCodes for chained operations (must match WXS files)
$UpgradeCodes = @{
    "core"          = "7A5B3E2F-1C4D-4F8A-9E6B-0D2A1F3C5E7D"
    "core-arm64"    = "B4C6D8E0-2F1A-4E3C-A5D7-9B0E1F2A3C4D"
    "cuda_v12"      = "3F8A2D1E-5B6C-4E7F-A9D0-1C2B3E4F5A6D"
    "cuda_v13"      = "9C7E3A1B-2D4F-4E5A-B6C8-0D1E2F3A4B5C"
    "rocm"          = "4B2E8F1A-6C3D-4A5E-9F7B-0D1C2E3A4B5D"
    "vulkan"        = "6D4A2E8F-1B3C-4F5E-A7D9-0C1B2E3F4A5D"
    "cuda_v12_deps" = "A1E3B5C7-2D4F-6A8E-9B0C-1D2E3F4A5B6C"
    "cuda_v13_deps" = "8F2A4E6C-1B3D-5C7E-A9F0-2D1E3B4A5C6D"
    "rocm_deps"     = "E5C7A9B1-3D2F-4E6A-8F0C-1B2D3E4A5F6C"
    "vulkan_deps"   = "2A4C6E8F-0B1D-3E5A-7C9F-1D2B3A4E5C6F"
    "mlx_cuda_v13"  = "3E7A1B5C-9D2F-4A6E-B8C0-1F2D3E4A5B6C"
    "mlx_deps"      = "4F8B2C6D-0E3A-5B7F-C9D1-2E3F4A5B6C7D"
}

$BackendDepsPackages = @{
    "cuda_v12"      = "cuda_v12_deps"
    "cuda_v13"      = "cuda_v13_deps"
    "rocm"          = "rocm_deps"
    "vulkan"        = "vulkan_deps"
    "mlx_cuda_v13"  = "mlx_deps"
}

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

function New-InstallCacheTarget {
    param(
        [string]$CacheRoot,
        [string]$CacheDir,
        [string]$CoreMsi,
        [string]$ETag = "",
        [bool]$UseExisting = $false,
        [bool]$ReplaceCacheRoot = $false,
        [bool]$Persistent = $false
    )

    $stagingCacheDir = if ($Persistent -and -not $UseExisting) { "${CacheDir}.download" } else { $CacheDir }
    return [PSCustomObject]@{
        CacheRoot        = $CacheRoot
        CacheDir         = $CacheDir
        StagingCacheDir  = $stagingCacheDir
        CoreMsi          = $CoreMsi
        ETag             = $ETag
        UseExisting      = $UseExisting
        ReplaceCacheRoot = $ReplaceCacheRoot
        Persistent       = $Persistent
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

function Get-InstallCacheTarget {
    param(
        [string]$CoreUrl,
        [string]$CoreMsi,
        [bool]$CacheOnlyMode = $false
    )

    $installerETag = Get-RemoteETag -Url $CoreUrl
    if ($installerETag) {
        $cacheRoot = Get-InstallerCacheRoot
        $cacheDir = Join-Path $cacheRoot (Get-InstallerCacheKey -ETag $installerETag)
        if (Test-Path -LiteralPath (Join-Path $cacheDir $CoreMsi) -PathType Leaf) {
            return New-InstallCacheTarget -CacheRoot $cacheRoot -CacheDir $cacheDir -CoreMsi $CoreMsi -ETag $installerETag -UseExisting $true -Persistent $true
        }
        if ($CacheOnlyMode) {
            return New-InstallCacheTarget -CacheRoot $cacheRoot -CacheDir $cacheDir -CoreMsi $CoreMsi -ETag $installerETag -ReplaceCacheRoot $true -Persistent $true
        }
    } elseif ($CacheOnlyMode) {
        Write-Status "  Installer ETag unavailable; refreshing installer cache without cache reuse."
        $cacheRoot = Get-InstallerCacheRoot
        $cacheDir = Join-Path $cacheRoot ([guid]::NewGuid().ToString("N"))
        return New-InstallCacheTarget -CacheRoot $cacheRoot -CacheDir $cacheDir -CoreMsi $CoreMsi -ReplaceCacheRoot $true -Persistent $true
    }

    $cacheRoot = Get-TemporaryInstallerCacheRoot
    $cacheDir = Join-Path $cacheRoot ([guid]::NewGuid().ToString("N"))
    return New-InstallCacheTarget -CacheRoot $cacheRoot -CacheDir $cacheDir -CoreMsi $CoreMsi
}

function Get-CachedInstallCacheTarget {
    param([string]$CoreMsi)

    $cacheRoot = Get-InstallerCacheRoot
    if (-not (Test-Path -LiteralPath $cacheRoot)) {
        throw "Cached installer payloads not found in $cacheRoot"
    }

    $matches = @()
    foreach ($entry in @(Get-ChildItem -LiteralPath $cacheRoot -Directory -ErrorAction SilentlyContinue)) {
        if ($entry.Name.EndsWith(".download", [System.StringComparison]::OrdinalIgnoreCase)) {
            continue
        }
        $candidate = Join-Path $entry.FullName $CoreMsi
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            $matches += $entry.FullName
        }
    }

    if ($matches.Count -eq 0) {
        throw "Cached installer payloads not found in $cacheRoot"
    }
    if ($matches.Count -gt 1) {
        throw "Multiple cached installer payloads found in $cacheRoot"
    }

    return New-InstallCacheTarget -CacheRoot $cacheRoot -CacheDir $matches[0] -CoreMsi $CoreMsi -UseExisting $true -Persistent $true
}

function Initialize-InstallCacheTarget {
    param($Target)

    if ($Target.UseExisting) {
        return $Target.CacheDir
    }

    if ($Target.ReplaceCacheRoot) {
        Remove-Item -LiteralPath $Target.CacheRoot -Recurse -Force -ErrorAction SilentlyContinue
    }
    Remove-Item -LiteralPath $Target.StagingCacheDir -Recurse -Force -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Path $Target.StagingCacheDir -Force | Out-Null
    return $Target.StagingCacheDir
}

function Complete-InstallCacheTarget {
    param($Target)

    if ($Target.StagingCacheDir -eq $Target.CacheDir) {
        return
    }
    Remove-Item -LiteralPath $Target.CacheDir -Recurse -Force -ErrorAction SilentlyContinue
    Move-Item -LiteralPath $Target.StagingCacheDir -Destination $Target.CacheDir -Force
}

function Remove-InstallCacheTarget {
    param($Target)

    Remove-Item -LiteralPath $Target.StagingCacheDir -Recurse -Force -ErrorAction SilentlyContinue
    if (-not $Target.UseExisting) {
        Remove-Item -LiteralPath $Target.CacheDir -Recurse -Force -ErrorAction SilentlyContinue
    }
}

function Get-Architecture {
    # Try .NET RuntimeInformation first (PowerShell 6+ / .NET Core)
    try {
        $osArch = [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture
        if ($null -ne $osArch) {
            switch ($osArch.ToString().ToLower()) {
                "x64"   { return "amd64" }
                "arm64" { return "arm64" }
            }
        }
    } catch { }

    # Fallback for Windows PowerShell 5.1
    $procArch = $env:PROCESSOR_ARCHITECTURE
    switch ($procArch) {
        "AMD64" { return "amd64" }
        "ARM64" { return "arm64" }
        default { return "amd64" }
    }
}

function Get-CoreMsiName {
    param([string]$Arch)
    if ($Arch -eq "arm64") { return "ollama-core-arm64.msi" }
    return "ollama-core.msi"
}

function Resolve-InstallDir {
    # 1. Explicit environment variable
    if ($InstallDir) {
        return $InstallDir
    }

    # 2. Registry (persisted from previous install)
    $regDir = $null
    try {
        $regDir = (Get-ItemProperty -Path $OllamaRegistryKey -Name "InstallDir" -ErrorAction SilentlyContinue).InstallDir
    } catch {}
    if ($regDir -and (Test-Path $regDir)) {
        return $regDir
    }

    # 3. Check PATH for existing ollama.exe
    $ollamaCmd = Get-Command "ollama" -ErrorAction SilentlyContinue
    if ($ollamaCmd) {
        $existingDir = Split-Path $ollamaCmd.Source -Parent
        if (Test-Path $existingDir) {
            return $existingDir
        }
    }

    # 4. Default
    return Join-Path $env:LOCALAPPDATA "Programs\Ollama"
}

function Test-Signature {
    param([string]$FilePath)

    $sig = Get-AuthenticodeSignature -FilePath $FilePath
    if ($sig.Status -ne "Valid") {
        Write-Status "  Signature status: $($sig.Status)"
        return $false
    }

    $subject = $sig.SignerCertificate.Subject
    if ($subject -notmatch "(^|, )O=Ollama Inc\.(,|$)") {
        Write-Status "  Unexpected signer: $subject"
        return $false
    }

    Write-Status "  Signature valid: $subject"
    return $true
}

function Assert-SignatureValid {
    param(
        [string]$FilePath,
        [string]$Label
    )

    if (-not (Test-Path -LiteralPath $FilePath -PathType Leaf)) {
        throw "Expected file not found for signature verification: $FilePath"
    }
    if (-not (Test-Signature -FilePath $FilePath)) {
        Remove-Item -LiteralPath $FilePath -Force -ErrorAction SilentlyContinue
        throw "Signature verification failed for $Label"
    }
}

function Invoke-Download {
    param(
        [string]$Url,
        [string]$OutFile,
        [string]$Label
    )

    $prefix = if ($Label) { "$Label " } else { "  " }
    Write-Status "  Downloading: $Url"
    $tempFile = "$OutFile.tmp"
    try {
        $request = [System.Net.HttpWebRequest]::Create($Url)
        $request.AllowAutoRedirect = $true
        $response = $request.GetResponse()
        $responseETag = $response.Headers["ETag"]
        $totalBytes = $response.ContentLength
        $stream = $response.GetResponseStream()
        $fileStream = [System.IO.FileStream]::new($tempFile, [System.IO.FileMode]::Create)
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
                        Write-Host -NoNewline "`r${prefix}[$bar] ${pctFmt}%"
                    } else {
                        $sizeMB = [math]::Round($totalRead / 1MB, 1)
                        Write-Host -NoNewline "`r${prefix}${sizeMB} MB downloaded..."
                    }
                    $lastUpdate = $now
                }
            }

            # Final progress update
            if ($totalBytes -gt 0) {
                $bar = '#' * $barWidth
                Write-Host "`r${prefix}[$bar] 100.0%"
            } else {
                $sizeMB = [math]::Round($totalRead / 1MB, 1)
                Write-Host "`r${prefix}${sizeMB} MB downloaded.          "
            }
        } finally {
            $fileStream.Close()
            $stream.Close()
            $response.Close()
        }

        Move-Item -Path $tempFile -Destination $OutFile -Force
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
    } finally {
        if (Test-Path $tempFile) {
            Remove-Item $tempFile -Force -ErrorAction SilentlyContinue
        }
    }
}

function Test-FileHashMatches {
    param(
        [string]$FilePath,
        [string]$ExpectedHash
    )

    if (-not $ExpectedHash -or -not (Test-Path $FilePath)) {
        return $false
    }

    $actualHash = (Get-FileHash -Path $FilePath -Algorithm SHA256).Hash.ToLower()
    return $actualHash -eq $ExpectedHash.ToLower()
}

function Assert-FileHashMatches {
    param(
        [string]$FilePath,
        [string]$ExpectedHash,
        [string]$Label
    )

    if (-not $ExpectedHash) {
        return
    }
    if (-not (Test-Path $FilePath)) {
        throw "Expected file not found for hash verification: $FilePath"
    }

    $actualHash = (Get-FileHash -Path $FilePath -Algorithm SHA256).Hash.ToLower()
    if ($actualHash -ne $ExpectedHash.ToLower()) {
        Remove-Item -Path $FilePath -Force -ErrorAction SilentlyContinue
        throw "SHA256 mismatch for $Label (expected $ExpectedHash, got $actualHash)"
    }
}

function Install-Msi {
    param(
        [string]$MsiPath,
        [string]$TargetDir = ""
    )

    $msiArgs = @("/i", "`"$MsiPath`"", "/quiet", "/norestart")
    if ($TargetDir) {
        # Use ROOTDIRECTORY (the WXS-defined install folder), not TARGETDIR.
        # The MSI directory tree is rooted under StandardDirectory LocalAppDataFolder,
        # so TARGETDIR won't redirect the install location.
        $msiArgs += "ROOTDIRECTORY=`"$TargetDir`""
    }

    $argStr = $msiArgs -join " "
    Write-Status "  Installing $(Split-Path $MsiPath -Leaf)..."

    $proc = Start-Process -FilePath "msiexec.exe" -ArgumentList $argStr -Wait -PassThru -NoNewWindow
    if ($proc.ExitCode -ne 0) {
        Write-Warning "msiexec returned exit code $($proc.ExitCode) for $(Split-Path $MsiPath -Leaf)"
        return $false
    }
    return $true
}

function Uninstall-MsiByUpgradeCode {
    param([string]$UpgradeCode)

    # msiexec /x only accepts ProductCodes, not UpgradeCodes.
    # Use the Windows Installer COM API to find the ProductCode from the UpgradeCode.
    $installer = New-Object -ComObject WindowsInstaller.Installer
    $relatedProducts = $installer.RelatedProducts("{$UpgradeCode}")
    $found = $false
    foreach ($productCode in $relatedProducts) {
        $found = $true
        Write-Status "  Uninstalling $productCode..."
        $proc = Start-Process -FilePath "msiexec.exe" `
            -ArgumentList "/x `"$productCode`" /quiet /norestart" `
            -Wait -PassThru -NoNewWindow -ErrorAction SilentlyContinue
        if ($proc -and $proc.ExitCode -ne 0 -and $proc.ExitCode -ne 1605) {
            Write-Warning "msiexec /x returned exit code $($proc.ExitCode) for $productCode"
        }
    }
    return $found
}

function Get-InstalledPackagesJson {
    param([string]$Dir)
    $jsonPath = Join-Path $Dir "packages.json"
    if (Test-Path $jsonPath) {
        return Get-Content $jsonPath -Raw | ConvertFrom-Json
    }
    return $null
}

function Get-InstalledBackends {
    param([string]$Dir)
    $backends = @()
    $libDir = Join-Path $Dir "lib\ollama"
    foreach ($name in @("cuda_v12", "cuda_v13", "rocm", "vulkan", "mlx_cuda_v13")) {
        $backendDir = Join-Path $libDir $name
        if (Test-Path $backendDir) {
            $backends += $name
        }
    }
    return $backends
}

function Detect-Hardware {
    Write-Status "Detecting GPU hardware..."
    $selected = @()

    try {
        $gpus = Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue
    } catch {
        Write-Status "  Could not detect GPUs, defaulting to CPU-only"
        return $selected
    }

    foreach ($gpu in $gpus) {
        $name = $gpu.Name
        if (-not $name) { continue }

        if ($name -match "NVIDIA") {
            Write-Status "  Detected: $name"
            # Check driver version via nvidia-smi to determine CUDA version
            $cudaVer = "cuda_v12"  # default
            try {
                $smiOutput = & nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>$null
                if ($smiOutput) {
                    $driverMajor = [int]($smiOutput.Trim().Split('.')[0])
                    # CUDA 13 requires driver >= 570
                    if ($driverMajor -ge 570) {
                        $cudaVer = "cuda_v13"
                    }
                }
            } catch {}
            if ($cudaVer -notin $selected) { $selected += $cudaVer }
            # MLX CUDA backend requires CUDA 13
            if ($cudaVer -eq "cuda_v13" -and "mlx_cuda_v13" -notin $selected) {
                $selected += "mlx_cuda_v13"
            }
            Write-Status "    -> $cudaVer"
        }
        elseif ($name -match "AMD|Radeon") {
            Write-Status "  Detected: $name -> rocm"
            if ("rocm" -notin $selected) { $selected += "rocm" }
        }
    }

    # Always include Vulkan on x64 as a fallback GPU backend
    $arch = Get-Architecture
    if ($arch -eq "amd64" -and "vulkan" -notin $selected) {
        $selected += "vulkan"
    }

    return $selected
}

function Stop-OllamaProcesses {
    Write-Status "Stopping Ollama processes..."
    $procs = Get-Process -Name "ollama", "Ollama app" -ErrorAction SilentlyContinue
    if ($procs) {
        $procs | Stop-Process -Force -ErrorAction SilentlyContinue
        # Wait briefly for processes to exit
        Start-Sleep -Seconds 2
    }
}

function Remove-InnoSetupInstall {
    # Check both HKCU (per-user) and HKLM (per-machine) locations
    $innoGuid = "{44E83376-CE68-45EB-8FC1-393500EB558C}_is1"
    $possibleKeys = @(
        "HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\$innoGuid",
        "HKLM:\Software\Microsoft\Windows\CurrentVersion\Uninstall\$innoGuid",
        "HKLM:\Software\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\$innoGuid"
    )

    $foundKey = $null
    foreach ($key in $possibleKeys) {
        if (Test-Path $key) {
            $foundKey = $key
            Write-Status "  Found Inno Setup at: $key"
            break
        }
    }

    if (-not $foundKey) {
        Write-Status "  No legacy Inno Setup installation found"
        return
    }

    Write-Step "Removing legacy Inno Setup installation"
    $uninstallString = (Get-ItemProperty -Path $foundKey).UninstallString
    if ($uninstallString) {
        # Strip quotes if present
        $uninstallExe = $uninstallString -replace '"', ''
        Write-Status "  Uninstall string: $uninstallExe"

        if (-not (Test-Path $uninstallExe)) {
            Write-Warning "  Uninstaller not found at: $uninstallExe"
            # Try to clean up the orphaned registry key
            Remove-Item -Path $foundKey -Force -ErrorAction SilentlyContinue
            return
        }

        Write-Status "  Running Inno Setup uninstaller..."
        $proc = Start-Process -FilePath $uninstallExe `
            -ArgumentList "/VERYSILENT /NORESTART /SUPPRESSMSGBOXES" `
            -Wait -PassThru -NoNewWindow
        if ($proc.ExitCode -ne 0) {
            Write-Warning "Inno Setup uninstaller returned exit code $($proc.ExitCode)"
        }
        # Wait for cleanup
        Start-Sleep -Seconds 2

        # Verify it was removed
        if (Test-Path $foundKey) {
            Write-Warning "  Inno Setup registry key still exists after uninstall"
        }
    } else {
        Write-Warning "  No UninstallString found in registry"
    }
}

# --------------------------------------------------------------------------
# Uninstall flow
# --------------------------------------------------------------------------

function Invoke-Uninstall {
    Write-Step "Uninstalling Ollama"
    if (-not $DebugInstall) {
        Write-Host "Uninstalling Ollama..."
    }

    Stop-OllamaProcesses

    # Uninstall all known MSI packages (backends first, then deps, then core)
    $uninstallOrder = @(
        "mlx_cuda_v13", "cuda_v12", "cuda_v13", "rocm", "vulkan",
        "mlx_deps", "cuda_v12_deps", "cuda_v13_deps", "rocm_deps", "vulkan_deps",
        "core", "core-arm64"
    )

    foreach ($pkg in $uninstallOrder) {
        $code = $UpgradeCodes[$pkg]
        Uninstall-MsiByUpgradeCode $code | Out-Null
    }

    # Also remove Inno Setup if present
    Remove-InnoSetupInstall

    # Model removal: OLLAMA_REMOVE_MODELS=1 removes without prompting,
    # OLLAMA_REMOVE_MODELS=0 preserves without prompting, unset prompts interactively.
    $removeModels = $false
    $modelsDir = Join-Path $env:USERPROFILE ".ollama\models"
    if (Test-Path $modelsDir) {
        if ($env:OLLAMA_REMOVE_MODELS -eq "1") {
            $removeModels = $true
        } elseif ($null -eq $env:OLLAMA_REMOVE_MODELS) {
            $response = Read-Host "Remove downloaded models at $modelsDir? [y/N]"
            $removeModels = $response -match '^[Yy]'
        }
    }

    if ($removeModels) {
        Write-Status "  Removing models..."
        Remove-Item -Path (Join-Path $env:USERPROFILE ".ollama\models") -Recurse -Force -ErrorAction SilentlyContinue
    }

    # Clean cache (check both possible locations)
    $cacheDirs = @(
        (Get-InstallerCacheRoot),
        (Get-TemporaryInstallerCacheRoot)
    )
    foreach ($dir in $cacheDirs) {
        if (Test-Path $dir) {
            Remove-Item -Path $dir -Recurse -Force -ErrorAction SilentlyContinue
        }
    }

    if ($DebugInstall) {
        Write-Host ""
    }
    Write-Host "Ollama has been uninstalled."
}

# --------------------------------------------------------------------------
# Main install flow
# --------------------------------------------------------------------------

function Invoke-Install {
    $arch = Get-Architecture
    $targetDir = Resolve-InstallDir
    $coreMsi = Get-CoreMsiName $arch
    $isUpgrade = Test-Path (Join-Path $targetDir "ollama.exe")
    $coreUrl = "$DownloadBaseURL/$coreMsi"
    if ($Version) {
        $coreUrl = "${coreUrl}?version=$Version"
    }

    if ($InstallCached) {
        $cacheTarget = Get-CachedInstallCacheTarget -CoreMsi $coreMsi
    } else {
        $cacheTarget = Get-InstallCacheTarget -CoreUrl $coreUrl -CoreMsi $coreMsi -CacheOnlyMode $CacheOnly
    }
    $CacheDir = Initialize-InstallCacheTarget -Target $cacheTarget
    $downloadedPayloads = -not $cacheTarget.UseExisting

    Write-Step "Ollama Installer"
    Write-Status "  Architecture: $arch"
    Write-Status "  Install directory: $targetDir"
    Write-Status "  Cache directory: $CacheDir"
    Write-Status "  Cache only: $CacheOnly"
    Write-Status "  Install cached: $InstallCached"
    Write-Status "  Upgrade: $isUpgrade"

    if (-not $DebugInstall) {
        $action = if ($isUpgrade) { "Updating" } else { "Installing" }
        Write-Host "$action Ollama..."
    }

    # ------------------------------------------------------------------
    # Step 1: Download core MSI
    # ------------------------------------------------------------------
    Write-Step "Downloading core package"
    $coreMsiPath = Join-Path $CacheDir $coreMsi

    if ($cacheTarget.UseExisting) {
        Write-Status "  Using cached $coreMsi"
    } else {
        if (-not $DebugInstall) {
            Write-Host ">>> Downloading Ollama for Windows..."
        }
        Write-Status "  Downloading $coreMsi..."
        $coreDownloadETag = Invoke-Download -Url $coreUrl -OutFile $coreMsiPath -Label "Downloading Ollama..."
        if ($cacheTarget.ETag -and $coreDownloadETag -and ($cacheTarget.ETag.Trim().Trim('"') -ne $coreDownloadETag.Trim().Trim('"'))) {
            Remove-InstallCacheTarget -Target $cacheTarget
            throw "Core MSI ETag changed while downloading"
        }
    }

    if (-not $DebugInstall) {
        Write-Host ">>> Verifying signature..."
    }
    Assert-SignatureValid -FilePath $coreMsiPath -Label $coreMsi

    # ------------------------------------------------------------------
    # Step 2: Extract packages.json from core MSI
    # ------------------------------------------------------------------
    Write-Step "Reading package manifest"
    $tempExtract = Join-Path $CacheDir "extract_temp"
    if (Test-Path $tempExtract) {
        Remove-Item -Path $tempExtract -Recurse -Force
    }
    New-Item -ItemType Directory -Path $tempExtract -Force | Out-Null

    # Administrative install extracts files without installing
    $extractProc = Start-Process -FilePath "msiexec.exe" `
        -ArgumentList "/a `"$coreMsiPath`" /qn TARGETDIR=`"$tempExtract`"" `
        -Wait -PassThru -NoNewWindow
    if ($extractProc.ExitCode -ne 0) {
        Write-Warning "Failed to extract packages.json from core MSI (exit code $($extractProc.ExitCode))"
    }

    # Find packages.json in extracted files
    $newManifestPath = Get-ChildItem -Path $tempExtract -Filter "packages.json" -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
    $newManifest = $null
    if ($newManifestPath) {
        $newManifest = Get-Content $newManifestPath.FullName -Raw | ConvertFrom-Json
        Write-Status "  Version: $($newManifest.version)"
        Write-Status "  Available packages: $(($newManifest.packages | ForEach-Object { $_.name }) -join ', ')"
    } else {
        Write-Status "  No packages.json found in core MSI (CPU-only install)"
        $newManifest = @{ version = $Version; packages = @() }
    }

    # Clean up temp extraction
    Remove-Item -Path $tempExtract -Recurse -Force -ErrorAction SilentlyContinue

    # ------------------------------------------------------------------
    # Step 3: Detect installed backends (for upgrade)
    # ------------------------------------------------------------------
    $installedBackends = @()
    $installedManifest = $null
    if ($isUpgrade) {
        $installedBackends = Get-InstalledBackends $targetDir
        $installedManifest = Get-InstalledPackagesJson $targetDir
        if ($installedBackends.Count -gt 0) {
            Write-Status "  Currently installed backends: $($installedBackends -join ', ')"
        }
    }

    # ------------------------------------------------------------------
    # Step 4: Backend selection
    # ------------------------------------------------------------------
    Write-Step "Selecting GPU backends"
    $selectedBackends = @()

    # Filter manifest packages to current architecture (packages without arch are assumed amd64)
    $archPackages = $newManifest.packages | Where-Object {
        $pkgArch = if ($_.arch) { $_.arch } else { "amd64" }
        $pkgArch -eq $arch
    }

    if ($All) {
        $selectedBackends = $archPackages | ForEach-Object { $_.name }
        Write-Status "  Mode: All backends ($arch)"
    } elseif ($Backends.Count -gt 0) {
        $archNames = $archPackages | ForEach-Object { $_.name }
        $selectedBackends = @()
        foreach ($b in $Backends) {
            if ($archNames -contains $b) {
                $selectedBackends += $b
            } else {
                Write-Warning "  Backend '$b' is not available for $arch, skipping"
            }
        }
        Write-Status "  Mode: Explicit ($($selectedBackends -join ', '))"
    } elseif ($Minimal) {
        $selectedBackends = @()
        Write-Status "  Mode: Minimal (CPU only)"
    } elseif ($isUpgrade -and $installedBackends.Count -gt 0) {
        $selectedBackends = $installedBackends
        Write-Status "  Mode: Upgrade (keeping $($selectedBackends -join ', '))"
    } else {
        $selectedBackends = Detect-Hardware
        if ($selectedBackends.Count -gt 0) {
            Write-Status "  Selected: $($selectedBackends -join ', ')"
        } else {
            Write-Status "  No GPU detected, CPU-only install"
        }
        Write-Status "  Set OLLAMA_INSTALL_ALL=1 to install all backends"
    }

    # ------------------------------------------------------------------
    # Step 5: Download GPU MSIs (comparing SHA256 to skip unchanged)
    # ------------------------------------------------------------------
    $gpuDownloads = @()  # List of @{name; msiPath; depsPath} for install

    if ($selectedBackends.Count -gt 0) {
        Write-Step "Downloading GPU packages"

        # Friendly display names for GPU backends
        $displayNames = @{
            'cuda_v12' = 'CUDA v12'
            'cuda_v13' = 'CUDA v13'
            'rocm'     = 'ROCm'
            'vulkan'       = 'Vulkan'
            'mlx_cuda_v13' = 'MLX CUDA v13'
        }

        # Pre-compute download labels for column alignment
        $downloadPlan = @()
        foreach ($backendName in $selectedBackends) {
            $pkg = $newManifest.packages | Where-Object { $_.name -eq $backendName }
            if (-not $pkg) {
                Write-Warning "  Backend '$backendName' not found in manifest, skipping"
                continue
            }

            $friendly = $displayNames[$backendName]
            if (-not $friendly) { $friendly = $backendName }

            # Extract version from deps filename (e.g., ollama-rocm-deps-6.3.2.msi -> 6.3.2)
            $depsVersion = $null
            if ($pkg.deps -and $pkg.deps -match '-deps-(.+)\.msi$') {
                $depsVersion = $Matches[1]
            }

            # Deps label: "{BaseName} {version} libraries"
            # Strip version qualifier from friendly name for deps (CUDA v12 -> CUDA)
            # since the full SDK version (12.8.1) is more informative
            $baseName = $friendly -replace ' v\d+$', ''
            $depsLabel = if ($depsVersion) { "$baseName $depsVersion libraries" } else { "$baseName libraries" }

            # Backend label: "Ollama {friendly} backend"
            # For backends without a version in their name (e.g., rocm), derive one
            # from the deps major version to disambiguate future multi-version scenarios
            $backendFriendly = $friendly
            if ($depsVersion -and $friendly -notmatch 'v\d') {
                $majorVer = ($depsVersion -split '\.')[0]
                $backendFriendly = "$friendly v$majorVer"
            }
            $backendLabel = "Ollama $backendFriendly backend"

            $downloadPlan += @{
                backendName  = $backendName
                pkg          = $pkg
                depsLabel    = $depsLabel
                backendLabel = $backendLabel
            }
        }

        # Find max formatted label width for column alignment
        $maxLabelLen = 0
        foreach ($plan in $downloadPlan) {
            if ($plan.depsLabel.Length -gt $maxLabelLen) { $maxLabelLen = $plan.depsLabel.Length }
            if ($plan.backendLabel.Length -gt $maxLabelLen) { $maxLabelLen = $plan.backendLabel.Length }
        }
        # Formatted label = "  " (indent) + label + ":" → pad all to same width
        $padWidth = $maxLabelLen + 3

        # Track whether GPU header has been printed (only print if downloads happen)
        $gpuHeaderPrinted = $false

        foreach ($plan in $downloadPlan) {
            $pkg = $plan.pkg
            $backendName = $plan.backendName
            $entry = @{ name = $backendName; msiPath = $null; depsPath = $null }
            $installedPkg = $null
            if ($installedManifest -and $installedManifest.packages) {
                $installedPkg = $installedManifest.packages | Where-Object { $_.name -eq $backendName } | Select-Object -First 1
            }
            $backendInstalled = $installedBackends -contains $backendName

            # Download deps MSI
            if ($pkg.deps) {
                $depsPath = Join-Path $CacheDir $pkg.deps
                $needsDepsDownload = $true
                $depsAlreadyCurrent = $backendInstalled -and $installedPkg -and $pkg.deps_sha256 -and `
                    $installedPkg.deps_sha256 -and ($installedPkg.deps_sha256.ToLower() -eq $pkg.deps_sha256.ToLower())

                # Check if cached deps matches new SHA256
                if (Test-Path $depsPath) {
                    if (-not $pkg.deps_sha256 -or (Test-FileHashMatches -FilePath $depsPath -ExpectedHash $pkg.deps_sha256)) {
                        Write-Status "  $($pkg.deps) available in cache"
                        $needsDepsDownload = $false
                    } elseif ($InstallCached) {
                        Assert-FileHashMatches -FilePath $depsPath -ExpectedHash $pkg.deps_sha256 -Label $pkg.deps
                    } else {
                        Write-Status "  $($pkg.deps) cache hash mismatch, redownloading"
                        Remove-Item -Path $depsPath -Force -ErrorAction SilentlyContinue
                    }
                }

                if ($needsDepsDownload) {
                    if ($InstallCached) {
                        if ($depsAlreadyCurrent) {
                            Write-Status "  $($pkg.deps) unchanged and already installed"
                        } else {
                            throw "Required deps MSI missing from cache: $depsPath"
                        }
                    } else {
                        if (-not $DebugInstall -and -not $gpuHeaderPrinted) {
                            Write-Host "Downloading GPU components..."
                            $gpuHeaderPrinted = $true
                        }
                        $depsUrl = "$DownloadBaseURL/$($pkg.deps)"
                        Write-Status "  Downloading $($pkg.deps)..."
                        $paddedLabel = ("  " + $plan.depsLabel + ":").PadRight($padWidth)
                        Invoke-Download -Url $depsUrl -OutFile $depsPath -Label $paddedLabel
                        Assert-FileHashMatches -FilePath $depsPath -ExpectedHash $pkg.deps_sha256 -Label $pkg.deps
                        Assert-SignatureValid -FilePath $depsPath -Label $pkg.deps
                        $entry.depsPath = $depsPath
                    }
                } else {
                    Assert-FileHashMatches -FilePath $depsPath -ExpectedHash $pkg.deps_sha256 -Label $pkg.deps
                    Assert-SignatureValid -FilePath $depsPath -Label $pkg.deps
                    $entry.depsPath = $depsPath
                }
            }

            # Download backend MSI
            $msiPath = Join-Path $CacheDir $pkg.file
            $needsMsiDownload = $true
            $backendAlreadyCurrent = $backendInstalled -and $installedPkg -and $pkg.sha256 -and `
                $installedPkg.sha256 -and ($installedPkg.sha256.ToLower() -eq $pkg.sha256.ToLower())

            if (Test-Path $msiPath) {
                if (-not $pkg.sha256 -or (Test-FileHashMatches -FilePath $msiPath -ExpectedHash $pkg.sha256)) {
                    Write-Status "  $($pkg.file) available in cache"
                    $needsMsiDownload = $false
                } elseif ($InstallCached) {
                    Assert-FileHashMatches -FilePath $msiPath -ExpectedHash $pkg.sha256 -Label $pkg.file
                } else {
                    Write-Status "  $($pkg.file) cache hash mismatch, redownloading"
                    Remove-Item -Path $msiPath -Force -ErrorAction SilentlyContinue
                }
            }

            if ($needsMsiDownload) {
                if ($InstallCached) {
                    if ($backendAlreadyCurrent) {
                        Write-Status "  $($pkg.file) unchanged and already installed"
                    } else {
                        throw "Required backend MSI missing from cache: $msiPath"
                    }
                } else {
                    if (-not $DebugInstall -and -not $gpuHeaderPrinted) {
                        Write-Host "Downloading GPU components..."
                        $gpuHeaderPrinted = $true
                    }
                    $msiUrl = "$DownloadBaseURL/$($pkg.file)"
                    Write-Status "  Downloading $($pkg.file)..."
                    $paddedLabel = ("  " + $plan.backendLabel + ":").PadRight($padWidth)
                    Invoke-Download -Url $msiUrl -OutFile $msiPath -Label $paddedLabel
                    Assert-FileHashMatches -FilePath $msiPath -ExpectedHash $pkg.sha256 -Label $pkg.file
                    Assert-SignatureValid -FilePath $msiPath -Label $pkg.file
                    $entry.msiPath = $msiPath
                }
            } else {
                Assert-FileHashMatches -FilePath $msiPath -ExpectedHash $pkg.sha256 -Label $pkg.file
                Assert-SignatureValid -FilePath $msiPath -Label $pkg.file
                $entry.msiPath = $msiPath
            }
            if ($entry.msiPath -or $entry.depsPath) {
                $gpuDownloads += $entry
            }
        }
    }

    if ($CacheOnly) {
        if ($downloadedPayloads) {
            Complete-InstallCacheTarget -Target $cacheTarget
            Write-Host ""
            if ($DebugInstall) {
                Write-Host "Downloads complete. MSIs cached in $($cacheTarget.CacheDir)"
            } else {
                Write-Host "Downloads complete."
            }
        } else {
            if ($DebugInstall) {
                Write-Host "MSI cache is current: $($cacheTarget.CacheDir)"
            } else {
                Write-Host "MSI cache is current."
            }
        }
        return
    }

    # ------------------------------------------------------------------
    # Step 6: Stop running Ollama processes
    # ------------------------------------------------------------------
    Stop-OllamaProcesses

    # ------------------------------------------------------------------
    # Step 7: Remove legacy Inno Setup install
    # ------------------------------------------------------------------
    Remove-InnoSetupInstall

    # ------------------------------------------------------------------
    # Step 8: Install deps MSIs
    # ------------------------------------------------------------------
    if (-not $DebugInstall) {
        Write-Host "Installing..." -NoNewline
    }
    if ($gpuDownloads.Count -gt 0) {
        Write-Step "Installing GPU dependencies"
        foreach ($dl in $gpuDownloads) {
            if ($dl.depsPath) {
                $ok = Install-Msi -MsiPath $dl.depsPath -TargetDir $targetDir
                if (-not $ok) {
                    Write-Warning "  Deps install failed for $($dl.name), skipping backend"
                    $dl.msiPath = $null  # Don't install backend if deps failed
                }
            }
        }
    }

    # ------------------------------------------------------------------
    # Step 9: Install backend MSIs
    # ------------------------------------------------------------------
    if ($gpuDownloads.Count -gt 0) {
        Write-Step "Installing GPU backends"
        foreach ($dl in $gpuDownloads) {
            if ($dl.msiPath) {
                Install-Msi -MsiPath $dl.msiPath -TargetDir $targetDir | Out-Null
            }
        }
    }

    # ------------------------------------------------------------------
    # Step 10: Install core MSI (last, minimizes downtime)
    # ------------------------------------------------------------------
    Write-Step "Installing Ollama core"
    $coreOk = Install-Msi -MsiPath $coreMsiPath -TargetDir $targetDir
    if (-not $coreOk) {
        Write-Error "Core MSI installation failed"
        return
    }

    # Persist install dir to registry if non-default
    $defaultDir = Join-Path $env:LOCALAPPDATA "Programs\Ollama"
    if ($targetDir -ne $defaultDir) {
        New-Item -Path $OllamaRegistryKey -Force -ErrorAction SilentlyContinue | Out-Null
        Set-ItemProperty -Path $OllamaRegistryKey -Name "InstallDir" -Value $targetDir
    }

    # ------------------------------------------------------------------
    # Step 11: Uninstall removed backends (if Minimal on upgrade)
    # ------------------------------------------------------------------
    if ($Minimal -and $isUpgrade -and $installedBackends.Count -gt 0) {
        Write-Step "Removing previously installed GPU backends"
        foreach ($backend in $installedBackends) {
            $backendCode = $UpgradeCodes[$backend]
            $depsPackage = $BackendDepsPackages[$backend]
            $depsCode = if ($depsPackage) { $UpgradeCodes[$depsPackage] } else { $null }
            if ($backendCode) {
                Write-Status "  Removing $backend..."
                Uninstall-MsiByUpgradeCode $backendCode | Out-Null
            }
            if ($depsCode) {
                Uninstall-MsiByUpgradeCode $depsCode | Out-Null
            }
        }
    }

    # ------------------------------------------------------------------
    # Step 12: Start Ollama
    # ------------------------------------------------------------------
    # The MSI updated the user PATH in the registry, but this process
    # still has the old PATH. Refresh it so the launched app inherits
    # a PATH that includes the install directory. The app uses
    # exec.LookPath to find the ollama binary, which requires PATH.
    $userPath = [Environment]::GetEnvironmentVariable("PATH", "User")
    $machinePath = [Environment]::GetEnvironmentVariable("PATH", "Machine")
    $env:PATH = "$userPath;$machinePath"

    # Ensure install dir is in PATH even if the MSI Environment
    # component didn't fire (e.g., repair install, custom dir edge
    # case, or dev/unsigned builds).
    $pathDirs = $env:PATH -split ';' | ForEach-Object { $_.TrimEnd('\') }
    $normalizedTarget = $targetDir.TrimEnd('\')
    if ($normalizedTarget -notin $pathDirs) {
        Write-Status "  Adding $targetDir to PATH for this session"
        $env:PATH = "$targetDir;$env:PATH"
    }

    $appExe = Join-Path $targetDir "Ollama app.exe"
    if (Test-Path $appExe) {
        Write-Step "Starting Ollama"
        Start-Process -FilePath $appExe -ArgumentList "--hide", "--fast-startup" -WindowStyle Hidden
    } else {
        Write-Warning "'Ollama app.exe' not found at $appExe - the MSI may not have installed correctly"
    }

    Remove-InstallCacheTarget -Target $cacheTarget

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    if ($DebugInstall) {
        Write-Host ""
        Write-Host "Ollama has been installed to $targetDir" -ForegroundColor Green
    } else {
        Write-Host "done."
    }
    Write-Host "Install complete. You can now run 'ollama'."
}

# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

if ($Uninstall) {
    Invoke-Uninstall
} else {
    Invoke-Install
}
