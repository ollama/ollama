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
        OLLAMA_DOWNLOAD_ONLY      Set to 1 to download MSIs without installing
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
$DownloadOnly = $env:OLLAMA_DOWNLOAD_ONLY -eq "1"
$DebugInstall = [bool]$env:OLLAMA_DEBUG

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

$DownloadBaseURL = if ($env:OLLAMA_DOWNLOAD_URL) { $env:OLLAMA_DOWNLOAD_URL.TrimEnd('/') } else { "https://ollama.com/download" }
$CacheDir = Join-Path $env:LOCALAPPDATA "Ollama\msi_cache"
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

function Get-Architecture {
    $arch = [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture.ToString().ToLower()
    switch ($arch) {
        "x64"   { return "amd64" }
        "arm64" { return "arm64" }
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

function Test-AuthenticodeSignature {
    param([string]$FilePath)
    $sig = Get-AuthenticodeSignature -FilePath $FilePath
    return $sig.Status -eq 'Valid'
}

function Invoke-Download {
    param(
        [string]$Url,
        [string]$OutFile
    )

    $tempFile = "$OutFile.tmp"
    try {
        Invoke-WebRequest -Uri $Url -OutFile $tempFile -UseBasicParsing
        Move-Item -Path $tempFile -Destination $OutFile -Force
    } catch {
        $status = $_.Exception.Response.StatusCode.value__
        if ($status) {
            throw "Download failed: HTTP $status for $Url"
        }
        throw "Download failed for ${Url}: $($_.Exception.Message)"
    } finally {
        if (Test-Path $tempFile) {
            Remove-Item $tempFile -Force -ErrorAction SilentlyContinue
        }
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
    foreach ($name in @("cuda_v12", "cuda_v13", "rocm", "vulkan")) {
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
    if (-not (Test-Path $InnoSetupUninstallKey)) {
        return
    }

    Write-Step "Removing legacy Inno Setup installation"
    $uninstallString = (Get-ItemProperty -Path $InnoSetupUninstallKey).UninstallString
    if ($uninstallString) {
        # Strip quotes if present
        $uninstallExe = $uninstallString -replace '"', ''
        Write-Status "  Running Inno Setup uninstaller..."
        $proc = Start-Process -FilePath $uninstallExe `
            -ArgumentList "/VERYSILENT /NORESTART /SUPPRESSMSGBOXES" `
            -Wait -PassThru -NoNewWindow
        if ($proc.ExitCode -ne 0) {
            Write-Warning "Inno Setup uninstaller returned exit code $($proc.ExitCode)"
        }
        # Wait for cleanup
        Start-Sleep -Seconds 2
    }
}

# --------------------------------------------------------------------------
# Uninstall flow
# --------------------------------------------------------------------------

function Invoke-Uninstall {
    Write-Step "Uninstalling Ollama"
    if (-not $DebugInstall) {
        Write-Host "Uninstalling Ollama... " -NoNewline
    }

    Stop-OllamaProcesses

    # Uninstall all known MSI packages (backends first, then deps, then core)
    $uninstallOrder = @(
        "cuda_v12", "cuda_v13", "rocm", "vulkan",
        "cuda_v12_deps", "cuda_v13_deps", "rocm_deps", "vulkan_deps",
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

    # Clean cache
    if (Test-Path $CacheDir) {
        Remove-Item -Path $CacheDir -Recurse -Force -ErrorAction SilentlyContinue
    }

    if ($DebugInstall) {
        Write-Host "Ollama has been uninstalled."
    } else {
        Write-Host "done."
    }
}

# --------------------------------------------------------------------------
# Main install flow
# --------------------------------------------------------------------------

function Invoke-Install {
    $arch = Get-Architecture
    $targetDir = Resolve-InstallDir
    $coreMsi = Get-CoreMsiName $arch
    $isUpgrade = Test-Path (Join-Path $targetDir "ollama.exe")

    Write-Step "Ollama Installer"
    Write-Status "  Architecture: $arch"
    Write-Status "  Install directory: $targetDir"
    Write-Status "  Upgrade: $isUpgrade"

    if (-not $DebugInstall) {
        $action = if ($isUpgrade) { "Updating" } else { "Installing" }
        Write-Host "$action Ollama..." -NoNewline
    }

    # Ensure cache directory exists
    if (-not (Test-Path $CacheDir)) {
        New-Item -ItemType Directory -Path $CacheDir -Force | Out-Null
    }

    # ------------------------------------------------------------------
    # Step 1: Download core MSI
    # ------------------------------------------------------------------
    Write-Step "Downloading core package"
    $coreMsiPath = Join-Path $CacheDir $coreMsi
    $coreUrl = "$DownloadBaseURL/$coreMsi"
    if ($Version) {
        $coreUrl = "${coreUrl}?version=$Version"
    }

    # Always download core MSI (it changes every release and there's no
    # SHA256 to compare against since core can't contain its own hash).
    # For local/dev testing, the download is fast anyway.
    Write-Status "  Downloading $coreMsi..."
    Invoke-Download -Url $coreUrl -OutFile $coreMsiPath

    # Verify signature (warn but continue for unsigned dev builds)
    if (-not (Test-AuthenticodeSignature $coreMsiPath)) {
        Write-Warning "  $coreMsi does not have a valid Authenticode signature"
    }

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

    if ($All) {
        $selectedBackends = $newManifest.packages | ForEach-Object { $_.name }
        Write-Status "  Mode: All backends"
    } elseif ($Backends.Count -gt 0) {
        $selectedBackends = $Backends
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
        if (-not $DebugInstall) {
            $backendList = ($selectedBackends | ForEach-Object { $_ -replace '_', ' ' }) -join ', '
            Write-Host "Downloading optional GPU components ($backendList), press Ctrl+C to skip..."
        }

        foreach ($backendName in $selectedBackends) {
            $pkg = $newManifest.packages | Where-Object { $_.name -eq $backendName }
            if (-not $pkg) {
                Write-Warning "  Backend '$backendName' not found in manifest, skipping"
                continue
            }

            $entry = @{ name = $backendName; msiPath = $null; depsPath = $null }

            # Download deps MSI
            if ($pkg.deps) {
                $depsPath = Join-Path $CacheDir $pkg.deps
                $needsDepsDownload = $true

                # Check if cached deps matches new SHA256
                if ((Test-Path $depsPath) -and $pkg.deps_sha256) {
                    $cachedHash = (Get-FileHash -Path $depsPath -Algorithm SHA256).Hash.ToLower()
                    if ($cachedHash -eq $pkg.deps_sha256) {
                        Write-Status "  $($pkg.deps) unchanged (cached)"
                        $needsDepsDownload = $false
                    }
                }

                if ($needsDepsDownload) {
                    $depsUrl = "$DownloadBaseURL/$($pkg.deps)"
                    Write-Status "  Downloading $($pkg.deps)..."
                    Invoke-Download -Url $depsUrl -OutFile $depsPath

                    # Verify SHA256
                    if ($pkg.deps_sha256) {
                        $dlHash = (Get-FileHash -Path $depsPath -Algorithm SHA256).Hash.ToLower()
                        if ($dlHash -ne $pkg.deps_sha256) {
                            Write-Warning "  SHA256 mismatch for $($pkg.deps) (expected $($pkg.deps_sha256), got $dlHash)"
                        }
                    }
                }
                $entry.depsPath = $depsPath
            }

            # Download backend MSI
            $msiPath = Join-Path $CacheDir $pkg.file
            $needsMsiDownload = $true

            if ((Test-Path $msiPath) -and $pkg.sha256) {
                $cachedHash = (Get-FileHash -Path $msiPath -Algorithm SHA256).Hash.ToLower()
                if ($cachedHash -eq $pkg.sha256) {
                    Write-Status "  $($pkg.file) unchanged (cached)"
                    $needsMsiDownload = $false
                }
            }

            if ($needsMsiDownload) {
                $msiUrl = "$DownloadBaseURL/$($pkg.file)"
                Write-Status "  Downloading $($pkg.file)..."
                Invoke-Download -Url $msiUrl -OutFile $msiPath

                # Verify SHA256
                if ($pkg.sha256) {
                    $dlHash = (Get-FileHash -Path $msiPath -Algorithm SHA256).Hash.ToLower()
                    if ($dlHash -ne $pkg.sha256) {
                        Write-Warning "  SHA256 mismatch for $($pkg.file) (expected $($pkg.sha256), got $dlHash)"
                    }
                }
            }
            $entry.msiPath = $msiPath
            $gpuDownloads += $entry
        }
    }

    if ($DownloadOnly) {
        Write-Host ""
        Write-Host "Downloads complete. MSIs cached in $CacheDir"
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
            $depsCode = $UpgradeCodes["${backend}_deps"]
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
        Start-Process -FilePath $appExe -WindowStyle Hidden
    } else {
        Write-Warning "'Ollama app.exe' not found at $appExe - the MSI may not have installed correctly"
    }

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    if ($DebugInstall) {
        Write-Host ""
        Write-Host "Ollama has been installed to $targetDir" -ForegroundColor Green
    } else {
        Write-Host "done." -ForegroundColor Green
    }
    Write-Host "Run 'ollama run llama3.2' to get started."
}

# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

if ($Uninstall) {
    Invoke-Uninstall
} else {
    Invoke-Install
}
