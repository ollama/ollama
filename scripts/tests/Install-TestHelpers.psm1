<#
.SYNOPSIS
    Shared test helpers for Ollama install script tests.
#>

# --------------------------------------------------------------------------
# Unit test helpers
# --------------------------------------------------------------------------

function Get-TestCacheDir {
    return Join-Path $env:TEMP "ollama-test-cache"
}

function Get-TestInstallDir {
    return Join-Path $env:TEMP "ollama-test-install"
}

function Initialize-TestEnvironment {
    $cacheDir = Get-TestCacheDir
    $installDir = Get-TestInstallDir

    if (Test-Path $cacheDir) { Remove-Item $cacheDir -Recurse -Force }
    if (Test-Path $installDir) { Remove-Item $installDir -Recurse -Force }

    New-Item -ItemType Directory -Path $cacheDir -Force | Out-Null
    New-Item -ItemType Directory -Path $installDir -Force | Out-Null

    return @{
        CacheDir = $cacheDir
        InstallDir = $installDir
    }
}

function Remove-TestEnvironment {
    $cacheDir = Get-TestCacheDir
    $installDir = Get-TestInstallDir

    if (Test-Path $cacheDir) { Remove-Item $cacheDir -Recurse -Force -ErrorAction SilentlyContinue }
    if (Test-Path $installDir) { Remove-Item $installDir -Recurse -Force -ErrorAction SilentlyContinue }
}

function New-MockPackagesJson {
    param(
        [string]$Version = "0.15.0",
        [string]$OutputPath
    )

    $manifest = @{
        version = $Version
        packages = @(
            @{
                name = "cuda_v12"
                file = "ollama-cuda-v12.msi"
                sha256 = "abc123def456"
                deps = "ollama-cuda-deps-12.8.1.msi"
                deps_sha256 = "789abc012def"
            },
            @{
                name = "cuda_v13"
                file = "ollama-cuda-v13.msi"
                sha256 = "111222333444"
                deps = "ollama-cuda-v13-deps-13.0.0.msi"
                deps_sha256 = "555666777888"
            },
            @{
                name = "rocm"
                file = "ollama-rocm.msi"
                sha256 = "aaa111bbb222"
                deps = "ollama-rocm-deps-6.3.2.msi"
                deps_sha256 = "ccc333ddd444"
            },
            @{
                name = "vulkan"
                file = "ollama-vulkan.msi"
                sha256 = "eee555fff666"
                deps = "ollama-vulkan-deps-1.4.0.msi"
                deps_sha256 = "777888999000"
            }
        )
    }

    $manifest | ConvertTo-Json -Depth 3 | Out-File -FilePath $OutputPath -Encoding utf8
    return $manifest
}

function New-MockInstalledDir {
    param(
        [string]$Dir,
        [string[]]$Backends = @()
    )

    # Create ollama.exe placeholder
    New-Item -ItemType File -Path (Join-Path $Dir "ollama.exe") -Force | Out-Null

    # Create backend directories
    foreach ($backend in $Backends) {
        $backendDir = Join-Path $Dir "lib\ollama\$backend"
        New-Item -ItemType Directory -Path $backendDir -Force | Out-Null
        New-Item -ItemType File -Path (Join-Path $backendDir "ggml-test.dll") -Force | Out-Null
    }
}

# --------------------------------------------------------------------------
# Integration test helpers
# --------------------------------------------------------------------------

# Known UpgradeCodes for all Ollama MSI packages.
# Used by the Windows Installer COM API (RelatedProducts) to find installed
# products. This approach works for both per-user and system-level installs.
$script:OllamaUpgradeCodes = @{
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

# Find the dist directory containing built MSIs
function Find-DistDir {
    # Walk up from test dir to find the repo root
    $dir = $PSScriptRoot
    while ($dir) {
        $candidate = Join-Path $dir "dist"
        if (Test-Path (Join-Path $candidate "ollama-core.msi")) {
            return $candidate
        }
        $parent = Split-Path $dir -Parent
        if ($parent -eq $dir) { break }
        $dir = $parent
    }
    return $null
}

# Start a Python HTTP server serving files from $Dir on a random port.
# Returns a hashtable with Port, Process, and BaseUrl.
function Start-LocalHttpServer {
    param([string]$Dir)

    # Find an open port
    $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, 0)
    $listener.Start()
    $port = $listener.LocalEndpoint.Port
    $listener.Stop()

    $pythonExe = (Get-Command python -ErrorAction SilentlyContinue).Source
    if (-not $pythonExe) {
        $pythonExe = (Get-Command python3 -ErrorAction SilentlyContinue).Source
    }
    if (-not $pythonExe) {
        throw "Python not found. Integration tests require Python for the local HTTP server."
    }

    $proc = Start-Process -FilePath $pythonExe `
        -ArgumentList "-m", "http.server", $port, "--bind", "127.0.0.1" `
        -WorkingDirectory $Dir `
        -WindowStyle Hidden `
        -PassThru

    # Wait briefly for server to start
    Start-Sleep -Seconds 1

    return @{
        Port = $port
        Process = $proc
        BaseUrl = "http://127.0.0.1:$port"
    }
}

function Stop-LocalHttpServer {
    param($Server)
    if ($Server -and $Server.Process -and -not $Server.Process.HasExited) {
        $Server.Process | Stop-Process -Force -ErrorAction SilentlyContinue
    }
}

# Run install.ps1 with given environment variables and OLLAMA_DOWNLOAD_URL set.
# Returns a hashtable with ExitCode, Output (string[]), and Duration.
function Invoke-InstallScript {
    param(
        [string]$BaseUrl,
        [hashtable]$EnvVars = @{}
    )

    $scriptPath = Join-Path (Split-Path -Parent $PSScriptRoot) "install.ps1"

    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $output = & {
        $env:OLLAMA_DOWNLOAD_URL = $BaseUrl
        foreach ($key in $EnvVars.Keys) {
            Set-Item -Path "Env:\$key" -Value $EnvVars[$key]
        }
        try {
            & powershell.exe -ExecutionPolicy Bypass -File $scriptPath 2>&1
        } finally {
            Remove-Item Env:\OLLAMA_DOWNLOAD_URL -ErrorAction SilentlyContinue
            foreach ($key in $EnvVars.Keys) {
                Remove-Item "Env:\$key" -ErrorAction SilentlyContinue
            }
        }
    }
    $sw.Stop()

    return @{
        ExitCode = $LASTEXITCODE
        Output   = $output
        Duration = $sw.Elapsed
    }
}

# Uninstall all Ollama MSI packages (cleanup after integration tests).
function Invoke-FullUninstall {
    # Use Windows Installer COM API with known UpgradeCodes to find and uninstall
    # all Ollama products. This works for both per-user and system-level installs.
    $installer = New-Object -ComObject WindowsInstaller.Installer
    # Uninstall backends first, then deps, then core (reverse of install order)
    $uninstallOrder = @("cuda_v12","cuda_v13","rocm","vulkan",
                        "cuda_v12_deps","cuda_v13_deps","rocm_deps","vulkan_deps",
                        "core","core-arm64")
    foreach ($name in $uninstallOrder) {
        $uc = $script:OllamaUpgradeCodes[$name]
        if (-not $uc) { continue }
        try {
            $related = $installer.RelatedProducts("{$uc}")
            foreach ($pc in $related) {
                Start-Process msiexec -ArgumentList "/x", $pc, "/quiet", "/norestart" `
                    -Wait -NoNewWindow -ErrorAction SilentlyContinue
            }
        } catch { }
    }

    # Also clean up any Ollama processes
    Get-Process -Name "ollama", "Ollama app" -ErrorAction SilentlyContinue |
        Stop-Process -Force -ErrorAction SilentlyContinue

    # Remove registry keys
    Remove-Item -Path "HKCU:\Software\Ollama" -Recurse -Force -ErrorAction SilentlyContinue

    # Remove default install dir
    $defaultDir = Join-Path $env:LOCALAPPDATA "Programs\Ollama"
    if (Test-Path $defaultDir) {
        Remove-Item $defaultDir -Recurse -Force -ErrorAction SilentlyContinue
    }

    # Remove cache
    $cacheDir = Join-Path $env:LOCALAPPDATA "Ollama\msi_cache"
    if (Test-Path $cacheDir) {
        Remove-Item $cacheDir -Recurse -Force -ErrorAction SilentlyContinue
    }
}

# Check if Ollama is installed by looking for the exe.
function Test-OllamaInstalled {
    param([string]$Dir = "")
    if (-not $Dir) {
        $Dir = Join-Path $env:LOCALAPPDATA "Programs\Ollama"
    }
    return (Test-Path (Join-Path $Dir "ollama.exe"))
}

# Get the list of installed Ollama MSI products using Windows Installer COM API.
# Returns an array of objects with Name, UpgradeCode, and ProductCode properties.
function Get-InstalledOllamaProducts {
    $installer = New-Object -ComObject WindowsInstaller.Installer
    $products = @()
    foreach ($name in $script:OllamaUpgradeCodes.Keys) {
        $uc = $script:OllamaUpgradeCodes[$name]
        try {
            $related = $installer.RelatedProducts("{$uc}")
            foreach ($pc in $related) {
                $products += [PSCustomObject]@{
                    Name        = $name
                    UpgradeCode = $uc
                    ProductCode = $pc
                }
            }
        } catch { }
    }
    return $products
}

# Check if a specific backend is installed.
function Test-BackendInstalled {
    param(
        [string]$Dir,
        [string]$Backend
    )
    if (-not $Dir) {
        $Dir = Join-Path $env:LOCALAPPDATA "Programs\Ollama"
    }
    return (Test-Path (Join-Path $Dir "lib\ollama\$Backend"))
}

Export-ModuleMember -Function *
