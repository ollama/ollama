<#
.SYNOPSIS
    Shared test helpers for Ollama install script tests.
#>

# --------------------------------------------------------------------------
# Unit test helpers
# --------------------------------------------------------------------------

$script:InstallScriptSignatureVerificationDisabled = $false

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

function Set-TestProcessEnvironment {
    $pathValue = [Environment]::GetEnvironmentVariable("Path", "Process")
    if (-not $pathValue) {
        $pathValue = [Environment]::GetEnvironmentVariable("PATH", "Process")
    }

    [Environment]::SetEnvironmentVariable("PATH", $null, "Process")
    if ($pathValue) {
        [Environment]::SetEnvironmentVariable("Path", $pathValue, "Process")
    }
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
                arch = "amd64"
                file = "ollama-cuda-v12.msi"
                sha256 = "abc123def456"
                deps = "ollama-cuda-deps-12.8.1.msi"
                deps_sha256 = "789abc012def"
            },
            @{
                name = "cuda_v13"
                arch = "amd64"
                file = "ollama-cuda-v13.msi"
                sha256 = "111222333444"
                deps = "ollama-cuda-v13-deps-13.0.0.msi"
                deps_sha256 = "555666777888"
            },
            @{
                name = "rocm_v7_1"
                arch = "amd64"
                file = "ollama-rocm.msi"
                sha256 = "aaa111bbb222"
                deps = "ollama-rocm-deps-6.3.2.msi"
                deps_sha256 = "ccc333ddd444"
            },
            @{
                name = "vulkan"
                arch = "amd64"
                file = "ollama-vulkan.msi"
                sha256 = "eee555fff666"
                deps = "ollama-vulkan-deps-1.4.0.msi"
                deps_sha256 = "777888999000"
            },
            @{
                name = "mlx_cuda_v13"
                arch = "amd64"
                file = "ollama-mlx-cuda-v13.msi"
                sha256 = "aaa222bbb333"
                deps = "ollama-cuda-v13-deps-13.0.0.msi"
                deps_sha256 = "ccc444ddd555"
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
    "mlx_cuda_v13"  = "3E7A1B5C-9D2F-4A6E-B8C0-1F2D3E4A5B6C"
}

$script:InnoSetupUninstallGuid = "{44E83376-CE68-45EB-8FC1-393500EB558C}_is1"

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

    Set-TestProcessEnvironment

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
    if ($Server -and $Server.TempDir -and (Test-Path $Server.TempDir)) {
        Remove-Item -LiteralPath $Server.TempDir -Recurse -Force -ErrorAction SilentlyContinue
    }
}

function Get-InnoSetupInstallKey {
    foreach ($key in @(
        "HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\$script:InnoSetupUninstallGuid",
        "HKLM:\Software\Microsoft\Windows\CurrentVersion\Uninstall\$script:InnoSetupUninstallGuid",
        "HKLM:\Software\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\$script:InnoSetupUninstallGuid"
    )) {
        if (Test-Path -LiteralPath $key) {
            return $key
        }
    }
    return ""
}

function Test-InnoSetupInstalled {
    return [bool](Get-InnoSetupInstallKey)
}

function Get-InnoSetupUninstallerPathForTest {
    param([string]$UninstallString)

    if (-not $UninstallString) {
        return ""
    }
    if ($UninstallString -match '^\s*"([^"]+)"') {
        return $Matches[1]
    }
    return ($UninstallString.Trim() -split '\s+', 2)[0]
}

function Install-InnoSetupForTest {
    param(
        [Parameter(Mandatory=$true)]
        [string]$InstallerPath,
        [int]$TimeoutSeconds = 120
    )

    if (-not (Test-Path -LiteralPath $InstallerPath -PathType Leaf)) {
        throw "Inno Setup installer not found: $InstallerPath"
    }

    $proc = Start-Process -FilePath $InstallerPath `
        -ArgumentList "/VERYSILENT /SUPPRESSMSGBOXES /NORESTART" `
        -WindowStyle Hidden `
        -PassThru
    $completed = $proc.WaitForExit($TimeoutSeconds * 1000)
    if (-not $completed) {
        $proc | Stop-Process -Force -ErrorAction SilentlyContinue
        throw "Inno Setup installer did not exit within $TimeoutSeconds seconds"
    }
    if ($proc.ExitCode -ne 0) {
        throw "Inno Setup installer exited with code $($proc.ExitCode)"
    }
    if (-not (Test-InnoSetupInstalled)) {
        throw "Inno Setup registry key was not created"
    }

    Get-Process -Name "ollama", "Ollama app" -ErrorAction SilentlyContinue |
        Stop-Process -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
}

function Invoke-InnoSetupInstallerForTest {
    param(
        [Parameter(Mandatory=$true)]
        [string]$InstallerPath,
        [int]$TimeoutSeconds = 120
    )

    if (-not (Test-Path -LiteralPath $InstallerPath -PathType Leaf)) {
        throw "Inno Setup installer not found: $InstallerPath"
    }

    $logDir = Get-InstallScriptTestBuildRoot
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
    $logPath = Join-Path $logDir "OllamaSetup-$([Guid]::NewGuid().ToString('N')).log"
    $proc = Start-Process -FilePath $InstallerPath `
        -ArgumentList @("/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART", "/LOG=`"$logPath`"") `
        -WindowStyle Hidden `
        -PassThru
    $completed = $proc.WaitForExit($TimeoutSeconds * 1000)
    if (-not $completed) {
        $proc | Stop-Process -Force -ErrorAction SilentlyContinue
        throw "Inno Setup installer did not exit within $TimeoutSeconds seconds. Log: $logPath"
    }

    $log = @()
    if (Test-Path -LiteralPath $logPath -PathType Leaf) {
        $log = Get-Content -LiteralPath $logPath
    }

    return @{
        ExitCode = $proc.ExitCode
        LogPath  = $logPath
        Log      = $log
    }
}

function Uninstall-InnoSetupForTest {
    $key = Get-InnoSetupInstallKey
    if (-not $key) {
        return
    }

    $uninstallString = (Get-ItemProperty -Path $key -ErrorAction SilentlyContinue).UninstallString
    $uninstallExe = Get-InnoSetupUninstallerPathForTest -UninstallString $uninstallString
    if ($uninstallExe -and (Test-Path -LiteralPath $uninstallExe -PathType Leaf)) {
        Start-Process -FilePath $uninstallExe `
            -ArgumentList "/VERYSILENT /NORESTART /SUPPRESSMSGBOXES" `
            -WindowStyle Hidden `
            -Wait `
            -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 2
    }
    if (Test-Path -LiteralPath $key) {
        Remove-Item -LiteralPath $key -Recurse -Force -ErrorAction SilentlyContinue
    }
}

function Set-InstallScriptDownloadBase {
    param(
        [Parameter(Mandatory=$true)]
        [string]$ScriptPath,

        [Parameter(Mandatory=$true)]
        [string]$BaseUrl
    )

    $content = Get-Content -Path $ScriptPath -Raw
    $replacement = '$DownloadBaseURL = "' + $BaseUrl + '"'
    $updated = $content.Replace('$DownloadBaseURL = "https://ollama.com/download"', $replacement)
    if ($updated -eq $content) {
        throw "Unable to patch DownloadBaseURL in test install.ps1: $ScriptPath"
    }
    $updated | Out-File -FilePath $ScriptPath -Encoding utf8 -Force
}

function Set-InstallScriptTestSignatureVerification {
    param([bool]$Disabled)

    $script:InstallScriptSignatureVerificationDisabled = $Disabled
}

function Get-InstallScriptTestBuildRoot {
    $repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
    return Join-Path $repoRoot "build\install-tests"
}

function Get-InstallTestCacheRoot {
    $repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
    return Join-Path $repoRoot ".cache\install-tests"
}

function Disable-InstallScriptSignatureVerification {
    param(
        [Parameter(Mandatory=$true)]
        [string]$ScriptPath
    )

    $content = Get-Content -Path $ScriptPath -Raw
    $replacement = @'
$script:SignatureVerificationWarningShown = $false

function Test-Signature {
    param([string]$FilePath)

    if (-not $script:SignatureVerificationWarningShown) {
        Write-Warning "Signature verification disabled for local integration tests"
        $script:SignatureVerificationWarningShown = $true
    }
    return $true
}
'@
    $pattern = '(?s)function Test-Signature \{.*?\r?\n\}\r?\n\r?\nfunction Assert-SignatureValid'
    $regex = New-Object System.Text.RegularExpressions.Regex($pattern)
    $updated = $regex.Replace($content, $replacement + "`r`nfunction Assert-SignatureValid", 1)
    if ($updated -eq $content) {
        throw "Unable to disable signature verification in test install.ps1: $ScriptPath"
    }
    $updated | Out-File -FilePath $ScriptPath -Encoding utf8 -Force
}

function Test-OllamaSignedArtifact {
    param([string]$Path)

    try {
        $sig = Get-AuthenticodeSignature -FilePath $Path
        if ($sig.Status -ne "Valid") {
            return $false
        }
        return ($sig.SignerCertificate.Subject -match '(^|, )O=Ollama Inc\.(,|$)')
    } catch {
        return $false
    }
}

function Test-InstallerArtifactsSigned {
    param(
        [Parameter(Mandatory=$true)]
        [string]$DistDir
    )

    $paths = @()
    foreach ($name in @("install.ps1", "OllamaSetup.exe")) {
        $path = Join-Path $DistDir $name
        if (Test-Path -LiteralPath $path -PathType Leaf) {
            $paths += $path
        }
    }
    $paths += @(Get-ChildItem -LiteralPath $DistDir -File -Filter "*.msi" -ErrorAction SilentlyContinue |
        ForEach-Object { $_.FullName })

    if ($paths.Count -eq 0) {
        return $false
    }
    foreach ($path in $paths) {
        if (-not (Test-OllamaSignedArtifact -Path $path)) {
            return $false
        }
    }
    return $true
}

function New-TestInstallScriptCopy {
    param(
        [Parameter(Mandatory=$true)]
        [string]$OutputPath,

        [string]$BaseUrl,
        [bool]$DisableSignatureVerification = $false
    )

    $sourceScript = Join-Path (Split-Path -Parent $PSScriptRoot) "install.ps1"
    Copy-Item -LiteralPath $sourceScript -Destination $OutputPath -Force
    if ($BaseUrl) {
        Set-InstallScriptDownloadBase -ScriptPath $OutputPath -BaseUrl $BaseUrl
    }
    if ($DisableSignatureVerification) {
        Disable-InstallScriptSignatureVerification -ScriptPath $OutputPath
    }
}

function Start-LocalMSIDownloadServer {
    param(
        [Parameter(Mandatory=$true)]
        [string]$DistDir
    )

    $serveDir = Join-Path (Get-InstallScriptTestBuildRoot) "download-$([Guid]::NewGuid().ToString('N'))"
    New-Item -ItemType Directory -Path $serveDir -Force | Out-Null
    Get-ChildItem -LiteralPath $DistDir -File | ForEach-Object {
        Copy-Item -LiteralPath $_.FullName -Destination $serveDir -Force
    }

    $server = Start-LocalHttpServer -Dir $serveDir
    $server["TempDir"] = $serveDir
    New-TestInstallScriptCopy `
        -OutputPath (Join-Path $serveDir "install.ps1") `
        -BaseUrl $server.BaseUrl `
        -DisableSignatureVerification $script:InstallScriptSignatureVerificationDisabled
    return $server
}

function New-InstallScriptForTest {
    param(
        [string]$BaseUrl,
        [object]$DisableSignatureVerification = $null
    )

    $scriptPath = Join-Path (Split-Path -Parent $PSScriptRoot) "install.ps1"
    $disableSignatures = $script:InstallScriptSignatureVerificationDisabled
    if ($null -ne $DisableSignatureVerification) {
        $disableSignatures = [bool]$DisableSignatureVerification
    }

    if (-not $BaseUrl -and -not $disableSignatures) {
        return @{
            Path = $scriptPath
            TempDir = $null
            SignatureVerificationDisabled = $false
        }
    }

    $tempDir = Join-Path (Get-InstallScriptTestBuildRoot) ([Guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $tempDir -Force | Out-Null
    $tempScript = Join-Path $tempDir "install.ps1"
    New-TestInstallScriptCopy `
        -OutputPath $tempScript `
        -BaseUrl $BaseUrl `
        -DisableSignatureVerification $disableSignatures

    return @{
        Path = $tempScript
        TempDir = $tempDir
        SignatureVerificationDisabled = $disableSignatures
    }
}

# Run install.ps1 with given environment variables.
# Returns a hashtable with ExitCode, Output (string[]), and Duration.
function Invoke-InstallScript {
    param(
        [string]$BaseUrl,
        [hashtable]$EnvVars = @{},
        [object]$DisableSignatureVerification = $null
    )

    $testScript = New-InstallScriptForTest -BaseUrl $BaseUrl -DisableSignatureVerification $DisableSignatureVerification
    $scriptPath = $testScript.Path

    $flagSummary = ($EnvVars.Keys | Sort-Object | ForEach-Object { "$_=$($EnvVars[$_])" }) -join ", "
    if ($flagSummary) {
        Write-Host "    Running install.ps1 ($flagSummary)" -ForegroundColor DarkGray
    } else {
        Write-Host "    Running install.ps1 (defaults)" -ForegroundColor DarkGray
    }

    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $stdoutPath = Join-Path $env:TEMP "ollama-install-$([Guid]::NewGuid()).out"
    $stderrPath = Join-Path $env:TEMP "ollama-install-$([Guid]::NewGuid()).err"
    $output = @()
    $exitCode = 0
    try {
        Set-TestProcessEnvironment

        foreach ($key in $EnvVars.Keys) {
            Set-Item -Path "Env:\$key" -Value $EnvVars[$key]
        }

        $proc = Start-Process -FilePath "powershell.exe" `
            -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "`"$scriptPath`"") `
            -WindowStyle Hidden `
            -RedirectStandardOutput $stdoutPath `
            -RedirectStandardError $stderrPath `
            -PassThru
        if ($proc) {
            $proc.WaitForExit()
            $exitCode = $proc.ExitCode
        }

        if (Test-Path $stdoutPath) {
            $output += Get-Content $stdoutPath
        }
        if (Test-Path $stderrPath) {
            $output += Get-Content $stderrPath
        }
    } finally {
        foreach ($key in $EnvVars.Keys) {
            Remove-Item "Env:\$key" -ErrorAction SilentlyContinue
        }
        Remove-Item $stdoutPath, $stderrPath -Force -ErrorAction SilentlyContinue
        if ($testScript.TempDir) {
            Remove-Item -LiteralPath $testScript.TempDir -Recurse -Force -ErrorAction SilentlyContinue
        }
    }
    $sw.Stop()
    Write-Host "    Install completed in $($sw.Elapsed.ToString('mm\:ss'))" -ForegroundColor DarkGray

    return @{
        ExitCode = $exitCode
        Output   = $output
        Duration = $sw.Elapsed
    }
}

# Uninstall all Ollama MSI packages (cleanup after integration tests).
function Invoke-FullUninstall {
    # Use Windows Installer COM API with known UpgradeCodes to find and uninstall
    # all Ollama products. This works for both per-user and system-level installs.
    $installer = New-Object -ComObject WindowsInstaller.Installer
    $windowsDir = if ($env:SystemRoot) { $env:SystemRoot } else { $env:windir }
    if (-not $windowsDir) {
        throw "Unable to resolve Windows system directory for msiexec.exe"
    }
    $msiExec = Join-Path $windowsDir "System32\msiexec.exe"
    if (-not (Test-Path -LiteralPath $msiExec -PathType Leaf)) {
        throw "Unable to find msiexec.exe at $msiExec"
    }
    # Uninstall backends first, then deps, then core (reverse of install order)
    $uninstallOrder = @("mlx_cuda_v13","cuda_v12","cuda_v13","rocm","vulkan",
                        "cuda_v12_deps","cuda_v13_deps","rocm_deps","vulkan_deps",
                        "core","core-arm64")
    foreach ($name in $uninstallOrder) {
        $uc = $script:OllamaUpgradeCodes[$name]
        if (-not $uc) { continue }
        try {
            $related = $installer.RelatedProducts("{$uc}")
            foreach ($pc in $related) {
                Start-Process -FilePath $msiExec -ArgumentList "/x", $pc, "/quiet", "/norestart" `
                    -Wait -NoNewWindow -ErrorAction SilentlyContinue
            }
        } catch { }
    }

    # Also clean up any Ollama processes
    Get-Process -Name "ollama", "Ollama app" -ErrorAction SilentlyContinue |
        Stop-Process -Force -ErrorAction SilentlyContinue

    Uninstall-InnoSetupForTest

    # Remove registry keys
    Remove-Item -Path "HKCU:\Software\Ollama" -Recurse -Force -ErrorAction SilentlyContinue

    # Remove default install dir
    $defaultDir = Join-Path $env:LOCALAPPDATA "Programs\Ollama"
    if (Test-Path $defaultDir) {
        Remove-Item $defaultDir -Recurse -Force -ErrorAction SilentlyContinue
    }

    # Remove cache
    $cacheDir = Join-Path $env:LOCALAPPDATA "Ollama\install_cache"
    if (Test-Path $cacheDir) {
        Remove-Item $cacheDir -Recurse -Force -ErrorAction SilentlyContinue
    }
}

# Check if Ollama is installed by looking for the required core payloads.
function Test-OllamaInstalled {
    param([string]$Dir = "")
    if (-not $Dir) {
        $Dir = Join-Path $env:LOCALAPPDATA "Programs\Ollama"
    }
    return (
        (Test-Path (Join-Path $Dir "ollama.exe")) -and
        (Test-Path (Join-Path $Dir "lib\ollama\llama-server.exe"))
    )
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
    if ($Backend -eq "rocm") {
        $libDir = Join-Path $Dir "lib\ollama"
        if (Test-Path (Join-Path $libDir "rocm")) {
            return $true
        }
        return (@(Get-ChildItem -LiteralPath $libDir -Directory -Filter "rocm_v*" -ErrorAction SilentlyContinue).Count -gt 0)
    }
    return (Test-Path (Join-Path $Dir "lib\ollama\$Backend"))
}

# --------------------------------------------------------------------------
# App-level test helpers (for testing upgrade via app API)
# --------------------------------------------------------------------------

# App runs on port 3001 in dev mode (--dev flag)
$script:AppApiPort = 3001
$script:AppApiBaseUrl = "http://127.0.0.1:$script:AppApiPort"

# Start the Ollama app in test mode (dev mode, hidden, no GUI).
# Dev mode: fixed port 3001, no token auth required.
# Returns hashtable with Process and info needed for cleanup.
function Start-OllamaAppTestMode {
    param(
        [string]$UpdateServerUrl,      # Mock update check server URL
        [string]$DownloadServerUrl,    # Mock download server URL
        [string]$AppExePath,           # Path to "Ollama app.exe" (optional)
        [int]$StartupTimeoutSeconds = 30
    )

    Set-TestProcessEnvironment

    # Find app executable
    if (-not $AppExePath) {
        $AppExePath = Join-Path $env:LOCALAPPDATA "Programs\Ollama\Ollama app.exe"
    }

    if (-not (Test-Path $AppExePath)) {
        throw "Ollama app not found at: $AppExePath"
    }

    # Set environment variables for the app process
    $envVars = @{}
    if ($UpdateServerUrl) {
        $envVars["OLLAMA_UPDATE_URL"] = $UpdateServerUrl
    }
    # Build process start info
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $AppExePath
    $psi.Arguments = "--dev --hide"
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true

    # Set environment variables
    foreach ($key in $envVars.Keys) {
        $psi.EnvironmentVariables[$key] = $envVars[$key]
    }

    # Start the process
    $proc = [System.Diagnostics.Process]::Start($psi)

    if (-not $proc) {
        throw "Failed to start Ollama app process"
    }

    # Wait for the API to become available
    $started = $false
    for ($i = 0; $i -lt $StartupTimeoutSeconds; $i++) {
        if ($proc.HasExited) {
            throw "Ollama app exited unexpectedly with code: $($proc.ExitCode)"
        }

        try {
            $response = Invoke-RestMethod -Uri "$script:AppApiBaseUrl/api/v1/update/check" `
                -Method GET -TimeoutSec 2 -ErrorAction Stop
            $started = $true
            break
        } catch {
            Start-Sleep -Seconds 1
        }
    }

    if (-not $started) {
        $proc | Stop-Process -Force -ErrorAction SilentlyContinue
        throw "Ollama app API failed to respond within $StartupTimeoutSeconds seconds"
    }

    return @{
        Process = $proc
        Port = $script:AppApiPort
        BaseUrl = $script:AppApiBaseUrl
        AppExePath = $AppExePath
    }
}

# Stop the Ollama app and clean up related processes.
function Stop-OllamaApp {
    param(
        [Parameter(ValueFromPipeline=$true)]
        $AppInstance
    )

    # Stop the specific process if provided
    if ($AppInstance -and $AppInstance.Process) {
        if (-not $AppInstance.Process.HasExited) {
            $AppInstance.Process | Stop-Process -Force -ErrorAction SilentlyContinue
        }
    }

    # Also kill any lingering Ollama processes
    Get-Process -Name "ollama", "Ollama app", "ollama app" -ErrorAction SilentlyContinue |
        Stop-Process -Force -ErrorAction SilentlyContinue

    # Give processes time to fully terminate
    Start-Sleep -Milliseconds 500
}

# Call the app's update check API.
# Returns the updateInfo object with: currentVersion, availableVersion, updateAvailable, updateDownloaded
function Invoke-AppUpdateCheck {
    param(
        [string]$BaseUrl = $script:AppApiBaseUrl,
        [int]$TimeoutSeconds = 10
    )

    try {
        $response = Invoke-RestMethod -Uri "$BaseUrl/api/v1/update/check" `
            -Method GET -TimeoutSec $TimeoutSeconds -ErrorAction Stop
        return $response.updateInfo
    } catch {
        throw "Failed to call update check API: $_"
    }
}

# Call the app's install update API to trigger an upgrade.
# The app will download MSIs (if not cached), spawn install.ps1, and exit.
# Returns the API response.
function Invoke-AppInstallUpdate {
    param(
        [string]$BaseUrl = $script:AppApiBaseUrl,
        [int]$TimeoutSeconds = 10
    )

    try {
        $response = Invoke-RestMethod -Uri "$BaseUrl/api/v1/update/install" `
            -Method POST -TimeoutSec $TimeoutSeconds -ErrorAction Stop
        return $response
    } catch {
        throw "Failed to call install update API: $_"
    }
}

# Wait for the app process to exit (after triggering an upgrade).
# Returns $true if exited within timeout, $false otherwise.
function Wait-AppExit {
    param(
        [Parameter(Mandatory=$true)]
        $AppInstance,
        [int]$TimeoutSeconds = 60
    )

    if (-not $AppInstance -or -not $AppInstance.Process) {
        return $true
    }

    $exited = $AppInstance.Process.WaitForExit($TimeoutSeconds * 1000)
    return $exited
}

# Wait for install.ps1 to complete after app triggers upgrade.
# Watches for the installer process and waits for it to finish.
function Wait-InstallerComplete {
    param(
        [int]$TimeoutSeconds = 120
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)

    # Wait for msiexec to appear (install.ps1 calls msiexec)
    $installerStarted = $false
    while ((Get-Date) -lt $deadline) {
        $msiProcs = Get-Process -Name "msiexec" -ErrorAction SilentlyContinue
        if ($msiProcs) {
            $installerStarted = $true
            break
        }
        Start-Sleep -Milliseconds 500
    }

    if (-not $installerStarted) {
        Write-Warning "msiexec process not detected within timeout"
        return $false
    }

    # Wait for all msiexec processes to complete
    while ((Get-Date) -lt $deadline) {
        $msiProcs = Get-Process -Name "msiexec" -ErrorAction SilentlyContinue
        if (-not $msiProcs) {
            return $true
        }
        Start-Sleep -Seconds 1
    }

    return $false
}

# Create a mock update server response file (update.json).
# This mimics what ollama.com/api/update returns.
function New-MockUpdateResponse {
    param(
        [Parameter(Mandatory=$true)]
        [string]$OutputDir,
        [string]$Version = "99.0.0",
        [string]$DownloadBaseUrl
    )

    # Ensure output directory exists
    if (-not (Test-Path $OutputDir)) {
        New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
    }

    # The Windows app updater rewrites this legacy URL shape to install.ps1.
    # The test server serves install.ps1 at the same root.
    $response = @{
        url = "$DownloadBaseUrl/OllamaSetup.exe"
        version = $Version
    }

    $updateJsonPath = Join-Path $OutputDir "update.json"
    $response | ConvertTo-Json | Out-File -FilePath $updateJsonPath -Encoding utf8

    return $updateJsonPath
}

# Create a mock download directory structure that mimics a release.
# This sets up the files that install.ps1 expects to download.
function New-MockDownloadDir {
    param(
        [Parameter(Mandatory=$true)]
        [string]$OutputDir,
        [string]$Version = "99.0.0",
        [string]$SourceDistDir  # Copy MSIs from here if provided
    )

    $versionDir = Join-Path $OutputDir $Version
    if (-not (Test-Path $versionDir)) {
        New-Item -ItemType Directory -Path $versionDir -Force | Out-Null
    }

    if ($SourceDistDir -and (Test-Path $SourceDistDir)) {
        # Copy actual MSIs from dist directory
        $msiFiles = Get-ChildItem -Path $SourceDistDir -Filter "*.msi" -ErrorAction SilentlyContinue
        foreach ($msi in $msiFiles) {
            Copy-Item -Path $msi.FullName -Destination $versionDir -Force
        }

        # Copy install.ps1 to root (not versioned)
        $installScript = Join-Path (Split-Path -Parent $PSScriptRoot) "install.ps1"
        if (Test-Path $installScript) {
            Copy-Item -Path $installScript -Destination $OutputDir -Force
        }
    } else {
        # Create placeholder files for testing
        @("ollama-core.msi", "ollama-cuda-v12.msi", "ollama-vulkan.msi") | ForEach-Object {
            $filePath = Join-Path $versionDir $_
            "placeholder" | Out-File -FilePath $filePath -Encoding utf8
        }
    }

    return $versionDir
}

# Version of OllamaSetup.exe to use for legacy Inno Setup migration tests.
# Change this version when testing against different releases.
$script:InnoSetupTestVersion = "0.21.0"

# Download OllamaSetup.exe from a specific release for migration testing.
# Returns the path to the downloaded file, or $null if download failed.
function Get-InnoSetupInstaller {
    param(
        [string]$Version = $script:InnoSetupTestVersion,
        [string]$OutputDir
    )

    if (-not $OutputDir) {
        $OutputDir = Join-Path (Get-InstallTestCacheRoot) "inno"
    }

    if (-not (Test-Path $OutputDir)) {
        New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
    }

    $outputPath = Join-Path $OutputDir "OllamaSetup-$Version.exe"

    # Return cached file if it exists and is valid
    if (Test-Path $outputPath) {
        $fileInfo = Get-Item $outputPath
        if ($fileInfo.Length -gt 50MB -and (Test-OllamaSignedArtifact -Path $outputPath)) {
            return $outputPath
        }
        # File is incomplete or not an official Ollama-signed installer; re-download it.
        Remove-Item $outputPath -Force
    }

    # Download from GitHub releases
    $downloadUrl = "https://github.com/ollama/ollama/releases/download/v$Version/OllamaSetup.exe"
    $tempOutputPath = "$outputPath.download"
    Remove-Item -LiteralPath $tempOutputPath -Force -ErrorAction SilentlyContinue

    try {
        Write-Host "Downloading OllamaSetup.exe version $Version from GitHub..."
        $request = [System.Net.HttpWebRequest]::Create($downloadUrl)
        $request.AllowAutoRedirect = $true
        $response = $request.GetResponse()
        $totalBytes = $response.ContentLength
        $stream = $response.GetResponseStream()
        $fileStream = [System.IO.FileStream]::new($tempOutputPath, [System.IO.FileMode]::Create)
        $buffer = [byte[]]::new(65536)
        $totalRead = 0
        $lastUpdate = [DateTime]::MinValue
        $barWidth = 40

        try {
            while (($read = $stream.Read($buffer, 0, $buffer.Length)) -gt 0) {
                $fileStream.Write($buffer, 0, $read)
                $totalRead += $read

                $now = [DateTime]::UtcNow
                if (($now - $lastUpdate).TotalMilliseconds -ge 500) {
                    $sizeMB = [math]::Round($totalRead / 1MB, 1)
                    if ($totalBytes -gt 0) {
                        $totalMB = [math]::Round($totalBytes / 1MB, 1)
                        $pct = [math]::Min(100.0, ($totalRead / $totalBytes) * 100)
                        $filled = [math]::Floor($barWidth * $pct / 100)
                        $empty = $barWidth - $filled
                        $bar = ('#' * $filled) + (' ' * $empty)
                        Write-Host -NoNewline "`r  [$bar] $($pct.ToString('0'))%  $sizeMB / $totalMB MB"
                    } else {
                        Write-Host -NoNewline "`r  $sizeMB MB downloaded..."
                    }
                    $lastUpdate = $now
                }
            }
            Write-Host ""  # newline after progress bar
        } finally {
            $fileStream.Close()
            $stream.Close()
            $response.Close()
        }

        # Verify download succeeded
        if (Test-Path $tempOutputPath) {
            $fileInfo = Get-Item $tempOutputPath
            if ($fileInfo.Length -gt 50MB) {
                if (-not (Test-OllamaSignedArtifact -Path $tempOutputPath)) {
                    throw "Downloaded OllamaSetup.exe does not have a valid Ollama signature"
                }
                Move-Item -LiteralPath $tempOutputPath -Destination $outputPath -Force
                $fileInfo = Get-Item $outputPath
                Write-Host "Downloaded OllamaSetup.exe ($([math]::Round($fileInfo.Length / 1MB, 1)) MB)"
                return $outputPath
            }
        }

        Write-Warning "Downloaded file is too small or missing"
        Remove-Item -LiteralPath $tempOutputPath -Force -ErrorAction SilentlyContinue
        return $null
    } catch {
        Remove-Item -LiteralPath $tempOutputPath -Force -ErrorAction SilentlyContinue
        Write-Warning "Failed to download OllamaSetup.exe: $_"
        return $null
    }
}

# Check if the MSI cache directory contains expected files.
function Test-MsiCachePopulated {
    param(
        [string[]]$ExpectedFiles = @("ollama-core.msi"),
        [string]$CacheDir
    )

    if (-not $CacheDir) {
        $CacheDir = Join-Path $env:LOCALAPPDATA "Ollama\install_cache"
    }

    if (-not (Test-Path $CacheDir)) {
        return $false
    }

    $cacheContents = Get-MsiCacheContents -CacheDir $CacheDir
    foreach ($file in $ExpectedFiles) {
        $found = $cacheContents | Where-Object { $_.Name -eq $file -or $_.RelativePath -eq $file } | Select-Object -First 1
        if (-not $found) {
            return $false
        }
    }

    return $true
}

# Get files from the MSI cache. install.ps1 stores complete payloads under
# hash-named subdirectories, so callers must not assume files live at the root.
function Get-MsiCacheContents {
    param(
        [string]$CacheDir
    )

    if (-not $CacheDir) {
        $CacheDir = Join-Path $env:LOCALAPPDATA "Ollama\install_cache"
    }

    if (-not (Test-Path $CacheDir)) {
        return @()
    }

    $root = (Resolve-Path -LiteralPath $CacheDir).Path
    $trimChars = [char[]]@([System.IO.Path]::DirectorySeparatorChar, [System.IO.Path]::AltDirectorySeparatorChar)
    return Get-ChildItem -LiteralPath $CacheDir -Recurse -File | Select-Object `
        Name,
        @{Name="RelativePath"; Expression={ $_.FullName.Substring($root.Length).TrimStart($trimChars) }},
        Length,
        LastWriteTime
}

# Clear the MSI cache directory.
function Clear-MsiCache {
    param(
        [string]$CacheDir
    )

    if (-not $CacheDir) {
        $CacheDir = Join-Path $env:LOCALAPPDATA "Ollama\install_cache"
    }

    if (Test-Path $CacheDir) {
        Remove-Item -Path $CacheDir -Recurse -Force -ErrorAction SilentlyContinue
    }
}

# Check if the upgrade marker file exists (indicates upgrade in progress).
function Test-UpgradeMarkerExists {
    $markerPath = Join-Path $env:LOCALAPPDATA "Ollama\upgraded"
    return (Test-Path $markerPath)
}

# Remove the upgrade marker file.
function Remove-UpgradeMarker {
    $markerPath = Join-Path $env:LOCALAPPDATA "Ollama\upgraded"
    if (Test-Path $markerPath) {
        Remove-Item -Path $markerPath -Force -ErrorAction SilentlyContinue
    }
}

# Get the installed Ollama version by running ollama --version.
function Get-InstalledOllamaVersion {
    param(
        [string]$InstallDir
    )

    if (-not $InstallDir) {
        $InstallDir = Join-Path $env:LOCALAPPDATA "Programs\Ollama"
    }

    $ollamaExe = Join-Path $InstallDir "ollama.exe"
    if (-not (Test-Path $ollamaExe)) {
        return $null
    }

    try {
        $version = & $ollamaExe --version 2>$null
        if ($version -match "ollama version (\S+)") {
            return $matches[1]
        }
        return $version
    } catch {
        return $null
    }
}

Export-ModuleMember -Function *
