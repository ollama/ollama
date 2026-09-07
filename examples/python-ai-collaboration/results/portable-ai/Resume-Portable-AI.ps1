[CmdletBinding()]
param(
    [switch]$InstallDriver
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "Initialize-PortableEnvironment.ps1")

$requiredDriver = [version]"616.64"
$driverInstaller = Join-Path $script:EnvironmentRoot `
    "drivers\616.64-desktop-win10-win11-64bit-international-dch-whql.exe"
$stateDirectory = Join-Path $script:EnvironmentRoot "state"
$rebootMarker = Join-Path $stateDirectory "driver-reboot-required"

function Get-NvidiaDriverVersion {
    $nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($null -eq $nvidiaSmi) {
        return $null
    }

    $output = @(& $nvidiaSmi.Source `
        "--query-gpu=driver_version" `
        "--format=csv,noheader")
    if ($LASTEXITCODE -ne 0 -or $output.Count -eq 0) {
        return $null
    }
    return [version]([string]$output[0]).Trim()
}

$installedDriver = Get-NvidiaDriverVersion
$driverUpdateNeeded = (
    $null -eq $installedDriver -or
    $installedDriver -lt $requiredDriver
)

if ($driverUpdateNeeded) {
    if (-not $InstallDriver) {
        Write-Error "NVIDIA driver $requiredDriver is required. Re-run with -InstallDriver."
        exit 2
    }
    if (-not (Test-Path -LiteralPath $driverInstaller -PathType Leaf)) {
        throw "Driver installer is missing: $driverInstaller"
    }

    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    $isAdministrator = $principal.IsInRole(
        [Security.Principal.WindowsBuiltInRole]::Administrator
    )
    if (-not $isAdministrator) {
        $arguments = @(
            "-NoProfile"
            "-ExecutionPolicy", "Bypass"
            "-File", "`"$PSCommandPath`""
            "-InstallDriver"
        ) -join " "
        Start-Process powershell.exe -Verb RunAs -ArgumentList $arguments
        Write-Host "Approve the administrator prompt to restore the NVIDIA driver."
        exit 0
    }

    Write-Host "Installing NVIDIA driver $requiredDriver..." -ForegroundColor Cyan
    $installer = Start-Process -FilePath $driverInstaller `
        -ArgumentList "-s", "-noreboot" `
        -Wait `
        -PassThru
    if ($installer.ExitCode -notin @(0, 3010)) {
        throw "NVIDIA installer failed with exit code $($installer.ExitCode)."
    }

    $installedDriver = Get-NvidiaDriverVersion
    if (
        $installer.ExitCode -eq 3010 -or
        $null -eq $installedDriver -or
        $installedDriver -lt $requiredDriver
    ) {
        New-Item -ItemType Directory -Path $stateDirectory -Force | Out-Null
        Set-Content -LiteralPath $rebootMarker `
            -Value "Driver $requiredDriver staged; reboot required." `
            -Encoding ASCII
        Write-Host "Driver update requires a reboot."
        Write-Host "Stop Ollama, eject the SSD, reboot, reconnect it, then run Resume-Portable-AI.cmd."
        exit 3010
    }
}

if (Test-Path -LiteralPath $rebootMarker) {
    Remove-Item -LiteralPath $rebootMarker -Force
}

Write-Host "NVIDIA driver $installedDriver is ready."
& (Join-Path $PSScriptRoot "Start-Portable-Ollama.ps1") -Background
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host "Portable AI environment is ready."
Write-Host "Models: $env:OLLAMA_MODELS"
