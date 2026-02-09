<#
.SYNOPSIS
    Generates packages.json manifest from built GPU backend and dependency MSIs.

.DESCRIPTION
    Scans the output directory for GPU backend and dependency MSIs, computes
    their SHA256 hashes, and generates a packages.json file. This manifest is
    embedded in the core MSI and used by the install script and app updater
    to determine which GPU packages need downloading.

.PARAMETER Version
    The Ollama version string (e.g., "0.15.0")

.PARAMETER DistDir
    Path to the dist directory containing built MSIs (default: ../../dist)

.PARAMETER OutputFile
    Path for the generated packages.json (default: <DistDir>/windows-amd64/packages.json)

.EXAMPLE
    .\generate-packages-json.ps1 -Version 0.15.0
    .\generate-packages-json.ps1 -Version 0.15.0 -DistDir C:\ollama\dist
#>

param(
    [Parameter(Mandatory=$true)]
    [string]$Version,

    [Parameter(Mandatory=$false)]
    [string]$DistDir = "$PSScriptRoot\..\..\dist",

    [Parameter(Mandatory=$false)]
    [string]$OutputFile = ""
)

$ErrorActionPreference = "Stop"

$DistDir = [System.IO.Path]::GetFullPath($DistDir)

if ([string]::IsNullOrEmpty($OutputFile)) {
    $OutputFile = Join-Path $DistDir "windows-amd64\packages.json"
}

Write-Host "Generating packages.json"
Write-Host "  Version: $Version"
Write-Host "  Dist dir: $DistDir"
Write-Host "  Output: $OutputFile"

# Backend package definitions
# Each entry maps a backend name to its MSI filename pattern and deps pattern
$backendDefs = @(
    @{
        Name = "cuda_v12"
        MsiPattern = "ollama-cuda-v12.msi"
        DepsPattern = "ollama-cuda-deps-*.msi"
    },
    @{
        Name = "cuda_v13"
        MsiPattern = "ollama-cuda-v13.msi"
        DepsPattern = "ollama-cuda-v13-deps-*.msi"
    },
    @{
        Name = "rocm"
        MsiPattern = "ollama-rocm.msi"
        DepsPattern = "ollama-rocm-deps-*.msi"
    },
    @{
        Name = "vulkan"
        MsiPattern = "ollama-vulkan.msi"
        DepsPattern = "ollama-vulkan-deps-*.msi"
    }
)

$packages = @()

foreach ($def in $backendDefs) {
    $msiPath = Join-Path $DistDir $def.MsiPattern
    if (-not (Test-Path $msiPath)) {
        Write-Host "  Skipping $($def.Name): $($def.MsiPattern) not found"
        continue
    }

    $msiHash = (Get-FileHash -Path $msiPath -Algorithm SHA256).Hash.ToLower()
    Write-Host "  Found $($def.MsiPattern): $msiHash"

    # Find the deps MSI (glob pattern may match versioned filename)
    $depsFiles = Get-ChildItem -Path $DistDir -Filter $def.DepsPattern -File -ErrorAction SilentlyContinue
    $depsFile = ""
    $depsHash = ""

    if ($depsFiles -and $depsFiles.Count -gt 0) {
        # Use the first match (there should only be one per backend)
        $depsPath = $depsFiles[0].FullName
        $depsFile = $depsFiles[0].Name
        $depsHash = (Get-FileHash -Path $depsPath -Algorithm SHA256).Hash.ToLower()
        Write-Host "  Found ${depsFile}: $depsHash"
    } else {
        Write-Host "  WARNING: No deps MSI found for $($def.Name) (pattern: $($def.DepsPattern))"
    }

    $pkg = @{
        name = $def.Name
        file = $def.MsiPattern
        sha256 = $msiHash
    }

    if ($depsFile) {
        $pkg.deps = $depsFile
        $pkg.deps_sha256 = $depsHash
    }

    $packages += $pkg
}

# Build the manifest object
$manifest = [ordered]@{
    version = $Version
    packages = $packages
}

# Convert to JSON and write
$json = $manifest | ConvertTo-Json -Depth 3
$json | Out-File -FilePath $OutputFile -Encoding utf8 -Force

# Also generate an ARM64 packages.json (empty packages array for now)
$arm64OutputFile = Join-Path $DistDir "windows-arm64\packages.json"
$arm64Dir = Split-Path $arm64OutputFile -Parent
if (Test-Path $arm64Dir) {
    $arm64Manifest = [ordered]@{
        version = $Version
        packages = @()
    }
    $arm64Json = $arm64Manifest | ConvertTo-Json -Depth 3
    $arm64Json | Out-File -FilePath $arm64OutputFile -Encoding utf8 -Force
    Write-Host "  Generated ARM64 packages.json (empty packages)"
}

Write-Host ""
Write-Host "Generated packages.json with $($packages.Count) package(s):"
Write-Host $json
