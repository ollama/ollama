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
    Path for the generated packages.json (default: <DistDir>/packages.json)

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
    $OutputFile = Join-Path $DistDir "packages.json"
}

Write-Host "Generating packages.json"
Write-Host "  Version: $Version"
Write-Host "  Dist dir: $DistDir"
Write-Host "  Output: $OutputFile"

function Resolve-BackendPath {
    param(
        [Parameter(Mandatory=$true)]
        [string]$Arch,

        [Parameter(Mandatory=$true)]
        [string]$Pattern,

        [Parameter(Mandatory=$true)]
        [string]$Default
    )

    $libRoot = Join-Path $DistDir ("windows-{0}\lib\ollama" -f $Arch)
    $candidates = @()
    if (Test-Path $libRoot) {
        $candidates = @(Get-ChildItem -Path $libRoot -Directory -Filter $Pattern -ErrorAction SilentlyContinue)
    }

    if ($candidates.Count -gt 1) {
        $names = ($candidates | ForEach-Object { $_.Name }) -join ", "
        throw "Multiple backend directories match ${Pattern}: $names"
    }
    if ($candidates.Count -eq 1) {
        return $candidates[0].Name
    }

    return $Default
}

$rocmBackendPath = Resolve-BackendPath -Arch "amd64" -Pattern "rocm_v*" -Default "rocm_v7_1"

# Backend package definitions
# Each entry maps a backend name to its MSI filename, dependency package stem,
# version breadcrumb, and target architecture.
# The arch field is embedded in packages.json so consumers (install.ps1, app updater) can filter
# packages for the running architecture. Today all backends are amd64-only, but when ARM64 GPU
# backends are added they will appear here with Arch = "arm64".
$backendDefs = @(
    @{
        Name = "cuda_v12"
        Arch = "amd64"
        MsiFile = "ollama-cuda-v12.msi"
        BackendPath = "cuda_v12"
        DepsStem = "cuda"
        VersionFile = "cuda-version.txt"
    },
    @{
        Name = "cuda_v13"
        Arch = "amd64"
        MsiFile = "ollama-cuda-v13.msi"
        BackendPath = "cuda_v13"
        DepsStem = "cuda-v13"
        VersionFile = "cuda-version.txt"
        DepsVersionBackendPath = "mlx_cuda_v13"
        DepsVersionFile = "mlx-version.txt"
    },
    @{
        Name = $rocmBackendPath
        Arch = "amd64"
        MsiFile = "ollama-rocm.msi"
        BackendPath = $rocmBackendPath
        DepsStem = "rocm"
        VersionFile = "rocm-version.txt"
    },
    @{
        Name = "vulkan"
        Arch = "amd64"
        MsiFile = "ollama-vulkan.msi"
        BackendPath = "vulkan"
        DepsStem = "vulkan"
        VersionFile = "vulkan-version.txt"
    },
    @{
        Name = "mlx_cuda_v13"
        Arch = "amd64"
        MsiFile = "ollama-mlx-cuda-v13.msi"
        BackendPath = "mlx_cuda_v13"
        DepsStem = "cuda-v13"
        VersionFile = "mlx-version.txt"
    }
)

function Get-VersionBreadcrumb {
    param(
        [Parameter(Mandatory=$true)]
        [string]$Arch,

        [Parameter(Mandatory=$true)]
        [string]$BackendPath,

        [Parameter(Mandatory=$true)]
        [string]$VersionFile
    )

    $versionPath = Join-Path $DistDir ("windows-{0}\lib\ollama\{1}\{2}" -f $Arch, $BackendPath, $VersionFile)
    if (-not (Test-Path $versionPath)) {
        return ""
    }

    $version = Get-Content -Path $versionPath -TotalCount 1 | Select-Object -First 1
    if ($version) {
        $version = $version.Trim()
    }

    return $version
}

function Get-DependencyVersion {
    param(
        [Parameter(Mandatory=$true)]
        [hashtable]$Definition
    )

    if ($Definition.ContainsKey("DepsVersionBackendPath")) {
        $depsVersion = Get-VersionBreadcrumb `
            -Arch $Definition.Arch `
            -BackendPath $Definition.DepsVersionBackendPath `
            -VersionFile $Definition.DepsVersionFile
        if ($depsVersion) {
            return $depsVersion
        }
    }

    return Get-VersionBreadcrumb `
        -Arch $Definition.Arch `
        -BackendPath $Definition.BackendPath `
        -VersionFile $Definition.VersionFile
}

function Get-DependencyMsi {
    param(
        [Parameter(Mandatory=$true)]
        [hashtable]$Definition
    )

    $depsPattern = "ollama-$($Definition.DepsStem)-deps-*.msi"
    $genericDepsFile = "ollama-$($Definition.DepsStem)-deps.msi"
    $genericDepsPath = Join-Path $DistDir $genericDepsFile

    $candidates = @()
    $candidates += @(Get-ChildItem -Path $DistDir -Filter $depsPattern -File -ErrorAction SilentlyContinue)
    if (Test-Path $genericDepsPath) {
        $candidates += @(Get-Item $genericDepsPath)
    }
    $candidates = @($candidates | Sort-Object Name -Unique)

    if ($candidates.Count -gt 1) {
        $names = ($candidates | ForEach-Object { $_.Name }) -join ", "
        throw "Multiple dependency MSIs found for $($Definition.Name): $names. Remove stale dependency MSIs before generating packages.json."
    }

    $depsVersion = Get-DependencyVersion -Definition $Definition
    if ($depsVersion) {
        $expectedDepsFile = "ollama-$($Definition.DepsStem)-deps-$depsVersion.msi"
        if ($candidates.Count -eq 0) {
            throw "Dependency MSI missing for $($Definition.Name): expected $expectedDepsFile"
        }
        if ($candidates[0].Name -ne $expectedDepsFile) {
            throw "Dependency MSI for $($Definition.Name) does not match version breadcrumb: found $($candidates[0].Name), expected $expectedDepsFile"
        }
        return $candidates[0]
    }

    if ($candidates.Count -eq 1) {
        Write-Host "  WARNING: No version breadcrumb found for $($Definition.Name); using $($candidates[0].Name)"
        return $candidates[0]
    }

    throw "Dependency MSI missing for $($Definition.Name): no version breadcrumb found and no generic $genericDepsFile exists"
}

$packages = @()

foreach ($def in $backendDefs) {
    $msiPath = Join-Path $DistDir $def.MsiFile
    if (-not (Test-Path $msiPath)) {
        Write-Host "  Skipping $($def.Name): $($def.MsiFile) not found"
        continue
    }

    $msiHash = (Get-FileHash -Path $msiPath -Algorithm SHA256).Hash.ToLower()
    Write-Host "  Found $($def.MsiFile): $msiHash"

    $depsMsi = Get-DependencyMsi -Definition $def
    $depsFile = ""
    $depsHash = ""

    if ($depsMsi) {
        $depsFile = $depsMsi.Name
        $depsHash = (Get-FileHash -Path $depsMsi.FullName -Algorithm SHA256).Hash.ToLower()
        Write-Host "  Found ${depsFile}: $depsHash"
    }

    $pkg = @{
        name = $def.Name
        arch = $def.Arch
        file = $def.MsiFile
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

# Convert to JSON
$json = $manifest | ConvertTo-Json -Depth 3

# Compare-and-write: only update the file if content changed.
# This preserves the file timestamp when hashes haven't changed,
# so downstream MSI builds (which check timestamps) skip rebuilding.
$needsWrite = $true
if (Test-Path $OutputFile) {
    $existing = Get-Content -Path $OutputFile -Raw -ErrorAction SilentlyContinue
    if ($existing -and ($existing.TrimEnd() -eq $json.TrimEnd())) {
        Write-Host "  packages.json unchanged, skipping write"
        $needsWrite = $false
    }
}
if ($needsWrite) {
    $json | Out-File -FilePath $OutputFile -Encoding utf8 -Force
}

Write-Host ""
Write-Host "Generated packages.json with $($packages.Count) package(s):"
Write-Host $json
