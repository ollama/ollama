#!powershell
#
# Download and unpack a Windows ROCm SDK tarball from AMD TheRock nightlies.
#
# Examples:
#   powershell -ExecutionPolicy Bypass -File .\scripts\fetch_therock_rocm_windows.ps1
#   powershell -ExecutionPolicy Bypass -File .\scripts\fetch_therock_rocm_windows.ps1 -Target gfx110X
#   powershell -ExecutionPolicy Bypass -File .\scripts\fetch_therock_rocm_windows.ps1 -Version 7.13.0a20260430
#
# The default prefix is repo-local and does not require administrator access.

[CmdletBinding()]
param(
    [string]$Prefix,
    [string]$Version = "latest",
    [ValidateSet("multiarch", "gfx103X", "gfx110X", "gfx1150", "gfx1151", "gfx120X", "gfx942", "gfx950")]
    [string]$Target = "multiarch",
    [string]$ArtifactBaseUrl = "https://rocm.nightlies.amd.com/tarball-multi-arch",
    [string]$HipSdkPath,
    [ValidateSet("auto", "curl", "bits", "webrequest")]
    [string]$DownloadMethod = "auto",
    [switch]$ResolveOnly,
    [switch]$Force
)

$ErrorActionPreference = "Stop"

function repoRoot {
    $root = $null
    try {
        $root = & git rev-parse --show-toplevel 2>$null
    } catch {
        $root = $null
    }
    if ($root) {
        return ([string]$root).Trim()
    }
    return (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}

function targetArtifactName {
    param([string]$Name)

    switch ($Name) {
        "multiarch" { return "multiarch" }
        "gfx103X" { return "gfx103X-all" }
        "gfx110X" { return "gfx110X-all" }
        "gfx120X" { return "gfx120X-all" }
        default { return $Name }
    }
}

function readArtifactKeys {
    param(
        [string]$BaseUrl,
        [string]$Prefix
    )

    $base = $BaseUrl.TrimEnd("/")
    $isS3List = $base -match "s3\.amazonaws\.com$"
    $uri = $(if ($isS3List) { "${base}/?prefix=${Prefix}" } else { "${base}/" })
    Write-Output "Querying $uri"
    $content = (Invoke-WebRequest -Uri $uri -UseBasicParsing).Content

    if ($isS3List) {
        [xml]$listing = $content
        $keys = @()
        foreach ($contents in $listing.ListBucketResult.Contents) {
            if ($contents.Key) {
                $keys += [string]$contents.Key
            }
        }
        return $keys
    }

    $matches = [regex]::Matches($content, "therock-dist-windows-[A-Za-z0-9.-]+\.tar\.gz")
    return @($matches | ForEach-Object { $_.Value } | Where-Object { $_ -like "$Prefix*" } | Select-Object -Unique)
}

function parseArtifactVersion {
    param(
        [string]$Key,
        [string]$ArtifactTarget
    )

    $escaped = [regex]::Escape($ArtifactTarget)
    if ($Key -match "^therock-dist-windows-${escaped}-(\d+)\.(\d+)\.(\d+)(a|rc)(\d+)\.tar\.gz$") {
        return [PSCustomObject]@{
            Key = $Key
            Version = "$($matches[1]).$($matches[2]).$($matches[3])$($matches[4])$($matches[5])"
            Major = [int]$matches[1]
            Minor = [int]$matches[2]
            Patch = [int]$matches[3]
            ChannelRank = $(if ($matches[4] -eq "a") { 1 } else { 0 })
            Build = [int]$matches[5]
        }
    }
    return $null
}

function resolveArtifact {
    param(
        [string]$BaseUrl,
        [string]$ArtifactTarget,
        [string]$RequestedVersion
    )

    $prefix = "therock-dist-windows-${ArtifactTarget}-"
    if ($RequestedVersion -ne "latest") {
        $key = "${prefix}${RequestedVersion}.tar.gz"
        return [PSCustomObject]@{
            Key = $key
            Version = $RequestedVersion
        Url = "$($BaseUrl.TrimEnd('/'))/${key}"
        }
    }

    $keys = readArtifactKeys $BaseUrl "${prefix}7"
    $artifacts = @()
    foreach ($key in $keys) {
        $artifact = parseArtifactVersion $key $ArtifactTarget
        if ($artifact) {
            $artifacts += $artifact
        }
    }
    if ($artifacts.Count -eq 0) {
        throw "No TheRock Windows tarballs found for target '${ArtifactTarget}' at ${BaseUrl}"
    }

    $latest = $artifacts |
        Sort-Object Major, Minor, Patch, ChannelRank, Build |
        Select-Object -Last 1

    return [PSCustomObject]@{
        Key = $latest.Key
        Version = $latest.Version
        Url = "$($BaseUrl.TrimEnd('/'))/$($latest.Key)"
    }
}

function ensureArchive {
    param(
        [string]$Url,
        [string]$ArchivePath,
        [string]$Method,
        [switch]$ForceDownload
    )

    if ((Test-Path -LiteralPath $ArchivePath) -and -not $ForceDownload) {
        Write-Output "Using cached archive $ArchivePath"
        return
    }

    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $ArchivePath) | Out-Null
    $partial = "${ArchivePath}.partial"
    if ($ForceDownload -and (Test-Path -LiteralPath $partial)) {
        Remove-Item -LiteralPath $partial -Force
    }
    Write-Output "Downloading $Url"

    $curl = Get-Command -Name "curl.exe" -ErrorAction SilentlyContinue | Select-Object -First 1
    $hasBits = [bool](Get-Command -Name "Start-BitsTransfer" -ErrorAction SilentlyContinue)
    $selected = $Method
    if ($selected -eq "auto") {
        if ($curl) {
            $selected = "curl"
        } elseif ($hasBits) {
            $selected = "bits"
        } else {
            $selected = "webrequest"
        }
    }

    Write-Output "Download method: $selected"
    if ($selected -eq "curl") {
        if (-not $curl) {
            throw "Download method 'curl' requested but curl.exe was not found"
        }
        & $curl.Path --location --fail --retry 5 --retry-delay 5 --continue-at - --output $partial $Url
        if ($LASTEXITCODE -ne 0) {
            throw "curl failed with exit code $LASTEXITCODE"
        }
    } elseif ($selected -eq "bits") {
        if (-not $hasBits) {
            throw "Download method 'bits' requested but Start-BitsTransfer is not available"
        }
        if (Test-Path -LiteralPath $partial) {
            Write-Output "BITS does not resume .partial downloads; restarting $partial"
            Remove-Item -LiteralPath $partial -Force
        }
        Start-BitsTransfer -Source $Url -Destination $partial
    } elseif ($selected -eq "webrequest") {
        if (Test-Path -LiteralPath $partial) {
            Write-Output "Invoke-WebRequest does not resume .partial downloads; restarting $partial"
            Remove-Item -LiteralPath $partial -Force
        }
        Invoke-WebRequest -Uri $Url -OutFile $partial -UseBasicParsing
    } else {
        throw "Unknown download method: $selected"
    }

    Move-Item -LiteralPath $partial -Destination $ArchivePath -Force
}

function clearDirectory {
    param([string]$Path)

    if (Test-Path -LiteralPath $Path) {
        Remove-Item -LiteralPath $Path -Recurse -Force
    }
    New-Item -ItemType Directory -Force -Path $Path | Out-Null
}

function installArchive {
    param(
        [string]$ArchivePath,
        [string]$InstallPath,
        [switch]$ForceInstall
    )

    $marker = Join-Path $InstallPath ".ollama-therock-ready"
    if ((Test-Path -LiteralPath $marker) -and -not $ForceInstall) {
        Write-Output "Using existing ROCm install $InstallPath"
        return
    }

    $parent = Split-Path -Parent $InstallPath
    New-Item -ItemType Directory -Force -Path $parent | Out-Null
    $tmp = Join-Path $parent ("." + (Split-Path -Leaf $InstallPath) + ".tmp")
    clearDirectory $tmp

    Write-Output "Extracting $ArchivePath"
    & tar -xzf $ArchivePath -C $tmp
    if ($LASTEXITCODE -ne 0) {
        throw "tar failed with exit code $LASTEXITCODE"
    }

    $children = @(Get-ChildItem -LiteralPath $tmp -Force)
    if ($children.Count -eq 1 -and $children[0].PSIsContainer) {
        $inner = $children[0].FullName
        foreach ($item in Get-ChildItem -LiteralPath $inner -Force) {
            Move-Item -LiteralPath $item.FullName -Destination $tmp
        }
        Remove-Item -LiteralPath $inner -Recurse -Force
    }

    if (Test-Path -LiteralPath $InstallPath) {
        Remove-Item -LiteralPath $InstallPath -Recurse -Force
    }
    Move-Item -LiteralPath $tmp -Destination $InstallPath
    Set-Content -LiteralPath $marker -Value (Get-Date -Format o) -Encoding ascii
}

function copyIfMissing {
    param(
        [string]$FromRoot,
        [string]$ToRoot,
        [string[]]$RelativePaths
    )

    foreach ($relative in $RelativePaths) {
        $src = Join-Path $FromRoot $relative
        $dst = Join-Path $ToRoot $relative
        if ((Test-Path -LiteralPath $src) -and -not (Test-Path -LiteralPath $dst)) {
            New-Item -ItemType Directory -Force -Path (Split-Path -Parent $dst) | Out-Null
            Copy-Item -LiteralPath $src -Destination $dst -Recurse -Force
            Write-Output "Augmented missing $relative from HIP SDK"
        }
    }
}

function augmentFromHipSdk {
    param(
        [string]$InstallPath,
        [string]$SdkPath
    )

    if (-not $SdkPath) {
        return
    }
    if (-not (Test-Path -LiteralPath $SdkPath)) {
        throw "HIP SDK path not found: $SdkPath"
    }

    copyIfMissing $SdkPath $InstallPath @(
        "bin\llvm-rc.exe",
        "bin\hipconfig.exe",
        "bin\hipcc.exe",
        "bin\amdhip64_7.dll",
        "bin\amdhip64.dll",
        "include\hip",
        "lib\cmake\hip",
        "lib\cmake\hip-lang",
        "lib\cmake\amd_comgr",
        "lib\cmake\hsa-runtime64"
    )
}

function promoteTheRockHostTools {
    param([string]$InstallPath)

    $toolNames = @(
        "amdgpu-arch.exe",
        "clang.exe",
        "clang++.exe",
        "clang-linker-wrapper.exe",
        "clang-offload-bundler.exe",
        "clang-offload-packager.exe",
        "clang-offload-wrapper.exe",
        "ld.lld.exe",
        "lld.exe",
        "lld-link.exe",
        "llvm-ar.exe",
        "llvm-lib.exe",
        "llvm-objcopy.exe",
        "llvm-readobj.exe",
        "llvm-symbolizer.exe"
    )
    $srcDir = Join-Path $InstallPath "lib\llvm\bin"
    $dstDir = Join-Path $InstallPath "bin"
    foreach ($tool in $toolNames) {
        $src = Join-Path $srcDir $tool
        $dst = Join-Path $dstDir $tool
        if ((Test-Path -LiteralPath $src) -and -not (Test-Path -LiteralPath $dst)) {
            New-Item -ItemType Directory -Force -Path $dstDir | Out-Null
            Copy-Item -LiteralPath $src -Destination $dst -Force
            Write-Output "Promoted TheRock tool $tool into bin"
        }
    }
}

function promoteTheRockDeviceLibraries {
    param([string]$InstallPath)

    $src = Join-Path $InstallPath "lib\llvm\amdgcn"
    $dst = Join-Path $InstallPath "amdgcn"
    if ((Test-Path -LiteralPath $src) -and -not (Test-Path -LiteralPath $dst)) {
        Copy-Item -LiteralPath $src -Destination $dst -Recurse -Force
        Write-Output "Promoted TheRock device libraries into amdgcn"
    }
}

function writeEnvironmentFile {
    param(
        [string]$InstallPath,
        [string]$EnvPath
    )

    $escaped = $InstallPath.Replace("'", "''")
    $content = @"
`$env:HIP_PATH = '$escaped'
`$env:HIP_DIR = `$env:HIP_PATH
`$env:ROCM_PATH = '$escaped'
`$env:OLLAMA_THEROCK_ROCM_PATH = `$env:HIP_PATH
`$env:HIP_PLATFORM = 'amd'

`$hipBin = Join-Path `$env:HIP_PATH 'bin'
`$hipLlvmBin = Join-Path `$env:HIP_PATH 'lib\llvm\bin'
`$hipDeviceLibPath = Join-Path `$env:HIP_PATH 'amdgcn\bitcode'
if (-not (Test-Path `$hipDeviceLibPath)) {
    `$hipDeviceLibPath = Join-Path `$env:HIP_PATH 'lib\llvm\amdgcn\bitcode'
}
`$env:HIP_CLANG_PATH = `$hipLlvmBin
if (Test-Path `$hipDeviceLibPath) {
    `$env:HIP_DEVICE_LIB_PATH = `$hipDeviceLibPath
    `$env:ROCM_DEVICE_LIB_PATH = `$hipDeviceLibPath
}

`$clangExe = Join-Path `$hipLlvmBin 'clang.exe'
`$clangxxExe = Join-Path `$hipLlvmBin 'clang++.exe'
if (-not (Test-Path `$clangExe)) {
    `$clangExe = Join-Path `$hipBin 'clang.exe'
}
if (-not (Test-Path `$clangxxExe)) {
    `$clangxxExe = Join-Path `$hipBin 'clang++.exe'
}
`$env:CC = `$clangExe
`$env:CXX = `$clangxxExe
`$env:HIPCXX = `$clangxxExe

`$env:CMAKE_PREFIX_PATH = `$env:HIP_PATH
`$env:CMAKE_PROGRAM_PATH = `$hipLlvmBin + ';' + `$hipBin

`$pathEntries = @(`$hipLlvmBin, `$hipBin)
`$existingPath = @()
`$currentPath = [Environment]::GetEnvironmentVariable('Path', 'Process')
if (-not `$currentPath) {
    `$currentPath = `$env:Path
}
if (-not `$currentPath) {
    `$currentPath = `$env:PATH
}
if (`$currentPath) {
    `$existingPath = `$currentPath -split ';' | Where-Object { `$_ }
}
`$normalizedPath = ((`$pathEntries + `$existingPath) | Select-Object -Unique) -join ';'
try {
    Get-ChildItem Env: | Out-Null
} catch {
    Remove-Item Env:PATH -ErrorAction SilentlyContinue
}
[Environment]::SetEnvironmentVariable('Path', `$normalizedPath, 'Process')
`$env:Path = `$normalizedPath

`$llvmRc = Join-Path `$hipBin 'llvm-rc.exe'
if (-not (Test-Path `$llvmRc)) {
    `$llvmRc = Join-Path `$hipLlvmBin 'llvm-rc.exe'
}
if (Test-Path `$llvmRc) {
    `$env:RC = `$llvmRc
}
"@
    Set-Content -LiteralPath $EnvPath -Value $content -Encoding ascii
}

$root = repoRoot
if (-not $Prefix) {
    $Prefix = Join-Path $root ".cache\therock-rocm"
}
$Prefix = [System.IO.Path]::GetFullPath($Prefix)

$artifactTarget = targetArtifactName $Target
$artifact = resolveArtifact $ArtifactBaseUrl $artifactTarget $Version
$archivePath = Join-Path $Prefix "archives\$($artifact.Key)"
$installPath = Join-Path $Prefix "windows-$Target-$($artifact.Version)"
$envPath = Join-Path $installPath "ollama-therock-env.ps1"

if ($ResolveOnly) {
    Write-Output "Resolved TheRock ROCm artifact:"
    Write-Output "  Version: $($artifact.Version)"
    Write-Output "  Target:  $Target ($artifactTarget)"
    Write-Output "  URL:     $($artifact.Url)"
    Write-Output "  ROCm:    $installPath"
    return
}

ensureArchive $artifact.Url $archivePath $DownloadMethod -ForceDownload:$Force
installArchive $archivePath $installPath -ForceInstall:$Force
promoteTheRockHostTools $installPath
promoteTheRockDeviceLibraries $installPath
augmentFromHipSdk $installPath $HipSdkPath
writeEnvironmentFile $installPath $envPath

$manifest = [PSCustomObject]@{
    target = $Target
    artifactTarget = $artifactTarget
    version = $artifact.Version
    url = $artifact.Url
    archive = $archivePath
    installPath = $installPath
    envFile = $envPath
}
$manifestPath = Join-Path $installPath "ollama-therock-manifest.json"
$manifest | ConvertTo-Json | Set-Content -LiteralPath $manifestPath -Encoding ascii

Write-Output ""
Write-Output "TheRock ROCm is ready:"
Write-Output "  Version: $($artifact.Version)"
Write-Output "  Target:  $Target ($artifactTarget)"
Write-Output "  ROCm:    $installPath"
Write-Output "  Env:     $envPath"
