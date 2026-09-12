#!powershell

param(
    [string]$Prefix
)

$ErrorActionPreference = "Stop"

$version = "7.14.0"
$url = "https://repo.amd.com/rocm/tarball-multi-arch/therock-dist-windows-multiarch-$version.tar.gz"

if (-not $Prefix) {
    $root = (& git rev-parse --show-toplevel 2>$null)
    if (-not $root) {
        $root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
    }
    $Prefix = Join-Path ([string]$root).Trim() ".cache\rocm"
}

$Prefix = [IO.Path]::GetFullPath($Prefix)
$archive = Join-Path $Prefix "archives\therock-dist-windows-multiarch-$version.tar.gz"
$install = Join-Path $Prefix "windows-multiarch-$version"
$envFile = Join-Path $install "ollama-rocm-env.ps1"

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $archive) | Out-Null
if (-not (Test-Path -LiteralPath $archive)) {
    curl.exe --location --fail --retry 5 --output $archive $url
    if ($LASTEXITCODE -ne 0) {
        throw "curl failed with exit code $LASTEXITCODE"
    }
}

if (-not (Test-Path -LiteralPath (Join-Path $install ".ollama-rocm-ready"))) {
    $tmp = Join-Path $Prefix ".windows-multiarch-$version.tmp"
    Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Force -Path $tmp | Out-Null

    tar -xzf $archive -C $tmp
    if ($LASTEXITCODE -ne 0) {
        throw "tar failed with exit code $LASTEXITCODE"
    }

    $children = @(Get-ChildItem -LiteralPath $tmp -Force)
    if ($children.Count -eq 1 -and $children[0].PSIsContainer) {
        foreach ($item in Get-ChildItem -LiteralPath $children[0].FullName -Force) {
            Move-Item -LiteralPath $item.FullName -Destination $tmp
        }
        Remove-Item -LiteralPath $children[0].FullName -Recurse -Force
    }

    Remove-Item -LiteralPath $install -Recurse -Force -ErrorAction SilentlyContinue
    Move-Item -LiteralPath $tmp -Destination $install
    Set-Content -LiteralPath (Join-Path $install ".ollama-rocm-ready") -Value (Get-Date -Format o) -Encoding ascii
}

$llvmAmdgcn = Join-Path $install "lib\llvm\amdgcn"
$rocmAmdgcn = Join-Path $install "amdgcn"
if ((Test-Path -LiteralPath $llvmAmdgcn) -and -not (Test-Path -LiteralPath $rocmAmdgcn)) {
    Copy-Item -LiteralPath $llvmAmdgcn -Destination $rocmAmdgcn -Recurse -Force
}

$escapedInstall = $install.Replace("'", "''")
$content = @"
`$env:HIP_PATH = '$escapedInstall'
`$env:HIP_DIR = `$env:HIP_PATH
`$env:ROCM_PATH = `$env:HIP_PATH
`$env:HIP_PLATFORM = 'amd'
`$env:HIP_CLANG_PATH = Join-Path `$env:HIP_PATH 'lib\llvm\bin'
`$env:HIP_DEVICE_LIB_PATH = Join-Path `$env:HIP_PATH 'amdgcn\bitcode'
if (-not (Test-Path `$env:HIP_DEVICE_LIB_PATH)) {
    `$env:HIP_DEVICE_LIB_PATH = Join-Path `$env:HIP_PATH 'lib\llvm\amdgcn\bitcode'
}
`$env:ROCM_DEVICE_LIB_PATH = `$env:HIP_DEVICE_LIB_PATH
`$env:CC = Join-Path `$env:HIP_CLANG_PATH 'clang.exe'
`$env:CXX = Join-Path `$env:HIP_CLANG_PATH 'clang++.exe'
`$env:HIPCXX = `$env:CXX
`$env:CMAKE_PREFIX_PATH = `$env:HIP_PATH
`$entries = @(`$env:HIP_CLANG_PATH, (Join-Path `$env:HIP_PATH 'bin'))
`$env:Path = ((`$entries + (`$env:Path -split ';' | Where-Object { `$_ })) | Select-Object -Unique) -join ';'
`$llvmRc = Join-Path `$env:HIP_CLANG_PATH 'llvm-rc.exe'
if (Test-Path `$llvmRc) {
    `$env:RC = `$llvmRc
}
"@
Set-Content -LiteralPath $envFile -Value $content -Encoding ascii

Write-Output "ROCm $version`: $install"
