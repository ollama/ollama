#!powershell

param(
    [string]$Prefix
)

$ErrorActionPreference = "Stop"

$version = "10.0.0"
$indexURL = "https://stable.repo.amd.com/rocm/whl-next/"

if (-not $Prefix) {
    $root = (& git rev-parse --show-toplevel 2>$null)
    if (-not $root) {
        $root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
    }
    $common = (& git -C $root rev-parse --path-format=absolute --git-common-dir 2>$null)
    if ($common -and (Split-Path -Leaf $common.Trim()) -eq ".git") {
        $root = Split-Path -Parent $common.Trim()
    }
    $Prefix = Join-Path $root.Trim() ".cache\rocm"
}

$install = Join-Path ([IO.Path]::GetFullPath($Prefix)) "windows-$version"
$venv = Join-Path $install ".venv"
$python = Join-Path $venv "Scripts\python.exe"
$rocmSDK = Join-Path $venv "Scripts\rocm-sdk.exe"
$ready = Join-Path $install ".ollama-rocm-ready"

if (-not (Test-Path -LiteralPath $ready)) {
    Remove-Item -LiteralPath $install -Recurse -Force -ErrorAction SilentlyContinue
    python -m venv $venv
    if ($LASTEXITCODE -ne 0) { throw "failed to create Python virtual environment" }
    & $python -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw "failed to upgrade pip" }
    & $python -m pip install --index-url $indexURL "rocm[libraries,devel,device-all]==$version"
    if ($LASTEXITCODE -ne 0) { throw "failed to install ROCm $version" }
    & $rocmSDK init
    if ($LASTEXITCODE -ne 0) { throw "rocm-sdk init failed" }
    (& $rocmSDK path --root).Trim() | Set-Content -LiteralPath (Join-Path $install "root.txt") -Encoding ascii
    Set-Content -LiteralPath $ready -Value (Get-Date -Format o) -Encoding ascii
}

$rocmRoot = (Get-Content -LiteralPath (Join-Path $install "root.txt") -Raw).Trim()
$escapedRoot = $rocmRoot.Replace("'", "''")
$envFile = Join-Path $install "ollama-rocm-env.ps1"
$content = @"
`$env:HIP_PATH = '$escapedRoot'
`$env:HIP_PLATFORM = 'amd'
`$env:CMAKE_PREFIX_PATH = `$env:HIP_PATH
`$env:Path = @((Join-Path `$env:HIP_PATH 'bin'), (Join-Path `$env:HIP_PATH 'lib\llvm\bin'), (Join-Path `$env:HIP_PATH 'llvm\bin'), `$env:Path) -join ';'
"@
Set-Content -LiteralPath $envFile -Value $content -Encoding ascii

Write-Output "ROCm $version`: $rocmRoot"
Write-Output "Environment: $envFile"
