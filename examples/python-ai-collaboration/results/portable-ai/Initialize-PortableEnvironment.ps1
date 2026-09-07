Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$script:OllamaRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$script:EnvironmentRoot = Join-Path $script:OllamaRoot "env"
$script:OllamaExecutable = Join-Path $script:OllamaRoot "ollama.exe"

if (-not (Test-Path -LiteralPath $script:OllamaExecutable -PathType Leaf)) {
    throw "Portable Ollama was not found at $script:OllamaExecutable"
}

$python = Get-ChildItem -LiteralPath (Join-Path $script:EnvironmentRoot "python") `
    -Filter "python.exe" -File -Recurse |
    Select-Object -First 1
if ($null -eq $python) {
    throw "Portable Python was not found under $script:EnvironmentRoot\python"
}

$cmake = Get-ChildItem -LiteralPath (Join-Path $script:EnvironmentRoot "cmake") `
    -Filter "cmake.exe" -File -Recurse |
    Select-Object -First 1
if ($null -eq $cmake) {
    throw "Portable CMake was not found under $script:EnvironmentRoot\cmake"
}

$gitDirectory = Join-Path $script:EnvironmentRoot "git\cmd"
$uvDirectory = Join-Path $script:EnvironmentRoot "uv"
$modelsDirectory = Join-Path $script:OllamaRoot "models"
$cacheDirectory = Join-Path $script:EnvironmentRoot "cache"
$tempDirectory = Join-Path $script:EnvironmentRoot "tmp"

@(
    $modelsDirectory
    $cacheDirectory
    $tempDirectory
    (Join-Path $script:EnvironmentRoot "logs")
) | ForEach-Object {
    New-Item -ItemType Directory -Path $_ -Force | Out-Null
}

$env:OLLAMA_MODELS = $modelsDirectory
$env:UV_CACHE_DIR = Join-Path $cacheDirectory "uv"
$env:PIP_CACHE_DIR = Join-Path $cacheDirectory "pip"
$env:HF_HOME = Join-Path $cacheDirectory "huggingface"
$env:TORCH_HOME = Join-Path $cacheDirectory "torch"
$env:TEMP = $tempDirectory
$env:TMP = $tempDirectory
$env:PATH = @(
    $script:OllamaRoot
    $gitDirectory
    $cmake.DirectoryName
    $python.DirectoryName
    $uvDirectory
    $env:PATH
) -join ";"
