[CmdletBinding()]
param(
    [switch]$Background
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "Initialize-PortableEnvironment.ps1")

try {
    $version = Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/version" `
        -TimeoutSec 2
    Write-Host "Ollama is already running (version $($version.version))."
    exit 0
}
catch {
    # A failed health check is expected when the portable service is not running.
}

if (-not $Background) {
    & $script:OllamaExecutable serve
    exit $LASTEXITCODE
}

$logsDirectory = Join-Path $script:EnvironmentRoot "logs"
$stdoutLog = Join-Path $logsDirectory "ollama-stdout.log"
$stderrLog = Join-Path $logsDirectory "ollama-stderr.log"
$process = Start-Process -FilePath $script:OllamaExecutable `
    -ArgumentList "serve" `
    -WorkingDirectory $script:OllamaRoot `
    -RedirectStandardOutput $stdoutLog `
    -RedirectStandardError $stderrLog `
    -PassThru

$deadline = (Get-Date).AddSeconds(30)
while ((Get-Date) -lt $deadline) {
    if ($process.HasExited) {
        throw "Ollama exited with code $($process.ExitCode). See $stderrLog"
    }

    try {
        $version = Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/version" `
            -TimeoutSec 2
        Write-Host "Portable Ollama $($version.version) is running as PID $($process.Id)."
        Write-Host "Models: $env:OLLAMA_MODELS"
        exit 0
    }
    catch {
        Start-Sleep -Milliseconds 500
    }
}

throw "Ollama did not become ready within 30 seconds. See $stderrLog"
