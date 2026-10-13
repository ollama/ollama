[CmdletBinding()]
param(
    [switch]$Background
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "Initialize-PortableEnvironment.ps1")

function Get-OllamaApiOwnerProcess {
    $connection = Get-NetTCPConnection -LocalAddress "127.0.0.1" `
        -LocalPort 11434 `
        -State Listen `
        -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($null -eq $connection) {
        return $null
    }

    return Get-CimInstance Win32_Process `
        -Filter "ProcessId=$($connection.OwningProcess)" `
        -ErrorAction SilentlyContinue
}

function Test-PortableOllamaOwner {
    param(
        [Parameter(Mandatory = $true)]
        [AllowNull()]
        [object]$Process
    )

    return $null -ne $Process -and
        [string]$Process.ExecutablePath -ceq $script:OllamaExecutable
}

try {
    $version = Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/version" `
        -TimeoutSec 2
    $ownerProcess = Get-OllamaApiOwnerProcess
    if (Test-PortableOllamaOwner -Process $ownerProcess) {
        Write-Host "Portable Ollama is already running (version $($version.version), PID $($ownerProcess.ProcessId))."
        exit 0
    }

    $ownerPath = if ($null -eq $ownerProcess) {
        "unknown process"
    }
    else {
        [string]$ownerProcess.ExecutablePath
    }
    throw "Port 11434 is already served by a different process ($ownerPath). Stop that service before starting portable Ollama."
}
catch [System.Net.WebException] {
    # A failed health check is expected when the portable service is not running.
}
catch {
    throw
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
        $ownerProcess = Get-OllamaApiOwnerProcess
        if (-not (Test-PortableOllamaOwner -Process $ownerProcess)) {
            $ownerPath = if ($null -eq $ownerProcess) {
                "unknown process"
            }
            else {
                [string]$ownerProcess.ExecutablePath
            }
            throw "Port 11434 became owned by a different process ($ownerPath)."
        }

        Write-Host "Portable Ollama $($version.version) is running as PID $($ownerProcess.ProcessId)."
        Write-Host "Models: $env:OLLAMA_MODELS"
        exit 0
    }
    catch [System.Net.WebException] {
        Start-Sleep -Milliseconds 500
    }
    catch {
        throw
    }
}

throw "Ollama did not become ready within 30 seconds. See $stderrLog"
