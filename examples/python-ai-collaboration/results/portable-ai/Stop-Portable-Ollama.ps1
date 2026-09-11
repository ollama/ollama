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

$runningModels = @()
$portableApiOwner = $null
try {
    $runningModels = @(
        (Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/ps" `
            -TimeoutSec 5).models
    )
    $portableApiOwner = Get-OllamaApiOwnerProcess
}
catch [System.Net.WebException] {
    Write-Host "Ollama API is already stopped."
}

if ($runningModels.Count -gt 0 -and -not (Test-PortableOllamaOwner -Process $portableApiOwner)) {
    $ownerPath = if ($null -eq $portableApiOwner) {
        "unknown process"
    }
    else {
        [string]$portableApiOwner.ExecutablePath
    }
    Write-Host "Detected a non-portable Ollama API owner ($ownerPath); skipping model unload."
}
else {
    foreach ($model in $runningModels) {
        & $script:OllamaExecutable stop $model.name
        if ($LASTEXITCODE -ne 0) {
            throw "Unable to unload model $($model.name)."
        }
    }
}

$processes = @(
    Get-CimInstance Win32_Process -Filter "Name='ollama.exe'" |
        Where-Object { $_.ExecutablePath -eq $script:OllamaExecutable }
)
foreach ($process in $processes) {
    Stop-Process -Id $process.ProcessId -ErrorAction Stop
    try {
        Wait-Process -Id $process.ProcessId -Timeout 15 -ErrorAction Stop
    }
    catch {
        if ($null -ne (Get-Process -Id $process.ProcessId -ErrorAction SilentlyContinue)) {
            throw "Portable Ollama process $($process.ProcessId) did not stop within 15 seconds."
        }
    }

    if ($null -ne (Get-Process -Id $process.ProcessId -ErrorAction SilentlyContinue)) {
        throw "Portable Ollama process $($process.ProcessId) is still running."
    }
}

Write-Host "Portable Ollama stopped. The SSD can now be safely ejected."
