Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "Initialize-PortableEnvironment.ps1")

$runningModels = (Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/ps" `
    -TimeoutSec 5).models
foreach ($model in $runningModels) {
    & $script:OllamaExecutable stop $model.name
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to unload model $($model.name)."
    }
}

$processes = @(
    Get-CimInstance Win32_Process -Filter "Name='ollama.exe'" |
        Where-Object { $_.ExecutablePath -eq $script:OllamaExecutable }
)
foreach ($process in $processes) {
    Stop-Process -Id $process.ProcessId -ErrorAction Stop
    Wait-Process -Id $process.ProcessId -Timeout 15 -ErrorAction SilentlyContinue
}

Write-Host "Portable Ollama stopped. The SSD can now be safely ejected."
