[CmdletBinding()]
param(
    [string]$Model = "qwen2.5:7b",
    [string]$Prompt
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "Initialize-PortableEnvironment.ps1")

if ([string]::IsNullOrWhiteSpace($Prompt)) {
    $encodedPrompt = "6KuL55SoIDMwMCDlrZfku4vntLnph4/lrZDlipvlrbg="
    $Prompt = [Text.Encoding]::UTF8.GetString(
        [Convert]::FromBase64String($encodedPrompt)
    )
}

$nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($null -eq $nvidiaSmi) {
    throw "nvidia-smi was not found."
}

try {
    Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/version" `
        -TimeoutSec 2 | Out-Null
}
catch {
    throw "Ollama is not running. Run Start-Portable-Ollama.ps1 -Background first."
}

function Get-ModelCatalogEntry {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ModelName
    )

    $catalog = Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/tags" `
        -TimeoutSec 5
    return $catalog.models |
        Where-Object {
            $_.name -ieq $ModelName -or $_.model -ieq $ModelName
        } |
        Select-Object -First 1
}

function Find-LoadedModel {
    param(
        [Parameter(Mandatory = $true)]
        [object[]]$LoadedModels,
        [Parameter(Mandatory = $true)]
        [AllowNull()]
        [object]$TargetModel
    )

    if ($null -eq $TargetModel) {
        return $null
    }

    $targetDigest = [string]$TargetModel.digest
    if (-not [string]::IsNullOrWhiteSpace($targetDigest)) {
        $digestMatch = $LoadedModels |
            Where-Object { [string]$_.digest -eq $targetDigest } |
            Select-Object -First 1
        if ($null -ne $digestMatch) {
            return $digestMatch
        }
    }

    $targetNames = @([string]$TargetModel.name, [string]$TargetModel.model) |
        Where-Object { -not [string]::IsNullOrWhiteSpace($_) } |
        Select-Object -Unique

    foreach ($name in $targetNames) {
        $nameMatch = $LoadedModels |
            Where-Object {
                $_.name -ieq $name -or $_.model -ieq $name
            } |
            Select-Object -First 1
        if ($null -ne $nameMatch) {
            return $nameMatch
        }
    }

    return $null
}

Write-Host "Pulling $Model to $env:OLLAMA_MODELS ..." -ForegroundColor Cyan
& $script:OllamaExecutable pull $Model
if ($LASTEXITCODE -ne 0) {
    throw "ollama pull failed with exit code $LASTEXITCODE."
}

$targetModel = Get-ModelCatalogEntry -ModelName $Model
if ($null -eq $targetModel) {
    throw "Unable to resolve model identity for '$Model' from /api/tags."
}

$loadedBeforeTest = Find-LoadedModel `
    -LoadedModels (Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/ps" -TimeoutSec 5).models `
    -TargetModel $targetModel
if ($null -ne $loadedBeforeTest) {
    $stopName = if ([string]::IsNullOrWhiteSpace([string]$loadedBeforeTest.name)) {
        $Model
    }
    else {
        [string]$loadedBeforeTest.name
    }
    & $script:OllamaExecutable stop $stopName
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to unload $Model before the VRAM baseline measurement."
    }

    $deadline = (Get-Date).AddSeconds(30)
    do {
        Start-Sleep -Milliseconds 500
        $stillLoaded = Find-LoadedModel `
            -LoadedModels (Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/ps" -TimeoutSec 5).models `
            -TargetModel $targetModel
    } while ($null -ne $stillLoaded -and (Get-Date) -lt $deadline)

    if ($null -ne $stillLoaded) {
        throw "$Model did not unload within 30 seconds."
    }
}

function Get-GpuMemoryMiB {
    $output = @(& $nvidiaSmi.Source `
        "--query-gpu=memory.used" `
        "--format=csv,noheader,nounits")
    $exitCode = $LASTEXITCODE
    $value = $output |
        Where-Object { $_ -match "^\s*\d+\s*$" } |
        Select-Object -First 1
    if ($exitCode -ne 0 -or $null -eq $value) {
        throw "Unable to read GPU memory usage."
    }
    return [int]([string]$value).Trim()
}

$logsDirectory = Join-Path $script:EnvironmentRoot "logs"
$stdoutLog = Join-Path $logsDirectory "qwen-test-output.txt"
$stderrLog = Join-Path $logsDirectory "qwen-test-error.txt"
$baselineMiB = Get-GpuMemoryMiB
$peakMiB = $baselineMiB

Write-Host "Running model test and monitoring VRAM..." -ForegroundColor Cyan
$generationJob = Start-Job -ScriptBlock {
    param($Executable, $ModelName, $ModelPrompt, $ErrorLog)

    $utf8 = New-Object Text.UTF8Encoding($false)
    [Console]::OutputEncoding = $utf8
    $OutputEncoding = $utf8
    $output = @(& $Executable run $ModelName $ModelPrompt 2> $ErrorLog)
    [pscustomobject]@{
        ExitCode = $LASTEXITCODE
        Output = [string]::Join([Environment]::NewLine, $output)
    }
} -ArgumentList $script:OllamaExecutable, $Model, $Prompt, $stderrLog

try {
    while ($generationJob.State -eq "Running") {
        $usedMiB = Get-GpuMemoryMiB
        if ($usedMiB -gt $peakMiB) {
            $peakMiB = $usedMiB
        }
        Start-Sleep -Milliseconds 500
    }
    $generationResult = Receive-Job -Job $generationJob -Wait -ErrorAction Stop
}
finally {
    Remove-Job -Job $generationJob -Force
}

if ($generationResult.ExitCode -ne 0) {
    throw "ollama run failed with code $($generationResult.ExitCode). See $stderrLog"
}
$cleanOutput = [regex]::Replace(
    [string]$generationResult.Output,
    "$([char]27)\[[0-?]*[ -/]*[@-~]",
    ""
).Trim()
[IO.File]::WriteAllText(
    $stdoutLog,
    $cleanOutput,
    [Text.UTF8Encoding]::new($false)
)

$runningModels = Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/ps" `
    -TimeoutSec 5
$loadedModel = Find-LoadedModel -LoadedModels $runningModels.models -TargetModel $targetModel
$ollamaVramMiB = if ($null -eq $loadedModel) {
    0
}
else {
    [math]::Round($loadedModel.size_vram / 1MB)
}

$response = $cleanOutput
Write-Host ""
Write-Host "--- Model response ---"
Write-Host $response
Write-Host ""
Write-Host "--- GPU report ---"
Write-Host "Baseline VRAM: $baselineMiB MiB"
Write-Host "Peak total VRAM: $peakMiB MiB"
Write-Host "Peak increase: $($peakMiB - $baselineMiB) MiB"
Write-Host "Ollama-reported model VRAM: $ollamaVramMiB MiB"
Write-Host "Model storage: $env:OLLAMA_MODELS"
