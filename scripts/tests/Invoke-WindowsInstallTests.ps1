[CmdletBinding()]
param(
    [ValidateSet("Unit", "Integration", "UpgradeMatrix", "AppIntegration")]
    [string[]]$Tag = @("Unit"),

    [ValidateSet("Auto", "Host", "Sandbox")]
    [string]$Isolation = "Auto",

    [ValidateRange(1, 240)]
    [int]$TimeoutMinutes = 90,

    [switch]$CI,

    [switch]$PrepareOnly
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$testPath = Join-Path $repoRoot "scripts\tests\install.Tests.ps1"
$stateRoot = Join-Path $repoRoot ".cache\windows-test-sandbox"
$configPath = Join-Path $stateRoot "config.json"
$destructiveTags = @("Integration", "UpgradeMatrix", "AppIntegration")
$isDestructive = @($Tag | Where-Object { $_ -in $destructiveTags }).Count -gt 0
. (Join-Path $PSScriptRoot "WindowsSandbox-TestHelpers.ps1")
$needsAppIntegration = "AppIntegration" -in $Tag

function Build-AppUpdaterIntegrationTest {
    param([string]$OutputPath)

    New-Item -ItemType Directory -Path (Split-Path -Parent $OutputPath) -Force | Out-Null
    Remove-Item -LiteralPath $OutputPath -Force -ErrorAction SilentlyContinue
    Write-Host "Building fresh Windows app updater integration test..."
    Push-Location $repoRoot
    try {
        & go test -count=1 -c -tags "updater_integration updater_unsigned" -o $OutputPath ./app/updater
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to build the Windows app updater integration test (exit code $LASTEXITCODE)."
        }
    } finally {
        Pop-Location
    }
}

function Invoke-HostTests {
    param([string]$AppIntegrationBinary = "")

    $pester = Get-Module -ListAvailable Pester |
        Where-Object { $_.Version -ge [version]"5.0.0" } |
        Sort-Object Version -Descending |
        Select-Object -First 1
    if (-not $pester) {
        throw "Pester 5 or newer is required. Install it with: Install-Module Pester -Scope CurrentUser -MinimumVersion 5.0"
    }

    Import-Module $pester.Path -Force
    Write-Host "Running Windows installer tests on the host (tags: $($Tag -join ', '))." -ForegroundColor Yellow
    if ($isDestructive) {
        Write-Warning "These tests make real installer changes. The test safety gate will abort if Ollama is already installed."
    }
    $pesterArgs = @{
        Path = $testPath
        Tag = $Tag
        Output = "Detailed"
        PassThru = $true
        CI = $CI
    }
    $savedUpdaterBinary = $env:OLLAMA_TEST_UPDATER_BINARY
    try {
        if ($AppIntegrationBinary) {
            $env:OLLAMA_TEST_UPDATER_BINARY = $AppIntegrationBinary
        }
        $result = Invoke-Pester @pesterArgs
        if ($result.FailedCount -gt 0 -or $result.Result -ne "Passed") {
            throw "Pester failed: result=$($result.Result), failed=$($result.FailedCount)"
        }
    } finally {
        if ($null -eq $savedUpdaterBinary) {
            Remove-Item Env:OLLAMA_TEST_UPDATER_BINARY -ErrorAction SilentlyContinue
        } else {
            $env:OLLAMA_TEST_UPDATER_BINARY = $savedUpdaterBinary
        }
    }
    Write-Host "Host result: passed=$($result.PassedCount), failed=$($result.FailedCount), skipped=$($result.SkippedCount)"
    return [PSCustomObject]@{
        Outcome = $result.Result
        PassedCount = $result.PassedCount
        FailedCount = $result.FailedCount
        SkippedCount = $result.SkippedCount
    }
}

$sandboxConfigured = Test-Path -LiteralPath $configPath -PathType Leaf
$selectedIsolation = $Isolation
if ($Isolation -eq "Auto") {
    $selectedIsolation = if ($isDestructive -and $sandboxConfigured) { "Sandbox" } else { "Host" }
}

if ($selectedIsolation -eq "Host") {
    $appIntegrationBinary = ""
    if ($needsAppIntegration) {
        $appIntegrationBinary = Join-Path $stateRoot "host\updater-integration.test.exe"
        Build-AppUpdaterIntegrationTest -OutputPath $appIntegrationBinary
    }
    Invoke-HostTests -AppIntegrationBinary $appIntegrationBinary
    return
}

if (-not $sandboxConfigured) {
    throw "Windows Sandbox is not configured for this checkout. Run scripts\tests\Setup-WindowsTestSandbox.ps1, or rerun with -Isolation Host."
}

$config = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
if ($config.schemaVersion -ne 1 -or -not $config.enabled) {
    throw "Unsupported or disabled Windows Sandbox configuration: $configPath"
}
if (-not (Test-Path -LiteralPath $config.sandboxExecutable -PathType Leaf)) {
    throw "Configured Windows Sandbox executable is missing: $($config.sandboxExecutable)"
}
if (-not (Test-Path -LiteralPath $config.pesterModulePath -PathType Container)) {
    throw "Configured Pester module is missing: $($config.pesterModulePath). Rerun Setup-WindowsTestSandbox.ps1."
}
if (-not $PrepareOnly) {
    Get-WindowsSandboxCliPath | Out-Null
    Assert-NoWindowsSandboxRunning
}

$runId = (Get-Date -Format "yyyyMMdd-HHmmss") + "-" + [guid]::NewGuid().ToString("N").Substring(0, 8)
$runRoot = Join-Path $stateRoot "runs\$runId"
$installerCache = Join-Path $repoRoot ".cache\install-tests"
New-Item -ItemType Directory -Path $runRoot, $installerCache -Force | Out-Null
if ($needsAppIntegration) {
    Build-AppUpdaterIntegrationTest -OutputPath (Join-Path $runRoot "updater-integration.test.exe")
}

$tagCsv = $Tag -join ","
$guestCommand = "powershell.exe -NoProfile -ExecutionPolicy Bypass -File C:\host\repo\scripts\tests\windows-sandbox\Invoke-WindowsInstallTestsGuest.ps1 -TagCsv $tagCsv"
$wsb = New-WindowsSandboxConfigurationXml -MemoryInMB $config.memoryInMB -LogonCommand $guestCommand -MappedFolders @(
    @{ HostFolder = $repoRoot; SandboxFolder = "C:\host\repo"; ReadOnly = $true },
    @{ HostFolder = $runRoot; SandboxFolder = "C:\host\run"; ReadOnly = $false },
    @{ HostFolder = $installerCache; SandboxFolder = "C:\host\installer-cache"; ReadOnly = $false },
    @{ HostFolder = $config.pesterModulePath; SandboxFolder = "C:\host\pester"; ReadOnly = $true }
)
$wsbPath = Join-Path $runRoot "run.wsb"
[System.IO.File]::WriteAllText($wsbPath, $wsb, [System.Text.Encoding]::UTF8)

Write-Host "Prepared isolated Windows installer test run: $runRoot"
if ($PrepareOnly) {
    Write-Host "Launch this file to start it: $wsbPath"
    return [PSCustomObject]@{ RunRoot = $runRoot; SandboxConfig = $wsbPath; Launched = $false }
}

Write-Host "Launching Windows Sandbox (tags: $($Tag -join ', '))..."
$resultPath = Join-Path $runRoot "result.json"
$startedPath = Join-Path $runRoot "started.txt"
$sandboxId = Start-WindowsSandboxTestInstance -ConfigurationXml $wsb
Write-Host "Windows Sandbox instance: $sandboxId"
try {
    Connect-WindowsSandboxTestInstance -Id $sandboxId -StartedPath $startedPath -TimeoutMinutes 5

    $deadline = [DateTime]::UtcNow.AddMinutes($TimeoutMinutes)
    while (-not (Test-Path -LiteralPath $resultPath -PathType Leaf) -and [DateTime]::UtcNow -lt $deadline) {
        Start-Sleep -Seconds 2
    }
    if (-not (Test-Path -LiteralPath $resultPath -PathType Leaf)) {
        throw "Windows Sandbox tests did not finish within $TimeoutMinutes minutes. Run artifacts: $runRoot"
    }

    $result = Get-Content -LiteralPath $resultPath -Raw | ConvertFrom-Json
} finally {
    Stop-WindowsSandboxTestInstance -Id $sandboxId
    Wait-WindowsSandboxTestInstanceStopped -Id $sandboxId
}

Write-Host "Sandbox result: $($result.outcome); passed=$($result.passedCount), failed=$($result.failedCount), skipped=$($result.skippedCount)"
Write-Host "Run artifacts: $runRoot"
if ($result.outcome -ne "Passed") {
    throw "Windows Sandbox tests failed: $($result.error)"
}
return $result
