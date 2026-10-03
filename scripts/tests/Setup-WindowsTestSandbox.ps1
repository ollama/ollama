[CmdletBinding()]
param(
    [switch]$EnableFeature,
    [switch]$SkipSmokeTest,
    [ValidateRange(2048, 65536)]
    [int]$MemoryInMB = 16384
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$stateRoot = Join-Path $repoRoot ".cache\windows-test-sandbox"
$configPath = Join-Path $stateRoot "config.json"
$sandboxExe = Join-Path $env:WINDIR "System32\WindowsSandbox.exe"
. (Join-Path $PSScriptRoot "WindowsSandbox-TestHelpers.ps1")

function Test-IsAdministrator {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = [Security.Principal.WindowsPrincipal]::new($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

if ($EnableFeature) {
    if (-not (Test-IsAdministrator)) {
        throw "Enabling Windows Sandbox requires an elevated PowerShell session."
    }
    $feature = Get-WindowsOptionalFeature -Online -FeatureName Containers-DisposableClientVM
    if ($feature.State -ne "Enabled") {
        $feature = Enable-WindowsOptionalFeature -Online -FeatureName Containers-DisposableClientVM -All -NoRestart
        if ($feature.RestartNeeded) {
            throw "Windows Sandbox was enabled, but Windows must be restarted before setup can continue."
        }
    }
}

if (-not (Test-Path -LiteralPath $sandboxExe -PathType Leaf)) {
    throw "Windows Sandbox is not enabled. Rerun this script from an elevated PowerShell with -EnableFeature."
}

$pester = Get-Module -ListAvailable Pester |
    Where-Object { $_.Version -ge [version]"5.0.0" } |
    Sort-Object Version -Descending |
    Select-Object -First 1
if (-not $pester) {
    throw "Pester 5 or newer is required. Install it with: Install-Module Pester -Scope CurrentUser -MinimumVersion 5.0"
}

New-Item -ItemType Directory -Path $stateRoot -Force | Out-Null
$validatedAt = $null

if (-not $SkipSmokeTest) {
    Get-WindowsSandboxCliPath | Out-Null
    Assert-NoWindowsSandboxRunning

    $smokeRoot = Join-Path $stateRoot ("setup-" + [guid]::NewGuid().ToString("N"))
    New-Item -ItemType Directory -Path $smokeRoot -Force | Out-Null
    $smokeScript = Join-Path $smokeRoot "smoke.ps1"
    $smokeConfig = Join-Path $smokeRoot "smoke.wsb"
    $smokeResult = Join-Path $smokeRoot "result.json"

    $scriptText = @'
$ErrorActionPreference = "Stop"
[System.IO.File]::WriteAllText("C:\host\started.txt", (Get-Date).ToString("o"), [System.Text.Encoding]::UTF8)
$client = [System.Net.Sockets.TcpClient]::new()
try {
    try {
        $task = $client.ConnectAsync("community.chocolatey.org", 443)
        $networkAvailable = $task.Wait([TimeSpan]::FromSeconds(10)) -and $client.Connected
    } catch {
        $networkAvailable = $false
    }
} finally {
    $client.Dispose()
}
$result = [ordered]@{
    completedAt = (Get-Date).ToString("o")
    userName = $env:USERNAME
    isAdministrator = ([Security.Principal.WindowsPrincipal]::new(
        [Security.Principal.WindowsIdentity]::GetCurrent()
    )).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
    mappedFolderAvailable = Test-Path -LiteralPath "C:\host"
    networkHttpsAvailable = $networkAvailable
}
$result | ConvertTo-Json | Set-Content -LiteralPath "C:\host\result.json" -Encoding UTF8
shutdown.exe /s /f /t 5
'@
    [System.IO.File]::WriteAllText($smokeScript, $scriptText, [System.Text.Encoding]::UTF8)

    $configText = New-WindowsSandboxConfigurationXml -MemoryInMB $MemoryInMB `
        -LogonCommand "powershell.exe -NoProfile -ExecutionPolicy Bypass -File C:\host\smoke.ps1" `
        -MappedFolders @(@{ HostFolder = $smokeRoot; SandboxFolder = "C:\host"; ReadOnly = $false })
    [System.IO.File]::WriteAllText($smokeConfig, $configText, [System.Text.Encoding]::UTF8)

    Write-Host "Launching the Windows Sandbox smoke test..."
    $smokeStarted = Join-Path $smokeRoot "started.txt"
    $sandboxId = Start-WindowsSandboxTestInstance -ConfigurationXml $configText
    try {
        Connect-WindowsSandboxTestInstance -Id $sandboxId -StartedPath $smokeStarted -TimeoutMinutes 5

        $deadline = [DateTime]::UtcNow.AddMinutes(3)
        while (-not (Test-Path -LiteralPath $smokeResult -PathType Leaf) -and [DateTime]::UtcNow -lt $deadline) {
            Start-Sleep -Seconds 2
        }
        if (-not (Test-Path -LiteralPath $smokeResult -PathType Leaf)) {
            throw "Windows Sandbox did not complete the smoke test. Inspect $smokeRoot"
        }

        $result = Get-Content -LiteralPath $smokeResult -Raw | ConvertFrom-Json
        if (-not $result.mappedFolderAvailable -or -not $result.networkHttpsAvailable -or -not $result.isAdministrator) {
            throw "Windows Sandbox started but failed one or more prerequisites. Result: $($result | ConvertTo-Json -Compress)"
        }
    } finally {
        Stop-WindowsSandboxTestInstance -Id $sandboxId
        Wait-WindowsSandboxTestInstanceStopped -Id $sandboxId
    }
    $validatedAt = (Get-Date).ToString("o")
    $cleanupError = $null
    for ($attempt = 0; $attempt -lt 30; $attempt++) {
        try {
            if (Test-Path -LiteralPath $smokeRoot) {
                Remove-Item -LiteralPath $smokeRoot -Recurse -Force -ErrorAction Stop
            }
            $cleanupError = $null
            break
        } catch {
            $cleanupError = $_
            Start-Sleep -Seconds 1
        }
    }
    if ($cleanupError) {
        Write-Warning "Sandbox smoke test passed, but its scratch directory is still in use: $smokeRoot"
    }
}

$config = [ordered]@{
    schemaVersion = 1
    enabled = $true
    configuredAt = (Get-Date).ToString("o")
    validatedAt = $validatedAt
    sandboxExecutable = $sandboxExe
    memoryInMB = $MemoryInMB
    pesterModulePath = $pester.ModuleBase
    pesterVersion = $pester.Version.ToString()
}
$configTemp = "$configPath.tmp"
[System.IO.File]::WriteAllText(
    $configTemp,
    ($config | ConvertTo-Json -Depth 3),
    [System.Text.Encoding]::UTF8
)
Move-Item -LiteralPath $configTemp -Destination $configPath -Force

Write-Host "Windows Sandbox installer testing is configured."
Write-Host "Config: $configPath"
Write-Host "Pester: $($pester.Version) at $($pester.ModuleBase)"
if ($SkipSmokeTest) {
    Write-Warning "The Sandbox smoke test was skipped. The first isolated test run is the validation."
}
