[CmdletBinding()]
param(
    [string]$TagCsv = "Integration",
    [string]$RepoRoot = "C:\host\repo",
    [string]$RunRoot = "C:\host\run",
    [string]$InstallerCache = "C:\host\installer-cache",
    [string]$PesterModuleRoot = "C:\host\pester",
    [switch]$NoShutdown
)

$ErrorActionPreference = "Stop"
$resultPath = Join-Path $RunRoot "result.json"
$resultTempPath = "$resultPath.tmp"
$transcriptPath = Join-Path $RunRoot "transcript.log"
$exitCode = 1
$summary = [ordered]@{
    completed = $false
    outcome = "Failed"
    startedAt = (Get-Date).ToString("o")
    completedAt = $null
    tags = @($TagCsv -split ',' | ForEach-Object { $_.Trim() } | Where-Object { $_ })
    totalCount = 0
    passedCount = 0
    failedCount = 0
    skippedCount = 0
    error = $null
}

Set-Content -LiteralPath (Join-Path $RunRoot "started.txt") -Value $summary.startedAt -Encoding UTF8

# Windows Sandbox ships with unusable WMI (microsoft/Windows-Sandbox#67), and
# taskkill.exe enumerates processes through WMI, so every taskkill invocation
# there hangs for minutes. Ollama's Inno Setup installers call
# `taskkill /f /im ...` before copying files and the uninstallers call it from
# [UninstallRun], which makes both hang inside the guest. Recent Inno Setup
# releases no longer resolve executables through the current directory, so the
# only reliable fix is to replace the guest's own taskkill.exe (System32 and
# SysWOW64, the installers are 32-bit) with a tiny WMI-free implementation.
# The guest is disposable; nothing outside Windows Sandbox is ever touched.
function Get-TaskkillShimPath {
    $shimDir = Join-Path $RunRoot "install-tests\taskkill-shim"
    $shimExe = Join-Path $shimDir "taskkill.exe"
    if (-not (Test-Path -LiteralPath $shimExe -PathType Leaf)) {
        New-Item -ItemType Directory -Path $shimDir -Force | Out-Null
        $source = @'
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;

// Minimal taskkill replacement for Windows Sandbox guests whose WMI is broken.
// Supports: /f /im <image>   and   /pid <pid> [/t] [/f]
static class TaskkillShim {
    [StructLayout(LayoutKind.Sequential)]
    struct PROCESSENTRY32 {
        public uint dwSize; public uint cntUsage; public uint th32ProcessID; public IntPtr th32DefaultHeapID;
        public uint th32ModuleID; public uint cntThreads; public uint th32ParentProcessID; public int pcPriClassBase;
        public uint dwFlags; [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 260)] public string szExeFile;
    }
    [DllImport("kernel32.dll", SetLastError = true)] static extern IntPtr CreateToolhelp32Snapshot(uint flags, uint pid);
    [DllImport("kernel32.dll", SetLastError = true)] static extern bool Process32First(IntPtr snap, ref PROCESSENTRY32 entry);
    [DllImport("kernel32.dll", SetLastError = true)] static extern bool Process32Next(IntPtr snap, ref PROCESSENTRY32 entry);
    [DllImport("kernel32.dll", SetLastError = true)] static extern bool CloseHandle(IntPtr handle);

    static Dictionary<uint, List<uint>> Children() {
        var map = new Dictionary<uint, List<uint>>();
        IntPtr snap = CreateToolhelp32Snapshot(2, 0);
        if (snap == IntPtr.Zero || snap == new IntPtr(-1)) return map;
        try {
            var entry = new PROCESSENTRY32(); entry.dwSize = (uint)Marshal.SizeOf(typeof(PROCESSENTRY32));
            if (Process32First(snap, ref entry)) {
                do {
                    List<uint> list;
                    if (!map.TryGetValue(entry.th32ParentProcessID, out list)) { list = new List<uint>(); map[entry.th32ParentProcessID] = list; }
                    list.Add(entry.th32ProcessID);
                } while (Process32Next(snap, ref entry));
            }
        } finally { CloseHandle(snap); }
        return map;
    }

    static int Kill(int pid) {
        try { Process.GetProcessById(pid).Kill(); Console.WriteLine("SUCCESS: The process with PID " + pid + " has been terminated."); return 0; }
        catch (ArgumentException) { Console.WriteLine("SUCCESS: The process with PID " + pid + " has already exited."); return 0; }
        catch (InvalidOperationException) { Console.WriteLine("SUCCESS: The process with PID " + pid + " has already exited."); return 0; }
        catch (Exception e) { Console.Error.WriteLine("ERROR: The process with PID " + pid + " could not be terminated: " + e.Message); return 1; }
    }

    static void KillTree(uint pid, Dictionary<uint, List<uint>> map, ref int failures) {
        List<uint> kids;
        if (map.TryGetValue(pid, out kids)) foreach (var kid in kids) KillTree(kid, map, ref failures);
        if (Kill((int)pid) != 0) failures++;
    }

    static int Main(string[] args) {
        string image = null; int pid = -1; bool tree = false;
        for (int i = 0; i < args.Length; i++) {
            string a = args[i].ToLowerInvariant();
            if (a == "/im" && i + 1 < args.Length) image = args[++i];
            else if (a == "/pid" && i + 1 < args.Length) pid = int.Parse(args[++i]);
            else if (a == "/t") tree = true;
        }
        if (image != null) {
            string name = image.EndsWith(".exe", StringComparison.OrdinalIgnoreCase) ? image.Substring(0, image.Length - 4) : image;
            var procs = Process.GetProcessesByName(name);
            if (procs.Length == 0) { Console.Error.WriteLine("ERROR: The process \"" + image + "\" not found."); return 128; }
            int failures = 0;
            foreach (var p in procs) if (Kill(p.Id) != 0) failures++;
            return failures == 0 ? 0 : 1;
        }
        if (pid >= 0) {
            int failures = 0;
            if (tree) KillTree((uint)pid, Children(), ref failures); else failures = Kill(pid);
            return failures == 0 ? 0 : 1;
        }
        Console.Error.WriteLine("ERROR: Invalid syntax. Supported: /f /im <image> | /pid <pid> [/t] [/f]");
        return 1;
    }
}
'@
        # Add-Type cannot emit executables on PowerShell 7, so use the .NET
        # Framework C# compiler that ships with Windows.
        $csc = Join-Path $env:SystemRoot "Microsoft.NET\Framework64\v4.0.30319\csc.exe"
        if (-not (Test-Path -LiteralPath $csc -PathType Leaf)) {
            $csc = Join-Path $env:SystemRoot "Microsoft.NET\Framework\v4.0.30319\csc.exe"
        }
        if (-not (Test-Path -LiteralPath $csc -PathType Leaf)) {
            throw "The .NET Framework C# compiler was not found; cannot build the taskkill shim for Windows Sandbox."
        }
        $sourcePath = Join-Path $shimDir "taskkill.cs"
        [System.IO.File]::WriteAllText($sourcePath, $source, [System.Text.UTF8Encoding]::new($false))
        $compileOutput = & $csc /nologo /target:exe /optimize+ "/out:$shimExe" $sourcePath 2>&1
        if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $shimExe -PathType Leaf)) {
            throw "Failed to compile the taskkill shim: $($compileOutput -join "`n")"
        }
    }
    return $shimExe
}

# Replace the guest's taskkill.exe with the WMI-free shim when WMI is broken.
function Install-WindowsSandboxTaskkillShim {
    try {
        Get-CimInstance -ClassName Win32_Process -Filter "ProcessId = $PID" -ErrorAction Stop | Out-Null
        Write-Host "WMI works in this guest; keeping the system taskkill.exe."
        return
    } catch {
        Write-Host "WMI is unavailable in this guest ($($_.Exception.Message.Trim())); replacing taskkill.exe with a WMI-free shim."
    }

    $shim = Get-TaskkillShimPath
    foreach ($dir in @("System32", "SysWOW64")) {
        $target = Join-Path $env:SystemRoot "$dir\taskkill.exe"
        if (-not (Test-Path -LiteralPath $target -PathType Leaf)) {
            continue
        }
        if (Test-Path -LiteralPath "$target.orig" -PathType Leaf) {
            continue
        }
        $notes = @()
        $notes += (cmd.exe /c "takeown /f `"$target`" 2>&1" | Out-String).Trim()
        $notes += (cmd.exe /c "icacls `"$target`" /grant Administrators:F 2>&1" | Out-String).Trim()
        try {
            Move-Item -LiteralPath $target -Destination "$target.orig" -Force
            Copy-Item -LiteralPath $shim -Destination $target -Force
            Write-Host "  replaced $target"
        } catch {
            Write-Warning "Could not replace ${target}: $($_.Exception.Message) ($($notes -join ' / '))"
        }
    }
}


# Windows Sandbox ships with Smart App Control in evaluation mode while the
# Defender antivirus service is disabled. Every msiexec /i or /a then stalls
# for 120 seconds while wintrust retries a Defender reputation lookup that can
# never be answered (microsoft/Windows-Sandbox#68, #85). Turning Smart App
# Control off and refreshing the Code Integrity policy removes the stall
# without a reboot. This is a functional-test guest, not a security boundary
# under test, so the policy is disabled before any installer work starts.
function Initialize-SandboxInstallerPerformance {
    param([string]$DistDir)

    Write-Host "Disabling Smart App Control evaluation in the Windows Sandbox guest..."
    foreach ($controlSet in @("CurrentControlSet", "ControlSet001")) {
        $policyKey = "HKLM:\SYSTEM\$controlSet\Control\CI\Policy"
        if (Test-Path -LiteralPath $policyKey) {
            Set-ItemProperty -Path $policyKey -Name VerifiedAndReputablePolicyState -Value 0 -Type DWord
        }
    }
    $ciTool = Join-Path $env:SystemRoot "System32\CiTool.exe"
    if (Test-Path -LiteralPath $ciTool -PathType Leaf) {
        $refresh = (& $ciTool --refresh --json 2>&1) -join " "
        Write-Host "  CiTool --refresh: $refresh"
    } else {
        Write-Warning "CiTool.exe is not available; installer runs may stall for two minutes each."
    }

    # Prove the fix took hold with the smallest MSI available before spending
    # minutes on real installs.
    $probeMsi = Get-ChildItem -LiteralPath $DistDir -Filter "*.msi" -File -ErrorAction SilentlyContinue |
        Sort-Object Length | Select-Object -First 1
    if (-not $probeMsi) {
        return
    }
    $probeRoot = Join-Path $env:TEMP ("sandbox-msi-probe-" + [guid]::NewGuid().ToString("N"))
    New-Item -ItemType Directory -Path $probeRoot -Force | Out-Null
    try {
        $probeCopy = Join-Path $probeRoot $probeMsi.Name
        Copy-Item -LiteralPath $probeMsi.FullName -Destination $probeCopy -Force
        $probeLog = Join-Path $probeRoot "probe.log"
        $probeTarget = Join-Path $probeRoot "extract"
        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        $proc = Start-Process -FilePath (Join-Path $env:SystemRoot "System32\msiexec.exe") `
            -ArgumentList @("/a", "`"$probeCopy`"", "/qn", "TARGETDIR=`"$probeTarget`"", "/L*v", "`"$probeLog`"") `
            -Wait -PassThru -NoNewWindow
        $sw.Stop()
        $seconds = [math]::Round($sw.Elapsed.TotalSeconds, 1)
        Write-Host "  msiexec probe ($($probeMsi.Name)): exit=$($proc.ExitCode) in ${seconds}s"
        if ($sw.Elapsed.TotalSeconds -ge 60) {
            Write-Warning "msiexec still stalls in this guest (${seconds}s for a $([math]::Round($probeMsi.Length / 1KB)) KB package); installer tests will be slow."
        }
    } finally {
        Remove-Item -LiteralPath $probeRoot -Recurse -Force -ErrorAction SilentlyContinue
    }
}



try {
    Start-Transcript -LiteralPath $transcriptPath -Force | Out-Null

    $env:OLLAMA_TEST_SANDBOX = "1"
    $env:OLLAMA_TEST_ARTIFACT_DIR = $RunRoot
    $destructiveTags = @("Integration", "UpgradeMatrix", "AppIntegration")
    if (@($summary.tags | Where-Object { $_ -in $destructiveTags }).Count -gt 0) {
        Initialize-SandboxInstallerPerformance -DistDir (Join-Path $RepoRoot "dist")
        Install-WindowsSandboxTaskkillShim
    }


    $pesterManifest = Join-Path $PesterModuleRoot "Pester.psd1"
    if (-not (Test-Path -LiteralPath $pesterManifest -PathType Leaf)) {
        throw "Mapped Pester module not found: $pesterManifest"
    }
    Import-Module $pesterManifest -MinimumVersion 5.0 -Force

    $testPath = Join-Path $RepoRoot "scripts\tests\install.Tests.ps1"
    if (-not (Test-Path -LiteralPath $testPath -PathType Leaf)) {
        throw "Installer tests not found: $testPath"
    }

    $env:OLLAMA_TEST_INSTALLER_CACHE_DIR = $InstallerCache
    $env:OLLAMA_TEST_IGNORE_DIST_INSTALLER = "1"
    $env:OLLAMA_TEST_LATEST_INSTALLER_VERSION = "0.1.48"
    $env:OLLAMA_TEST_PINNED_UPGRADE_VERSION = "0.1.47"
    $distDir = Join-Path $RepoRoot "dist"
    if (Test-Path -LiteralPath $distDir -PathType Container) {
        $env:OLLAMA_TEST_DIST_DIR = $distDir
    }

    $configuration = New-PesterConfiguration
    $configuration.Run.Path = $testPath
    $configuration.Run.PassThru = $true
    $configuration.Filter.Tag = $summary.tags
    $configuration.Output.Verbosity = "Detailed"

    # Pester's NUnit writer queries Win32_OperatingSystem through CIM. Windows
    # Sandbox denies that query even though the tests themselves can run, so
    # keep the portable JSON summary and transcript as the guest artifacts.
    $configuration.TestResult.Enabled = $false

    $result = Invoke-Pester -Configuration $configuration
    $summary.totalCount = $result.TotalCount
    $summary.passedCount = $result.PassedCount
    $summary.failedCount = $result.FailedCount
    $summary.skippedCount = $result.SkippedCount

    if ($result.FailedCount -eq 0 -and $result.Result -eq "Passed") {
        $summary.outcome = "Passed"
        $exitCode = 0
    } else {
        $summary.error = "Pester completed with result $($result.Result) and $($result.FailedCount) failed tests."
    }
} catch {
    $summary.error = ($_ | Out-String).Trim()
} finally {
    $summary.completed = $true
    $summary.completedAt = (Get-Date).ToString("o")
    $ollamaLogDir = Join-Path $env:LOCALAPPDATA "Ollama"
    if (Test-Path -LiteralPath $ollamaLogDir -PathType Container) {
        $logArtifactDir = Join-Path $RunRoot "localappdata-logs"
        New-Item -ItemType Directory -Path $logArtifactDir -Force -ErrorAction SilentlyContinue | Out-Null
        Get-ChildItem -LiteralPath $ollamaLogDir -File -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -match '\.log(?:-|$)' } |
            Copy-Item -Destination $logArtifactDir -Force -ErrorAction SilentlyContinue
    }
    try { Stop-Transcript | Out-Null } catch { }
    $json = $summary | ConvertTo-Json -Depth 4
    [System.IO.File]::WriteAllText($resultTempPath, $json, [System.Text.Encoding]::UTF8)
    Move-Item -LiteralPath $resultTempPath -Destination $resultPath -Force
    if (-not $NoShutdown) {
        shutdown.exe /s /f /t 5 | Out-Null
    }
}

exit $exitCode
