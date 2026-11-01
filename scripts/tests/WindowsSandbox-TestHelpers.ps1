# Shared Windows Sandbox lifecycle helpers for the installer test runner and
# the setup smoke test. Dot-source this file.
#
# The Store-distributed Windows Sandbox ships a command-line interface
# (wsb.exe). It is used for everything except triggering the guest user logon:
#
#   wsb start   creates the VM headless from an inline .wsb-style XML config
#   wsb connect opens the RDP client, which is what logs WDAGUtilityAccount on
#               and fires <LogonCommand>; the session survives closing the client
#   wsb exec    runs commands in the guest (used for diagnostics)
#   wsb stop    deterministically tears the VM down, including on failure
#
# The RDP client window is parked off-screen without activation while the
# guest logs on, then closed as soon as the guest signals that its logon
# command started, so nothing steals the foreground for the rest of the run.

$script:WindowsSandboxClientProcessNames = @(
    "WindowsSandbox",
    "WindowsSandboxClient",
    "WindowsSandboxRemoteSession"
)

function Initialize-WindowsSandboxWindowControl {
    if ("Ollama.WindowsSandboxWindow" -as [type]) {
        return
    }

    Add-Type -TypeDefinition @"
using System;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text;

namespace Ollama {
    public static class WindowsSandboxWindow {
        private delegate bool EnumWindowsProc(IntPtr window, IntPtr parameter);

        [DllImport("user32.dll")]
        private static extern bool EnumWindows(EnumWindowsProc callback, IntPtr parameter);

        [DllImport("user32.dll")]
        private static extern bool EnumChildWindows(IntPtr parent, EnumWindowsProc callback, IntPtr parameter);

        [DllImport("user32.dll")]
        private static extern bool IsWindowVisible(IntPtr window);

        [DllImport("user32.dll")]
        private static extern uint GetWindowThreadProcessId(IntPtr window, out uint processId);

        [DllImport("user32.dll", CharSet = CharSet.Unicode)]
        private static extern int GetWindowText(IntPtr window, StringBuilder text, int count);

        [DllImport("user32.dll")]
        private static extern bool ShowWindowAsync(IntPtr window, int command);

        [DllImport("user32.dll")]
        private static extern bool SetWindowPos(
            IntPtr window,
            IntPtr insertAfter,
            int x,
            int y,
            int width,
            int height,
            uint flags);

        [DllImport("user32.dll")]
        private static extern IntPtr GetForegroundWindow();

        [DllImport("user32.dll")]
        private static extern bool SetForegroundWindow(IntPtr window);

        private static bool IsSandboxWindow(IntPtr window) {
            if (window == IntPtr.Zero || !IsWindowVisible(window)) return false;

            uint processId;
            GetWindowThreadProcessId(window, out processId);
            string processName = "";
            try { processName = Process.GetProcessById((int)processId).ProcessName; } catch { }
            return processName.StartsWith("WindowsSandbox", StringComparison.OrdinalIgnoreCase);
        }

        private static int ForEachSandboxWindow(Action<IntPtr> action) {
            int count = 0;
            EnumWindows(delegate(IntPtr window, IntPtr parameter) {
                if (IsSandboxWindow(window)) {
                    action(window);
                    count++;
                }
                return true;
            }, IntPtr.Zero);
            return count;
        }

        public static IntPtr ForegroundWindow() {
            return GetForegroundWindow();
        }

        public static int ParkOffscreen(IntPtr restoreFocusTo) {
            const uint SWP_NOSIZE = 0x0001;
            const uint SWP_NOZORDER = 0x0004;
            const uint SWP_NOACTIVATE = 0x0010;
            const uint SWP_ASYNCWINDOWPOS = 0x4000;
            bool sandboxHadFocus = IsSandboxWindow(GetForegroundWindow());
            int count = ForEachSandboxWindow(delegate(IntPtr window) {
                SetWindowPos(
                    window,
                    IntPtr.Zero,
                    -32000,
                    -32000,
                    0,
                    0,
                    SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE | SWP_ASYNCWINDOWPOS);
            });
            if (count > 0 && sandboxHadFocus && restoreFocusTo != IntPtr.Zero) {
                SetForegroundWindow(restoreFocusTo);
            }
            return count;
        }

        public static void Minimize() {
            ForEachSandboxWindow(delegate(IntPtr window) { ShowWindowAsync(window, 6); });
        }

        public static string GetErrorText() {
            string error = "";
            ForEachSandboxWindow(delegate(IntPtr window) {
                var text = new StringBuilder();
                EnumChildWindows(window, delegate(IntPtr child, IntPtr parameter) {
                    var childText = new StringBuilder(1024);
                    GetWindowText(child, childText, childText.Capacity);
                    if (childText.Length > 0) text.AppendLine(childText.ToString());
                    return true;
                }, IntPtr.Zero);
                string candidate = text.ToString().Trim();
                if (candidate.IndexOf("Error 0x", StringComparison.OrdinalIgnoreCase) >= 0 ||
                    candidate.IndexOf("Windows Sandbox failed", StringComparison.OrdinalIgnoreCase) >= 0) {
                    error = candidate;
                }
            });
            return error;
        }
    }
}
"@
}

function Get-ForegroundWindowHandle {
    Initialize-WindowsSandboxWindowControl
    return [Ollama.WindowsSandboxWindow]::ForegroundWindow()
}

function Set-WindowsSandboxWindowParked {
    param([IntPtr]$RestoreFocusTo = [IntPtr]::Zero)

    Initialize-WindowsSandboxWindowControl
    return [Ollama.WindowsSandboxWindow]::ParkOffscreen($RestoreFocusTo)
}

function Set-WindowsSandboxWindowMinimized {
    Initialize-WindowsSandboxWindowControl
    [Ollama.WindowsSandboxWindow]::Minimize()
}

function Get-WindowsSandboxErrorText {
    Initialize-WindowsSandboxWindowControl
    return [Ollama.WindowsSandboxWindow]::GetErrorText()
}

function Get-WindowsSandboxCliPath {
    $cli = Get-Command wsb.exe -ErrorAction SilentlyContinue
    if (-not $cli) {
        throw ("The Windows Sandbox command-line interface (wsb.exe) was not found. " +
               "Installer tests need the Store-distributed Windows Sandbox on Windows 11 24H2 or newer; " +
               "update Windows Sandbox from the Microsoft Store, or rerun with -Isolation Host.")
    }
    return $cli.Source
}

function Invoke-WindowsSandboxCli {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    $cli = Get-WindowsSandboxCliPath
    $output = & $cli @Arguments --raw 2>&1 | ForEach-Object { $_.ToString() }
    $exitCode = $LASTEXITCODE
    $text = ($output -join "`n").Trim()
    if ($exitCode -ne 0) {
        throw "wsb $($Arguments -join ' ') failed (exit code $exitCode): $text"
    }
    if (-not $text) {
        return $null
    }
    return ($text | ConvertFrom-Json)
}

function Get-WindowsSandboxRunningIds {
    $result = Invoke-WindowsSandboxCli -Arguments @("list")
    if (-not $result -or -not $result.WindowsSandboxEnvironments) {
        return @()
    }
    return @($result.WindowsSandboxEnvironments | ForEach-Object { [string]$_.Id })
}

function Get-WindowsSandboxClientProcesses {
    return @(Get-Process -Name $script:WindowsSandboxClientProcessNames -ErrorAction SilentlyContinue)
}

function Assert-NoWindowsSandboxRunning {
    $running = Get-WindowsSandboxRunningIds
    if ($running.Count -gt 0) {
        throw ("Another Windows Sandbox instance is already running ($($running -join ', ')). " +
               "Stop it first with: wsb stop --id <id>")
    }
    $clients = Get-WindowsSandboxClientProcesses
    if ($clients.Count -gt 0) {
        throw ("A Windows Sandbox client window is still open ($(($clients | ForEach-Object { $_.Name } | Select-Object -Unique) -join ', ')). " +
               "Close it before starting installer tests.")
    }
}

function Start-WindowsSandboxTestInstance {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ConfigurationXml
    )

    $result = Invoke-WindowsSandboxCli -Arguments @("start", "--config", $ConfigurationXml)
    if (-not $result -or -not $result.Id) {
        throw "wsb start did not return a sandbox ID."
    }
    return [string]$result.Id
}

function Stop-WindowsSandboxClient {
    Get-WindowsSandboxClientProcesses | Stop-Process -Force -ErrorAction SilentlyContinue
}

function Stop-WindowsSandboxTestInstance {
    param([string]$Id = "")

    Stop-WindowsSandboxClient
    $ids = if ($Id) { @($Id) } else { @() }
    if (-not $Id) {
        try { $ids = Get-WindowsSandboxRunningIds } catch { $ids = @() }
    }
    foreach ($sandboxId in $ids) {
        try {
            Invoke-WindowsSandboxCli -Arguments @("stop", "--id", $sandboxId) | Out-Null
        } catch {
            Write-Verbose "wsb stop --id $sandboxId reported: $($_.Exception.Message)"
        }
    }
}

function Wait-WindowsSandboxTestInstanceStopped {
    param(
        [string]$Id = "",
        [ValidateRange(1, 120)][int]$TimeoutSeconds = 30
    )

    $deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
    do {
        $running = @()
        try { $running = Get-WindowsSandboxRunningIds } catch { }
        $stillRunning = if ($Id) { $Id -in $running } else { $running.Count -gt 0 }
        if (-not $stillRunning -and (Get-WindowsSandboxClientProcesses).Count -eq 0) {
            return
        }
        Start-Sleep -Milliseconds 250
    } while ([DateTime]::UtcNow -lt $deadline)

    Stop-WindowsSandboxTestInstance -Id $Id
}

# Open the RDP client so the guest user session is created and <LogonCommand>
# runs. Returns once $StartedPath exists (written by the logon command) and the
# client has been closed again; the guest session keeps running headless.
function Connect-WindowsSandboxTestInstance {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Id,

        [Parameter(Mandatory = $true)]
        [string]$StartedPath,

        [ValidateRange(1, 60)][int]$TimeoutMinutes = 5
    )

    $foregroundWindow = Get-ForegroundWindowHandle
    $cli = Get-WindowsSandboxCliPath
    Start-Process -FilePath $cli -ArgumentList @("connect", "--id", $Id) -WindowStyle Hidden | Out-Null

    $deadline = [DateTime]::UtcNow.AddMinutes($TimeoutMinutes)
    while (-not (Test-Path -LiteralPath $StartedPath -PathType Leaf) -and [DateTime]::UtcNow -lt $deadline) {
        Set-WindowsSandboxWindowParked -RestoreFocusTo $foregroundWindow | Out-Null
        $sandboxError = Get-WindowsSandboxErrorText
        if ($sandboxError) {
            Stop-WindowsSandboxTestInstance -Id $Id
            throw "Windows Sandbox failed before the guest logon command started:`n$sandboxError"
        }
        Start-Sleep -Milliseconds 50
    }
    if (-not (Test-Path -LiteralPath $StartedPath -PathType Leaf)) {
        Stop-WindowsSandboxTestInstance -Id $Id
        throw "Windows Sandbox $Id started but its logon command did not begin within $TimeoutMinutes minutes."
    }

    Set-WindowsSandboxWindowMinimized
    Stop-WindowsSandboxClient
}

function New-WindowsSandboxConfigurationXml {
    param(
        [Parameter(Mandatory = $true)]
        [hashtable[]]$MappedFolders,

        [Parameter(Mandatory = $true)]
        [string]$LogonCommand,

        [ValidateRange(2048, 65536)][int]$MemoryInMB = 8192
    )

    $folders = foreach ($folder in $MappedFolders) {
        $readOnly = if ($folder.ReadOnly) { "true" } else { "false" }
        "    <MappedFolder>`n" +
        "      <HostFolder>$([Security.SecurityElement]::Escape($folder.HostFolder))</HostFolder>`n" +
        "      <SandboxFolder>$([Security.SecurityElement]::Escape($folder.SandboxFolder))</SandboxFolder>`n" +
        "      <ReadOnly>$readOnly</ReadOnly>`n" +
        "    </MappedFolder>"
    }
    return @"
<Configuration>
  <VGpu>Disable</VGpu>
  <Networking>Enable</Networking>
  <MemoryInMB>$MemoryInMB</MemoryInMB>
  <MappedFolders>
$($folders -join "`n")
  </MappedFolders>
  <LogonCommand>
    <Command>$([Security.SecurityElement]::Escape($LogonCommand))</Command>
  </LogonCommand>
</Configuration>
"@
}
