<#
.SYNOPSIS
    Pester tests for the Windows install.ps1 script.

.DESCRIPTION
    IMPORTANT: Integration tests (Tag: Integration) install and uninstall
    Ollama for real. Run them only on a machine where Ollama is not currently
    installed; the tests abort if they detect an existing install.

    Unit tests (Tag: Unit) are fast, mock downloads/signatures/process launch,
    and do not touch the real install state.

    Integration tests (Tag: Integration) use a signed dist\OllamaSetup.exe
    when present. Otherwise, they download and cache the latest shipped
    OllamaSetup.exe. They install Ollama for real, verify the installed state,
    run a pinned-version upgrade from 0.21.0, and then run the real uninstaller.

.EXAMPLE
    Import-Module Pester -MinimumVersion 5.0
    Invoke-Pester scripts\tests\install.Tests.ps1 -Tag Unit -Output Detailed

.EXAMPLE
    # Compatibility check: run unit tests under Windows PowerShell.
    powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "Import-Module Pester -MinimumVersion 5.0; Invoke-Pester scripts\tests\install.Tests.ps1 -Tag Unit -Output Detailed"

.EXAMPLE
    # Destructive: requires no existing Ollama install.
    Invoke-Pester scripts\tests\install.Tests.ps1 -Tag Integration -Output Detailed
#>

BeforeAll {
    $script:RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
    $script:InstallScript = Join-Path $script:RepoRoot "scripts\install.ps1"
    $script:UpdateServerScript = Join-Path $script:RepoRoot "scripts\tests\update-server.py"
    $script:InstallerDownloadCacheDir = Join-Path $script:RepoRoot ".cache\install-tests"
    $script:DistInstaller = Join-Path $script:RepoRoot "dist\OllamaSetup.exe"
    $script:OriginalLocalAppData = [Environment]::GetEnvironmentVariable("LOCALAPPDATA", "Process")
    $script:LatestInstallerUrl = "https://ollama.com/download/OllamaSetup.exe"
    $script:PinnedUpgradeVersion = "0.21.0"
    $script:PinnedUpgradeInstallerUrl = "https://github.com/ollama/ollama/releases/download/v$($script:PinnedUpgradeVersion)/OllamaSetup.exe"
    $script:InnoSetupUninstallGuid = "{44E83376-CE68-45EB-8FC1-393500EB558C}_is1"
    $script:EnvNames = @(
        "OLLAMA_VERSION",
        "OLLAMA_INSTALL_DIR",
        "OLLAMA_UNINSTALL",
        "OLLAMA_CACHE_ONLY",
        "OLLAMA_INSTALL_CACHED",
        "OLLAMA_DEBUG",
        "LOCALAPPDATA",
        "TEMP"
    )

    function Save-TestEnvironment {
        $script:SavedEnv = @{}
        foreach ($name in $script:EnvNames) {
            $script:SavedEnv[$name] = [Environment]::GetEnvironmentVariable($name, "Process")
            Remove-Item "Env:$name" -ErrorAction SilentlyContinue
        }
    }

    function Restore-TestEnvironment {
        foreach ($name in $script:EnvNames) {
            Remove-Item "Env:$name" -ErrorAction SilentlyContinue
            if ($null -ne $script:SavedEnv[$name]) {
                Set-Item "Env:$name" $script:SavedEnv[$name]
            }
        }
    }

    function Set-TestProcessEnvironment {
        $pathValue = [Environment]::GetEnvironmentVariable("Path", "Process")
        if (-not $pathValue) {
            $pathValue = [Environment]::GetEnvironmentVariable("PATH", "Process")
        }

        [Environment]::SetEnvironmentVariable("PATH", $null, "Process")
        if ($pathValue) {
            [Environment]::SetEnvironmentVariable("Path", $pathValue, "Process")
        }
    }

    function Get-WindowsPowerShellModulePath {
        $paths = @()
        $documents = [Environment]::GetFolderPath("MyDocuments")
        if ($documents) {
            $paths += Join-Path $documents "WindowsPowerShell\Modules"
        }
        if ($env:ProgramFiles) {
            $paths += Join-Path $env:ProgramFiles "WindowsPowerShell\Modules"
        }
        if ($env:SystemRoot) {
            $paths += Join-Path $env:SystemRoot "system32\WindowsPowerShell\v1.0\Modules"
        }
        return ($paths -join ";")
    }

    function Set-TestEnvironment {
        param(
            [hashtable]$Values = @{}
        )

        $testRoot = Join-Path $TestDrive ([guid]::NewGuid().ToString("N"))
        $localAppData = Join-Path $testRoot "LocalAppData"
        $tempDir = Join-Path $testRoot "Temp"
        New-Item -ItemType Directory -Path $localAppData, $tempDir -Force | Out-Null
        Set-Item Env:LOCALAPPDATA $localAppData
        Set-Item Env:TEMP $tempDir

        foreach ($name in $Values.Keys) {
            Set-Item "Env:$name" $Values[$name]
        }
    }

    function New-FakeProcess {
        $process = [PSCustomObject]@{ ExitCode = 0 }
        $process | Add-Member -MemberType ScriptMethod -Name WaitForExit -Value { }
        return $process
    }

    function Get-TestInstallerCacheDir {
        param([string]$ETag)

        $normalizedETag = $ETag.Trim().Trim('"')
        $sha256 = [System.Security.Cryptography.SHA256]::Create()
        try {
            $bytes = [System.Text.Encoding]::UTF8.GetBytes($normalizedETag)
            $hash = $sha256.ComputeHash($bytes)
            $cacheKey = -join ($hash | ForEach-Object { $_.ToString("x2") })
            return Join-Path $env:LOCALAPPDATA "Ollama\install_cache\$cacheKey"
        } finally {
            $sha256.Dispose()
        }
    }

    function Get-TestInstallerStagingCacheDir {
        param([string]$ETag)

        return "$(Get-TestInstallerCacheDir $ETag).download"
    }

    function Get-TestInstallerStagingPath {
        param([string]$ETag)

        return Join-Path (Get-TestInstallerStagingCacheDir $ETag) "OllamaSetup.exe"
    }

    function New-TestInstallScript {
        param([string]$DownloadBaseUrl)

        $scriptPath = Join-Path $TestDrive ("install-" + [guid]::NewGuid().ToString("N") + ".ps1")
        $content = Get-Content -LiteralPath $script:InstallScript -Raw
        $replacement = '$DownloadBaseURL = "' + $DownloadBaseUrl.TrimEnd('/') + '"'
        $content = $content.Replace('$DownloadBaseURL = "https://ollama.com/download"', $replacement)
        Set-Content -LiteralPath $scriptPath -Value $content -Encoding UTF8
        return $scriptPath
    }

    function Get-IntegrationInstaller {
        if (Test-Path -LiteralPath $script:DistInstaller) {
            $installer = (Resolve-Path -LiteralPath $script:DistInstaller).Path
            try {
                Assert-TestInstallerSignature -FilePath $installer
                Write-Host "    Using signed dist installer: $installer" -ForegroundColor DarkGray
                return $installer
            } catch {
                Write-Host "    Ignoring unsigned dist installer: $installer ($($_.Exception.Message))" -ForegroundColor DarkYellow
            }
        }

        return Save-TestInstallerDownload -Url $script:LatestInstallerUrl -FileName "OllamaSetup-latest.exe"
    }

    function Get-PinnedUpgradeInstaller {
        return Save-TestInstallerDownload -Url $script:PinnedUpgradeInstallerUrl -FileName "OllamaSetup-$($script:PinnedUpgradeVersion).exe"
    }

    function Save-TestInstallerDownload {
        param(
            [string]$Url,
            [string]$FileName
        )

        New-Item -ItemType Directory -Path $script:InstallerDownloadCacheDir -Force | Out-Null
        $installerPath = Join-Path $script:InstallerDownloadCacheDir $FileName
        if (-not (Test-Path -LiteralPath $installerPath)) {
            Write-Host "    Downloading $Url" -ForegroundColor DarkGray
            $oldProgressPreference = $ProgressPreference
            $ProgressPreference = "SilentlyContinue"
            try {
                Invoke-WebRequest -Uri $Url -OutFile $installerPath -UseBasicParsing
            } finally {
                $ProgressPreference = $oldProgressPreference
            }
        } else {
            Write-Host "    Using cached installer: $installerPath" -ForegroundColor DarkGray
        }
        Assert-TestInstallerSignature -FilePath $installerPath
        return $installerPath
    }

    function Assert-TestInstallerSignature {
        param([string]$FilePath)

        $sig = Get-AuthenticodeSignature -FilePath $FilePath
        if ($sig.Status -ne "Valid") {
            throw "Installer signature is not valid: $FilePath ($($sig.Status))"
        }
        $subject = $sig.SignerCertificate.Subject
        if ($subject -notmatch "(^|, )O=Ollama Inc\.(,|$)") {
            throw "Installer is not signed by Ollama Inc.: $subject"
        }
    }

    function Get-TestPython {
        foreach ($candidate in @("python.exe", "python")) {
            $command = Get-Command $candidate -ErrorAction SilentlyContinue | Select-Object -First 1
            if ($command) {
                return $command.Source
            }
        }
        return $null
    }

    function Get-FreeTcpPort {
        $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Parse("127.0.0.1"), 0)
        try {
            $listener.Start()
            return $listener.LocalEndpoint.Port
        } finally {
            $listener.Stop()
        }
    }

    function Wait-TestHttpServer {
        param($Server)

        $url = "$($Server.BaseUrl)/download/OllamaSetup.exe"
        for ($i = 0; $i -lt 50; $i++) {
            try {
                Invoke-WebRequest -Uri $url -Method Head -UseBasicParsing -TimeoutSec 2 | Out-Null
                return
            } catch {
                Start-Sleep -Milliseconds 100
            }
        }

        $stderr = if ($Server.Process.HasExited) { $Server.Process.StandardError.ReadToEnd() } else { "" }
        throw "Local HTTP server did not start. $stderr"
    }

    function Join-TestProcessArguments {
        param([string[]]$Arguments)

        return (($Arguments | ForEach-Object {
            if ($null -eq $_ -or $_.Length -eq 0) {
                '""'
            } elseif ($_.IndexOfAny(@([char]32, [char]9, [char]34)) -lt 0) {
                $_
            } else {
                '"' + ($_.Replace('\', '\\').Replace('"', '\"')) + '"'
            }
        }) -join " ")
    }

    function Start-LocalHttpServer {
        param(
            [string]$InstallerPath,
            [string]$ETag = ""
        )

        $python = Get-TestPython
        if (-not $python) {
            Set-ItResult -Skipped -Because "Python is required to run local HTTP integration tests"
            return $null
        }

        $port = Get-FreeTcpPort
        $args = @(
            $script:UpdateServerScript,
            "--host", "127.0.0.1",
            "--port", [string]$port,
            "--installer", $InstallerPath,
            "--install-script", $script:InstallScript
        )
        if ($ETag) {
            $args += @("--installer-etag", $ETag)
        }

        $startInfo = [System.Diagnostics.ProcessStartInfo]::new()
        $startInfo.FileName = $python
        $startInfo.Arguments = Join-TestProcessArguments $args
        $startInfo.WorkingDirectory = $script:RepoRoot
        $startInfo.UseShellExecute = $false
        $startInfo.CreateNoWindow = $true
        $startInfo.RedirectStandardOutput = $true
        $startInfo.RedirectStandardError = $true

        $proc = [System.Diagnostics.Process]::new()
        $proc.StartInfo = $startInfo
        [void]$proc.Start()

        $server = [PSCustomObject]@{
            Process = $proc
            BaseUrl = "http://127.0.0.1:$port"
            DownloadBaseUrl = "http://127.0.0.1:$port/download"
        }
        Wait-TestHttpServer -Server $server
        return $server
    }

    function Stop-LocalHttpServer {
        param($Server)

        if ($Server -and $Server.Process -and -not $Server.Process.HasExited) {
            Stop-Process -Id $Server.Process.Id -Force -ErrorAction SilentlyContinue
        }
        if ($Server -and $Server.Process) {
            $Server.Process.Dispose()
        }
    }

    function Get-TestOllamaInstallDir {
        param([switch]$UseOriginalLocalAppData)

        if (-not $UseOriginalLocalAppData -and $env:OLLAMA_INSTALL_DIR) {
            return $env:OLLAMA_INSTALL_DIR
        }
        $localAppData = if ($UseOriginalLocalAppData -or -not $env:LOCALAPPDATA) {
            $script:OriginalLocalAppData
        } else {
            $env:LOCALAPPDATA
        }
        return Join-Path $localAppData "Programs\Ollama"
    }

    function Get-TestOllamaRegistryKey {
        $possibleKeys = @(
            "HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\$script:InnoSetupUninstallGuid",
            "HKLM:\Software\Microsoft\Windows\CurrentVersion\Uninstall\$script:InnoSetupUninstallGuid",
            "HKLM:\Software\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\$script:InnoSetupUninstallGuid"
        )

        foreach ($key in $possibleKeys) {
            if (Test-Path $key) {
                return $key
            }
        }
        return $null
    }

    function Get-TestOllamaInstalledVersion {
        param([string]$InstallDir)

        $regKey = Get-TestOllamaRegistryKey
        if ($regKey) {
            $displayVersion = (Get-ItemProperty -Path $regKey).DisplayVersion
            if ($displayVersion) {
                return $displayVersion
            }
        }

        $ollamaExe = Join-Path $InstallDir "ollama.exe"
        if (Test-Path -LiteralPath $ollamaExe) {
            try {
                $output = & $ollamaExe --version 2>$null
                $text = if ($output -is [array]) { $output -join "`n" } else { [string]$output }
                if ($text -match "(\d+\.\d+\.\d+)") {
                    return $matches[1]
                }
            } catch { }
        }

        return $null
    }

    function Test-OllamaRealInstallPresent {
        $installDir = Get-TestOllamaInstallDir -UseOriginalLocalAppData
        return [bool]((Get-TestOllamaRegistryKey) -or (Test-Path -LiteralPath (Join-Path $installDir "Ollama app.exe")))
    }

    function Stop-TestOllamaProcesses {
        Get-Process -Name "ollama", "Ollama app" -ErrorAction SilentlyContinue |
            Stop-Process -Force -ErrorAction SilentlyContinue
    }

    function Split-TestCommandLine {
        param([string]$CommandLine)

        if ($CommandLine -match '^\s*"([^"]+)"\s*(.*)$') {
            return [PSCustomObject]@{ FilePath = $matches[1]; Arguments = $matches[2] }
        }
        $parts = $CommandLine -split '\s+', 2
        return [PSCustomObject]@{
            FilePath = $parts[0]
            Arguments = if ($parts.Count -gt 1) { $parts[1] } else { "" }
        }
    }

    function Start-TestProcessAndWait {
        param(
            [string]$FilePath,
            [string]$Arguments = "",
            [int]$TimeoutSeconds = 180
        )

        $startInfo = [System.Diagnostics.ProcessStartInfo]::new()
        $startInfo.FileName = $FilePath
        $startInfo.Arguments = $Arguments
        $startInfo.UseShellExecute = $false
        $startInfo.CreateNoWindow = $true

        $proc = [System.Diagnostics.Process]::new()
        $proc.StartInfo = $startInfo
        [void]$proc.Start()
        try {
            if (-not $proc.WaitForExit($TimeoutSeconds * 1000)) {
                $proc.Kill()
                throw "Process did not exit within $TimeoutSeconds seconds: $FilePath"
            }
            return $proc.ExitCode
        } finally {
            $proc.Dispose()
        }
    }

    function Invoke-InstallScript {
        param(
            [string]$InstallScriptPath = $script:InstallScript,
            [hashtable]$EnvVars = @{},
            [int]$TimeoutSeconds = 600
        )

        $flagSummary = ($EnvVars.Keys | Sort-Object | ForEach-Object { "$_=$($EnvVars[$_])" }) -join ", "
        if ($flagSummary) {
            Write-Host "    Running install.ps1 ($flagSummary)" -ForegroundColor DarkGray
        } else {
            Write-Host "    Running install.ps1 (defaults)" -ForegroundColor DarkGray
        }

        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        $exitCode = 0
        $stdout = ""
        $stderr = ""
        $proc = $null
        try {
            Set-TestProcessEnvironment

            foreach ($key in $EnvVars.Keys) {
                Set-Item -Path "Env:\$key" -Value $EnvVars[$key]
            }

            $startInfo = [System.Diagnostics.ProcessStartInfo]::new()
            $startInfo.FileName = "powershell.exe"
            $startInfo.Arguments = Join-TestProcessArguments @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $InstallScriptPath)
            $startInfo.WorkingDirectory = $script:RepoRoot
            $startInfo.UseShellExecute = $false
            $startInfo.CreateNoWindow = $true
            $startInfo.RedirectStandardOutput = $true
            $startInfo.RedirectStandardError = $true
            $startInfo.EnvironmentVariables["PSModulePath"] = Get-WindowsPowerShellModulePath

            $proc = [System.Diagnostics.Process]::new()
            $proc.StartInfo = $startInfo

            [void]$proc.Start()
            $stdoutTask = $proc.StandardOutput.ReadToEndAsync()
            $stderrTask = $proc.StandardError.ReadToEndAsync()

            if (-not $proc.WaitForExit($TimeoutSeconds * 1000)) {
                try {
                    $proc.Kill()
                } catch { }
                throw "install.ps1 did not exit within $TimeoutSeconds seconds"
            }
            $proc.WaitForExit()
            $exitCode = $proc.ExitCode
            $stdout = $stdoutTask.GetAwaiter().GetResult()
            $stderr = $stderrTask.GetAwaiter().GetResult()
        } finally {
            foreach ($key in $EnvVars.Keys) {
                Remove-Item "Env:\$key" -ErrorAction SilentlyContinue
            }
            if ($proc) {
                $proc.Dispose()
            }
        }
        $sw.Stop()
        Write-Host "    Install completed in $($sw.Elapsed.ToString('mm\:ss'))" -ForegroundColor DarkGray
        if ($exitCode -ne 0) {
            if ($stdout) {
                Write-Host "    install.ps1 stdout:" -ForegroundColor DarkGray
                Write-Host $stdout
            }
            if ($stderr) {
                Write-Host "    install.ps1 stderr:" -ForegroundColor DarkGray
                Write-Host $stderr
            }
        }

        return @{
            ExitCode = $exitCode
            Output   = $stdout
            Error    = $stderr
            Duration = $sw.Elapsed
        }
    }

    function Invoke-TestInnoInstall {
        param(
            [string]$InstallerPath,
            [string]$InstallDir
        )

        Assert-TestInstallerSignature -FilePath $InstallerPath
        $args = Join-TestProcessArguments @(
            "/VERYSILENT",
            "/SUPPRESSMSGBOXES",
            "/NORESTART",
            "/DIR=$InstallDir"
        )
        Write-Host "    Installing pinned base installer..." -ForegroundColor DarkGray
        $exitCode = Start-TestProcessAndWait -FilePath $InstallerPath -Arguments $args -TimeoutSeconds 600
        if ($exitCode -ne 0) {
            throw "Ollama install failed with exit code $exitCode"
        }
        Write-Host "    Pinned base installer completed" -ForegroundColor DarkGray
        Stop-TestOllamaProcesses
    }

    function Invoke-TestOllamaUninstall {
        Stop-TestOllamaProcesses

        $regKey = Get-TestOllamaRegistryKey
        if (-not $regKey) {
            return
        }

        $props = Get-ItemProperty -Path $regKey
        $commandLine = if ($props.QuietUninstallString) { $props.QuietUninstallString } else { $props.UninstallString }
        if (-not $commandLine) {
            throw "No uninstall command found in $regKey"
        }

        $command = Split-TestCommandLine -CommandLine $commandLine
        $args = "$($command.Arguments) /VERYSILENT /SUPPRESSMSGBOXES /NORESTART".Trim()

        Write-Host "    Running uninstall cleanup..." -ForegroundColor DarkGray
        $exitCode = Start-TestProcessAndWait -FilePath $command.FilePath -Arguments $args -TimeoutSeconds 300
        if ($exitCode -ne 0) {
            throw "Ollama uninstall failed with exit code $exitCode"
        }
        Stop-TestOllamaProcesses
    }

    function Require-RealInstallIntegration {
        if (Test-OllamaRealInstallPresent) {
            throw @"
INTEGRATION TEST SAFETY ABORT:
An existing Ollama installation was detected. Real integration tests install and
uninstall Ollama and would interfere with your install.

Uninstall Ollama first, then rerun the integration tests.
"@
        }
        $signedInstaller = Get-IntegrationInstaller
        if (-not $signedInstaller) {
            return $null
        }
        return $signedInstaller
    }
}

Describe "Windows install.ps1" -Tag Unit {
    BeforeEach {
        Save-TestEnvironment
    }

    AfterEach {
        Restore-TestEnvironment
    }

    Context "syntax" {
    It "parses without errors" {
        $errors = $null
        $null = [System.Management.Automation.Language.Parser]::ParseFile(
            $script:InstallScript, [ref]$null, [ref]$errors
        )
        $errors.Count | Should -Be 0
    }

    It "documents the two-phase cache surface without exposing a cache path knob" {
        $content = Get-Content $script:InstallScript -Raw
        $content | Should -Match 'OLLAMA_CACHE_ONLY'
        $content | Should -Match 'OLLAMA_INSTALL_CACHED'
        $content | Should -Not -Match 'OLLAMA_DOWNLOAD_ONLY'
    }

    It "rejects invalid cache mode combinations" {
        Set-TestEnvironment @{
            OLLAMA_CACHE_ONLY = "1"
            OLLAMA_INSTALL_CACHED = "1"
        }

        { . $script:InstallScript } | Should -Throw "*OLLAMA_CACHE_ONLY and OLLAMA_INSTALL_CACHED cannot both be set*"
    }

    It "rejects uninstall combined with cache modes" {
        Set-TestEnvironment @{
            OLLAMA_UNINSTALL = "1"
            OLLAMA_CACHE_ONLY = "1"
        }

        { . $script:InstallScript } | Should -Throw "*OLLAMA_UNINSTALL cannot be combined*"
    }
    }

    Context "signature verification" {
    It "rejects unsigned files" {
        Set-TestEnvironment
        . $script:InstallScript

        $unsignedFile = Join-Path $env:TEMP "unsigned.exe"
        Set-Content -Path $unsignedFile -Value "unsigned payload" -Encoding ASCII

        Test-Signature -FilePath $unsignedFile | Should -Be $false
    }

    It "accepts valid Ollama signatures" {
        . $script:InstallScript

        Mock Get-AuthenticodeSignature {
            [PSCustomObject]@{
                Status = "Valid"
                SignerCertificate = [PSCustomObject]@{
                    Subject = "CN=Ollama Inc., O=Ollama Inc., L=San Francisco, S=California, C=US"
                }
            }
        }

        Test-Signature -FilePath "OllamaSetup.exe" | Should -Be $true
    }

    It "rejects valid signatures from other organizations" {
        . $script:InstallScript

        Mock Get-AuthenticodeSignature {
            [PSCustomObject]@{
                Status = "Valid"
                SignerCertificate = [PSCustomObject]@{
                    Subject = "CN=Example Corp, O=Example Corp, C=US"
                }
            }
        }

        Test-Signature -FilePath "OllamaSetup.exe" | Should -Be $false
    }

    It "rejects organization-name lookalikes" {
        . $script:InstallScript

        Mock Get-AuthenticodeSignature {
            [PSCustomObject]@{
                Status = "Valid"
                SignerCertificate = [PSCustomObject]@{
                    Subject = "CN=Example, O=Not Ollama Inc., C=US"
                }
            }
        }

        Test-Signature -FilePath "OllamaSetup.exe" | Should -Be $false
    }
    }

    Context "argument quoting" {
    It "quotes installer arguments without allowing install dir switch injection" {
        . $script:InstallScript

        $args = @(
            "/VERYSILENT",
            "/DIR=C:\Apps\Ollama `"Preview`"\"
        )

        (($args | ForEach-Object { Quote-ProcessArgument $_ }) -join " ") | Should -Be '/VERYSILENT "/DIR=C:\Apps\Ollama \"Preview\"\\"'
    }
    }

    Context "cache modes" {
    It "cache-only without a version caches latest using the installer ETag" {
        Set-TestEnvironment @{ OLLAMA_CACHE_ONLY = "1" }
        . $script:InstallScript

        Mock Get-RemoteETag { '"latest-etag"' }
        Mock Invoke-Download {
            Set-Content -Path $OutFile -Value "payload" -Encoding ASCII
            return '"latest-etag"'
        }
        Mock Test-Signature { $true }
        Mock Start-Process { throw "installer should not start in cache-only mode" }

        Invoke-Install

        $cacheDir = Get-TestInstallerCacheDir '"latest-etag"'
        $stagingInstaller = Get-TestInstallerStagingPath '"latest-etag"'
        Test-Path (Join-Path $cacheDir "OllamaSetup.exe") | Should -Be $true
        Test-Path (Split-Path $stagingInstaller -Parent) | Should -Be $false
        Should -Invoke Get-RemoteETag -Times 1 -Exactly -ParameterFilter {
            $Url -eq "https://ollama.com/download/OllamaSetup.exe"
        }
        Should -Invoke Invoke-Download -Times 1 -Exactly -ParameterFilter {
            $Url -eq "https://ollama.com/download/OllamaSetup.exe" -and
            $OutFile -eq $stagingInstaller
        }
        Should -Invoke Test-Signature -Times 1 -Exactly -ParameterFilter {
            $FilePath -eq $stagingInstaller
        }
        Should -Invoke Start-Process -Times 0 -Exactly
    }

    It "downloads the requested version into the prescribed cache" {
        Set-TestEnvironment @{
            OLLAMA_CACHE_ONLY = "1"
            OLLAMA_VERSION = "v0.21.0"
        }
        . $script:InstallScript

        Mock Get-RemoteETag { '"version-etag"' }
        Mock Invoke-Download {
            Set-Content -Path $OutFile -Value "payload" -Encoding ASCII
            return '"version-etag"'
        }
        Mock Test-Signature { $true }
        Mock Start-Process { throw "installer should not start in cache-only mode" }

        Invoke-Install

        $cacheDir = Get-TestInstallerCacheDir '"version-etag"'
        $stagingInstaller = Get-TestInstallerStagingPath '"version-etag"'
        Test-Path (Join-Path $cacheDir "OllamaSetup.exe") | Should -Be $true
        Test-Path (Split-Path $stagingInstaller -Parent) | Should -Be $false
        Should -Invoke Invoke-Download -Times 1 -Exactly -ParameterFilter {
            $Url -eq "https://ollama.com/download/OllamaSetup.exe?version=v0.21.0" -and
            $OutFile -eq $stagingInstaller
        }
        Should -Invoke Test-Signature -Times 1 -Exactly -ParameterFilter {
            $FilePath -eq $stagingInstaller
        }
        Should -Invoke Start-Process -Times 0 -Exactly
    }

    It "cache-only replaces a stale cached ETag directory instead of accumulating cache entries" {
        Set-TestEnvironment @{
            OLLAMA_CACHE_ONLY = "1"
            OLLAMA_VERSION = "0.21.0"
        }
        $cacheRoot = Join-Path $env:LOCALAPPDATA "Ollama\install_cache"
        $oldCacheDir = Get-TestInstallerCacheDir '"old-etag"'
        $newCacheDir = Get-TestInstallerCacheDir '"new-etag"'
        $installer = Join-Path $newCacheDir "OllamaSetup.exe"
        $stagingCacheDir = Get-TestInstallerStagingCacheDir '"new-etag"'
        New-Item -ItemType Directory -Path $oldCacheDir -Force | Out-Null
        Set-Content -Path (Join-Path $oldCacheDir "OllamaSetup.exe") -Value "old payload" -Encoding ASCII
        . $script:InstallScript

        Mock Get-RemoteETag { '"new-etag"' }
        Mock Invoke-Download {
            Test-Path $OutFile | Should -Be $false
            Test-Path $oldCacheDir | Should -Be $false
            Set-Content -Path $OutFile -Value "new payload" -Encoding ASCII
            return '"new-etag"'
        }
        Mock Test-Signature { $true }
        Mock Start-Process { throw "installer should not start in cache-only mode" }

        Invoke-Install

        (Get-Content $installer -Raw).Trim() | Should -Be "new payload"
        Test-Path $stagingCacheDir | Should -Be $false
        Get-ChildItem -Path $cacheRoot -Directory | Should -HaveCount 1
        Should -Invoke Invoke-Download -Times 1 -Exactly
        Should -Invoke Start-Process -Times 0 -Exactly
    }

    It "cache-only removes stale cache before a failed newer download" {
        Set-TestEnvironment @{
            OLLAMA_CACHE_ONLY = "1"
            OLLAMA_VERSION = "0.21.0"
        }
        $oldCacheDir = Get-TestInstallerCacheDir '"old-etag"'
        $newCacheDir = Get-TestInstallerCacheDir '"new-etag"'
        $stagingCacheDir = Get-TestInstallerStagingCacheDir '"new-etag"'
        New-Item -ItemType Directory -Path $oldCacheDir -Force | Out-Null
        Set-Content -Path (Join-Path $oldCacheDir "OllamaSetup.exe") -Value "old payload" -Encoding ASCII
        . $script:InstallScript

        Mock Get-RemoteETag { '"new-etag"' }
        Mock Invoke-Download {
            Test-Path $oldCacheDir | Should -Be $false
            Set-Content -Path $OutFile -Value "partial payload" -Encoding ASCII
            throw "download failed"
        }
        Mock Test-Signature { throw "signature verification should not run after download failure" }
        Mock Start-Process { throw "installer should not start after download failure" }

        { Invoke-Install } | Should -Throw "*download failed*"
        Test-Path $oldCacheDir | Should -Be $false
        Test-Path $newCacheDir | Should -Be $false
        Test-Path $stagingCacheDir | Should -Be $false
        Should -Invoke Invoke-Download -Times 1 -Exactly
        Should -Invoke Test-Signature -Times 0 -Exactly
        Should -Invoke Start-Process -Times 0 -Exactly
    }

    It "cache-only refreshes the persistent cache when ETags are unavailable" {
        Set-TestEnvironment @{ OLLAMA_CACHE_ONLY = "1" }
        $cacheRoot = Join-Path $env:LOCALAPPDATA "Ollama\install_cache"
        $oldCacheDir = Get-TestInstallerCacheDir '"old-etag"'
        New-Item -ItemType Directory -Path $oldCacheDir -Force | Out-Null
        Set-Content -Path (Join-Path $oldCacheDir "OllamaSetup.exe") -Value "old payload" -Encoding ASCII
        . $script:InstallScript

        Mock Get-RemoteETag { "" }
        Mock Invoke-Download {
            Test-Path $oldCacheDir | Should -Be $false
            Set-Content -Path $OutFile -Value "payload" -Encoding ASCII
            return ""
        }
        Mock Test-Signature { $true }
        Mock Start-Process { throw "installer should not start in cache-only mode" }

        Invoke-Install

        $cacheEntries = @(Get-ChildItem -Path $cacheRoot -Directory | Where-Object { -not $_.Name.EndsWith(".download") })
        $cacheEntries | Should -HaveCount 1
        Test-Path (Join-Path $cacheEntries[0].FullName "OllamaSetup.exe") | Should -Be $true
        Test-Path $oldCacheDir | Should -Be $false
        Should -Invoke Invoke-Download -Times 1 -Exactly
        Should -Invoke Test-Signature -Times 1 -Exactly
        Should -Invoke Start-Process -Times 0 -Exactly
    }

    It "cache-only tolerates a missing GET ETag after HEAD identified the cache" {
        Set-TestEnvironment @{ OLLAMA_CACHE_ONLY = "1" }
        $cacheDir = Get-TestInstallerCacheDir '"head-etag"'
        $stagingInstaller = Get-TestInstallerStagingPath '"head-etag"'
        . $script:InstallScript

        Mock Get-RemoteETag { '"head-etag"' }
        Mock Invoke-Download {
            Set-Content -Path $OutFile -Value "payload" -Encoding ASCII
            return ""
        }
        Mock Test-Signature { $true }
        Mock Start-Process { throw "installer should not start in cache-only mode" }

        Invoke-Install

        Test-Path (Join-Path $cacheDir "OllamaSetup.exe") | Should -Be $true
        Test-Path (Split-Path $stagingInstaller -Parent) | Should -Be $false
        Should -Invoke Invoke-Download -Times 1 -Exactly -ParameterFilter {
            $OutFile -eq $stagingInstaller
        }
        Should -Invoke Test-Signature -Times 1 -Exactly -ParameterFilter {
            $FilePath -eq $stagingInstaller
        }
    }

    It "cache-only rejects downloads that return a different ETag" {
        Set-TestEnvironment @{ OLLAMA_CACHE_ONLY = "1" }
        $cacheDir = Get-TestInstallerCacheDir '"head-etag"'
        $stagingInstaller = Get-TestInstallerStagingPath '"head-etag"'
        . $script:InstallScript

        Mock Get-RemoteETag { '"head-etag"' }
        Mock Invoke-Download {
            Set-Content -Path $OutFile -Value "payload" -Encoding ASCII
            return '"other-etag"'
        }
        Mock Test-Signature { throw "signature verification should not run after ETag mismatch" }

        { Invoke-Install } | Should -Throw "*Downloaded installer ETag mismatch*"
        Test-Path $cacheDir | Should -Be $false
        Test-Path (Split-Path $stagingInstaller -Parent) | Should -Be $false
        Should -Invoke Invoke-Download -Times 1 -Exactly -ParameterFilter {
            $OutFile -eq $stagingInstaller
        }
        Should -Invoke Test-Signature -Times 0 -Exactly
    }

    It "install-cached ignores partial staging downloads when the cache is missing" {
        Set-TestEnvironment @{ OLLAMA_INSTALL_CACHED = "1" }
        $stagingCacheDir = Get-TestInstallerStagingCacheDir '"missing-etag"'
        New-Item -ItemType Directory -Path $stagingCacheDir -Force | Out-Null
        Set-Content -Path (Join-Path $stagingCacheDir "OllamaSetup.exe") -Value "partial payload" -Encoding ASCII
        . $script:InstallScript

        Mock Get-RemoteETag { throw "install-cached should not resolve remote ETags" }
        Mock Invoke-Download { throw "download should not run in install-cached mode" }
        Mock Test-Signature { throw "signature verification should not run without a complete cached installer" }
        Mock Start-Process { throw "installer should not start without a complete cached installer" }

        { Invoke-Install } | Should -Throw "*Cached installer not found*"
        Should -Invoke Get-RemoteETag -Times 0 -Exactly
        Should -Invoke Invoke-Download -Times 0 -Exactly
        Should -Invoke Test-Signature -Times 0 -Exactly
        Should -Invoke Start-Process -Times 0 -Exactly
    }

    It "install-cached installs the existing cached installer without network resolution" {
        Set-TestEnvironment @{ OLLAMA_INSTALL_CACHED = "1" }
        $cacheDir = Get-TestInstallerCacheDir '"cached-etag"'
        $installer = Join-Path $cacheDir "OllamaSetup.exe"
        New-Item -ItemType Directory -Path $cacheDir -Force | Out-Null
        Set-Content -Path $installer -Value "payload" -Encoding ASCII
        . $script:InstallScript

        Mock Get-RemoteETag { throw "install-cached should not resolve remote ETags" }
        Mock Invoke-Download { throw "download should not run in install-cached mode" }
        Mock Test-Signature { $true }
        Mock Start-Process { return New-FakeProcess }
        Mock Update-SessionPath { }

        Invoke-Install

        Should -Invoke Invoke-Download -Times 0 -Exactly
        Should -Invoke Get-RemoteETag -Times 0 -Exactly
        Should -Invoke Start-Process -Times 1 -Exactly -ParameterFilter {
            $FilePath -eq $installer
        }
        Test-Path $cacheDir | Should -Be $false
    }

    It "install-cached ignores OLLAMA_VERSION and uses the final cached installer" {
        Set-TestEnvironment @{
            OLLAMA_INSTALL_CACHED = "1"
            OLLAMA_VERSION = "0.21.0"
        }
        $cacheDir = Get-TestInstallerCacheDir '"cached-etag"'
        $installer = Join-Path $cacheDir "OllamaSetup.exe"
        New-Item -ItemType Directory -Path $cacheDir -Force | Out-Null
        Set-Content -Path $installer -Value "payload" -Encoding ASCII
        . $script:InstallScript

        Mock Get-RemoteETag { throw "install-cached should not resolve remote ETags" }
        Mock Test-Signature { $true }
        Mock Start-Process { return New-FakeProcess }
        Mock Update-SessionPath { }

        Invoke-Install

        Should -Invoke Get-RemoteETag -Times 0 -Exactly
        Should -Invoke Start-Process -Times 1 -Exactly -ParameterFilter {
            $FilePath -eq $installer
        }
    }

    It "install-cached verifies the cached installer immediately before launch" {
        Set-TestEnvironment @{
            OLLAMA_INSTALL_CACHED = "1"
            OLLAMA_VERSION = "0.21.0"
        }
        $cacheDir = Get-TestInstallerCacheDir '"cached-etag"'
        $installer = Join-Path $cacheDir "OllamaSetup.exe"
        New-Item -ItemType Directory -Path $cacheDir -Force | Out-Null
        Set-Content -Path $installer -Value "payload" -Encoding ASCII
        . $script:InstallScript

        Mock Get-RemoteETag { throw "install-cached should not resolve remote ETags" }
        $script:SignatureChecks = 0
        $script:StartedInstaller = $null
        Mock Test-Signature {
            $script:SignatureChecks++
            return $true
        }
        Mock Start-Process {
            $script:StartedInstaller = @{
                FilePath = $FilePath
                ArgumentList = $ArgumentList
            }
            return New-FakeProcess
        }
        Mock Update-SessionPath { }

        Invoke-Install

        Should -Invoke Get-RemoteETag -Times 0 -Exactly
        $script:SignatureChecks | Should -Be 1
        $script:StartedInstaller.FilePath | Should -Be $installer
        $script:StartedInstaller.ArgumentList | Should -Match "/VERYSILENT"
        $script:StartedInstaller.ArgumentList | Should -Match "/LOG="
        $script:StartedInstaller.ArgumentList | Should -Match "OllamaSetup\.log"
        Test-Path $cacheDir | Should -Be $false
    }

    It "install-cached deletes the persistent cache when signature verification fails" {
        Set-TestEnvironment @{
            OLLAMA_INSTALL_CACHED = "1"
            OLLAMA_VERSION = "0.21.0"
        }
        $cacheDir = Get-TestInstallerCacheDir '"cached-etag"'
        $installer = Join-Path $cacheDir "OllamaSetup.exe"
        New-Item -ItemType Directory -Path $cacheDir -Force | Out-Null
        Set-Content -Path $installer -Value "payload" -Encoding ASCII
        . $script:InstallScript

        Mock Get-RemoteETag { throw "install-cached should not resolve remote ETags" }
        Mock Test-Signature { $false }
        Mock Start-Process { throw "installer should not start after signature failure" }

        { Invoke-Install } | Should -Throw "*Installer signature verification failed*"
        Should -Invoke Get-RemoteETag -Times 0 -Exactly
        Test-Path $cacheDir | Should -Be $false
        Should -Invoke Start-Process -Times 0 -Exactly
    }
    }

    Context "normal install path" {
    It "downloads default installs through the installer cache" {
        Set-TestEnvironment
        $cacheDir = Get-TestInstallerCacheDir '"default-etag"'
        $installer = Join-Path $cacheDir "OllamaSetup.exe"
        $stagingInstaller = Get-TestInstallerStagingPath '"default-etag"'
        . $script:InstallScript

        $script:DownloadedTo = $null
        $script:StartedInstaller = $null
        Mock Get-RemoteETag { '"default-etag"' }
        Mock Invoke-Download {
            $script:DownloadedTo = $OutFile
            Set-Content -Path $OutFile -Value "payload" -Encoding ASCII
        }
        Mock Test-Signature { $true }
        Mock Start-Process {
            $script:StartedInstaller = $FilePath
            return New-FakeProcess
        }
        Mock Update-SessionPath { }

        Invoke-Install

        $script:DownloadedTo | Should -Be $stagingInstaller
        $script:StartedInstaller | Should -Be $installer
        Test-Path $cacheDir | Should -Be $false
    }

    It "reuses a warm cache for normal installs and cleans it up after success" {
        Set-TestEnvironment
        $cacheDir = Get-TestInstallerCacheDir '"warm-etag"'
        $installer = Join-Path $cacheDir "OllamaSetup.exe"
        New-Item -ItemType Directory -Path $cacheDir -Force | Out-Null
        Set-Content -Path $installer -Value "payload" -Encoding ASCII
        . $script:InstallScript

        Mock Get-RemoteETag { '"warm-etag"' }
        Mock Invoke-Download { throw "download should not run with warm cache" }
        Mock Test-Signature { $true }
        Mock Start-Process { return New-FakeProcess }
        Mock Update-SessionPath { }

        Invoke-Install

        Should -Invoke Invoke-Download -Times 0 -Exactly
        Should -Invoke Start-Process -Times 1 -Exactly -ParameterFilter {
            $FilePath -eq $installer
        }
        Test-Path $cacheDir | Should -Be $false
    }

    It "falls back to a throwaway temp cache for normal installs when ETags are unavailable" {
        Set-TestEnvironment
        . $script:InstallScript

        $script:DownloadedTo = $null
        $script:StartedInstaller = $null
        Mock Get-RemoteETag { "" }
        Mock Invoke-Download {
            $script:DownloadedTo = $OutFile
            Set-Content -Path $OutFile -Value "payload" -Encoding ASCII
        }
        Mock Test-Signature { $true }
        Mock Start-Process {
            $script:StartedInstaller = $FilePath
            return New-FakeProcess
        }
        Mock Update-SessionPath { }

        Invoke-Install

        $script:StartedInstaller | Should -Not -BeNullOrEmpty
        $script:DownloadedTo | Should -Be (Join-Path ("$(Split-Path $script:StartedInstaller -Parent).download") "OllamaSetup.exe")
        $script:StartedInstaller | Should -Match ([regex]::Escape((Join-Path $env:TEMP "Ollama\install_cache")))
        Test-Path (Split-Path $script:StartedInstaller -Parent) | Should -Be $false
        Test-Path (Split-Path $script:DownloadedTo -Parent) | Should -Be $false
    }

    It "deletes a newly downloaded cache entry when signature verification fails" {
        Set-TestEnvironment
        $cacheDir = Get-TestInstallerCacheDir '"bad-signature-etag"'
        $stagingCacheDir = Get-TestInstallerStagingCacheDir '"bad-signature-etag"'
        . $script:InstallScript

        Mock Get-RemoteETag { '"bad-signature-etag"' }
        Mock Invoke-Download {
            Set-Content -Path $OutFile -Value "payload" -Encoding ASCII
        }
        Mock Test-Signature { $false }
        Mock Start-Process { throw "installer should not start after signature failure" }

        { Invoke-Install } | Should -Throw "*Installer signature verification failed*"
        Test-Path $cacheDir | Should -Be $false
        Test-Path $stagingCacheDir | Should -Be $false
        Should -Invoke Start-Process -Times 0 -Exactly
    }
    }
}

Describe "Windows install.ps1 integration" -Tag Integration {
    BeforeEach {
        Save-TestEnvironment
        $script:RealInstallTestStarted = $false
    }

    AfterEach {
        if ($script:RealInstallTestStarted) {
            Invoke-TestOllamaUninstall
            Remove-Item -LiteralPath (Join-Path $env:LOCALAPPDATA "Ollama\install_cache") -Recurse -Force -ErrorAction SilentlyContinue
            Remove-Item -LiteralPath (Join-Path $env:LOCALAPPDATA "Ollama\upgraded") -Force -ErrorAction SilentlyContinue
        }
        Restore-TestEnvironment
    }

    It "installs from a local signed installer" {
        $signedInstaller = Require-RealInstallIntegration
        if (-not $signedInstaller) { return }

        $server = $null
        try {
            $installDir = Join-Path $TestDrive ("OllamaInstall-" + [guid]::NewGuid().ToString("N"))
            Set-TestEnvironment @{ OLLAMA_INSTALL_DIR = $installDir }
            $script:RealInstallTestStarted = $true
            $server = Start-LocalHttpServer -InstallerPath $signedInstaller
            if (-not $server) { return }
            $installScript = New-TestInstallScript -DownloadBaseUrl $server.DownloadBaseUrl

            $result = Invoke-InstallScript -InstallScriptPath $installScript
            $result.ExitCode | Should -Be 0

            Test-Path -LiteralPath (Join-Path $installDir "Ollama app.exe") | Should -Be $true
            Test-Path -LiteralPath (Join-Path $installDir "ollama.exe") | Should -Be $true
            Get-TestOllamaRegistryKey | Should -Not -BeNullOrEmpty
        } finally {
            Stop-LocalHttpServer $server
        }
    }

    It "upgrades from pinned 0.21.0 through the two-phase app cache flow" {
        $signedInstaller = Require-RealInstallIntegration
        if (-not $signedInstaller) { return }
        $oldInstaller = Get-PinnedUpgradeInstaller

        $server = $null
        try {
            $installDir = Join-Path $TestDrive ("OllamaInstall-" + [guid]::NewGuid().ToString("N"))
            Set-TestEnvironment @{ OLLAMA_INSTALL_DIR = $installDir }
            $script:RealInstallTestStarted = $true

            Invoke-TestInnoInstall -InstallerPath $oldInstaller -InstallDir $installDir
            Test-Path -LiteralPath (Join-Path $installDir "Ollama app.exe") | Should -Be $true
            $oldVersion = Get-TestOllamaInstalledVersion -InstallDir $installDir
            $oldVersion | Should -Not -BeNullOrEmpty

            $etag = "real-install-cache-etag"
            $cacheDir = Get-TestInstallerCacheDir $etag
            $server = Start-LocalHttpServer -InstallerPath $signedInstaller -ETag $etag
            if (-not $server) { return }
            $installScript = New-TestInstallScript -DownloadBaseUrl $server.DownloadBaseUrl

            $result = Invoke-InstallScript -InstallScriptPath $installScript -EnvVars @{ OLLAMA_CACHE_ONLY = "1" }
            $result.ExitCode | Should -Be 0

            $installer = Join-Path $cacheDir "OllamaSetup.exe"
            Test-Path -LiteralPath $installer | Should -Be $true

            $result = Invoke-InstallScript -InstallScriptPath $installScript -EnvVars @{ OLLAMA_INSTALL_CACHED = "1" }
            $result.ExitCode | Should -Be 0

            Test-Path -LiteralPath (Join-Path $installDir "Ollama app.exe") | Should -Be $true
            Get-TestOllamaRegistryKey | Should -Not -BeNullOrEmpty
            Test-Path -LiteralPath $cacheDir | Should -Be $false
            $newVersion = Get-TestOllamaInstalledVersion -InstallDir $installDir
            $newVersion | Should -Not -BeNullOrEmpty
            $newVersion | Should -Not -Be $oldVersion
        } finally {
            Stop-LocalHttpServer $server
        }
    }
}
