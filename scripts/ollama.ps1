$ErrorActionPreference = 'Stop'

$configDefaults = @{
    OLLAMA_FLASH_ATTENTION = $null
    OLLAMA_KV_CACHE_TYPE = $null
    OLLAMA_KEEP_ALIVE = $null
    OLLAMA_LOAD_TIMEOUT = $null
    OLLAMA_NUM_PARALLEL = $null
    OLLAMA_MAX_LOADED_MODELS = $null
    OLLAMA_GPU_OVERHEAD = $null
    OLLAMA_DEBUG = $null
    OLLAMA_SCHED_SPREAD = $null
    CUDA_VISIBLE_DEVICES = $null
    HIP_VISIBLE_DEVICES = $null
    ROCR_VISIBLE_DEVICES = $null
    GGML_VK_VISIBLE_DEVICES = $null
}
foreach ($name in $configDefaults.Keys) {
    if (-not [string]::IsNullOrEmpty($configDefaults[$name]) -and [string]::IsNullOrEmpty([Environment]::GetEnvironmentVariable($name))) {
        [Environment]::SetEnvironmentVariable($name, $configDefaults[$name])
    }
}
if ([string]::IsNullOrEmpty($env:OLLAMA_HOST)) { $env:OLLAMA_HOST = '127.0.0.1:11434' }

function Write-WrapperError([string]$Message) {
    [Console]::Error.WriteLine("ERROR: $Message")
}

function Resolve-OllamaBinary {
    $candidate = $env:OLLAMA_WRAPPER_BIN
    if ([string]::IsNullOrEmpty($candidate)) {
        $candidate = Join-Path $PSScriptRoot 'ollama.exe'
    }
    if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) {
        $command = Get-Command ollama.exe -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($command) { $candidate = $command.Source }
    }
    if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) {
        throw 'Ollama executable not found; set OLLAMA_WRAPPER_BIN or install ollama.exe.'
    }
    $resolved = (Resolve-Path -LiteralPath $candidate).Path
    $scriptPath = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot 'ollama.bat'))
    if ([string]::Equals($resolved, $scriptPath, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Resolved Ollama binary points to the wrapper itself: $resolved"
    }
    return $resolved
}

function Get-HostInfo {
    $value = $env:OLLAMA_HOST.Trim()
    $scheme = 'http'
    if ($value -match '^([^:]+)://(.*)$') { $scheme = $Matches[1]; $value = $Matches[2] }
    if ($scheme -notin @('http', 'https') -or [string]::IsNullOrWhiteSpace($value)) {
        throw "Invalid OLLAMA_HOST: $($env:OLLAMA_HOST)"
    }
    $defaultPort = if ($scheme -eq 'https') { '443' } elseif ($env:OLLAMA_HOST -match '^http://') { '80' } else { '11434' }
    $authority = ($value -split '[?#]', 2)[0]
    if ($authority.Contains('/')) {
        $path = '/' + ($authority -split '/', 2)[1]
        $authority = ($authority -split '/', 2)[0]
    }
    if ([string]::IsNullOrWhiteSpace($authority) -or $authority -match '[@?#]') { throw "Invalid OLLAMA_HOST: $($env:OLLAMA_HOST)" }
    if ($authority -match '^\[([^]]+)\](?::(\d+))?$') { $hostName = "[$($Matches[1])]"; $port = $Matches[2] }
    elseif ($authority -match '^([^:]+)(?::(\d+))?$') { $hostName = $Matches[1]; $port = $Matches[2] }
    else { throw "Invalid OLLAMA_HOST: $($env:OLLAMA_HOST)" }
    if ([string]::IsNullOrEmpty($port)) { $port = $defaultPort }
    if ($hostName -eq '0.0.0.0') { $hostName = '127.0.0.1' }
    elseif ($hostName -eq '[::]') { $hostName = '[::1]' }
    $connectHost = "$hostName`:$port"
    $local = $hostName -in @('127.0.0.1', 'localhost', '[::1]')
    $key = $connectHost -replace '[:\[\]]', '_'
    return [pscustomobject]@{ Scheme = $scheme; ConnectHost = $connectHost; Path = $path.TrimEnd('/'); IsLocal = $local; Key = $key }
}

function Test-ServerHealth([string]$Scheme, [string]$ConnectHost, [string]$Path = '') {
    $url = "$Scheme`://$ConnectHost$Path/api/version"
    try {
        $request = [System.Net.HttpWebRequest]::Create($url)
        $request.Timeout = 1000
        $request.ReadWriteTimeout = 1000
        $request.Proxy = $null
        $response = $request.GetResponse()
        $response.Close()
        return $true
    } catch {
        return $false
    }
}

function Get-ProcessIdentity([int]$Id) {
    try {
        return Get-CimInstance Win32_Process -Filter "ProcessId = $Id" -ErrorAction Stop
    } catch {
        return $null
    }
}

function Test-ManagedProcess($State, [string]$Binary) {
    if (-not (Test-Path -LiteralPath $State.JsonPath -PathType Leaf)) { return $null }
    try { $record = Get-Content -LiteralPath $State.JsonPath -Raw | ConvertFrom-Json -ErrorAction Stop } catch { return $null }
    if ($record.pid -isnot [int] -and $record.pid -isnot [long]) { return $null }
    $process = Get-ProcessIdentity ([int]$record.pid)
    if (-not $process -or -not $process.ExecutablePath -or -not $process.CommandLine) { return $null }
    if (-not [string]::Equals([IO.Path]::GetFullPath($process.ExecutablePath), $Binary, [StringComparison]::OrdinalIgnoreCase)) { return $null }
    if ($process.CommandLine -notmatch '(?i)(^|\s)serve(\s|$)') { return $null }
    if ($record.binary -and -not [string]::Equals([IO.Path]::GetFullPath([string]$record.binary), $Binary, [StringComparison]::OrdinalIgnoreCase)) { return $null }
    if ($record.started_at) {
        try { $actualStart = (Get-Process -Id ([int]$record.pid) -ErrorAction Stop).StartTime.ToUniversalTime() } catch { return $null }
        $recordedStart = ([DateTime]::Parse([string]$record.started_at)).ToUniversalTime()
        if ($actualStart.Ticks -ne $recordedStart.Ticks) { return $null }
    }
    return [pscustomobject]@{ Process = $process; StartedAt = [DateTime]::Parse([string]$record.started_at).ToUniversalTime() }
}

function Get-StatePaths($HostInfo) {
    $root = $env:OLLAMA_WRAPPER_STATE_DIR
    if ([string]::IsNullOrEmpty($root)) { $root = Join-Path $env:LOCALAPPDATA 'OllamaWrapper' }
    $directory = Join-Path $root $HostInfo.Key
    return [pscustomobject]@{
        Directory = $directory
        PidPath = Join-Path $directory 'server.pid'
        JsonPath = Join-Path $directory 'server.json'
        LockPath = Join-Path $directory 'lock'
        StdoutPath = Join-Path $directory 'server.stdout.log'
        StderrPath = Join-Path $directory 'server.stderr.log'
    }
}

function Enter-WrapperLock($State) {
    New-Item -ItemType Directory -Path $State.Directory -Force | Out-Null
    try {
        New-Item -ItemType Directory -Path $State.LockPath -ErrorAction Stop | Out-Null
        [IO.File]::WriteAllText((Join-Path $State.LockPath 'pid'), "$PID`n")
        return $true
    } catch {
        Write-WrapperError "Lifecycle operation is locked at $($State.LockPath)."
        return $false
    }
}

function Exit-WrapperLock($State) {
    Remove-Item -LiteralPath $State.LockPath -Recurse -Force -ErrorAction SilentlyContinue
}

function Start-ManagedServer($State, [string]$Binary, [string]$Scheme, [string]$ConnectHost, [string]$Path) {
    if (-not (Enter-WrapperLock $State)) { return 5 }
    try {
        $managed = Test-ManagedProcess $State $Binary
        if ($managed) { Write-Host "Server is already running (PID $($managed.Process.ProcessId))."; return 0 }
        if (Test-ServerHealth $Scheme $ConnectHost $Path) { Write-WrapperError 'Server is already running, but is not managed by this wrapper.'; return 4 }
        Remove-Item -LiteralPath $State.PidPath, $State.JsonPath -Force -ErrorAction SilentlyContinue
        $process = Start-Process -FilePath $Binary -ArgumentList @('serve') -PassThru -RedirectStandardOutput $State.StdoutPath -RedirectStandardError $State.StderrPath
        [IO.File]::WriteAllText($State.PidPath + '.tmp', "$($process.Id)`n")
        Move-Item -LiteralPath ($State.PidPath + '.tmp') -Destination $State.PidPath -Force
        $record = [ordered]@{ pid = $process.Id; host = $ConnectHost; binary = $Binary; started_at = $process.StartTime.ToUniversalTime().ToString('o') }
        $json = $record | ConvertTo-Json -Compress
        [IO.File]::WriteAllText($State.JsonPath + '.tmp', $json + "`n")
        Move-Item -LiteralPath ($State.JsonPath + '.tmp') -Destination $State.JsonPath -Force
        for ($i = 0; $i -lt 50; $i++) {
            if (Test-ServerHealth $Scheme $ConnectHost $Path) { Write-Host "Server started (PID $($process.Id))."; return 0 }
            if ($process.HasExited) { Write-WrapperError 'Ollama server exited before becoming healthy.'; return 3 }
            Start-Sleep -Milliseconds 100
        }
        Write-WrapperError 'Ollama server did not become healthy within 10 seconds.'
        return 3
    } finally { Exit-WrapperLock $State }
}

function Stop-ManagedServer($State, [string]$Binary, [string]$Scheme, [string]$ConnectHost, [string]$Path) {
    if (-not (Enter-WrapperLock $State)) { return 5 }
    try {
        $managed = Test-ManagedProcess $State $Binary
        $process = if ($managed) { $managed.Process } else { $null }
        if (-not $process) {
            if (Test-ServerHealth $Scheme $ConnectHost $Path) { Write-WrapperError 'Server is running, but it is not managed by this wrapper.'; return 4 }
            Remove-Item -LiteralPath $State.PidPath, $State.JsonPath -Force -ErrorAction SilentlyContinue
            Write-Host 'Server is not running.'
            return 0
        }
        $target = Get-Process -Id $process.ProcessId -ErrorAction Stop
        if ($target.StartTime.ToUniversalTime().Ticks -ne $managed.StartedAt.Ticks) {
            Write-WrapperError "Cannot safely stop process $($process.ProcessId): process identity changed."
            return 4
        }
        try { $target.Kill() } catch { Write-WrapperError "Cannot safely stop process $($process.ProcessId): process identity could not be verified."; return 4 }
        try { $target.WaitForExit(10000) | Out-Null; if (-not $target.HasExited) { throw 'timeout' } } catch { Write-WrapperError "Process $($process.ProcessId) did not exit within 10 seconds."; return 1 }
        Remove-Item -LiteralPath $State.PidPath, $State.JsonPath -Force -ErrorAction SilentlyContinue
        if (Test-ServerHealth $Scheme $ConnectHost $Path) { Write-WrapperError 'Server remains available after the managed process exited.'; return 1 }
        Write-Host 'Server stopped.'
        return 0
    } finally { Exit-WrapperLock $State }
}

function Invoke-UpdateAll([string]$Binary, [string]$Scheme, [string]$ConnectHost, [string]$Path) {
    if (-not (Test-ServerHealth $Scheme $ConnectHost $Path)) { Write-WrapperError "Ollama server is not running.`nRun 'ollama start' or 'ollama serve'."; return 3 }
    $lines = & $Binary list
    if ($LASTEXITCODE -ne 0) { Write-WrapperError 'Could not list local models.'; return 1 }
    $failed = $false
    $first = $true
    foreach ($line in $lines) {
        if ([string]::IsNullOrWhiteSpace($line)) { continue }
        if ($first) { $first = $false; continue }
        $model = ($line -split '\s+', 2)[0]
        if ([string]::IsNullOrWhiteSpace($model)) { Write-WrapperError "Cannot parse model row: $line"; return 1 }
        & $Binary pull $model | ForEach-Object { Write-Host $_ }
        if ($LASTEXITCODE -ne 0) { Write-WrapperError "Failed to update model: $model"; $failed = $true }
    }
    if ($failed) { return 1 }
    return 0
}

function Invoke-Native([string]$Binary, [string[]]$Arguments) {
    $quoted = foreach ($argument in $Arguments) {
        if ([string]::IsNullOrEmpty($argument)) { '""'; continue }
        if ($argument -notmatch '[\s"]') { $argument; continue }
        $escaped = [regex]::Replace($argument, '(\\*)"', '$1$1\"')
        $escaped = [regex]::Replace($escaped, '(\\+)$', '$1$1')
        '"' + $escaped + '"'
    }
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $Binary
    $psi.Arguments = [string]::Join(' ', [string[]]$quoted)
    $psi.UseShellExecute = $false
    $process = [System.Diagnostics.Process]::Start($psi)
    $process.WaitForExit()
    return $process.ExitCode
}

try {
    $binary = Resolve-OllamaBinary
    $hostInfo = Get-HostInfo
    $state = Get-StatePaths $hostInfo
} catch {
    Write-WrapperError $_.Exception.Message
    exit 2
}

$argsList = @($args)
$command = if ($argsList.Count) { [string]$argsList[0] } else { '' }

switch ($command) {
    'start' {
        if ($argsList.Count -ne 1) { Write-WrapperError 'Usage: ollama start'; exit 2 }
        if (-not $hostInfo.IsLocal) { Write-WrapperError 'start, stop, and restart only supports local loopback hosts.'; exit 4 }
        exit (Start-ManagedServer $state $binary $hostInfo.Scheme $hostInfo.ConnectHost $hostInfo.Path)
    }
    'stop' {
        if ($argsList.Count -eq 1) {
            if (-not $hostInfo.IsLocal) { Write-WrapperError 'start, stop, and restart only supports local loopback hosts.'; exit 4 }
            exit (Stop-ManagedServer $state $binary $hostInfo.Scheme $hostInfo.ConnectHost $hostInfo.Path)
        }
        if (-not (Test-ServerHealth $hostInfo.Scheme $hostInfo.ConnectHost $hostInfo.Path)) { Write-WrapperError "Ollama server is not running.`nRun 'ollama start' or 'ollama serve'."; exit 3 }
    }
    'restart' {
        if ($argsList.Count -ne 1) { Write-WrapperError 'Usage: ollama restart'; exit 2 }
        if (-not $hostInfo.IsLocal) { Write-WrapperError 'start, stop, and restart only supports local loopback hosts.'; exit 4 }
        $result = Stop-ManagedServer $state $binary $hostInfo.Scheme $hostInfo.ConnectHost $hostInfo.Path
        if ($result -ne 0) { exit $result }
        exit (Start-ManagedServer $state $binary $hostInfo.Scheme $hostInfo.ConnectHost $hostInfo.Path)
    }
    'status' {
        if ($argsList.Count -ne 1) { Write-WrapperError 'Usage: ollama status'; exit 2 }
        $managed = Test-ManagedProcess $state $binary
        if ($managed) { Write-Output "Server is running, managed by this wrapper (PID $($managed.Process.ProcessId))."; exit 0 }
        if (Test-ServerHealth $hostInfo.Scheme $hostInfo.ConnectHost $hostInfo.Path) { Write-Output 'Server is running, but it is not managed by this wrapper.'; Write-Output 'Wrapper configuration is not applied to this instance.'; exit 0 }
        Write-Output 'Server is not running.'
        exit 0
    }
    'update' {
        if ($argsList.Count -eq 2 -and $argsList[1] -ceq 'all') { exit (Invoke-UpdateAll $binary $hostInfo.Scheme $hostInfo.ConnectHost $hostInfo.Path) }
    }
}

if ($command -in @('run', 'pull', 'push', 'ps', 'list', 'show', 'rm') -and $argsList.Count -gt 1 -and $argsList[1] -notin @('--help', '-h') -and -not (Test-ServerHealth $hostInfo.Scheme $hostInfo.ConnectHost $hostInfo.Path)) {
    Write-WrapperError "Ollama server is not running.`nRun 'ollama start' or 'ollama serve'."
    exit 3
}
$nativeExitCode = Invoke-Native $binary ([string[]]$argsList)
exit $nativeExitCode
