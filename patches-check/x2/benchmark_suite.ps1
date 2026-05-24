# benchmark_suite.ps1 - Comprehensive benchmark for gfx12 WMMA kernel

param(
    [switch]$Quick,
    [switch]$LongContext
)

$ErrorActionPreference = "Continue"

$env:HSA_OVERRIDE_GFX_VERSION = "12.0.1"
$env:ROCR_VISIBLE_DEVICES = "0"
$env:HIP_VISIBLE_DEVICES = "0"
$env:OLLAMA_FLASH_ATTENTION = "1"

$testPrompt = "Write a detailed technical analysis of RISC-V vs ARM architectures."
$warmupPrompt = "What is 2+2?"

$serverOutLog = "C:\Users\rr\Desktop\Ollama\patches-check\x2\server_out.log"
$serverErrLog = "C:\Users\rr\Desktop\Ollama\patches-check\x2\server_err.log"

function Clean-Ollama-Processes {
    Write-Host "Cleaning up all Ollama server and runner processes..." -ForegroundColor Gray
    Get-Process ollama -ErrorAction SilentlyContinue | Stop-Process -Force
    Start-Sleep -Seconds 2

    # Robust port cleanup: ensure 11434 is 100% free before proceeding
    $portFree = $false
    for ($i = 0; $i -lt 15; $i++) {
        $conns = Get-NetTCPConnection -LocalPort 11434 -ErrorAction SilentlyContinue
        if (-not $conns) {
            $portFree = $true
            break
        }
        
        # Kill the processes holding the port
        foreach ($c in $conns) {
            $targetPid = $c.OwningProcess
            if ($targetPid -gt 0) {
                Write-Host "  Forcibly stopping process holding port 11434 (PID: $targetPid)..." -ForegroundColor Yellow
                Stop-Process -Id $targetPid -Force -ErrorAction SilentlyContinue
            }
        }
        Start-Sleep -Seconds 1
    }
    
    if (-not $portFree) {
        Write-Host "  Warning: Port 11434 is still bound! Socket release timed out." -ForegroundColor Red
    } else {
        Write-Host "  Port 11434 is clean and free." -ForegroundColor Gray
    }

    if (Test-Path $serverOutLog) {
        Remove-Item $serverOutLog -Force -ErrorAction SilentlyContinue
    }
    if (Test-Path $serverErrLog) {
        Remove-Item $serverErrLog -Force -ErrorAction SilentlyContinue
    }
}

if ($Quick) {
    $tests = @(
        @{ Model="qwen2.5-coder:latest"; Layers=28; Ctx=4096;  KV="f16" },
        @{ Model="gemma-4-e4b:latest";   Layers=28; Ctx=4096;  KV="f16" }
    )
} elseif ($LongContext) {
    $tests = @(
        @{ Model="qwen2.5-coder:latest"; Layers=28; Ctx=16384; KV="q8_0" },
        @{ Model="qwen2.5-coder:latest"; Layers=28; Ctx=32768; KV="q8_0" }
    )
} else {
    $tests = @(
        # Gemma-4-e4b:latest (25-28 and 33)
        @{ Model="gemma-4-e4b:latest";   Layers=25; Ctx=4096;  KV="f16" },
        @{ Model="gemma-4-e4b:latest";   Layers=26; Ctx=4096;  KV="f16" },
        @{ Model="gemma-4-e4b:latest";   Layers=27; Ctx=4096;  KV="f16" },
        @{ Model="gemma-4-e4b:latest";   Layers=28; Ctx=4096;  KV="f16" },
        @{ Model="gemma-4-e4b:latest";   Layers=33; Ctx=4096;  KV="f16" },

        # Qwen2.5-coder:latest (25-28 and 33)
        @{ Model="qwen2.5-coder:latest"; Layers=25; Ctx=4096;  KV="f16" },
        @{ Model="qwen2.5-coder:latest"; Layers=26; Ctx=4096;  KV="f16" },
        @{ Model="qwen2.5-coder:latest"; Layers=27; Ctx=4096;  KV="f16" },
        @{ Model="qwen2.5-coder:latest"; Layers=28; Ctx=4096;  KV="f16" },
        @{ Model="qwen2.5-coder:latest"; Layers=33; Ctx=4096;  KV="f16" },

        # Devstral Small (devstral:latest) (25, 28, 33)
        @{ Model="devstral:latest";      Layers=25; Ctx=4096;  KV="f16" },
        @{ Model="devstral:latest";      Layers=28; Ctx=4096;  KV="f16" },
        @{ Model="devstral:latest";      Layers=33; Ctx=4096;  KV="f16" }
    )
}

$results = @()

Write-Host "`n========================================================" -ForegroundColor Cyan
Write-Host "         gfx12 WMMA Benchmark Suite - RX 9070 XT                     " -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan

# Initially ensure clean state
Clean-Ollama-Processes

foreach ($test in $tests) {
    $model = $test.Model
    $layers = $test.Layers
    $ctx = $test.Ctx
    $kv = $test.KV

    Write-Host "`n-----------------------------------------------------" -ForegroundColor Gray
    Write-Host "Testing: $model (layers=$layers, ctx=$ctx, KV=$kv)" -ForegroundColor White

    # Set environment variables for the server process
    $env:OLLAMA_NUM_GPU = "$layers"
    $env:OLLAMA_CTX_SIZE = "$ctx"
    $env:OLLAMA_KV_CACHE_TYPE = $kv

    Write-Host "  Starting Ollama server with OLLAMA_NUM_GPU=$layers..." -ForegroundColor DarkGray
    # Redirect server standard out and err to separate logs to parse diagnostics later
    $ServerProc = Start-Process -FilePath "C:\Users\rr\AppData\Local\AMD\AI_Bundle\Ollama\ollama.exe" -ArgumentList "serve" -RedirectStandardOutput $serverOutLog -RedirectStandardError $serverErrLog -PassThru
    
    # Wait until the HTTP endpoint responds to guarantee it is listening
    $ready = $false
    for ($i = 0; $i -lt 30; $i++) {
        $check = Invoke-RestMethod -Uri "http://127.0.0.1:11434/" -ErrorAction SilentlyContinue
        if ($check -match "Ollama is running") {
            $ready = $true
            break
        }
        Start-Sleep -Seconds 1
    }
    # Wait another 5 seconds for full internal runner startup and model loading
    Start-Sleep -Seconds 5

    try {
        $clientOut = "C:\Users\rr\Desktop\Ollama\patches-check\x2\client_out.log"
        $clientErr = "C:\Users\rr\Desktop\Ollama\patches-check\x2\client_err.log"
        if (Test-Path $clientOut) { Remove-Item $clientOut -Force -ErrorAction SilentlyContinue }
        if (Test-Path $clientErr) { Remove-Item $clientErr -Force -ErrorAction SilentlyContinue }

        Write-Host "  Warming up GPU..." -ForegroundColor DarkGray
        cmd /c "echo $warmupPrompt | C:\Users\rr\AppData\Local\AMD\AI_Bundle\Ollama\ollama.exe run $model --verbose > `"$clientOut`" 2> `"$clientErr`""
        Start-Sleep -Seconds 2

        if (Test-Path $clientOut) { Remove-Item $clientOut -Force -ErrorAction SilentlyContinue }
        if (Test-Path $clientErr) { Remove-Item $clientErr -Force -ErrorAction SilentlyContinue }

        Write-Host "  Running benchmark..." -ForegroundColor DarkGray
        cmd /c "echo $testPrompt | C:\Users\rr\AppData\Local\AMD\AI_Bundle\Ollama\ollama.exe run $model --verbose > `"$clientOut`" 2> `"$clientErr`""
        
        $output = Get-Content $clientOut, $clientErr -ErrorAction SilentlyContinue

        $prefillMatch = $output | Select-String "prompt eval rate:\s*([\d.]+)\s*tokens/s"
        $decodeMatch = $output | Select-String -Pattern "(?<!prompt )eval rate:\s*([\d.]+)\s*tokens/s"
        $totalMatch = $output | Select-String "total duration:\s*([\d.]+)\s*ms"
        
        # Read the server logs to extract offload details and check for real CPU fallback
        Start-Sleep -Seconds 1
        $serverOutput = Get-Content $serverOutLog, $serverErrLog -Raw -ErrorAction SilentlyContinue

        $layersMatch = $serverOutput | Select-String "offloading\s+(\d+)\s+repeating layers to GPU"
        $offloadedLayers = if ($layersMatch) { [int]$layersMatch.Matches[0].Groups[1].Value } else { 
            $layersMatch2 = $serverOutput | Select-String "offloaded\s+(\d+)/(\d+)\s+layers to GPU"
            if ($layersMatch2) { [int]$layersMatch2.Matches[0].Groups[1].Value } else { 0 }
        }

        # Real CPU fallback is when weights are loaded to CPU (CPU model buffer size, ignoring CPU_Mapped)
        $cpuMatch = $serverOutput | Select-String "CPU model buffer size" | Where-Object { $_ -notmatch "CPU_Mapped" }
        $cpuFallback = if ($cpuMatch) { $true } else { $false }

        $prefill = if ($prefillMatch) { [double]$prefillMatch.Matches[0].Groups[1].Value } else { 0 }
        $decode = if ($decodeMatch) { [double]$decodeMatch.Matches[0].Groups[1].Value } else { 0 }
        $totalMs = if ($totalMatch) { [double]$totalMatch.Matches[0].Groups[1].Value } else { 0 }
        
        $status = "OK"
        if ($cpuFallback) {
            $status = "CPU_FALLBACK"
        } elseif ($prefill -eq 0) {
            $status = "PARSE_ERROR"
        }

        $result = [PSCustomObject]@{
            Model = $model
            Layers = $layers
            Offloaded = $offloadedLayers
            Context = $ctx
            KV_Type = $kv
            Prefill_tok_s = $prefill
            Decode_tok_s = $decode
            Total_ms = $totalMs
            CPU_Fallback = $cpuFallback
            Status = $status
        }

        $results += $result

        $color = if ($cpuFallback) { "Red" } elseif ($prefill -gt 1000) { "Green" } else { "Yellow" }
        Write-Host "  Result: $([math]::Round($prefill,1)) tok/s prefill, $([math]::Round($decode,1)) tok/s decode (Offloaded: $offloadedLayers/$layers)" -ForegroundColor $color
        if ($cpuFallback) {
            Write-Host "  WARNING: CPU FALLBACK DETECTED!" -ForegroundColor Red
        }
    }
    catch {
        Write-Host "  Error running benchmark: $_" -ForegroundColor Red
    }
    finally {
        # Shutdown server process to ensure clean environment for next run
        Clean-Ollama-Processes
    }
}

Write-Host "`n========================================================" -ForegroundColor Cyan
Write-Host "                      BENCHMARK RESULTS SUMMARY                       " -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan

Write-Host ("`n{0,-25} {1,6} {2,8} {3,8} {4,12} {5,10} {6,15}" -f "Model", "Layers", "Ctx", "KV", "Prefill", "Decode", "Status") -ForegroundColor White
Write-Host ("-" * 85) -ForegroundColor Gray

foreach ($r in $results) {
    $color = switch ($r.Status) {
        "OK" { "Green" }
        "CPU_FALLBACK" { "Red" }
        default { "Yellow" }
    }
    # Enclose formatting in parentheses to avoid prefix matching operator -f with parameter -ForegroundColor
    Write-Host ("{0,-25} {1,6} {2,8} {3,8} {4,12:F1} {5,10:F1} {6,15}" -f $r.Model, $r.Layers, $r.Context, $r.KV_Type, $r.Prefill_tok_s, $r.Decode_tok_s, $r.Status) -ForegroundColor $color
}

$csvPath = "C:\Users\rr\Desktop\Ollama\patches-check\x2\benchmark_results_$(Get-Date -Format 'yyyyMMdd_HHmmss').csv"
$results | Export-Csv -Path $csvPath -NoTypeInformation
Write-Host "`nResults exported to: $csvPath" -ForegroundColor Green

$successful = $results | Where-Object { $_.Status -eq "OK" }
if ($successful) {
    $avgPrefill = ($successful | Measure-Object -Property Prefill_tok_s -Average).Average
    $avgDecode = ($successful | Measure-Object -Property Decode_tok_s -Average).Average
    $maxPrefill = ($successful | Measure-Object -Property Prefill_tok_s -Maximum).Maximum
    $maxDecode = ($successful | Measure-Object -Property Decode_tok_s -Maximum).Maximum

    Write-Host "`n=== Performance Analysis ===" -ForegroundColor Cyan
    Write-Host "Average Prefill: $([math]::Round($avgPrefill,1)) tok/s" -ForegroundColor White
    Write-Host "Average Decode:  $([math]::Round($avgDecode,1)) tok/s" -ForegroundColor White
    Write-Host "Max Prefill:     $([math]::Round($maxPrefill,1)) tok/s" -ForegroundColor White
    Write-Host "Max Decode:      $([math]::Round($maxDecode,1)) tok/s" -ForegroundColor White
}

$cpuFallbacks = $results | Where-Object { $_.Status -eq "CPU_FALLBACK" }
if ($cpuFallbacks) {
    Write-Host "`nWARNING: CPU Fallbacks detected:" -ForegroundColor Red
    foreach ($c in $cpuFallbacks) {
        Write-Host "   $($c.Model) (layers=$($c.Layers), ctx=$($c.Context), KV=$($c.KV_Type))" -ForegroundColor Red
    }
    Write-Host "   Fix: Reduce context, use smaller quant, or Q8_0/Q4_0 KV." -ForegroundColor Yellow
}
