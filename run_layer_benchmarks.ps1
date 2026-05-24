$layers = @(25, 26, 28, 29, 30, 31, 32, 33)
$model = "devstral:latest"
$prompt = "Write a comprehensive Python script that sorts a list using quicksort and explains every step."

$resultsFile = "C:\Users\rr\Desktop\Ollama\ollama-for-amd\layer_benchmarks.txt"
"=== Devstral Benchmark Results (RX 9070 XT gfx1201) ===" | Out-File -FilePath $resultsFile

foreach ($l in $layers) {
    Write-Host "Benchmarking with $l GPU layers..."
    
    # Restart ollama server with the specific number of GPU layers
    Stop-Process -Name "ollama" -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
    
    $env:HSA_OVERRIDE_GFX_VERSION = "12.0.1"
    $env:ROCR_VISIBLE_DEVICES = "0"
    $env:HIP_VISIBLE_DEVICES = "0"
    $env:OLLAMA_FLASH_ATTENTION = "1"
    $env:OLLAMA_NUM_GPU = $l.ToString()
    $env:OLLAMA_KEEP_ALIVE = "1m"
    
    $proc = Start-Process -FilePath "C:\Users\rr\AppData\Local\AMD\AI_Bundle\Ollama\ollama.exe" `
        -ArgumentList "serve" `
        -WindowStyle Hidden -PassThru
        
    Start-Sleep -Seconds 5
    
    "`n`n--- LAYER COUNT: $l ---" | Out-File -FilePath $resultsFile -Append
    
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $output = & "C:\Users\rr\AppData\Local\AMD\AI_Bundle\Ollama\ollama.exe" run $model --verbose $prompt 2>&1
    $sw.Stop()
    
    $stats = $output | Select-String "eval rate:|prompt eval rate:|total duration:|load duration:"
    
    if ($stats) {
        $stats | Out-File -FilePath $resultsFile -Append
        "Wall Time: $($sw.Elapsed.TotalSeconds.ToString('F2'))s" | Out-File -FilePath $resultsFile -Append
    } else {
        "Failed or no stats returned." | Out-File -FilePath $resultsFile -Append
        $output[-10..-1] | Out-File -FilePath $resultsFile -Append
    }
}

Write-Host "Benchmarks complete. Results in $resultsFile"
