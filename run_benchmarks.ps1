$layers = @(25, 28, 33)
$models = @("devstral:latest")
$prompt = "Write a comprehensive Python script that sorts a list using quicksort and explains every step."

$resultsFile = "C:\Users\rr\Desktop\Ollama\ollama-for-amd\multi_model_benchmarks.txt"
"=== Multi-Model Benchmark Results (RX 9070 XT gfx1201) ===" | Out-File -FilePath $resultsFile -Encoding ascii

foreach ($model in $models) {
    "`n=======================================" | Out-File -FilePath $resultsFile -Append -Encoding ascii
    "    MODEL: $model    " | Out-File -FilePath $resultsFile -Append -Encoding ascii
    "=======================================" | Out-File -FilePath $resultsFile -Append -Encoding ascii

    foreach ($l in $layers) {
        Write-Host "Benchmarking $model with $l GPU layers..."
    
    # Restart ollama server with the specific number of GPU layers
    Stop-Process -Name "ollama" -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
    
    $env:HSA_OVERRIDE_GFX_VERSION = "12.0.1"
    $env:ROCR_VISIBLE_DEVICES = "0"
    $env:HIP_VISIBLE_DEVICES = "0"
    $env:OLLAMA_FLASH_ATTENTION = "1"
    $env:OLLAMA_NUM_GPU = $l.ToString()
    $env:OLLAMA_KEEP_ALIVE = "1m"
    
    $proc = Start-Process -FilePath "C:\Users\rr\Desktop\Ollama\ollama-for-amd\dist\windows-amd64\ollama.exe" `
        -ArgumentList "serve" `
        -WorkingDirectory "C:\Users\rr\Desktop\Ollama\ollama-for-amd\dist\windows-amd64" `
        -WindowStyle Hidden -PassThru
        
    Start-Sleep -Seconds 5
    
    "`n`n--- LAYER COUNT: $l ---" | Out-File -FilePath $resultsFile -Append
    
    $payload = @{
        model = $model
        prompt = $prompt
        stream = $false
    } | ConvertTo-Json

    $payloadFile = "payload_$l.json"
    $payload | Out-File -FilePath $payloadFile -Encoding ascii -NoNewline

    $output = curl.exe -s -X POST http://localhost:11434/api/generate -H "Content-Type: application/json" -d "@$payloadFile"
    
    try {
        $statsObj = $output | ConvertFrom-Json
        if ($statsObj -and $statsObj.eval_count) {
            $promptRate = $statsObj.prompt_eval_count / ($statsObj.prompt_eval_duration / 1000000000)
            $evalRate = $statsObj.eval_count / ($statsObj.eval_duration / 1000000000)
            
            "Prompt Eval Rate: $($promptRate.ToString('F2')) tokens/s" | Out-File -FilePath $resultsFile -Append
            "Eval Rate: $($evalRate.ToString('F2')) tokens/s" | Out-File -FilePath $resultsFile -Append
            "Total Time: $(($statsObj.total_duration / 1000000000).ToString('F2'))s" | Out-File -FilePath $resultsFile -Append
        } else {
            "Failed or no stats returned. Output: $output" | Out-File -FilePath $resultsFile -Append
        }
    } catch {
        "Failed to parse JSON. Output: $output" | Out-File -FilePath $resultsFile -Append
    }
    
    if (Test-Path $payloadFile) { Remove-Item $payloadFile }
}
}

Write-Host "Benchmarks complete. Results in $resultsFile"
