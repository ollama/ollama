Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$usbRoot = $PSScriptRoot
Set-Location $usbRoot

$env:OLLAMA_MODELS = Join-Path $usbRoot 'models'
$env:HF_HOME = Join-Path $usbRoot 'whisper_models'
$env:XDG_CACHE_HOME = Join-Path $usbRoot 'whisper_models'
$healthCheckTimeoutSeconds = 3
$ollamaStartupTimeoutSeconds = 30

function Get-OllamaCommand {
  $cmd = Get-Command ollama -ErrorAction SilentlyContinue
  if ($cmd) { return $cmd.Source }

  $portableExe = Join-Path $usbRoot 'ollama\ollama.exe'
  if (Test-Path $portableExe) { return $portableExe }

  throw '找不到 ollama 執行檔，請先執行 Launch-AI-FirstTime.cmd。'
}

function Test-OllamaReady {
  try {
    $null = Invoke-WebRequest -Uri 'http://localhost:11434/api/tags' -UseBasicParsing -TimeoutSec $healthCheckTimeoutSeconds
    return $true
  } catch {
    return $false
  }
}

$ollamaExe = Get-OllamaCommand
if (-not (Test-OllamaReady)) {
  Write-Host '啟動 Ollama 服務...'
  Start-Process -FilePath $ollamaExe -ArgumentList 'serve' -WindowStyle Hidden
  for ($i = 0; $i -lt $ollamaStartupTimeoutSeconds; $i++) {
    Start-Sleep -Seconds 1
    if (Test-OllamaReady) { break }
  }
}

if (-not (Test-OllamaReady)) {
  throw 'Ollama 啟動失敗，請檢查安裝與防火牆設定。'
}

Write-Host 'Ollama 服務已就緒。'
Write-Host '提示：本腳本不會關閉 Ollama 服務，讓後續模型請求可直接使用。'

$pythonExe = Join-Path $usbRoot 'python_embed\python.exe'
$coreScript = Join-Path $usbRoot 'ai_core.py'
if ((Test-Path $pythonExe) -and (Test-Path $coreScript)) {
  Write-Host '啟動 ai_core.py 自動化流程...'
  & $pythonExe $coreScript
} else {
  Write-Host '尚未偵測到 ai_core.py 或可攜 Python，略過自動化流程。'
}
