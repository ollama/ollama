Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$usbRoot = $PSScriptRoot
$env:OLLAMA_MODELS = Join-Path $usbRoot 'models'

function Get-OllamaCommand {
  $cmd = Get-Command ollama -ErrorAction SilentlyContinue
  if ($cmd) { return $cmd.Source }

  $portableExe = Join-Path $usbRoot 'ollama\ollama.exe'
  if (Test-Path $portableExe) { return $portableExe }

  throw '找不到 ollama 執行檔。'
}

$ollamaExe = Get-OllamaCommand
$models = @('llama3')

foreach ($model in $models) {
  Write-Host "拉取模型: $model"
  & $ollamaExe pull $model
}
