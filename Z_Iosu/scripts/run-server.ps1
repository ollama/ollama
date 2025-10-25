param(
  [switch]$Rebuild,
  [string]$CudaArch="89",
  [string]$Configuration="Release",
  [switch]$Verbose,
  [switch]$TestAPI,
  [string]$ModelName="llama3.2",
  [string]$Prompt="Hola GPU?",
  [int]$StartupTimeoutSec=25
)
$ErrorActionPreference='Stop'
function Log($m){ if($Verbose){ Write-Host "[run] $m" -ForegroundColor Cyan } }
if($Rebuild -or -not (Test-Path .\ollama.exe)){
  Log "Compilando ($Configuration)"
  if(-not (Test-Path build)){
    cmake -S . -B build -DCMAKE_BUILD_TYPE=$Configuration -DCMAKE_CUDA_ARCHITECTURES=$CudaArch | Out-Null
  }
  # Build con config explícita (Visual Studio es multi-config)
  cmake --build build --config $Configuration --parallel
  # Copiar librerías generadas (ruta build/lib/ollama)
  $libSrc = Join-Path build "lib/ollama"
  if(Test-Path $libSrc){
    if(-not (Test-Path .\lib\ollama)){ New-Item -ItemType Directory -Path .\lib\ollama | Out-Null }
    Get-ChildItem $libSrc | ForEach-Object { Copy-Item $_.FullName .\lib\ollama -Recurse -Force }
  } else {
    Log "No se encontró $libSrc (puede que layout haya cambiado)"
  }
  # Limpiar flags CGO que rompen MSVC (/Werror) si vienen heredados
  if($env:CGO_CFLAGS){ Log "CGO_CFLAGS original: $($env:CGO_CFLAGS)" }
  $env:CGO_CFLAGS=""
  $env:CGO_ENABLED=1
  $ldflags="-s -w"
  go build -trimpath -ldflags $ldflags -o ollama.exe .
}
Log 'Iniciando servidor'
$env:OLLAMA_HOST='0.0.0.0:11434'
Start-Process -FilePath .\ollama.exe -ArgumentList 'serve' -NoNewWindow
Write-Host 'Servidor lanzado. Esperando disponibilidad...' -ForegroundColor Green

if($TestAPI){
  $deadline = (Get-Date).AddSeconds($StartupTimeoutSec)
  $ready=$false
  while((Get-Date) -lt $deadline){
    try {
      $resp = Invoke-WebRequest -Method GET -Uri 'http://localhost:11434' -TimeoutSec 3 -ErrorAction Stop
      if($resp.StatusCode -ge 200){ $ready=$true; break }
    } catch { Start-Sleep -Milliseconds 800 }
  }
  if(-not $ready){ Write-Warning 'No se recibió respuesta HTTP base antes del timeout'; }
  else {
    Write-Host 'Servidor responde, lanzando generate...' -ForegroundColor Cyan
    try {
      $body = @{ model=$ModelName; prompt=$Prompt; stream=$false } | ConvertTo-Json -Compress
      $gen = Invoke-WebRequest -Method POST -Body $body -Uri 'http://localhost:11434/api/generate' -ContentType 'application/json' -TimeoutSec 120
      Write-Host 'Respuesta generate:' -ForegroundColor Yellow
      Write-Host $gen.Content
    } catch { Write-Warning "Fallo petición generate: $_" }
  }
  Write-Host 'Fin prueba API. El servidor sigue ejecutándose.' -ForegroundColor Green
}
Write-Host 'Para detener: Get-Process ollama | Stop-Process' -ForegroundColor Magenta
