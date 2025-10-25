Param(
  [string]$OutputDir = "Z_Iosu/release",
  [string]$Name = "ollama-win64",
  [switch]$Zip,
  [switch]$SkipBuild,
  [switch]$Quiet
)

$ErrorActionPreference = 'Stop'
function Info($m){ if(-not $Quiet){ Write-Host "[package-release] $m" -ForegroundColor Cyan } }
function Warn($m){ if(-not $Quiet){ Write-Host "[package-release] $m" -ForegroundColor Yellow } }
function Die($m){ Write-Host "[package-release][ERROR] $m" -ForegroundColor Red; exit 1 }

# 1. Construir binario si no existe o si no se especifica SkipBuild
if (-not $SkipBuild) {
  if (Test-Path .\ollama.exe) { Info "Eliminando binario previo para build limpio"; Remove-Item .\ollama.exe -Force }
  Info "Compilando ollama.exe (CGO habilitado)"
  $env:CGO_ENABLED = 1
  go build -o ollama.exe .
  if ($LASTEXITCODE -ne 0 -or -not (Test-Path .\ollama.exe)) { Die "Fallo compilación Go" }
} else {
  if (-not (Test-Path .\ollama.exe)) { Die "No existe ollama.exe y se solicitó SkipBuild" }
}

# 2. Recolectar dependencias (DLL) mínimas
# Nota: Se asume que las DLL relevantes (CUDA/cuBLAS) ya están en PATH del runtime donde se lanzará.
# Aquí copiamos sólo lo propio (si existiera) y dejamos instrucciones.
$releaseRoot = Join-Path $OutputDir $Name
if (Test-Path $releaseRoot) { Remove-Item $releaseRoot -Recurse -Force }
New-Item -ItemType Directory -Path $releaseRoot | Out-Null

Copy-Item ollama.exe (Join-Path $releaseRoot 'ollama.exe')

# 3. Incluir README de uso rápido
$readme = @"
# Paquete $Name

Contenido:
- ollama.exe

## Uso rápido

```powershell
# Lanzar servidor
./ollama.exe serve

# Probar generación (requiere que haya modelos descargados o use el flujo normal de pull)
Invoke-WebRequest -Uri http://localhost:11434/api/generate -Method POST -Body '{"model":"llama3","prompt":"hola"}' -ContentType 'application/json'
```

## Notas
- Asegúrate de tener las librerías CUDA en tu sistema si quieres GPU.
- Este paquete es un binario directo; para Docker usa `Z_Iosu/docker/Dockerfile.simple`.
- Para hot-swap en contenedor: ver `Z_Iosu/docs/hot-swap-workflow.md`.

"@
$readme | Out-File -FilePath (Join-Path $releaseRoot 'README.txt') -Encoding UTF8 -Force

# 4. Zip opcional
if ($Zip) {
  $zipPath = Join-Path $OutputDir ("$Name.zip")
  if (Test-Path $zipPath) { Remove-Item $zipPath -Force }
  Info "Creando ZIP $zipPath"
  Compress-Archive -Path (Join-Path $releaseRoot '*') -DestinationPath $zipPath -Force
  Info "ZIP listo"
}

Info "Package listo en $releaseRoot"
