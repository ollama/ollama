param(
    [string]$CudaArch="89",              # Compute capability(s) e.g. 89;90
    [switch]$Reconfigure,                 # Forzar regenerar build dir
    [switch]$UseNinja,                    # Usar Ninja
    [switch]$Install,                     # Ejecutar cmake --install
    [switch]$SkipGo,                      # Saltar build Go
    [switch]$Verbose
)

$ErrorActionPreference = 'Stop'
function Log($m){ if($Verbose){ Write-Host "[build] $m" -ForegroundColor Cyan } }

# 1. Verificar prerequisitos
$req = @('cmake','go')
foreach($r in $req){ if(-not (Get-Command $r -ErrorAction SilentlyContinue)){ throw "No se encontró '$r' en PATH" } }

# NVIDIA Toolkit
if(-not (Test-Path Env:CUDA_PATH)){ Write-Warning 'CUDA_PATH no definido; intentando ruta por defecto'; $defaultCuda='C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA'; if(Test-Path $defaultCuda){ $latest = Get-ChildItem $defaultCuda | Sort-Object Name -Descending | Select-Object -First 1; $env:CUDA_PATH=$latest.FullName } }
if(-not (Test-Path "$env:CUDA_PATH\bin\nvcc.exe")){ Write-Warning 'nvcc no encontrado. Verifica instalación de CUDA Toolkit.' }

# 2. Preparar directorios
$buildDir = Join-Path (Get-Location) 'build'
if($Reconfigure -and (Test-Path $buildDir)){ Log 'Eliminando directorio build anterior'; Remove-Item -Recurse -Force $buildDir }
if(-not (Test-Path $buildDir)){ New-Item -ItemType Directory -Path $buildDir | Out-Null }

# 3. Configurar CMake
$cmakeArgs = @('-B','build','-DCMAKE_BUILD_TYPE=Release',"-DCMAKE_CUDA_ARCHITECTURES=$CudaArch")
if($UseNinja){ $cmakeArgs += @('-G','Ninja') }
Log "Configurando CMake con arquitecturas: $CudaArch"
cmake @cmakeArgs

# 4. Build librerías CUDA
Log 'Compilando (CUDA)'
cmake --build build --parallel

if($Install){
  Log 'Instalando componente CUDA'
  cmake --install build --component CUDA --strip
}
else {
  # Si no se instala, asegurar ruta dist mínima
  if(-not (Test-Path 'dist/lib/ollama')){ New-Item -ItemType Directory -Path 'dist/lib/ollama' -Force | Out-Null }
  # Copiar lib resultante si existe (pattern genérico)
  Get-ChildItem build -Recurse -Include 'ggml*cuda*.dll','ggml*cuda*.lib' -ErrorAction SilentlyContinue | ForEach-Object { Copy-Item $_.FullName dist/lib/ollama -Force }
}

# 5. Build Go
if(-not $SkipGo){
  Log 'Compilando binario Go (ollama.exe)'
  $env:CGO_ENABLED=1
  go build -trimpath -o ollama.exe .
  if(-not (Test-Path .\lib\ollama)){ New-Item -ItemType Directory -Path .\lib\ollama | Out-Null }
  Copy-Item dist\lib\ollama\* .\lib\ollama -Recurse -Force
}

Log 'Hecho.'
Write-Host "Ejecuta: .\ollama.exe serve" -ForegroundColor Green
