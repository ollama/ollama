Param(
  [string]$Preset = "CUDA 12",
  [switch]$Clean,
  [switch]$SkipCMake,
  [switch]$SkipGo,
  [string]$OutDir = "dist\\cuda-win",
  [switch]$InstallerLayout,   # Prepara estructura dist/windows-amd64 para instalador
  [string]$Arch = "amd64",    # Solo amd64 en este flujo
  [switch]$Quiet
)
$ErrorActionPreference='Stop'
function Info($m){ if(-not $Quiet){ Write-Host "[build-win-cuda] $m" -ForegroundColor Cyan } }
function Die($m){ Write-Host "[build-win-cuda][ERROR] $m" -ForegroundColor Red; exit 1 }

if ($Clean -and (Test-Path build)) { Info "Limpiando build/"; Remove-Item build -Recurse -Force }
if (-not $SkipCMake) {
  Info "Configurando CMake preset '$Preset'"
  cmake --preset "$Preset"; if ($LASTEXITCODE -ne 0) { Die "Fallo cmake configure" }
  Info "Compilando librerías ($Preset)"
  cmake --build --preset "$Preset"; if ($LASTEXITCODE -ne 0) { Die "Fallo build" }
  Info "Instalando componentes CUDA"
  cmake --install build --component CUDA; if ($LASTEXITCODE -ne 0) { Die "Fallo install" }
}
if (-not $SkipGo) {
  Info "Compilando binario Go (CGO habilitado)"
  $env:CGO_ENABLED=1
  go build -o ollama.exe .; if ($LASTEXITCODE -ne 0) { Die "Fallo go build" }
  if (-not (Test-Path ollama.exe)) { Die "No se generó ollama.exe" }
}
if (-not (Test-Path $OutDir)) { New-Item -ItemType Directory -Path $OutDir | Out-Null }
Copy-Item ollama.exe (Join-Path $OutDir 'ollama.exe') -Force
Info "Listo: $(Join-Path $OutDir 'ollama.exe')"

# Layout opcional para Instalador (Inno Setup)
if ($InstallerLayout) {
  if ($Arch -ne 'amd64') { Die "Por ahora sólo amd64 soportado en InstallerLayout" }
  $archRoot = Join-Path 'dist' 'windows-amd64'
  if (-not (Test-Path $archRoot)) { New-Item -ItemType Directory -Path $archRoot | Out-Null }
  # Binario principal
  Copy-Item ollama.exe (Join-Path $archRoot 'ollama.exe') -Force
  # Wrapper esperado por .iss (ollama app.exe -> windows-amd64-app.exe en dist raíz original). Simplificamos creando windows-amd64-app.exe
  $wrapper = Join-Path 'dist' 'windows-amd64-app.exe'
  if (-not (Test-Path $wrapper)) { Copy-Item ollama.exe $wrapper -Force }
  # Librerías CUDA instaladas: si existen en dist/lib/ollama copiarlas a dist/windows-amd64/lib/ollama
  $srcLib = Join-Path 'dist' 'lib/ollama'
  $dstLib = Join-Path $archRoot 'lib/ollama'
  if (Test-Path $srcLib) {
    if (-not (Test-Path $dstLib)) { New-Item -ItemType Directory -Path $dstLib -Force | Out-Null }
    Get-ChildItem -Path $srcLib -File -Recurse | ForEach-Object {
      $rel = $_.FullName.Substring($srcLib.Length).TrimStart('\\','/')
      $target = Join-Path $dstLib $rel
      $tDir = Split-Path $target -Parent
      if (-not (Test-Path $tDir)) { New-Item -ItemType Directory -Path $tDir -Force | Out-Null }
      Copy-Item $_.FullName $target -Force
    }
  } else { Info "(Aviso) No se encontró dist/lib/ollama para copiar a layout instalador" }
  Info "Estructura instalador preparada en dist/windows-amd64"
}
