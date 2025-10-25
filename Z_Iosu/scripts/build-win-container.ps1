Param(
  [string]$Version = "0.0.0",
  [string]$Arch = "amd64",           # amd64 | arm64
  [switch]$Installer,                # Generar installer
  [string]$ImageTag = "ollama-winbuilder:local",
  [switch]$NoCache,
  [string]$Out = "dist",
  [string]$GoFlags = "",
  [switch]$RebuildImage,
  [switch]$CCache,
  [switch]$CpuLibs,                  # Compilar librerías CPU (ggml)
  [switch]$NoApp                     # No compilar app GUI
)

$ErrorActionPreference = 'Stop'

function Invoke-Cmd {
  param([string]$Cmd)
  Write-Host "[RUN] $Cmd" -ForegroundColor Cyan
  & powershell -NoLogo -NoProfile -Command $Cmd
  if ($LASTEXITCODE -ne 0) { throw "Fallo comando: $Cmd" }
}

$root = (Resolve-Path "$PSScriptRoot/.." ).Path
Set-Location $root

$dockerfile = "Z_Iosu/docker/Dockerfile.winbuilder"
if (-not (Test-Path $dockerfile)) { throw "No existe $dockerfile" }

if ($RebuildImage -or -not (docker images -q $ImageTag)) {
  Write-Host "==> Construyendo imagen $ImageTag" -ForegroundColor Yellow
  $noCacheArg = ""
  if ($NoCache) { $noCacheArg = "--no-cache" }
  docker build -f $dockerfile -t $ImageTag $noCacheArg .
  if ($LASTEXITCODE -ne 0) { throw "Error build imagen" }
}

$workdir = "/workspace"
$uid = (Get-ChildItem Env:UID).Value
if (-not $uid) { $uid = 1000 }

$dockerArgs = @("-v", "${PWD}:$workdir", "-w", $workdir)
$dockerArgs += @("-e", "PKG_VERSION=$Version")
if ($CCache) { $dockerArgs += @("-e","CCACHE_DIR=/workspace/.ccache") }

$buildCmd = "/usr/local/bin/build-win.sh -v $Version -a $Arch -o $Out"
if ($Installer) { $buildCmd += " -I" }
if ($CCache) { $buildCmd += " --ccache" }
if ($CpuLibs) { $buildCmd += " --cpu-libs" }
if ($NoApp) { $buildCmd += " --no-app" }
if ($GoFlags) { $buildCmd += " -- $GoFlags" }

Write-Host "==> Ejecutando build dentro del contenedor" -ForegroundColor Green
docker run --rm @dockerArgs $ImageTag bash -lc "$buildCmd"
if ($LASTEXITCODE -ne 0) { throw "Build falló dentro del contenedor" }

Write-Host "==> Artefactos en $Out" -ForegroundColor Green