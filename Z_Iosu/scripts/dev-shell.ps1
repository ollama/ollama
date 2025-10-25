Param(
  [string]$Image = "ollama-devfull",
  [string]$Name = "ollama-devshell",
  [switch]$Rebuild,
  [switch]$Keep,
  [switch]$Quiet
)
$ErrorActionPreference='Stop'
function Info($m){ if(-not $Quiet){ Write-Host "[dev-shell] $m" -ForegroundColor Cyan } }
function Die($m){ Write-Host "[dev-shell][ERROR] $m" -ForegroundColor Red; exit 1 }

# 1. Build image if missing or forced
$needBuild = $Rebuild
if (-not $needBuild) {
  $exists = docker images --format '{{.Repository}}:{{.Tag}}' | Select-String -SimpleMatch $Image -Quiet
  if (-not $exists) { $needBuild = $true }
}
if ($needBuild){
  Info "Construyendo imagen devfull ($Image)" 
  docker build -f Z_Iosu/docker/Dockerfile.devfull -t $Image .
  if ($LASTEXITCODE -ne 0) { Die "Fallo build devfull" }
}

# 2. Si contenedor existe y no está corriendo, iniciar; si no existe crear
$running = docker ps --format '{{.Names}}' | Select-String -SimpleMatch $Name -Quiet
if (-not $running) {
  $exists = docker ps -a --format '{{.Names}}' | Select-String -SimpleMatch $Name -Quiet
  if ($exists) { docker start $Name | Out-Null }
  else {
    Info "Creando contenedor $Name"
    docker run -d --name $Name -v "${PWD}:/src" -w /src $Image sleep infinity | Out-Null
  }
}

# 3. Entrar interactive shell
Info "Entrando en shell (bash) dentro de $Name"
docker exec -it $Name bash

# 4. Cleanup opcional
if (-not $Keep) {
  Info "Eliminando contenedor (usar -Keep para conservarlo)"
  docker rm -f $Name | Out-Null
}
