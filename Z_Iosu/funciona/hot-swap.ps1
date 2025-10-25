# SNAPSHOT FUNCIONA - hot-swap.ps1 (2025-09-05)
# Copia de referencia estable. No editar aquí; editar versión activa en Z_Iosu/scripts/hot-swap.ps1

Param(
  [switch]$ForceBuilder,
  [string]$ContainerName = "ollama-gpu",
  [string]$ImageName = "ollama:0.11.10-custom",
  [switch]$Restart,
  [switch]$Verify,
  [switch]$NoWSL,
  [switch]$RollbackOnFail,
  [switch]$KeepBackup,
  [switch]$Quiet
)

$ErrorActionPreference = 'Stop'
function Info($m){ if(-not $Quiet){ Write-Host "[hot-swap] $m" -ForegroundColor Cyan } }
function Warn($m){ if(-not $Quiet){ Write-Host "[hot-swap] $m" -ForegroundColor Yellow } }
function Die($m){ Write-Host "[hot-swap][ERROR] $m" -ForegroundColor Red; exit 1 }
function HashFile($path){ if(Test-Path $path){ (Get-FileHash $path -Algorithm SHA256).Hash.ToLower() } }

$repoWin = (Resolve-Path "$PSScriptRoot\..\").Path
$repoWsl = $null
if (-not $ForceBuilder -and -not $NoWSL) {
  $wslOk = $false
  try { wsl -e bash -lc 'echo wsl_ok' | Out-Null; $wslOk = $true } catch { $wslOk = $false }
  if ($wslOk) {
    $repoWsl = '/mnt/' + ($repoWin.Substring(0,1).ToLower()) + '/' + $repoWin.Substring(3) -replace '\\','/'
    Info "Ruta repo WSL: $repoWsl"
    $buildCmd = @(
      "cd $repoWsl",
      "export CGO_ENABLED=1",
      "export GOOS=linux GOARCH=amd64",
      "go build -o ollama-dev ./cmd/ollama || go build -o ollama-dev . || exit 97"
    ) -join ' && '
    Info "Compilando en WSL..."
    wsl -e bash -lc "$buildCmd" | Out-Null
    $exit = $LASTEXITCODE
    if ($exit -eq 0) { Info "Build Go (WSL) OK" }
    elseif ($exit -eq 97) { Warn "Fallo build nativo. Usaré builder."; $ForceBuilder = $true }
    else { Warn "Build WSL fallo ($exit). Iré a builder."; $ForceBuilder = $true }
  } else { Warn "WSL no disponible; usaré builder contenedor."; $ForceBuilder = $true }
}

$builderName = 'ollama-build'
if ($ForceBuilder) {
  if (-not (docker ps -a --format '{{.Names}}' | Select-String -SimpleMatch $builderName)) {
    Info "Creando builder (golang:1.24-bookworm)."
    docker run -d --name $builderName -w /src -v "${repoWin}:/src" golang:1.24-bookworm sleep infinity | Out-Null
  } elseif (-not (docker ps --format '{{.Names}}' | Select-String -SimpleMatch $builderName)) {
    docker start $builderName | Out-Null
  }
  Info "Compilando binario dentro del builder..."
  docker exec $builderName bash -lc "cd /src; export CGO_ENABLED=1 GOOS=linux GOARCH=amd64; go build -o ollama-dev ./cmd/ollama || go build -o ollama-dev ."; if ($LASTEXITCODE -ne 0) { Die "Fallo build en builder" }
}

$binPath = Join-Path $repoWin 'ollama-dev'
if (-not (Test-Path $binPath)) { Die "No se generó ollama-dev" }
$remotePath = '/usr/bin/ollama'
$backupName = "ollama.backup_" + (Get-Date -Format 'yyyyMMdd_HHmmss')
$tmpDir = New-Item -ItemType Directory -Path (Join-Path $env:TEMP ("hot-swap-" + [guid]::NewGuid())) -Force
$oldHash = $null
if (docker ps --format '{{.Names}}' | Select-String -SimpleMatch $ContainerName) {
  try { $oldHash = (docker exec $ContainerName bash -lc "sha256sum $remotePath 2>/dev/null" 2>$null) -split ' ' | Select-Object -First 1 } catch {}
}
if (-not (docker ps --format '{{.Names}}' | Select-String -SimpleMatch $ContainerName)) {
  Warn "Contenedor $ContainerName no está corriendo; lo crearé."; docker run -d --gpus all -p 11434:11434 --name $ContainerName $ImageName | Out-Null; Start-Sleep -Seconds 3
}
try {
  if ($RollbackOnFail -or $KeepBackup) {
    Info "Backup previo ($backupName)..."; docker exec $ContainerName bash -lc "if [ -f $remotePath ]; then cp $remotePath /tmp/$backupName; fi"
  }
} catch { Warn "No se pudo crear backup ($_)." }
Info "Copiando binario nuevo..."; docker cp $binPath "${ContainerName}:$remotePath"
if ($Restart.IsPresent) { Info "Reiniciando..."; docker restart $ContainerName | Out-Null; Start-Sleep -Seconds 2 }
$newHash = (docker exec $ContainerName bash -lc "sha256sum $remotePath" 2>$null) -split ' ' | Select-Object -First 1
if ($Verify) {
  $localHash = HashFile $binPath
  Info "Hash local:  $localHash"; Info "Hash previo: $oldHash"; Info "Hash nuevo:  $newHash"
  if ($localHash -ne $newHash -and $RollbackOnFail) {
    Warn "Verificación fallida. Rollback..."; docker exec $ContainerName bash -lc "if [ -f /tmp/$backupName ]; then cp /tmp/$backupName $remotePath; fi"; if ($Restart.IsPresent) { docker restart $ContainerName | Out-Null }; Die "Rollback aplicado"
  }
}
if (-not $KeepBackup -and ($RollbackOnFail -or $Verify)) { docker exec $ContainerName bash -lc "rm -f /tmp/$backupName" 2>$null | Out-Null }
if (-not $Quiet) { docker logs --tail 20 $ContainerName | Out-String | Write-Host }
Remove-Item -Recurse -Force $tmpDir 2>$null | Out-Null
Info "Hot-swap snapshot completado (hash nuevo: $newHash)"
