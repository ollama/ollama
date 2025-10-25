Param(
  [string]$Tag = "ollama:0.11.10-custom",
  [switch]$NoCache,
  [switch]$Push,
  [string]$Dockerfile = "Z_Iosu/docker/Dockerfile.simple",
  [string]$Context = ".",
  [switch]$Quiet
)

$ErrorActionPreference = 'Stop'
function Info($m){ if(-not $Quiet){ Write-Host "[build-simple] $m" -ForegroundColor Cyan } }
function Warn($m){ if(-not $Quiet){ Write-Host "[build-simple] $m" -ForegroundColor Yellow } }
function Die($m){ Write-Host "[build-simple][ERROR] $m" -ForegroundColor Red; exit 1 }

if (-not (Test-Path $Dockerfile)) { Die "No existe Dockerfile en ruta $Dockerfile" }

$cmd = @('docker','build','-f', $Dockerfile, '-t', $Tag)
if ($NoCache) { $cmd += '--no-cache' }
$cmd += $Context

Info "Ejecutando: $($cmd -join ' ')"
$process = Start-Process -FilePath $cmd[0] -ArgumentList $cmd[1..($cmd.Length-1)] -NoNewWindow -Wait -PassThru
if ($process.ExitCode -ne 0) { Die "Fallo build (exit=$($process.ExitCode))" }
Info "Imagen construida: $Tag"

if ($Push) {
  Info "Haciendo push de $Tag"
  docker push $Tag | Out-Null
  if ($LASTEXITCODE -ne 0) { Die "Fallo docker push" }
  Info "Push completado"
}
