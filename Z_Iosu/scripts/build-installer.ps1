<#
build-installer.ps1 (refactor)
Objetivo: flujo robusto y legible para generar instalador Inno Setup.
Mantiene: -AutoVersion, -Version, -ForceClangGnu, -SkipBuild, -SkipWrapper, -InnoPath, -Verbose.
Simplificaciones: funciones claras, sin here-strings problemáticos.
#>

  param(
    [string]$Version,
    [switch]$AutoVersion,
    [string]$Arch = 'amd64',
    [string]$IssFile = 'app/ollama.iss',
    [string]$OutDir = 'Z_Iosu/release/installer',
    [string]$InnoPath,
    [switch]$SkipBuild,
    [switch]$ForceClangGnu,
    [switch]$SkipWrapper,
    [switch]$DebugGo,
  [switch]$Portable,
  [switch]$AlsoPortable,
  [switch]$NoAlsoPortable,
    [switch]$Quiet,
    [switch]$Verbose
  )

  Set-StrictMode -Version Latest
  $ErrorActionPreference = 'Stop'

  function LogInfo($m){ if(-not $Quiet){ Write-Host "[installer] $m" -ForegroundColor Cyan } }
  function LogWarn($m){ Write-Host "[installer][WARN] $m" -ForegroundColor Yellow }
  function LogErr($m){ Write-Host "[installer][ERR] $m" -ForegroundColor Red }
  function LogDbg($m){ if($Verbose){ Write-Host "[installer][DBG] $m" -ForegroundColor DarkGray } }
  function Fail($m){ LogErr $m; exit 1 }

  # 1. Raíz repo
  $repoRootCandidate = Join-Path (Join-Path $PSScriptRoot '..') '..'
  try { $root = (Resolve-Path $repoRootCandidate).Path } catch { Fail "No se pudo resolver raíz repo" }
  Push-Location $root
  LogDbg "Root: $root"

  if (-not (Test-Path $IssFile)) { Fail "No se encuentra $IssFile" }

  # 2. Versionado
  if ($AutoVersion) {
    $gitShort = (git rev-parse --short HEAD) 2>$null
    if (-not $gitShort) { $gitShort = 'nogit' }
    $Version = (Get-Date -Format 'yyyy.MM.dd') + "+$gitShort"
  }
  if (-not $Version) { $Version = '0.0.0' }
  $env:PKG_VERSION = $Version
  LogInfo "Versión: $Version"

  function Get-CleanVersionInfo([string]$v){
    if ($v -notmatch '^[0-9]+(\.[0-9]+){1,3}$') {
      $m = [regex]::Match($v,'[0-9]+(\.[0-9]+){1,3}')
      if ($m.Success) { $v = $m.Value } else { $v = '0.0.0.0' }
    }
    $parts = $v.Split('.')
    while($parts.Count -lt 3){ $parts += '0' }
    if ($parts.Count -eq 3){ $parts += '0' }
    if ($parts.Count -gt 4){ $parts = $parts[0..3] }
    return ($parts -join '.')
  }
  $VersionInfoVersion = Get-CleanVersionInfo $Version
  LogDbg "VersionInfoVersion: $VersionInfoVersion"

  # 3. Localizar ISCC
  function Find-ISCC([string]$Hint){
    if ($Hint){ $h = $Hint.Trim('"'); if (Test-Path $h){ return (Resolve-Path $h).Path } LogWarn "Hint ISCC no válido: $h" }
    foreach($c in @('ISCC.exe','C:\\ProgramData\\chocolatey\\bin\\ISCC.exe',"$Env:ProgramFiles(x86)\Inno Setup 6\ISCC.exe","$Env:ProgramFiles\Inno Setup 6\ISCC.exe")){
      if (Test-Path $c){ return (Resolve-Path $c).Path }
    }
    return $null
  }
  $ISCC = Find-ISCC $InnoPath
  if (-not $ISCC) { Fail 'No se encontró ISCC.exe (instala Inno Setup 6 o usa -InnoPath)' }
  LogDbg "ISCC: $ISCC"

  # 4. Compilación (opcional)
  if (-not $SkipBuild) {
    if ($Arch -ne 'amd64'){ Fail 'Solo amd64 implementado' }
    $devScript = Join-Path (Join-Path (Join-Path $root 'Z_Iosu') 'scripts') 'dev-run.ps1'
    if (-not (Test-Path $devScript)) { Fail 'dev-run.ps1 no encontrado' }
  $devArgs = @('-GoRelease','-Clean','-BuildOnly')
    if ($DebugGo) { $devArgs = @('-Clean') }
    if ($ForceClangGnu){ $devArgs += @('-ForceClangGnu','-ResetGoEnv') }
    if ($DebugGo) { $devArgs += @('-ShowEnv') }
    LogInfo "Compilando release (dev-run.ps1)"
    LogDbg  "Args: $($devArgs -join ' ')"
    powershell -ExecutionPolicy Bypass -File $devScript @devArgs
    if ($LASTEXITCODE -ne 0){ Fail 'Fallo compilación release' }
  } else { LogInfo 'SkipBuild activo: reutilizando artefactos.' }

  # 5. Layout dist
  $dist = Join-Path $root 'dist'
  if (-not (Test-Path $dist)) { New-Item -ItemType Directory -Path $dist | Out-Null }
  $distArch = Join-Path $dist 'windows-amd64'
  if (-not (Test-Path $distArch)) { New-Item -ItemType Directory -Path $distArch | Out-Null }

  $serverExe = @('ollama-dev.exe','ollama.exe' ) | ForEach-Object { Join-Path $root $_ } | Where-Object { Test-Path $_ } | Select-Object -First 1
  if (-not $serverExe){ Fail 'No se encontró ollama-dev.exe ni ollama.exe' }
  Copy-Item $serverExe (Join-Path $distArch 'ollama.exe') -Force
  LogDbg "Servidor: $serverExe"

  # Wrapper
  $wrapperTarget = Join-Path $dist 'windows-amd64-app.exe'
  if (-not $SkipWrapper){
    if (-not (Test-Path $wrapperTarget)){
      LogInfo 'Creando wrapper'
      $wrapperDir = Join-Path (Join-Path (Join-Path $root 'Z_Iosu') 'temp_wrapper') 'app'
      if (Test-Path $wrapperDir){ Remove-Item -Recurse -Force $wrapperDir }
      New-Item -ItemType Directory -Path $wrapperDir | Out-Null
          $wrapperLines = @(
            'package main',
            'import (',
            '  "os"',
            '  "os/exec"',
            '  "fmt"',
            ')',
            'func main(){',
            '  cmd := exec.Command("ollama.exe","serve")',
            '  cmd.Stdout = os.Stdout',
            '  cmd.Stderr = os.Stderr',
            '  if err := cmd.Start(); err != nil { fmt.Println("error start:", err); return }',
            '  cmd.Wait()',
            '}'
          )
          $wrapperMain = ($wrapperLines -join [Environment]::NewLine)
          Set-Content -LiteralPath (Join-Path $wrapperDir 'main.go') -Value $wrapperMain -Encoding UTF8
      $goMod="module ollama_wrapper`n`ngo 1.21`n"
      [System.IO.File]::WriteAllText((Join-Path $wrapperDir 'go.mod'), $goMod, (New-Object System.Text.UTF8Encoding($false)))
      Push-Location $wrapperDir; go build -o $wrapperTarget .; $rc=$LASTEXITCODE; Pop-Location
      if ($rc -ne 0){ Fail 'Fallo compilación wrapper' }
    } else { LogDbg 'Wrapper existente reutilizado' }
  } else { LogWarn 'SkipWrapper activo' }

  # Copiar script welcome
  $welcome = Join-Path (Join-Path $root 'app') 'ollama_welcome.ps1'
  if (Test-Path $welcome){ Copy-Item $welcome (Join-Path $dist 'ollama_welcome.ps1') -Force }

  # 6. Preparar .iss temporal
  $issOriginal = $IssFile
  $issTemp = Join-Path $env:TEMP ('ollama_' + [guid]::NewGuid().ToString('N') + '.iss')
  $issText = Get-Content -LiteralPath $issOriginal -Raw
  if ($issText -match '(?m)^VersionInfoVersion=') {
    $issText = [regex]::Replace($issText,'(?m)^VersionInfoVersion=.*$',("VersionInfoVersion=" + $VersionInfoVersion))
  } else {
    $issText = [regex]::Replace($issText,'(?m)^(AppVersion=.*$)',("$1`r`nVersionInfoVersion=" + $VersionInfoVersion))
  }
   # Ajustar ruta de welcome script a absoluta para evitar problemas de '..\\dist' relativo al TEMP
   $welcomeAbs = Join-Path $dist 'ollama_welcome.ps1'
   if (Test-Path $welcomeAbs) {
     $escaped = [regex]::Escape('..\dist\ollama_welcome.ps1')
     $issText = [regex]::Replace($issText, $escaped, $welcomeAbs)
   }
   Set-Content -LiteralPath $issTemp -Value $issText -Encoding UTF8

  # Copiar assets junto a .iss temporal para rutas relativas
  $assetsSrc = Join-Path (Join-Path $root 'app') 'assets'
  if (Test-Path $assetsSrc){
    $assetsDst = Join-Path (Split-Path $issTemp -Parent) 'assets'
    if (Test-Path $assetsDst){ Remove-Item -Recurse -Force $assetsDst }
    Copy-Item -Recurse -Force $assetsSrc $assetsDst
    LogDbg "Assets copiados a temp"
  } else { LogWarn 'Assets no encontrados (icono / bmp pueden fallar)' }

  if (-not $Portable) {
    LogInfo 'Invocando Inno Setup (sin modo quiet para depurar)'
    & $ISCC $issTemp
    if ($LASTEXITCODE -ne 0){ Fail "Fallo ISCC (exit $LASTEXITCODE)" }
  }

  # 7. Post-proceso
  if (-not (Test-Path $OutDir)){ New-Item -ItemType Directory -Path $OutDir | Out-Null }

  if ($Portable) {
    $zipName = "Ollama-portable-$Version.zip"
    $zipPath = Join-Path $OutDir $zipName
    LogInfo "Creando paquete portable: $zipPath"
    $tempPort = Join-Path $env:TEMP ('ollama_port_' + [guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $tempPort | Out-Null
    Copy-Item (Join-Path $distArch 'ollama.exe') (Join-Path $tempPort 'ollama.exe') -Force
    if (Test-Path $wrapperTarget) { Copy-Item $wrapperTarget (Join-Path $tempPort 'ollama-app.exe') -Force }
    if (Test-Path $welcome){ Copy-Item $welcome (Join-Path $tempPort 'ollama_welcome.ps1') -Force }
    if (Test-Path (Join-Path $distArch 'lib')) { Copy-Item -Recurse -Force (Join-Path $distArch 'lib') (Join-Path $tempPort 'lib') }
    if (Test-Path (Join-Path $root 'LICENSE')) { Copy-Item (Join-Path $root 'LICENSE') (Join-Path $tempPort 'LICENSE') }
    if (Test-Path (Join-Path $root 'README.md')) { Copy-Item (Join-Path $root 'README.md') (Join-Path $tempPort 'README.md') }
    if (Test-Path $zipPath) { Remove-Item $zipPath -Force }
    Compress-Archive -Path (Join-Path $tempPort '*') -DestinationPath $zipPath
    LogInfo "Paquete portable listo: $zipPath"
  } else {
    $setup = Get-ChildItem $dist -Filter 'OllamaSetup.exe' -File -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $setup){
      $altDist = Join-Path $env:LOCALAPPDATA 'dist'
      if (Test-Path $altDist){
        $setup = Get-ChildItem $altDist -Filter 'OllamaSetup.exe' -File -ErrorAction SilentlyContinue | Select-Object -First 1
      }
    }
    if (-not $setup){ Fail 'No se generó OllamaSetup.exe' }
    $final = Join-Path $OutDir ("OllamaSetup-$Version.exe")
    Copy-Item $setup.FullName $final -Force
    LogInfo "Instalador listo: $final"

    if ($AlsoPortable -or (-not $NoAlsoPortable)) {
      $zipName = "Ollama-portable-$Version.zip"
      $zipPath = Join-Path $OutDir $zipName
      LogInfo "Creando paquete portable adicional: $zipPath"
      $tempPort = Join-Path $env:TEMP ('ollama_port_' + [guid]::NewGuid().ToString('N'))
      New-Item -ItemType Directory -Path $tempPort | Out-Null
      Copy-Item (Join-Path $distArch 'ollama.exe') (Join-Path $tempPort 'ollama.exe') -Force
      if (Test-Path $wrapperTarget) { Copy-Item $wrapperTarget (Join-Path $tempPort 'ollama-app.exe') -Force }
      if (Test-Path $welcome){ Copy-Item $welcome (Join-Path $tempPort 'ollama_welcome.ps1') -Force }
      if (Test-Path (Join-Path $distArch 'lib')) { Copy-Item -Recurse -Force (Join-Path $distArch 'lib') (Join-Path $tempPort 'lib') }
      if (Test-Path (Join-Path $root 'LICENSE')) { Copy-Item (Join-Path $root 'LICENSE') (Join-Path $tempPort 'LICENSE') }
      if (Test-Path (Join-Path $root 'README.md')) { Copy-Item (Join-Path $root 'README.md') (Join-Path $tempPort 'README.md') }
      if (Test-Path $zipPath) { Remove-Item $zipPath -Force }
      Compress-Archive -Path (Join-Path $tempPort '*') -DestinationPath $zipPath
      LogInfo "Paquete portable listo: $zipPath"
    }
  }

  Pop-Location
