<#!
.SYNOPSIS
  Script de conveniencia para ejecutar `go run . serve` en Windows asegurando toolchain C/C++ correcto.

.DESCRIPTION
  Detecta si está disponible MSVC (cl.exe) o clang-cl. Configura variables de entorno CC/CXX para Go.
  Permite limpiar caché y forzar regeneración. Opción para usar build cmake previa o solo go.

.PARAMETER Clean
  Limpia caché de compilación de Go y elimina carpeta build opcionalmente.

.PARAMETER UseCMake
  Intenta configurar (si no existe) y construir con CMake antes de ejecutar.

.PARAMETER Release
  Compila CMake en configuración Release (por defecto si UseCMake) en vez de Debug.

.PARAMETER Port
  Puerto en el que servir (default 11434). Se exporta como OLLAMA_HOST=127.0.0.1:PORT.

.PARAMETER ShowEnv
  Muestra variables de entorno CC/CXX elegidas.

.PARAMETER DryRun
  Solo muestra lo que haría.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File Z_Iosu\scripts\dev-run.ps1 -Clean -UseCMake -Release -ShowEnv

#>
[CmdletBinding()]
param(
  [switch]$Clean,
  [switch]$UseCMake,
  [switch]$Release,
  [int]$Port = 11434,
  [switch]$ShowEnv,
  [switch]$DryRun,
  [switch]$ResetGoEnv,
  [switch]$PreferClangCL,
  [switch]$ForceClangGnu,
  [switch]$GoRelease,
  [switch]$BuildOnly
)

$ErrorActionPreference = 'Stop'

function Find-Exe($names) {
  foreach ($n in $names) {
    $p = (Get-Command $n -ErrorAction SilentlyContinue | Select-Object -First 1).Path
    if ($p) { return $p }
  }
  return $null
}

Write-Host "[dev-run] Detectando toolchain..." -ForegroundColor Cyan

# Asegurar que el directorio de configuración de Go existe
$goEnvDir = "$env:APPDATA\go"
if (-not (Test-Path $goEnvDir)) {
  Write-Host "[dev-run] Creando directorio de configuración Go: $goEnvDir" -ForegroundColor Cyan
  New-Item -ItemType Directory -Path $goEnvDir -Force | Out-Null
}

$goCC = ''
$goCXX = ''
try {
  $goCC = (go env CC) 2>$null
  $goCXX = (go env CXX) 2>$null
} catch {
  Write-Host "[dev-run] Advertencia: Error leyendo configuración Go, continuando..." -ForegroundColor Yellow
}

if ($ResetGoEnv) {
  Write-Host '[dev-run] Limpiando CC/CXX persistentes en go env' -ForegroundColor Cyan
  if (-not $DryRun) {
    try {
      go env -u CC 2>$null | Out-Null
      go env -u CXX 2>$null | Out-Null
    } catch {
      Write-Host "[dev-run] Advertencia: Error limpiando variables Go env, continuando..." -ForegroundColor Yellow
    }
  }
  $goCC = ''
  $goCXX = ''
}
$cl = Find-Exe @('cl.exe')
$clangcl = Find-Exe @('clang-cl.exe','clang-cl')
$clangGnu = Find-Exe @('clang.exe')
$clangTriple = Find-Exe @('x86_64-w64-mingw32-clang.exe','x86_64-w64-mingw32-clang','llvm-mingw-clang.exe')
$gccLike = Find-Exe @('x86_64-w64-mingw32-gcc.exe','x86_64-w64-mingw32-gcc','gcc.exe','gcc')
$chosen = $null
if ($cl) { $chosen = 'cl' }
elseif ($clangcl) { $chosen = 'clang-cl' }
else {
  Write-Warning 'No se encontró cl.exe ni clang-cl.exe. Intenta instalar Visual Studio (Desktop C++) o LLVM con clang-cl.'
}

if ($ForceClangGnu) {
  $goArch = (go env GOARCH) 2>$null
  $ccPath = $null
  if ($goArch -eq 'amd64' -and $clangTriple) { $ccPath = $clangTriple }
  elseif ($clangGnu) { $ccPath = $clangGnu }
  elseif ($clangTriple) { $ccPath = $clangTriple } # por si GOARCH no es amd64
  elseif ($gccLike) { $ccPath = $gccLike }
  if (-not $ccPath) {
  Write-Warning 'No se encontró clang (gnu) ni gcc en PATH pese a usar -ForceClangGnu.'
    Write-Host 'Instala llvm-mingw y añade su carpeta bin al PATH. Ejemplo:' -ForegroundColor Cyan
    Write-Host '  C:\llvm-mingw\bin (ver https://github.com/mstorsjo/llvm-mingw/releases)' -ForegroundColor DarkCyan
    Write-Host 'Luego reintenta:  .\Z_Iosu\scripts\dev-run.ps1 -ForceClangGnu -ResetGoEnv -Clean -ShowEnv' -ForegroundColor Cyan
    exit 1
  }
  $chosen = 'clang'
  if ($ccPath -match 'gcc') { $chosen = 'gcc' }
  # Citar rutas que contienen espacios para evitar problemas con Go
  if ($ccPath -match '\s') { $env:CC = "`"$ccPath`"" } else { $env:CC = $ccPath }
  # Derivar CXX desde el mismo prefijo
  if ($ccPath -match 'x86_64-w64-mingw32-clang') {
    $cxxPath = ($ccPath -replace 'clang.exe','clang++.exe')
    if ($cxxPath -match '\s') { $env:CXX = "`"$cxxPath`"" } else { $env:CXX = $cxxPath }
  } elseif ($ccPath -match 'clang.exe$') {
    $cxxPath = $ccPath -replace 'clang.exe','clang++.exe'
    if (Test-Path $cxxPath) { 
      if ($cxxPath -match '\s') { $env:CXX = "`"$cxxPath`"" } else { $env:CXX = $cxxPath }
    } else { 
      if ($ccPath -match '\s') { $env:CXX = "`"$ccPath`"" } else { $env:CXX = $ccPath }
    }
  } elseif ($ccPath -match 'gcc(.exe)?$') {
    $cxxPath = $ccPath -replace 'gcc.exe','g++.exe' -replace 'gcc$','g++'
    if (Test-Path $cxxPath) { 
      if ($cxxPath -match '\s') { $env:CXX = "`"$cxxPath`"" } else { $env:CXX = $cxxPath }
    } else { 
      if ($ccPath -match '\s') { $env:CXX = "`"$ccPath`"" } else { $env:CXX = $ccPath }
    }
  }
  if ($ShowEnv) {
    $archMsg = if ($goArch) { " GOARCH=$goArch" } else { '' }
    Write-Host "[dev-run] Forzando clang (driver gnu)$archMsg -> $ccPath" -ForegroundColor Yellow
  }
} elseif ($chosen) {
  if ($PreferClangCL -and $clangcl) { $chosen = 'clang-cl' }
  elseif ($clangcl -and ($goCC -match 'clang.exe' -or $goCXX -match 'clang\+\+\.exe')) {
    Write-Host '[dev-run] Reemplazando clang.exe por clang-cl para flags MSVC' -ForegroundColor Cyan
    $chosen = 'clang-cl'
  }
  $env:CC = $chosen
  $env:CXX = $chosen
  if ($ShowEnv) { Write-Host "[dev-run] Toolchain elegido: $chosen" -ForegroundColor Yellow }
}

# Exportar puerto (formato que Ollama espera: host:port) si aplica
$env:OLLAMA_HOST = "127.0.0.1:$Port"

if ($ShowEnv) {
  Write-Host "[dev-run] CC=$($env:CC) CXX=$($env:CXX)" -ForegroundColor Yellow
  Write-Host "[dev-run] OLLAMA_HOST=$($env:OLLAMA_HOST)" -ForegroundColor Yellow
}

# Aviso: cgo en Windows oficialmente soporta toolchains estilo GCC/MinGW (gcc, clang/llvm-mingw), no MSVC cl.exe.
if ($env:CC -match '(?i)\\?cl(.exe)?$' -and -not $ForceClangGnu) {
  Write-Warning 'Se ha seleccionado cl.exe (MSVC). cgo en Windows usa flags GCC (-Wall -Werror) que MSVC no entiende y fallará (D8021).'
  Write-Host '[dev-run] Instala y prioriza un toolchain MinGW o llvm-mingw en PATH. Ejemplos:' -ForegroundColor Cyan
  Write-Host '          * TDM-GCC:    C:\tdm-gcc\bin añadido a PATH' -ForegroundColor DarkCyan
  Write-Host '          * llvm-mingw: C:\llvm-mingw\bin añadido a PATH' -ForegroundColor DarkCyan
  Write-Host '          Luego reintenta:  .\Z_Iosu\scripts\dev-run.ps1 -ForceClangGnu -ResetGoEnv -Clean -ShowEnv' -ForegroundColor Cyan
  Write-Host '          (O set CC=gcc && set CXX=g++)' -ForegroundColor Cyan
  Write-Host '[dev-run] Abortando para evitar error repetido.' -ForegroundColor Yellow
  exit 1
}

if ($Clean) {
  Write-Host '[dev-run] Limpiando caché Go...' -ForegroundColor Cyan
  if (-not $DryRun) { go clean -cache | Out-Null }
  if (Test-Path build) {
    Write-Host '[dev-run] Eliminando carpeta build/ ...' -ForegroundColor Cyan
    if (-not $DryRun) { Remove-Item -Recurse -Force build }
  }
}

if ($UseCMake) {
  if (-not (Test-Path build)) {
    Write-Host '[dev-run] Configurando CMake...' -ForegroundColor Cyan
    $cfgCmd = 'cmake -B build'
    if ($chosen -eq 'clang-cl') { $cfgCmd += ' -T ClangCL' }
    if ($DryRun) { Write-Host "DRY: $cfgCmd" } else { Invoke-Expression $cfgCmd }
  }
  Write-Host '[dev-run] Compilando CMake...' -ForegroundColor Cyan
  $conf = if ($Release) { 'Release' } else { 'Debug' }
  $buildCmd = "cmake --build build --config $conf"
  if ($DryRun) { Write-Host "DRY: $buildCmd" } else { Invoke-Expression $buildCmd }
}

$cgoFlagsOriginal = $env:CGO_CFLAGS
if ($cgoFlagsOriginal) {
  # Eliminar variantes problemáticas para cl.exe: /Werror y -Werror
  $sanitized = ($cgoFlagsOriginal -replace '/Werror','' -replace '-Werror','')
  if ($sanitized -ne $cgoFlagsOriginal) {
    Write-Host '[dev-run] Sanitizando CGO_CFLAGS removiendo Werror para MSVC' -ForegroundColor Cyan
    $env:CGO_CFLAGS = $sanitized.Trim()
  }
}

$env:CGO_ENABLED = 1

# Sanitizar variable de entorno CL (MSVC la usa para flags globales) si inyecta /Werror o -Werror
if ($env:CL) {
  if ($env:CL -match '/Werror' -or $env:CL -match '-Werror') {
    Write-Host '[dev-run] Sanitizando variable CL removiendo /Werror/-Werror' -ForegroundColor Cyan
    $env:CL = ($env:CL -replace '/Werror','' -replace '-Werror','').Trim()
  }
}

# Traducción de flags MSVC -> GCC para clang gnu (llvm-mingw)
if ($ForceClangGnu -and $chosen -eq 'clang') {
  $translated = $false
  foreach ($var in 'CGO_CXXFLAGS','CGO_CFLAGS') {
    $val = Get-Item Env:$var -ErrorAction SilentlyContinue | ForEach-Object { $_.Value }
    if ($val) {
      $orig = $val
      # Reemplazar /std:c++17 por -std=c++17
      $val = $val -replace '/std:c\+\+17','-std=c++17'
      # Quitar /EHsc (no lo entiende driver gnu y no es necesario con clang)
      $val = ($val -split '\s+') | Where-Object { $_ -ne '/EHsc' -and $_ -ne '/EHs' } | ForEach-Object { $_ } | Sort-Object -Unique -CaseSensitive | ForEach-Object { $_ } -join ' '
      # Eliminar duplicados triviales y limpiar espacios
      $val = ($val -replace '\s+',' ').Trim()
      if ($val -ne $orig) {
        Set-Item -Path "Env:$var" -Value $val
  Write-Host "[dev-run] Traduciendo flags ${var}: '$orig' -> '$val'" -ForegroundColor Cyan
        $translated = $true
      }
    }
  }
  # Si no había CGO_CXXFLAGS y no se detecta -std= añadirlo explícitamente
  if (-not $env:CGO_CXXFLAGS -or ($env:CGO_CXXFLAGS -notmatch '-std=c\+\+')) {
    $env:CGO_CXXFLAGS = ("$($env:CGO_CXXFLAGS) -std=c++17" -replace '\s+',' ').Trim()
    Write-Host "[dev-run] Asegurando -std=c++17 en CGO_CXXFLAGS => '$($env:CGO_CXXFLAGS)'" -ForegroundColor Cyan
  }
  if ($ShowEnv -and $translated) {
    Write-Host "[dev-run] Después de traducción: CGO_CFLAGS='$($env:CGO_CFLAGS)' CGO_CXXFLAGS='$($env:CGO_CXXFLAGS)'" -ForegroundColor Yellow
  }
}

# Inyección de flags extra para clang gnu en modo release u optimizado
if ($ForceClangGnu) {
  if ($chosen -eq 'clang') {
    $tFlags = '--target=x86_64-w64-windows-gnu -fuse-ld=lld'
    foreach ($pair in @('CGO_CFLAGS','CGO_CXXFLAGS','CGO_LDFLAGS')) {
      $cur = (Get-Item Env:$pair -ErrorAction SilentlyContinue | ForEach-Object { $_.Value })
      if (-not $cur) { $cur = '' }
      if ($cur -notmatch '--target=x86_64-w64-windows-gnu') { $cur = "$tFlags $cur" }
      if ($GoRelease -and $pair -ne 'CGO_LDFLAGS') {
        if ($cur -notmatch '-O3') { $cur = "-O3 -DNDEBUG $cur" }
      }
      $cur = ($cur -replace '\s+',' ').Trim()
      Set-Item -Path "Env:$pair" -Value $cur
    }
  } elseif ($chosen -eq 'gcc') {
    # Para GCC mingw: no usar --target ni -fuse-ld=lld (no válidos). Sólo optimizaciones y std si aplica.
    foreach ($pair in @('CGO_CFLAGS','CGO_CXXFLAGS')) {
      $cur = (Get-Item Env:$pair -ErrorAction SilentlyContinue | ForEach-Object { $_.Value })
      if (-not $cur) { $cur = '' }
      if ($GoRelease -and $cur -notmatch '-O3') { $cur = "-O3 -DNDEBUG $cur" }
      if ($pair -eq 'CGO_CXXFLAGS' -and ($cur -notmatch '-std=c\+\+')) { $cur = ("$cur -std=c++17" -replace '\s+',' ').Trim() }
      $cur = ($cur -replace '\s+',' ').Trim()
      Set-Item -Path "Env:$pair" -Value $cur
    }
    # CGO_LDFLAGS: dejar vacío o respetar existente para mingw-gcc
  }
  if ($ShowEnv) {
    Write-Host "[dev-run] Flags finales: CGO_CFLAGS='$($env:CGO_CFLAGS)'" -ForegroundColor Yellow
    Write-Host "[dev-run]              CGO_CXXFLAGS='$($env:CGO_CXXFLAGS)'" -ForegroundColor Yellow
    Write-Host "[dev-run]              CGO_LDFLAGS='$($env:CGO_LDFLAGS)'" -ForegroundColor Yellow
  }
}

$runCmd = 'go run . serve'
if ($GoRelease) {
  # Construimos binario optimizado (similar a Release) y luego ejecutamos
  $exe = 'ollama-dev.exe'
  $buildCmd = "go build -trimpath -ldflags '-s -w' -o $exe ."
  Write-Host "[dev-run] Compilando binario Release Go: $buildCmd" -ForegroundColor Cyan
  if (-not $DryRun) { & go build -trimpath -ldflags '-s -w' -o $exe .; $rc=$LASTEXITCODE; if ($rc -ne 0) { Write-Error "go build falló con código $rc"; exit $rc } }
  $runCmd = "./$exe serve"
  if ($BuildOnly) {
    Write-Host '[dev-run] BuildOnly activo: finalizando sin ejecutar el servidor' -ForegroundColor Yellow
    exit 0
  }
}
Write-Host "[dev-run] Ejecutando: $runCmd" -ForegroundColor Green
if ($DryRun) {
  Write-Host 'DRY: (no se ejecuta go run)'
  exit 0
}

# Ejecutar servidor según modo
if ($GoRelease) {
  & .\ollama-dev.exe serve
} else {
  & go run . serve
}
