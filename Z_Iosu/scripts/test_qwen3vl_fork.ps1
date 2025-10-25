# ============================================================================
# SCRIPT: Probar Fork LETS-BEE/llama.cpp Branch qwen3vl
# ============================================================================
# Proposito: Compilar y probar el soporte Qwen3-VL en entorno aislado
# Sin afectar tu instalacion actual de Ollama 0.12.6.99
# ============================================================================

param(
    [switch]$SkipClone,
    [switch]$OnlyCPU,
    [switch]$IncludeCUDA,
    [switch]$IncludeVulkan,
    [switch]$Verbose
)

$ErrorActionPreference = "Stop"

# ============================================================================
# CONFIGURACION
# ============================================================================

$TestDir = "C:\IA\tools\ollama-qwen3vl-test"
$RepoUrl = "https://github.com/LETS-BEE/llama.cpp.git"
$Branch = "qwen3vl"

function Write-ColorOutput {
    param([string]$Message, [string]$Color = "White")
    Write-Host $Message -ForegroundColor $Color
}

# ============================================================================
# PASO 1: CLONAR REPOSITORIO
# ============================================================================

Write-ColorOutput "`n[PASO 1] Preparando entorno de prueba..." "Cyan"

if (-not $SkipClone) {
    if (Test-Path $TestDir) {
        Write-ColorOutput "Directorio $TestDir ya existe." "Yellow"
        $response = Read-Host "Eliminar y clonar de nuevo? (S/N)"
        if ($response -eq 'S') {
            Remove-Item -Path $TestDir -Recurse -Force
        } else {
            Write-ColorOutput "Usando directorio existente" "Green"
            Set-Location $TestDir
            $SkipClone = $true
        }
    }
    
    if (-not $SkipClone) {
        Write-ColorOutput "Clonando LETS-BEE/llama.cpp branch qwen3vl..." "Yellow"
        git clone --branch $Branch --depth 1 $RepoUrl $TestDir
        
        if ($LASTEXITCODE -ne 0) {
            Write-ColorOutput "Error clonando repositorio" "Red"
            exit 1
        }
        
        Set-Location $TestDir
        Write-ColorOutput "Repositorio clonado exitosamente" "Green"
    }
} else {
    if (-not (Test-Path $TestDir)) {
        Write-ColorOutput "Directorio $TestDir no existe. Elimina -SkipClone" "Red"
        exit 1
    }
    Set-Location $TestDir
}

$currentBranch = git branch --show-current
Write-ColorOutput "Branch actual: $currentBranch" "Cyan"

# ============================================================================
# PASO 2: ANALIZAR CAMBIOS
# ============================================================================

Write-ColorOutput "`n[PASO 2] Analizando cambios especificos de Qwen3-VL..." "Cyan"

$qwenFiles = Get-ChildItem -Recurse -Filter "*qwen*" -ErrorAction SilentlyContinue

if ($qwenFiles.Count -gt 0) {
    Write-ColorOutput "Archivos Qwen3-VL encontrados:" "Green"
    foreach ($file in $qwenFiles | Select-Object -First 10) {
        Write-Host "   - $($file.FullName.Replace($TestDir, '.'))"
    }
    if ($qwenFiles.Count -gt 10) {
        Write-Host "   ... y $($qwenFiles.Count - 10) mas"
    }
} else {
    Write-ColorOutput "No se encontraron archivos especificos de Qwen3-VL" "Yellow"
}

Write-ColorOutput "`nUltimos 5 commits del branch:" "Cyan"
git log --oneline -5

# ============================================================================
# PASO 3: GENERAR DIFF
# ============================================================================

Write-ColorOutput "`n[PASO 3] Generando diff con llama.cpp oficial..." "Cyan"

$hasRemote = git remote | Select-String "ggml-org"
if (-not $hasRemote) {
    Write-ColorOutput "Agregando remoto ggml-org..." "Yellow"
    git remote add ggml-org https://github.com/ggml-org/llama.cpp.git
    if ($LASTEXITCODE -ne 0) {
        Write-ColorOutput "Error agregando remoto ggml-org" "Red"
        Write-ColorOutput "Continuando sin generar diff..." "Yellow"
        $skipDiff = $true
    }
} else {
    Write-ColorOutput "Remoto ggml-org ya existe" "Green"
}

if (-not $skipDiff) {
    Write-ColorOutput "Obteniendo informacion de ggml-org..." "Yellow"
    git fetch ggml-org master --depth 1
    
    if ($LASTEXITCODE -ne 0) {
        Write-ColorOutput "Advertencia: No se pudo obtener informacion de ggml-org" "Yellow"
        Write-ColorOutput "Continuando sin generar diff..." "Yellow"
        $skipDiff = $true
    }
}

$patchFile = "$TestDir\qwen3vl-changes.patch"

if (-not $skipDiff) {
    Write-ColorOutput "Generando patch de cambios..." "Yellow"
    git diff ggml-org/master...HEAD > $patchFile
}

if (Test-Path $patchFile) {
    $patchSize = (Get-Item $patchFile).Length
    if ($patchSize -gt 0) {
        Write-ColorOutput "Patch generado: $patchFile ($([math]::Round($patchSize/1KB, 2)) KB)" "Green"
        
        $patchContent = Get-Content $patchFile -Raw
        $filesChanged = ($patchContent | Select-String "diff --git" -AllMatches).Matches.Count
        
        Write-ColorOutput "`nEstadisticas del fork:" "Cyan"
        Write-Host "   - Archivos modificados: $filesChanged"
        
        $qwenMentions = ($patchContent | Select-String "qwen" -AllMatches -CaseSensitive:$false).Matches.Count
        if ($qwenMentions -gt 0) {
            Write-ColorOutput "   - Referencias a 'qwen': $qwenMentions" "Yellow"
        }
    } else {
        Write-ColorOutput "Patch vacio o no se pudo generar" "Yellow"
        $filesChanged = "N/A"
        $qwenMentions = "N/A"
    }
} else {
    Write-ColorOutput "No se genero archivo de patch (continuando de todas formas)" "Yellow"
    $filesChanged = "N/A"
    $qwenMentions = "N/A"
}

# ============================================================================
# PASO 4: VERIFICAR REQUISITOS
# ============================================================================

Write-ColorOutput "`n[PASO 4] Verificando requisitos de compilacion..." "Cyan"

try {
    $cmakeVersion = cmake --version | Select-Object -First 1
    Write-ColorOutput "OK: $cmakeVersion" "Green"
} catch {
    Write-ColorOutput "CMake no encontrado" "Red"
    exit 1
}

$vsPath = "${env:ProgramFiles}\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvarsall.bat"
if (Test-Path $vsPath) {
    Write-ColorOutput "OK: Visual Studio 2022 Professional" "Green"
} else {
    Write-ColorOutput "Visual Studio 2022 no encontrado" "Red"
    exit 1
}

if ($IncludeCUDA) {
    if ($env:CUDA_PATH_V13_0) {
        Write-ColorOutput "OK: CUDA 13.0 - $env:CUDA_PATH_V13_0" "Green"
    } else {
        Write-ColorOutput "CUDA_PATH_V13_0 no configurado, compilando solo CPU" "Yellow"
        $IncludeCUDA = $false
        $OnlyCPU = $true
    }
}

# ============================================================================
# PASO 5: COMPILAR
# ============================================================================

Write-ColorOutput "`n[PASO 5] Compilando llama.cpp con soporte Qwen3-VL..." "Cyan"

$buildDir = "$TestDir\build"
if (Test-Path $buildDir) {
    Remove-Item -Path $buildDir -Recurse -Force
}
New-Item -ItemType Directory -Path $buildDir | Out-Null

Set-Location $buildDir

Write-ColorOutput "Configurando CMake..." "Yellow"

$cmakeArgs = @(
    ".."
    "-G", "Visual Studio 17 2022"
    "-A", "x64"
    "-DLLAMA_CURL=OFF"
)

if ($OnlyCPU) {
    Write-ColorOutput "   Modo: Solo CPU (compilacion rapida)" "Cyan"
    $cmakeArgs += "-DGGML_CUDA=OFF"
    $cmakeArgs += "-DGGML_VULKAN=OFF"
} else {
    if ($IncludeCUDA) {
        Write-ColorOutput "   Habilitando soporte CUDA 13.0" "Cyan"
        $cmakeArgs += "-DGGML_CUDA=ON"
    } else {
        $cmakeArgs += "-DGGML_CUDA=OFF"
    }
    if ($IncludeVulkan) {
        Write-ColorOutput "   Habilitando soporte Vulkan" "Cyan"
        $cmakeArgs += "-DGGML_VULKAN=ON"
    } else {
        $cmakeArgs += "-DGGML_VULKAN=OFF"
    }
}

& cmake @cmakeArgs

if ($LASTEXITCODE -ne 0) {
    Write-ColorOutput "Error en configuracion CMake" "Red"
    exit 1
}

Write-ColorOutput "`nCompilando... (esto puede tardar 5-10 minutos)" "Yellow"
cmake --build . --config Release -j $env:NUMBER_OF_PROCESSORS

if ($LASTEXITCODE -ne 0) {
    Write-ColorOutput "Error en compilacion" "Red"
    exit 1
}

Write-ColorOutput "Compilacion exitosa" "Green"

# ============================================================================
# PASO 6: VERIFICAR BINARIOS
# ============================================================================

Write-ColorOutput "`n[PASO 6] Verificando binarios generados..." "Cyan"

$binDir = "$buildDir\bin\Release"
if (Test-Path $binDir) {
    $binaries = Get-ChildItem $binDir -Filter "*.exe"
    
    Write-ColorOutput "Binarios encontrados en $binDir :" "Green"
    foreach ($bin in $binaries) {
        $sizeMB = [math]::Round($bin.Length / 1MB, 2)
        Write-Host "   - $($bin.Name) ($sizeMB MB)"
    }
    
    $llamaCli = Get-ChildItem $binDir -Filter "llama-cli.exe" -ErrorAction SilentlyContinue
    if (-not $llamaCli) {
        $llamaCli = Get-ChildItem $binDir -Filter "main.exe" -ErrorAction SilentlyContinue
    }
    
    if ($llamaCli) {
        Write-ColorOutput "`nEjecutable principal: $($llamaCli.Name)" "Green"
        
        Write-ColorOutput "`nProbando ejecutable..." "Yellow"
        & $llamaCli.FullName --version
        
        if ($LASTEXITCODE -eq 0) {
            Write-ColorOutput "Ejecutable funciona correctamente" "Green"
        }
    }
} else {
    Write-ColorOutput "Directorio de binarios no encontrado" "Red"
    exit 1
}

# ============================================================================
# PASO 7: GENERAR REPORTE
# ============================================================================

Write-ColorOutput "`n[PASO 7] Generando reporte de compilacion..." "Cyan"

$reportFile = "$TestDir\QWEN3VL_TEST_REPORT.md"

$lastCommit = git log -1 --oneline

@"
# Reporte de Prueba: LETS-BEE/llama.cpp Branch qwen3vl

**Fecha:** $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")
**Branch:** $Branch
**Directorio:** $TestDir

---

## Informacion del Fork

**Ultimo commit:**
``````
$lastCommit
``````

**Archivos modificados respecto a ggml-org/master:**
``````
$filesChanged archivos
``````

**Referencias a 'qwen' en el codigo:**
``````
$qwenMentions menciones
``````

---

## Configuracion de Compilacion

- **Solo CPU:** $OnlyCPU
- **CUDA:** $IncludeCUDA
- **Vulkan:** $IncludeVulkan
- **Compilador:** Visual Studio 2022 Professional

---

## Binarios Generados

**Ubicacion:** $binDir

**Archivos:**
$(foreach ($bin in $binaries) { "- $($bin.Name) ($([math]::Round($bin.Length/1MB, 2)) MB)" } -join "`n")

---

## Proximos Pasos

### 1. Probar con Modelo Qwen3-VL

Descarga un modelo Qwen3-VL compatible de HuggingFace

Ejemplo de uso:
``````powershell
cd "$binDir"
.\llama-cli.exe -m "ruta/al/modelo.gguf" -p "Describe esta imagen:" --image "ruta/imagen.jpg"
``````

### 2. Comparar con tu Ollama Actual

**Tu version actual:**
- Ollama 0.12.6.99
- llama.cpp commit 1deee0f8
- Branch: 12-07-b5

**Esta version de prueba:**
- LETS-BEE fork
- Branch: qwen3vl
- Commits del 23 Oct 2025

---

## Archivos Generados

- **Patch:** $patchFile
- **Binarios:** $binDir
- **Reporte:** $reportFile

**Este entorno es completamente independiente de tu instalacion Ollama actual**

---

*Generado automaticamente por test_qwen3vl_fork.ps1*
"@ | Out-File -FilePath $reportFile -Encoding UTF8

Write-ColorOutput "Reporte guardado en: $reportFile" "Green"

# ============================================================================
# RESUMEN FINAL
# ============================================================================

Write-ColorOutput "`n" "White"
Write-ColorOutput "========================================" "Green"
Write-ColorOutput "  COMPILACION DE PRUEBA COMPLETADA     " "Green"
Write-ColorOutput "========================================" "Green"

Write-ColorOutput "`nDirectorio de prueba:" "Cyan"
Write-Host "   $TestDir"

Write-ColorOutput "`nArchivos importantes:" "Cyan"
Write-Host "   - Patch: $patchFile"
Write-Host "   - Reporte: $reportFile"
Write-Host "   - Binarios: $binDir"

Write-ColorOutput "`nProximos pasos:" "Yellow"
Write-Host "   1. Revisar el reporte: notepad `"$reportFile`""
Write-Host "   2. Probar con un modelo Qwen3-VL pequeno"
Write-Host "   3. Comparar resultados con tu Ollama actual"
Write-Host "   4. Decidir si integrar cambios a tu branch principal"

Write-ColorOutput "`nIMPORTANTE:" "Yellow"
Write-Host "   - Esta compilacion NO afecta tu Ollama 0.12.6.99 instalado"
Write-Host "   - Los binarios estan en: $binDir"
Write-Host "   - Tu instalacion actual sigue funcionando normalmente"

Write-ColorOutput "`nPara limpiar despues de probar:" "Gray"
Write-Host "   Remove-Item -Path `"$TestDir`" -Recurse -Force"

Write-ColorOutput "`nListo para probar Qwen3-VL!" "Green"
