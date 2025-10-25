# Quick Build Guide - Ollama Windows (test-llamacpp-bump)

## 🚀 Compilación Rápida (Un Solo Comando)

```powershell
$env:VERSION = "0.12.41"
powershell -ExecutionPolicy Bypass -File Z_Iosu\scripts\build_windows.ps1 buildCPU buildCUDA12 buildCUDA13 buildOllama buildApp buildInstaller
```

**Tiempo**: ~20 minutos  
**Resultado**: `dist\OllamaSetup.exe` (420 MB)

---

## ✅ Pre-requisitos Mínimos

1. **Visual Studio 2022 Professional**
2. **CUDA 13.0** en `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0`
3. **llvm-mingw UCRT** en `C:\llvm-mingw-ucrt\`
4. **Go** (versión según go.mod)
5. **Inno Setup 6** en `C:\Program Files (x86)\Inno Setup 6\`

---

## 📦 Instalar llvm-mingw UCRT (CRÍTICO)

```powershell
# Descargar
Invoke-WebRequest -Uri "https://github.com/mstorsjo/llvm-mingw/releases/download/20240619/llvm-mingw-20240619-ucrt-x86_64.zip" -OutFile "$env:TEMP\llvm-mingw-ucrt.zip"

# Extraer
Expand-Archive -Path "$env:TEMP\llvm-mingw-ucrt.zip" -DestinationPath "C:\llvm-mingw-ucrt" -Force

# Verificar
$llvmPath = (Resolve-Path "C:\llvm-mingw-ucrt\llvm-mingw-*").Path
& "$llvmPath\bin\gcc.exe" --version
# Debe mostrar: clang version 18.1.8
```

---

## ⚡ Optimizaciones Opcionales

### RAM Disk (acelera ~30%):
```powershell
# Crear disco R: de 20GB
imdisk -a -s 20G -m R: -p "/fs:ntfs /q /y"

# Configurar TEMP
New-Item -ItemType Directory -Path "R:\Temp" -Force
$env:TEMP = "R:\Temp"
$env:TMP = "R:\Temp"
[System.Environment]::SetEnvironmentVariable('TEMP', 'R:\Temp', 'User')
[System.Environment]::SetEnvironmentVariable('TMP', 'R:\Temp', 'User')
```

### ccache (opcional):
```powershell
choco install ccache
ccache -M 10G
```

---

## 🔧 Solo Compilar ollama.exe (Sin Instalador)

```powershell
# 1. Iniciar VS 2022
$vcvarsPath = "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvarsall.bat"
$tempFile = [System.IO.Path]::GetTempFileName()
cmd /c "`"$vcvarsPath`" amd64 && set > `"$tempFile`""
Get-Content $tempFile | ForEach-Object { 
    if ($_ -match "^(.*?)=(.*)$") { 
        Set-Content env:\"$($matches[1])" $matches[2] 
    } 
}
Remove-Item $tempFile

# 2. Configurar llvm-mingw
$llvmPath = (Resolve-Path "C:\llvm-mingw-ucrt\llvm-mingw-*").Path
$env:PATH = "$llvmPath\bin;$env:PATH"
$env:CC = "$llvmPath\bin\gcc.exe"
$env:CXX = "$llvmPath\bin\g++.exe"
$env:CGO_CFLAGS = "-I$llvmPath\include"
$env:CGO_CXXFLAGS = "-I$llvmPath\include"
$env:CGO_ENABLED = "1"

# 3. Compilar
go build -a -trimpath -ldflags "-s -w -X=github.com/ollama/ollama/version.Version=0.12.41 -X=github.com/ollama/ollama/server.mode=release" .
```

**Resultado**: `ollama.exe` (36.5 MB) en directorio actual

---

## 🧪 Probar Compilación

```powershell
# Verificar versión
.\ollama.exe --version
# Output: ollama version is 0.12.41

# Instalar
.\dist\OllamaSetup.exe

# Probar servidor
ollama serve

# (En otra terminal)
ollama run llama3.2
```

---

## ❌ Errores Comunes

### Error: `stdlib.h file not found`
**Solución**: Falta configurar `CGO_CFLAGS`
```powershell
$llvmPath = (Resolve-Path "C:\llvm-mingw-ucrt\llvm-mingw-*").Path
$env:CGO_CFLAGS = "-I$llvmPath\include"
$env:CGO_CXXFLAGS = "-I$llvmPath\include"
```

### Error: `undefined reference to __stdio_common_vfprintf`
**Solución**: Estás usando MSYS2 MinGW en lugar de llvm-mingw UCRT
```powershell
# Verificar que gcc es clang:
gcc --version
# Debe mostrar "clang version 18.1.8"
# Si muestra "gcc (GCC)", estás usando el compilador equivocado
```

### Error: `CUDA not compatible with MSVC 19.50`
**Solución**: Estás usando VS 2026 Insiders en lugar de VS 2022
```powershell
# Verificar versión de MSVC
cl.exe 2>&1 | Select-String "Version"
# Debe ser 19.44.x, NO 19.50.x
```

---

## 📁 Estructura de Archivos Generados

```
dist/
├── OllamaSetup.exe                    (420 MB - Instalador completo)
├── windows-amd64-app.exe              (6.81 MB - App de bandeja)
├── ollama_welcome.ps1
└── windows-amd64/
    ├── ollama.exe                     (36.5 MB - CLI principal)
    └── lib/ollama/
        ├── ggml-base.dll
        ├── ggml-cpu-*.dll             (8 variantes CPU)
        └── cuda_v13/
            ├── ggml-cuda.dll
            ├── cublas64_13.dll
            └── cublasLt64_13.dll
```

---

## 🎯 Comandos Útiles

```powershell
# Limpiar build anterior
Remove-Item -Recurse -Force "build","dist" -ErrorAction SilentlyContinue
go clean -cache

# Ver tamaño de archivos compilados
Get-ChildItem "dist" -Recurse -Include "*.exe","*.dll" | 
    Select-Object Name, @{N="Size (MB)";E={[math]::Round($_.Length/1MB, 2)}} | 
    Sort-Object "Size (MB)" -Descending

# Verificar dependencias DLL
dumpbin /dependents dist\windows-amd64\ollama.exe

# Ver commits de llama.cpp bump
git log --oneline 364a7a6d..1deee0f8 llama/llama.cpp
```

---

## 📚 Más Información

Ver documentación completa en: `Z_Iosu/OJO.md`
