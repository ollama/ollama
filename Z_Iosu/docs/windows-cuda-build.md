# Build Windows + CUDA (Local)

Objetivo: generar `ollama.exe` enlazando librerías CUDA ya instaladas en el sistema.

## Prerrequisitos
- Visual Studio 2022 (Desktop C++)
- CUDA Toolkit (coincidir con preset, ej. 12.x)
- CMake y Ninja (opcional pero recomendado)
- Go 1.24+

## Script Rápido
```powershell
pwsh Z_Iosu/scripts/build-win-cuda.ps1 -Preset "CUDA 12"
```

Opciones:
- `-Clean` elimina carpeta `build/` antes de configurar.
- `-SkipCMake` salta la parte CMake (solo recompila Go).
- `-SkipGo` solo reconstruye librerías nativas.
- `-OutDir` destino final del binario copiado.

## Manual (equivalente)
```powershell
cmake --preset "CUDA 12"
cmake --build --preset "CUDA 12"
cmake --install build --component CUDA
$env:CGO_ENABLED=1
go build -o ollama.exe .
```

## Output
- Binario principal: `ollama.exe`
- Librerías nativas: en `dist/lib/ollama` (tras el install CMake)

## Verificación rápida
```powershell
./ollama.exe serve
curl http://localhost:11434/api/tags
```

## Problemas comunes
- Falta nvml.dll: revisar PATH de CUDA `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vXX\bin`.
- Error link: asegurar que VS Dev Shell está cargado (o abrir terminal de VS x64 Native Tools).
