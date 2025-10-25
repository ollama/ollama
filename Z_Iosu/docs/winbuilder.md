Winbuilder (cross build Windows dentro de contenedor Linux)
==========================================================

Este flujo permite generar binarios `ollama.exe`, opcionalmente `windows-<arch>-app.exe` (GUI) y el instalador Inno Setup sin instalar toolchains Windows en tu máquina host. Ahora también puede compilar las librerías CPU (ggml) con `--cpu-libs`. No incluye CUDA/ROCm (solo CPU). Para CUDA sigue usando el flujo nativo Windows (`build-win-cuda.ps1`).

Componentes
-----------
- `docker/Dockerfile.winbuilder`: Imagen base (Ubuntu) con:
  - Go
  - mingw-w64 (gcc/g++) para amd64 y arm64
  - clang, cmake, ninja (reservado para futuras libs)
  - wine + Inno Setup extraído (ISCC.exe)
  - ccache opcional
- `scripts/build-win-container.ps1`: Wrapper que construye/actualiza la imagen y lanza el build.
- `docker/winbuilder/build-win.sh`: Script dentro del contenedor.

Novedades recientes:
- `--cpu-libs` compila ggml (CPU) y las coloca en `dist/lib/ollama`.
- `--no-app` omite la compilación de la GUI (`app/`).
- Se genera `windows-<arch>-app.exe` real (flags `-H windowsgui`) si no se especifica `--no-app`.

Uso rápido
----------
PowerShell (host):
```
# Binario amd64 + app GUI + (sin libs C)
powershell -ExecutionPolicy Bypass -File Z_Iosu/scripts/build-win-container.ps1 -Version 0.1.0

# Con installer
powershell -ExecutionPolicy Bypass -File Z_Iosu/scripts/build-win-container.ps1 -Version 0.1.0 -Installer

# Con libs CPU
powershell -ExecutionPolicy Bypass -File Z_Iosu/scripts/build-win-container.ps1 -Version 0.1.0 -CpuLibs

# ARM64
powershell -ExecutionPolicy Bypass -File Z_Iosu/scripts/build-win-container.ps1 -Version 0.1.0 -Arch arm64
```
Artefactos en `dist/`:
- `dist/ollama.exe`
- `dist/windows-<arch>-app.exe` (si se compila la app GUI)
- `dist/lib/ollama/*` (si `--cpu-libs`)
- `dist/windows-<arch>/` (layout para installer cuando se usa -Installer)
- `dist/OllamaSetup.exe` (si se genera installer)

Flags wrapper PowerShell
------------------------
- `-Version`: Define `PKG_VERSION` usado por Inno Setup.
- `-Arch`: `amd64` (default) o `arm64`.
- `-Installer`: Genera instalador (usa wine).
- `-RebuildImage`: Fuerza reconstrucción de la imagen.
- `-NoCache`: Construye imagen sin cache Docker.
- `-CCache`: Activa ccache (`.ccache` en repo montado).
- `-GoFlags`: Passthrough tras `--` hacia `go build`.
- `-CpuLibs`: Compila libs CPU (equivalente a componente CPU del CMake install).
- `-NoApp`: Omite build de la app GUI (solo servidor/CLI).

Ejemplo flags Go:
```
powershell -ExecutionPolicy Bypass -File Z_Iosu/scripts/build-win-container.ps1 -Version 0.2.0 -GoFlags "-ldflags='-s -w'"
```

Limitaciones actuales
---------------------
- Sin soporte CUDA/ROCm cross (solo CPU).
- Librerías compiladas: únicamente componente CPU (ggml). No se incluye CUDA/HIP.
- Sin firma de código.

Futuras mejoras sugeridas
-------------------------
- Añadir build libs CPU vía CMake.
- Integrar assets reales para `app.exe`.
- Pipeline CI reutilizando imagen.
- Variante multi-stage más ligera.

Limpieza
--------
```
docker rmi ollama-winbuilder:local
```

Depuración
----------
```
docker run -it --rm -v ${PWD}:/workspace -w /workspace ollama-winbuilder:local bash
```

Relación con otros scripts
--------------------------
- `build-win-cuda.ps1`: Build nativo Windows + CUDA.
- `build-installer.ps1`: Installer nativo Windows.
- Este flujo: cross build CPU desde Linux/WSL.

Nota CUDA
---------
Soporte CUDA dentro del contenedor requeriría SDK y toolchain nativo Windows o cross improbable; mantener flujo separado.