# Dockerfile.simple (Z_Iosu/docker)

Este Dockerfile crea una imagen derivada de la imagen oficial ya compilada (`ollama/ollama:<tag>`), evitando recompilar todo el proyecto y aplicando sólo personalizaciones.

## Estructura
```
Z_Iosu/
  docker/
    Dockerfile.simple
    Dockerfile.devfull (entorno desarrollo)
    overlay/ (archivos que se copiarán dentro del contenedor)
  scripts/
    build-simple.ps1
    hot-swap.ps1
    dev-shell.ps1
    build-win-cuda.ps1
    build-installer.ps1
```

## Construcción
```bash
docker build -f Z_Iosu/docker/Dockerfile.simple -t ollama:0.11.10-custom .
```

### Helper script rápido
En lugar de escribir el comando completo puedes usar:
```powershell
pwsh Z_Iosu/scripts/build-simple.ps1 -Tag ollama:0.11.10-custom
```
Flags útiles:
- `-NoCache` fuerza rebuild de capas.
- `-Push` hace `docker push` tras construir.

### Pasos detallados (cómo creamos la imagen)
1. Posiciónate en la raíz del repo (donde está `Z_Iosu/`).  
2. Ejecuta el build minimalista (no recompila Ollama, sólo añade ENV + overlay):  
   ```powershell
   docker build -f Z_Iosu/docker/Dockerfile.simple -t ollama:0.11.10-custom .
   ```
3. (Opcional) Verifica la imagen:  
   ```powershell
   docker image inspect ollama:0.11.10-custom --format '{{ .Id }}'
   ```
4. (Opcional) Lista variables incrustadas:  
   ```powershell
   docker run --rm ollama:0.11.10-custom env | findstr OLLAMA_
   ```

## Entorno Dev Completo (Opcional)
Si necesitas compilar con toolchain completo dentro de contenedor:
```powershell
pwsh Z_Iosu/scripts/dev-shell.ps1 -Rebuild -Keep
```
Documentado en: `Z_Iosu/docs/devfull.md`.

## Build Windows CUDA Local
Script para compilar librerías CUDA + binario:
```powershell
pwsh Z_Iosu/scripts/build-win-cuda.ps1 -Preset "CUDA 12"
```
Doc: `Z_Iosu/docs/windows-cuda-build.md`.

## Instalador Windows (Inno Setup)
```powershell
pwsh Z_Iosu/scripts/build-installer.ps1 -Version 0.11.10-local -Arch amd64
```
Doc: `Z_Iosu/docs/windows-installer.md`.

## Ejecución
```bash
docker run --rm -p 11434:11434 ollama:0.11.10-custom
docker run -d --gpus all `
  -p 11434:11434 `
  -v "C:\Users\iosuc\.ollama\models:/models" `
  --name ollama-gpu `
  ollama:0.11.10-custom
```

### Cómo arrancamos normalmente el contenedor GPU (Windows PowerShell)
Comando usado (variables ya están en la imagen, por eso no repetimos `-e`):
```powershell
docker run -d --gpus all `
  -p 11434:11434 `
  -v "C:\Users\iosuc\.ollama\models:/models" `
  --name ollama-gpu `
  ollama:0.11.10-custom
```

Si quieres forzar/overrides de las mismas variables (ejemplo):
```powershell
docker run -d --gpus all `
  -p 11434:11434 `
  -e OLLAMA_MODELS=/models `
  -e GIN_MODE=release `
  -e OLLAMA_DEBUG=2 `
  -v "C:\Users\iosuc\.ollama\models:/models" `
  --name ollama-gpu `
  ollama:0.11.10-custom
```

### Alternativa: montar toda la carpeta .ollama
Proporciona también pulls, blobs y metadata completos:
```powershell
$OLLAMA_HOME="$env:USERPROFILE\.ollama"
mkdir $OLLAMA_HOME -Force | Out-Null
docker run -d --gpus all `
  -p 11434:11434 `
  -v "$OLLAMA_HOME:/root/.ollama" `
  --name ollama-gpu `
  ollama:0.11.10-custom
```
En este caso la ruta de modelos interna sería `/root/.ollama/models`; puedes añadir `-e OLLAMA_MODELS=/root/.ollama/models` si quieres mantener coherencia.

### Verificación rápida tras arrancar
```powershell
curl http://localhost:11434/api/tags
docker logs -f ollama-gpu
docker exec -it ollama-gpu env | findstr OLLAMA_CONTEXT_LENGTH
```

### Ciclo de recreación limpio
```powershell
docker rm -f ollama-gpu 2>$null
docker build -f Z_Iosu/docker/Dockerfile.simple -t ollama:0.11.10-custom .
```

## Persistencia de modelos
```bash
docker volume create ollama_models
docker run -d --name ollama -p 11434:11434 -v ollama_models:/root/.ollama ollama:0.11.10-custom
```

## Cambios Rápidos
- Añade/edita variables de entorno en `Dockerfile.simple`.
- Coloca ficheros en `overlay/` (configs, scripts) para que se copien a `/opt/ollama-overlay/`.
- Ajusta `HEALTHCHECK` o `ENTRYPOINT` si lo necesitas.

## Hot-Swap del Binario (Iteración Rápida)
Si modificas código Go y quieres reemplazar el binario dentro de un contenedor en ejecución sin reconstruir toda la imagen:

```powershell
pwsh Z_Iosu/scripts/hot-swap.ps1 -Restart -Verify
```

Características:
- Compila vía WSL o (fallback) contenedor builder `golang:1.24-bookworm`.
- Copia el binario a `/usr/bin/ollama` dentro de `ollama-gpu`.
- `-Verify` compara hash local vs contenedor.
- `-RollbackOnFail` restaura backup si la verificación falla.

Documentación completa: `Z_Iosu/docs/hot-swap-workflow.md`.

## Empaquetado rápido Windows (binario)
Para generar un paquete local con `ollama.exe` y README:
```powershell
pwsh Z_Iosu/scripts/package-release.ps1 -Name ollama-win64 -Zip
```
Output en `Z_Iosu/release/<Name>` y zip opcional.

## Recomendaciones
- Fija digest: `FROM ollama/ollama@sha256:<digest>` para reproducibilidad.
- Evita `--pull` en desarrollo local; úsalo sólo en CI.
