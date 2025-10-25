# Entorno Dev Completo (Dockerfile.devfull)

Imagen de desarrollo interactivo con toolchain completo para compilar binario Linux dentro de contenedor sin afectar al host.

## Construcción
```powershell
docker build -f Z_Iosu/docker/Dockerfile.devfull -t ollama-devfull .
```

## Uso Rápido
```powershell
pwsh Z_Iosu/scripts/dev-shell.ps1 -Rebuild -Keep
# Dentro del contenedor:
export CGO_ENABLED=1 GOOS=linux GOARCH=amd64
go build -o ollama-dev ./cmd/ollama
./ollama-dev serve
```

## Script dev-shell.ps1
Parámetros:
- `-Image`: nombre/tag (default `ollama-devfull`).
- `-Name`: nombre contenedor (default `ollama-devshell`).
- `-Rebuild`: fuerza rebuild de la imagen.
- `-Keep`: no elimina el contenedor al salir.

## Flujo con Hot-Swap
1. Compila cambios rápido vía WSL / builder (`hot-swap.ps1`).
2. Sólo usa devfull si necesitas herramientas extra (cmake, clang, debug). 
3. Mantén el runtime limpio usando la imagen simple y reemplazo de binario.

## Volúmenes
El script monta `${PWD}:/src`. Cualquier build genera artefactos en tu árbol local.

## Debug
Puedes instalar extras dentro del contenedor (temporal):
```bash
apt-get update && apt-get install -y gdb strace lsof
```
No lo incluyo por defecto para mantener la imagen ligera.
