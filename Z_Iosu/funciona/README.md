# Snapshot FUNCIONA (2025-09-05)

Copia congelada de los artefactos que se validaron como funcionales:

Incluye:
- `Dockerfile.simple` (imagen base extendida)
- `Dockerfile.dev` (multi-stage para binario /ollama-dev)
- `hot-swap.ps1` (script de reemplazo en contenedor)

Uso recomendado (no modificar estos archivos; sirven de referencia estable):
```powershell
# Construir imagen funcional snapshot
docker build -f Z_Iosu/funciona/Dockerfile.simple -t ollama:0.11.10-funciona .

# Construir artifact dev y extraer binario
docker build -f Z_Iosu/funciona/Dockerfile.dev -t ollama-dev-artifact .
$cid = docker create ollama-dev-artifact sh
docker cp $cid:/ollama-dev ollama-dev
docker rm $cid

# Hot-swap (usar versión activa en scripts/, snapshot sólo como respaldo)
pwsh Z_Iosu/scripts/hot-swap.ps1 -Restart -Verify
```

Si en el futuro algo se rompe, comparar contra este snapshot (`git diff Z_Iosu/funciona/...`).
