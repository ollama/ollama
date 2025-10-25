# Workflow Hot-Swap de Binario (Linux dentro del contenedor)

Este flujo permite probar rápidamente cambios en el código Go sin reconstruir toda la imagen base ni reinstanciar pesos/modelos. Está optimizado para iteraciones rápidas sobre `cmd/ollama` y librerías asociadas.

---
## Objetivo
Reemplazar el binario `/usr/bin/ollama` dentro de un contenedor en ejecución (`ollama-gpu`) con una versión recién compilada a partir de los fuentes locales.

## Componentes Clave
- `Z_Iosu/docker/Dockerfile.dev`: construye una imagen *artifact* mínima que sólo contiene `/ollama-dev`.
- `Z_Iosu/scripts/hot-swap.ps1`: orquesta la compilación (WSL o contenedor builder) y copia el binario al contenedor runtime.
- Imagen base de ejecución: derivada de `Dockerfile.simple`.

---
## Modos de Compilación
1. WSL (más rápido, si tienes toolchain local y dependencias C disponibles).
2. Contenedor *builder* (fallback o forzado) basado en `golang:1.24-bookworm` con bind mount del repo.
3. (Opcional) Multi-stage artifact (`Dockerfile.dev`) si se desea una construcción completamente aislada.

El script intenta WSL primero (a menos que pases `-NoWSL` o `-ForceBuilder`). Si falla por dependencias, cambia automáticamente a builder.

---
## Script `hot-swap.ps1` (Parámetros)
| Flag | Uso | Descripción |
|------|-----|-------------|
| `-ForceBuilder` | opcional | Fuerza uso del contenedor builder aunque WSL esté disponible. |
| `-NoWSL` | opcional | Desactiva intento WSL. |
| `-Restart` | opcional | Reinicia el contenedor tras copiar el binario. |
| `-Verify` | opcional | Compara hash local vs hash dentro del contenedor. |
| `-RollbackOnFail` | opcional | Si la verificación falla, restaura backup. Requiere `-Verify`. |
| `-KeepBackup` | opcional | Conserva copia `/tmp/ollama.backup_*` dentro del contenedor. |
| `-ContainerName <name>` | opcional | (default: `ollama-gpu`). |
| `-ImageName <image>` | opcional | Imagen que se auto-arranca si el contenedor no existe. |
| `-Quiet` | opcional | Reduce la salida (sólo errores críticos). |

---
## Flujo Paso a Paso
1. Detectar entorno de compilación (WSL vs builder).
2. Compilar binario Linux (`ollama-dev`).
3. (Opcional) Capturar hash previo del binario en el contenedor.
4. (Opcional) Crear backup `/tmp/ollama.backup_<timestamp>`.
5. Copiar nuevo binario a `/usr/bin/ollama` (sobrescribe).
6. (Opcional) Reiniciar contenedor (`docker restart`).
7. (Opcional) Verificar hash y hacer rollback si falla (`-RollbackOnFail`).
8. Mostrar logs recientes para validar arranque.

---
## Ejemplos de Uso
### Iteración rápida sin reinicio
```powershell
pwsh Z_Iosu/scripts/hot-swap.ps1
```
El proceso sobrescribe el binario; si el servidor reutiliza procesos, se recomienda reiniciar manualmente para cargar código.

### Con reinicio y verificación de integridad
```powershell
pwsh Z_Iosu/scripts/hot-swap.ps1 -Restart -Verify
```

### Forzar builder (evitar WSL)
```powershell
pwsh Z_Iosu/scripts/hot-swap.ps1 -ForceBuilder -Restart
```

### Rollback automático si el hash no coincide
```powershell
pwsh Z_Iosu/scripts/hot-swap.ps1 -Verify -RollbackOnFail -Restart
```

### Mantener backup para inspección
```powershell
pwsh Z_Iosu/scripts/hot-swap.ps1 -Verify -KeepBackup
```

---
## Uso del Artifact Image (Alternativa)
Si prefieres construir aislado vía multi-stage:
```powershell
docker build -f Z_Iosu/docker/Dockerfile.dev -t ollama-dev-artifact .
# Extraer binario
$cid = docker create ollama-dev-artifact sh
docker cp $cid:/ollama-dev ollama-dev
docker rm $cid
# Copiar al runtime
docker cp .\ollama-dev ollama-gpu:/usr/bin/ollama
```
Aplica verificación opcional:
```powershell
docker exec ollama-gpu bash -lc 'sha256sum /usr/bin/ollama'
```

---
## Cuándo Usar Cada Método
| Situación | Recomendado |
|-----------|-------------|
| Cambios sólo Go puros | WSL build (rápido) |
| Problemas con toolchain local | Builder contenedor |
| Necesitas aislamiento reproducible | Artifact multi-stage |
| Debug de crash tras swap | Usar backup + rollback |

---
## Buenas Prácticas
- Recompilar siempre con `CGO_ENABLED=1` si interactúas con bindings C/CUDA.
- Limitar reinicios: agrupa varios cambios antes de `-Restart`.
- Conservar al menos un backup en entornos críticos (`-KeepBackup`).
- Verificar hashes en pipeline CI local (`-Verify`).

---
## Problemas Frecuentes
| Síntoma | Causa | Solución |
|---------|-------|----------|
| Hash difiere siempre | Cadenas embed (timestamp) | Asegura build determinístico, revisa `-ldflags -buildid=` |
| Falla build WSL | Falta toolchain/headers | Usa `-ForceBuilder` |
| Servidor no refleja cambio | Proceso cacheado | Ejecuta `-Restart` |
| Rollback no restaura | Backup no creado | Añade `-RollbackOnFail` (crea backup antes) |

---
## Integración con Desarrollo Diario
1. Edita fuentes (`cmd/ollama` o libs).
2. Lanza: `pwsh Z_Iosu/scripts/hot-swap.ps1 -Restart -Verify`.
3. Observa logs. Si hay regresión, repite.
4. Antes de commit grande: `-Verify -RollbackOnFail` para asegurar consistencia.

---
## Futuras Extensiones (Ideas)
- Target de Make: `make hot-swap`.
- Compilación selectiva basada en `git diff`.
- Notificación desktop al terminar.

---
## Resumen Rápido
```powershell
# Flujo estándar recomendado
yolo> pwsh Z_Iosu/scripts/hot-swap.ps1 -Restart -Verify
```
Cambia binario, reinicia, verifica hash y lista últimos logs.

---
© Workflow interno – Ajustar según evolución del repositorio.
