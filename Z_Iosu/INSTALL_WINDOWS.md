# Instalación de Ollama (Windows)

Esta guía resume cómo instalar el binario generado (modo desarrollador) y cómo usar el instalador/paquete portable creados con los scripts del fork.

## 1. Opciones de Instalación

1. Instalador (.exe) – recomendado para integración con PATH y desinstalador.
2. Paquete portable (.zip) – ejecutar sin instalación (ideal para pruebas rápidas o entornos restringidos).
3. Ejecución directa de binario de desarrollo (`ollama-dev.exe`).

## 2. Requisitos Rápidos
- Windows 10 22H2 o superior recomendado.
- PowerShell 5.1+.
- Para usar el instalador: Inno Setup solo es necesario al crear el instalador, no para ejecutar.

## 3. Obtener Artefactos
Los artefactos se generan con el script `Z_Iosu/scripts/build-installer.ps1`.
powershell -ExecutionPolicy Bypass -File Z_Iosu\scripts\build-installer.ps1 -AutoVersion -ForceClangGnu -Verbose

Ejemplos de generación:
```powershell
# Instalador con versión automática (fecha+commit)
powershell -ExecutionPolicy Bypass -File Z_Iosu\scripts\build-installer.ps1 -ForceClangGnu -AutoVersion

# Solo paquete portable (sin Inno Setup)
powershell -ExecutionPolicy Bypass -File Z_Iosu\scripts\build-installer.ps1 -ForceClangGnu -AutoVersion -Portable

# Reutilizando build existente para crear solo el instalador
your_previous_build_here
powershell -ExecutionPolicy Bypass -File Z_Iosu\scripts\build-installer.ps1 -SkipBuild -Version 1.2.3
```
Salida típica:
- Instalador: `Z_Iosu/release/installer/OllamaSetup-<VERSION>.exe`
- Portable:   `Z_Iosu/release/installer/Ollama-portable-<VERSION>.zip`

## 4. Instalación con el Instalador
1. Ejecuta `OllamaSetup-<VERSION>.exe`.
2. El instalador coloca los archivos en `%LOCALAPPDATA%\Programs\Ollama`.
3. Añade la ruta a PATH del usuario (HKCU Environment Path merge).
4. Crea accesos directos en menú inicio e inicio automático.
5. Al finalizar puedes abrir una nueva ventana de PowerShell y ejecutar:
   ```powershell
   ollama run llama3.2
   ```

Desinstalación:
- Desde “Aplicaciones instaladas” o ejecutando el desinstalador en la misma carpeta.
- Limpia modelos en `%USERPROFILE%\.ollama` salvo overrides personalizados.

## 5. Uso del Paquete Portable
1. Extrae el zip a una carpeta: `C:\OllamaPortable` (ejemplo).
2. Contenido clave:
   - `ollama.exe` (servidor)
   - `ollama-app.exe` (wrapper simple si se generó)
   - `ollama_welcome.ps1` (script auxiliar opcional)
   - `lib\` (bibliotecas compartidas si existen)
3. Para lanzar el servidor:
   ```powershell
   cd C:\OllamaPortable
   .\ollama.exe serve
   ```
4. (Opcional) Añadir carpeta al PATH temporalmente:
   ```powershell
   $env:PATH = "C:\OllamaPortable;" + $env:PATH
   ```

## 6. Verificación Post-Instalación
Prueba HTTP local:
```powershell
Invoke-RestMethod -Uri http://127.0.0.1:11434/api/tags -Method GET
```
Si responde JSON vacío o lista de modelos, el servicio está operativo.

## 7. Actualización
Para actualizar:
1. Cerrar procesos `ollama.exe` / `ollama-app.exe`.
2. Ejecutar nuevo instalador (reemplaza binarios) o reemplazar archivos en carpeta portable.
3. Reiniciar terminal para refrescar PATH si cambió.

## 8. Solución de Problemas
| Problema | Causa | Mitigación |
|----------|-------|-----------|
| `ollama` no se reconoce | PATH no actualizado aún | Cerrar y reabrir terminal o verificar `%LOCALAPPDATA%\Programs\Ollama` en PATH usuario |
| Puerto 11434 ocupado | Otro proceso usando puerto | Cambiar variable `OLLAMA_HOST` antes de lanzar (`$env:OLLAMA_HOST='127.0.0.1:11500'`) |
| Modelos no persisten | Carpeta `.ollama` limpiada | Revisar `%USERPROFILE%\.ollama\models` y permisos |
| Instalador no genera exe | Inno Setup no disponible o fallo en build | Re-ejecutar script sin `-Portable` y revisar logs verbosos |
| ZIP incompleto | Binario no estaba en `dist/windows-amd64` | No usar `-SkipBuild` en la primera generación |

## 9. Limpieza Manual
Portable: borrar carpeta donde lo extrajiste y `%USERPROFILE%\.ollama` si deseas eliminar modelos.
Instalador: usar desinstalador, luego eliminar `%LOCALAPPDATA%\Ollama` residual si quedó algo.

## 10. Firma de Código (Opcional Futuro)
El script soporta ajuste de `VersionInfoVersion` y prepara campo para futura integración con herramientas de firmado (e.g. `signtool`). Aún no implementado.

---
Para soporte adicional durante el desarrollo: usar `-Verbose` y/o `-ShowEnv` en los scripts.
