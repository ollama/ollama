# Instalador Windows (Inno Setup)

Genera un instalador `OllamaSetup-<version>.exe` usando `app/ollama.iss` sin modificar archivos fuera de `Z_Iosu`.

## Prerrequisitos
- Inno Setup (ISCC.exe en PATH)
- Go y binarios construidos (el script puede construirlos)

## Script
```powershell
pwsh Z_Iosu/scripts/build-installer.ps1 -Version 0.11.10-local -Arch amd64
```

Parámetros:
- `-Version`: exporta `PKG_VERSION` para que `ollama.iss` la use.
- `-Arch`: actualmente sólo `amd64` en este flujo simplificado.
- `-SkipBuild`: reutiliza artefactos ya presentes en `dist/windows-amd64`.
- `-OutDir`: destino final (por defecto `Z_Iosu/release/installer`).

## Qué hace
1. (Opcional) Compila `dist/windows-amd64/ollama.exe`.
2. Asegura wrapper `windows-amd64-app.exe` (copia simple si no existe) para el nombre esperado `ollama app.exe`.
3. Invoca `ISCC.exe app/ollama.iss` con la variable de entorno `PKG_VERSION`.
4. Copia resultado renombrado a `OllamaSetup-<version>.exe`.

## Artefactos esperados
- `Z_Iosu/release/installer/OllamaSetup-<version>.exe`

## Verificación
Ejecutar el instalador, comprobar:
- Instalación en `%LOCALAPPDATA%\Programs\Ollama`.
- Presencia de `ollama app.exe` y `ollama.exe`.

## Troubleshooting
- "No se encontró ISCC.exe": añadir la carpeta de instalación de Inno Setup al PATH.
- Errores de versión: confirmar variable `PKG_VERSION` refleja el valor deseado.
- Falta `dist/windows-amd64/ollama.exe`: ejecutar primero build CUDA o CPU.
