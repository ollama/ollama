# Guía de Build en Windows (Fork / Entorno Local)

Esta guía documenta los pasos reproducibles para compilar y ejecutar el servidor en Windows usando Go + cgo con toolchain **llvm-mingw** (recomendado) y el script de conveniencia `Z_Iosu/scripts/dev-run.ps1`.

## 1. Requisitos Previos

1. **Go** >= 1.21 (detectado: probado con 1.24.x)
2. **llvm-mingw** (recomendado) o toolchain MinGW que provea `clang`/`gcc` compatibles:
   - Descargar release: https://github.com/mstorsjo/llvm-mingw/releases
   - Extraer, por ejemplo en: `C:\llvm-mingw`
   - Añadir al PATH (inicio de sesión o sesión actual):
     ```powershell
     $env:PATH = 'C:\llvm-mingw\bin;' + $env:PATH
     ```
3. (Opcional) Visual Studio Build Tools – solo si se quiere probar `clang-cl` / `cl.exe` (NO recomendado para cgo aquí).
4. PowerShell 5.1 o superior.

## 2. Verificación Inicial

```powershell
where go
where clang
where x86_64-w64-mingw32-clang
```
Debes ver rutas dentro de `C:\llvm-mingw\bin`. Si solo aparece `clang.exe` sin el prefijo triple, el script igualmente lo detecta.

## 3. Script de Conveniencia
El archivo: `Z_Iosu/scripts/dev-run.ps1`

Parámetros principales:
- `-ForceClangGnu`   Fuerza clang (llvm-mingw) y si no existe intenta clang genérico o gcc (fallback). Ignora `cl.exe`.
- `-ResetGoEnv`      Elimina CC/CXX persistidos en `go env` (evita arrastre de toolchain viejo).
- `-Clean`           Limpia caché de compilación (`go clean -cache`) y borra `build/`.
- `-ShowEnv`         Imprime CC, CXX, flags cgo y host.
- `-GoRelease`       Compila binario optimizado (`-trimpath -ldflags '-s -w'`) y lo ejecuta.
- `-UseCMake` / `-Release` Flujo CMake opcional (debug/release) si se necesita integración nativa.
- `-PreferClangCL`   Prioriza `clang-cl` si está, para pruebas (no recomendado con cgo puro).
- `-DryRun`          No ejecuta acciones, solo muestra.

Orden de fallback con `-ForceClangGnu`:
1. `x86_64-w64-mingw32-clang.exe`
2. `clang.exe`
3. `x86_64-w64-mingw32-gcc.exe` / `gcc`
Si ninguno aparece: aborta indicando instalar llvm-mingw.

## 4. Primer Arranque (Modo Desarrollo)
```powershell
powershell -ExecutionPolicy Bypass -File Z_Iosu\scripts\dev-run.ps1 -ForceClangGnu -ResetGoEnv -Clean -ShowEnv
```
IosuUltimo
powershell -ExecutionPolicy Bypass -File Z_Iosu\scripts\dev-run.ps1 -ForceClangGnu -ResetGoEnv -GoRelease -Clean -ShowEnv

Esto:
1. Limpia CC/CXX persistentes.
2. Detecta `x86_64-w64-mingw32-clang.exe` (si existe) y lo asigna a `CC` y su par `clang++.exe` a `CXX`.
3. Traduce flags MSVC si apareciesen (/std:c++17, /EHsc) a formato GCC.
4. Inyecta: `--target=x86_64-w64-windows-gnu -fuse-ld=lld` y asegura `-std=c++17`.
5. Ejecuta `go run . serve`.
===================================================================================
===================================================================================
===================================================================================
Iosu  FINAL
powershell -ExecutionPolicy Bypass -File Z_Iosu\scripts\build-installer.ps1 -Version 0.11.102 -ForceClangGnu -InnoPath "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" -AlsoPortable -Verbose
===================================================================================
===================================================================================
===================================================================================


## 5. Build Optimizado (GoRelease)
```powershell
powershell -ExecutionPolicy Bypass -File Z_Iosu\scripts\dev-run.ps1 -ForceClangGnu -ResetGoEnv -GoRelease -Clean -ShowEnv
```
Genera `ollama-dev.exe` y lo lanza con `serve`.

## 6. Variables CGO Añadidas Automáticamente
Cuando se usa `-ForceClangGnu` el script ajusta (pre-pend):
- `CGO_CFLAGS`: `--target=x86_64-w64-windows-gnu -fuse-ld=lld -O3 -DNDEBUG ...`
- `CGO_CXXFLAGS`: idem + `-std=c++17` (si faltaba)
- `CGO_LDFLAGS`: `--target=x86_64-w64-windows-gnu -fuse-ld=lld ...`

Esto evita el error de formato `_cgo_.o` y el rechazo de flags MSVC.

## 7. Problemas Frecuentes
| Síntoma | Causa | Solución |
|---------|-------|----------|
| `cl : Command line error D8021` | Se usó `cl.exe` con flags GCC (`-Werror`) | Añadir `-ForceClangGnu` y asegurar llvm-mingw primero en PATH |
| `cgo: cannot parse _cgo_.o` | Mezcla de toolchain 32/64 o falta `--target` | Confirmar uso de `x86_64-w64-mingw32-clang.exe`; usar script actualizado |
| No se encuentra `clang.exe` | PATH sin llvm-mingw | Añadir `C:\llvm-mingw\bin` al PATH | 
| Flags `/std:c++17` inválidos | Se ejecutó clang GNU con flags MSVC | Script ya traduce; re-ejecutar con `-ResetGoEnv` |
| Código salida 1 tras servir | Cierre manual/context canceled | Ver logs y decidir si ignorar exit code (future improvement) |

## 8. Limpieza Manual
Para limpiar completamente (incluyendo módulos Go):
```powershell
go clean -cache -modcache
Remove-Item -Recurse -Force build -ErrorAction SilentlyContinue
```

## 9. Ejecución sin Script (Referencia)
Si necesitas reproducir manualmente:
```powershell
$env:PATH = 'C:\llvm-mingw\bin;' + $env:PATH
$env:CC='C:\llvm-mingw\bin\x86_64-w64-mingw32-clang.exe'
$env:CXX='C:\llvm-mingw\bin\x86_64-w64-mingw32-clang++.exe'
$env:CGO_ENABLED=1
$env:CGO_CFLAGS='--target=x86_64-w64-windows-gnu -fuse-ld=lld -O3 -DNDEBUG'
$env:CGO_CXXFLAGS='--target=x86_64-w64-windows-gnu -fuse-ld=lld -O3 -DNDEBUG -std=c++17'
$env:CGO_LDFLAGS='--target=x86_64-w64-windows-gnu -fuse-ld=lld'
$env:OLLAMA_HOST='127.0.0.1:11434'
# Desarrollo
go run . serve
# Release
go build -trimpath -ldflags '-s -w' -o ollama-dev.exe .
./ollama-dev.exe serve
```

## 10. Próximas Mejoras (Opcionales)
- Parámetro `-IgnoreExitCode` para no propagar código distinto de 0 al cerrar.
- `-LogFile` para redirigir stdout/stderr.
- `-Model <name>` para precarga controlada.

## 11. Checklist Rápido
```
[ ] Go instalado
[ ] llvm-mingw en PATH (x86_64-w64-mingw32-clang.exe responde)
[ ] go env CC/CXX no persistentes (usar -ResetGoEnv si dudas)
[ ] Script ejecutado con -ForceClangGnu
[ ] Puerto accesible: http://127.0.0.1:11434/api/ps
```

## 12. Generar Instalador Windows
Script: `Z_Iosu/scripts/build-installer.ps1`

Ejemplos:
```powershell
# Versión automática (fecha+commit), compilando con clang gnu
powershell -ExecutionPolicy Bypass -File Z_Iosu\scripts\build-installer.ps1 -ForceClangGnu -AutoVersion
powershell -ExecutionPolicy Bypass -File Z_Iosu\scripts\build-installer.ps1 -AutoVersion -ForceClangGnu -Verbose

# Versión fija y reutilizando binarios existentes
powershell -ExecutionPolicy Bypass -File Z_Iosu\scripts\build-installer.ps1 -Version 0.11.101  -ForceClangGnu -Verbose
powershell -ExecutionPolicy Bypass -File Z_Iosu\scripts\build-installer.ps1 -Version 0.1.0 -SkipBuild

# Generar instalador y también ZIP portable
powershell -ExecutionPolicy Bypass -File Z_Iosu\scripts\build-installer.ps1 -Version 0.11.101 -ForceClangGnu -AlsoPortable -Verbose

# Con salida verbosa
powershell -ExecutionPolicy Bypass -File Z_Iosu\scripts\build-installer.ps1 -ForceClangGnu -AutoVersion -Verbose
```
Salida:
- Instalador base (cuando se usa Inno): `dist/OllamaSetup.exe` (o `%LOCALAPPDATA%\dist\OllamaSetup.exe` si el OutputDir relativo cae allí).
- Copia versionada: `Z_Iosu/release/installer/OllamaSetup-<VERSION>.exe`
- Paquete portable (sin Inno con `-Portable` o adicional con `-AlsoPortable`): `Z_Iosu/release/installer/Ollama-portable-<VERSION>.zip`

Parámetros clave:
- `-AutoVersion`    Usa `YYYY.MM.DD+<shortCommit>`.
- `-Version 1.2.3`  Fija versión manual.
- `-ForceClangGnu`  Recompila usando clang gnu / fallback (dev-run).
- `-SkipBuild`      Empaqueta sin recompilar (usa binario existente).
- `-InnoPath C:\ruta\ISCC.exe` Ruta directa a `ISCC.exe`.
- `-SkipWrapper`    No (re)crea `windows-amd64-app.exe`.
- `-Portable`       Genera zip portable en lugar de instalador.
- `-AlsoPortable`   Fuerza generar zip además del instalador.
- `-NoAlsoPortable` No generar zip adicional (por defecto ahora se crea si hay instalador).
- `-Verbose`        Log extendido (muestra versión limpia, rutas, etc.).

Requisitos para el instalador:
1. Inno Setup 6 (ISCC.exe en PATH o usar `-InnoPath`).
2. Binario `ollama-dev.exe` o `ollama.exe` listo (si `-SkipBuild`).
3. `app/ollama.iss` presente (el script crea copia temporal para ajustar `VersionInfoVersion`).

Requisitos para `-Portable`:
1. Binario `ollama.exe` disponible (se toma `ollama-dev.exe` si existe y se renombra).
2. Wrapper generado (`windows-amd64-app.exe`) salvo `-SkipWrapper`.
3. LICENSE y README si se quieren empaquetar (se incluyen si existen).

Troubleshooting:
- "No se localizó ISCC.exe": Instala Inno Setup (o `choco install innosetup`) o pasa `-InnoPath`.
- "No se encontró binario de servidor": Ejecuta build previo con `dev-run.ps1 -ForceClangGnu -GoRelease`.
- Instalador sin versión correcta: Revisa `-Version` / `-AutoVersion` y que no haya quedado un valor previo en entorno.
- No aparece `OllamaSetup.exe` en `dist/`: El script busca también en `%LOCALAPPDATA%\dist`; si no está, revisar logs de ISCC.
- ZIP incompleto: Asegura que `dist/windows-amd64/ollama.exe` exista antes de usar `-Portable` (no omitir build si falta).

---
Si algo falla, ejecuta con `-ShowEnv -DryRun` y comparte la salida para diagnóstico.
