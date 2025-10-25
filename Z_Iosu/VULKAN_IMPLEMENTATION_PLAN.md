# Plan de Implementación Vulkan - Sin Tocar Archivos Originales

## Regla Fundamental
🚨 **NO MODIFICAR ARCHIVOS FUERA DE `Z_Iosu/`** 🚨
- Todos los cambios van en `Z_Iosu/`
- Solo tocar archivos originales si es ABSOLUTAMENTE necesario
- Evitar conflictos de merge con upstream

## Estado Actual
- Versión: 0.12.6-b1
- Soporte Vulkan: Implementado en commit `2aba569` (pero no en nuestra rama)
- Necesidad: Integrar soporte sin contaminar repo

## Estrategia: Scripts de Parche Inteligentes

### 1. Script de Verificación de Estado
**Archivo**: `Z_Iosu/scripts/check-vulkan-status.ps1`

```powershell
# Verificar si Vulkan está disponible SIN modificar nada
# - Revisar si existe ggml-vulkan/
# - Verificar commit actual vs 2aba569
# - Revisar instalación Vulkan SDK
# - Reportar qué falta
```

### 2. Script de Integración Controlada
**Archivo**: `Z_Iosu/scripts/integrate-vulkan.ps1`

```powershell
# OPCIÓN A: Cherry-pick limpio del commit
git fetch origin
git cherry-pick 2aba569a2a593f56651ded7f5011480ece70c80f --no-commit
# Revisar cambios antes de commitear
git status
# Solo commitear si usuario confirma

# OPCIÓN B: Aplicar solo archivos necesarios
# Copiar archivos específicos de Vulkan sin tocar otros
```

### 3. Script de Build con Vulkan
**Archivo**: `Z_Iosu/scripts/build-with-vulkan.ps1`

```powershell
# Extend dev-run.ps1 con soporte Vulkan
# - Detectar Vulkan SDK automáticamente
# - Configurar variables de entorno
# - Añadir flags específicos Vulkan
# - Compilar con -DGGML_USE_VULKAN=ON
```

### 4. Script de Instalación Vulkan SDK
**Archivo**: `Z_Iosu/scripts/install-vulkan-sdk.ps1`

```powershell
# Descargar e instalar Vulkan SDK 1.4.321.1
# Configurar variables de entorno automáticamente
# Verificar instalación
# Todo sin tocar configuraciones de sistema permanentes
```

## Pasos de Implementación

### Paso 1: Verificación
```powershell
powershell -ExecutionPolicy Bypass -File Z_Iosu\scripts\check-vulkan-status.ps1
```

### Paso 2: Preparar Vulkan SDK (si necesario)
```powershell
powershell -ExecutionPolicy Bypass -File Z_Iosu\scripts\install-vulkan-sdk.ps1
```

### Paso 3: Integrar código (solo si usuario acepta)
```powershell
powershell -ExecutionPolicy Bypass -File Z_Iosu\scripts\integrate-vulkan.ps1 -DryRun
# Revisar cambios propuestos
powershell -ExecutionPolicy Bypass -File Z_Iosu\scripts\integrate-vulkan.ps1 -Confirm
```

### Paso 4: Compilar con Vulkan
```powershell
powershell -ExecutionPolicy Bypass -File Z_Iosu\scripts\build-with-vulkan.ps1
```

## Archivos a Crear (SOLO en Z_Iosu/)

1. **Z_Iosu/scripts/check-vulkan-status.ps1** - Verificación sin modificar
2. **Z_Iosu/scripts/install-vulkan-sdk.ps1** - Instalación SDK
3. **Z_Iosu/scripts/integrate-vulkan.ps1** - Integración controlada
4. **Z_Iosu/scripts/build-with-vulkan.ps1** - Build con Vulkan
5. **Z_Iosu/docs/VULKAN_SETUP.md** - Documentación completa
6. **Z_Iosu/patches/vulkan-integration.patch** - Parche backup
7. **Z_Iosu/config/vulkan-env.ps1** - Variables de entorno

## Ventajas de Este Enfoque

✅ **Seguridad**: No tocar archivos originales  
✅ **Reversible**: Fácil vuelta atrás  
✅ **Documentado**: Todo en nuestro directorio  
✅ **Controlado**: Usuario decide cada paso  
✅ **Mantenible**: Scripts reutilizables  

## Próximo Paso
¿Quieres que implemente estos scripts en `Z_Iosu/scripts/`?