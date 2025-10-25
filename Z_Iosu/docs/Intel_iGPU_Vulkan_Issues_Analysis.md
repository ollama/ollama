# Intel iGPU Vulkan Issues Analysis & Solutions

## Problema Principal

**Error identificado**: Exception 0xe06d7363 en `ggml_backend_sched_graph_compute_async`

### Stack Trace Crítico
```
Exception 0xe06d7363 0x19930520 0x3efedff6a0 0x7ff97e9780aa
PC=0x7ff97e9780aa
signal arrived during external code execution

runtime.cgocall(0x7ff7bd3ed9a0, 0xc001571ab8)
github.com/ollama/ollama/ml/backend/ggml._Cfunc_ggml_backend_sched_graph_compute_async(0x16f7048f1e0, 0x17f0c157040)
github.com/ollama/ollama/ml/backend/ggml.(*Context).ComputeWithNotify.func1(...)
```

## Modelos Problemáticos Identificados

### GPT-OSS Series
- **gpt-oss:20b** - Crash confirmado durante inferencia
- **Síntomas**: Exception 0xe06d7363 en compute graph
- **Hardware afectado**: Intel iGPU con driver Vulkan

## Dos Problemas Distintos

### 1. Bugcheck 0x10e (VIDEO_MEMORY_MANAGEMENT_INTERNAL)
- **Estado**: ✅ **RESUELTO** con `GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1`
- **Causa**: Driver Intel Arc manejo incorrecto de host-visible video memory
- **Síntoma**: Sistema crash completo, BSOD
- **Solución**: Workaround implementado en scripts

### 2. Exception 0xe06d7363 en Compute Graph
- **Estado**: 🔴 **ACTIVO** - Requiere detección y fallback a CPU
- **Causa**: Incompatibilidad específica entre ciertos modelos y Intel Vulkan backend
- **Síntoma**: Crash de aplicación (no sistema)
- **Solución**: Detección automática y fallback a CPU

## Soluciones Implementadas

### Script Inteligente: `start_ollama_intel_smart.ps1`

#### Características
1. **Detección automática** de modelos problemáticos
2. **Configuración inteligente** de variables de entorno
3. **Fallback seguro** a CPU para modelos conflictivos
4. **Workarounds preventivos** para crashes conocidos

#### Uso
```powershell
# Detección automática
.\start_ollama_intel_smart.ps1

# Modelo específico
.\start_ollama_intel_smart.ps1 -Model "gpt-oss:20b"

# Forzar CPU (seguro)
.\start_ollama_intel_smart.ps1 -Model "any-model" -ForceCPU

# Forzar Vulkan (riesgoso para modelos problemáticos)
.\start_ollama_intel_smart.ps1 -Model "gpt-oss:20b" -ForceVulkan
```

#### Lógica de Decisión
```
¿Modelo problemático?
├─ SÍ → ¿ForceVulkan?
│  ├─ SÍ → Vulkan + Warning
│  └─ NO → CPU (seguro)
└─ NO → Vulkan + Safety flags
```

## Variables de Entorno Críticas

### Para Modelos Problemáticos (CPU Fallback)
```bash
OLLAMA_LLM_LIBRARY=cpu
```

### Para Modelos Compatibles (Vulkan Seguro)
```bash
OLLAMA_LLM_LIBRARY=vulkan
OLLAMA_INTEL_GPU=1
GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1  # Previene bugcheck 0x10e
```

### Variables Opcionales
```bash
OLLAMA_VK_IGPU_MEMORY_LIMIT_MB=16384   # Limitar memoria (si necesario)
GGML_VK_VISIBLE_DEVICES=0              # Seleccionar GPU específica
```

## Modelos de Compatibilidad

### ✅ Compatibles con Intel iGPU Vulkan
- **Llama 3.x series** (8B, 70B)
- **Gemma 2/3 series** (después de PR #12552)
- **Qwen series**
- **Mistral series**

### ⚠️ Problemáticos (Requieren CPU)
- **GPT-OSS series** (todas las variantes)
- **Modelos GPT4** (algunos)
- **Otros modelos grandes** (>20B en ciertas arquitecturas)

### 🔍 Sin Confirmar
- Modelos nuevos requieren testing individual
- El script detecta patrones conocidos automáticamente

## Diagnóstico Manual

### Síntomas de Modelo Problemático
```
time=...:...:....... level=TRACE source=runner.go:450 msg="forwardBatch compute started"
time=...:...:....... level=TRACE source=runner.go:623 msg="computeBatch: waiting for inputs"
Exception 0xe06d7363 0x19930520 0x3efedff6a0 0x7ff97e9780aa
```

### Síntomas de Bugcheck 0x10e
```
DRIVER_VERIFIER_IOMANAGER_VIOLATION (c9)
VIDEO_MEMORY_MANAGEMENT_INTERNAL (10e)
```

## Mejores Prácticas

### 1. Usar Script Inteligente
- Evita configuración manual
- Detección automática de problemas
- Aplicación de workarounds apropiados

### 2. Para Desarrollo/Testing
```powershell
# Probar modelo nuevo de forma segura
.\start_ollama_intel_smart.ps1 -Model "nuevo-modelo" -ForceCPU
```

### 3. Para Producción
```powershell
# Configuración automática óptima
.\start_ollama_intel_smart.ps1 -Model "modelo-conocido"
```

## Status del PR #11835

### ✅ Resuelto en Upstream
- **Commit**: 2aba569a2 (merged 5 días atrás)
- **Fixes incluidos**:
  - `fix vulkan handle releasing` (b6554e9)
  - Gemma3 Intel iGPU fixes (#12552)
  - Memory estimation improvements
  - Intel GPU ID detection en Windows

### ✅ Presente en Branch Actual
- Tu branch `12.7.b1` incluye todos los fixes
- Commit #12552 presente desde Oct 13, 2025
- Vulkan backend completamente integrado

## Recomendaciones Finales

### Para Usuarios
1. **Usar `start_ollama_intel_smart.ps1`** para configuración automática
2. **Evitar manual override** de modelos problemáticos
3. **Reportar nuevos modelos problemáticos** para añadir a la lista

### Para Desarrollo
1. **Mantener lista de modelos problemáticos** actualizada
2. **Considerar detección en runtime** dentro de GGML backend
3. **Monitor upstream fixes** para nuevas soluciones

### Estado Actual
- ✅ Bugcheck 0x10e: **Resuelto**
- ⚠️  Exception 0xe06d7363: **Mitigado** con CPU fallback
- ✅ Detección automática: **Implementada**
- ✅ Scripts de automatización: **Listos**

El sistema está ahora **robusto y seguro** para uso en Intel iGPU con Vulkan.