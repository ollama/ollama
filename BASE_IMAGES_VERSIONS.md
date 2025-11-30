# Актуальные версии базовых образов Docker

## Дата проверки
30 ноября 2025

## Текущие версии в проекте

### Базовые образы операционных систем

#### Ubuntu
- **Текущая версия в проекте:** `ubuntu:24.04`
- **Последняя доступная версия:** `ubuntu:26.04` (ноябрь 2025)
- **Рекомендуемая LTS версия:** `ubuntu:24.04` (LTS до апреля 2029)
- **Альтернативные LTS версии:**
  - `ubuntu:22.04` (LTS до апреля 2027)
  - `ubuntu:20.04` (LTS до апреля 2025)

**Рекомендация:** Оставить `ubuntu:24.04` как LTS версию, но использовать конкретный тег с датой для воспроизводимости (например, `ubuntu:24.04-20251130`).

#### AlmaLinux
- **Текущая версия в проекте:** `almalinux:8`
- **Последняя доступная версия:** `almalinux:10.1` (ноябрь 2025)
- **Актуальные версии:**
  - `almalinux:10` / `almalinux:10.1` - последняя стабильная версия
  - `almalinux:9` / `almalinux:9.7` - предыдущая стабильная версия
  - `almalinux:8` / `almalinux:8.10` - устаревшая версия (EOL в 2029)

**Рекомендация:** 
- Для новых сборок рассмотреть переход на `almalinux:9` или `almalinux:10`
- Если требуется совместимость с ROCm, проверить поддержку в новых версиях
- Использовать конкретные версии: `almalinux:9.7` или `almalinux:10.1`

### Специализированные образы

#### ROCm (AMD GPU)
- **Текущая версия в проекте:** `rocm/dev-almalinux-8:6.3.3-complete`
- **Последние доступные версии:**
  - `rocm/dev-almalinux-8:7.1.1-complete` (последняя для almalinux-8)
  - `rocm/dev-almalinux-9:7.1.1-complete` (для almalinux-9)
  - `rocm/dev-almalinux-8:6.4.4-complete` (более новая версия 6.x)

**Рекомендация:** 
- Обновить до `rocm/dev-almalinux-8:7.1.1-complete` для последних исправлений безопасности
- Или до `rocm/dev-almalinux-8:6.4.4-complete` если требуется версия 6.x
- Рассмотреть переход на `rocm/dev-almalinux-9` для лучшей поддержки

#### NVIDIA JetPack (ARM64)
- **Текущие версии в проекте:**
  - `nvcr.io/nvidia/l4t-jetpack:r35.4.1` (JetPack 5)
  - `nvcr.io/nvidia/l4t-jetpack:r36.4.0` (JetPack 6)

**Рекомендация:** 
- Проверить наличие обновлений через NVIDIA NGC Catalog
- Использовать последние доступные версии для каждой серии JetPack
- JetPack версии привязаны к конкретным версиям L4T (Linux for Tegra)

## Рекомендации по обновлению

### Приоритет 1 (Критично)
1. **Ubuntu:** Оставить `ubuntu:24.04`, но использовать конкретный тег с датой
2. **ROCm:** Обновить до `rocm/dev-almalinux-8:7.1.1-complete` или `6.4.4-complete`

### Приоритет 2 (Важно)
1. **AlmaLinux base:** Рассмотреть переход на `almalinux:9.7` для новых сборок
2. **ROCm base:** Рассмотреть переход на `rocm/dev-almalinux-9` если поддерживается

### Приоритет 3 (Опционально)
1. **AlmaLinux:** Долгосрочный переход на `almalinux:10` после тестирования
2. **JetPack:** Проверить и обновить до последних версий через NGC

## Примеры обновленных версий

### Минимальные обновления безопасности
```dockerfile
# Обновить только версии с исправлениями безопасности
ARG ROCMVERSION=7.1.1  # или 6.4.4
FROM --platform=linux/amd64 rocm/dev-almalinux-8:${ROCMVERSION}-complete AS base-amd64

FROM --platform=linux/arm64 almalinux:8.10 AS base-arm64

FROM ubuntu:24.04
```

### Полное обновление
```dockerfile
# Обновить до последних стабильных версий
ARG ROCMVERSION=7.1.1
FROM --platform=linux/amd64 rocm/dev-almalinux-9:${ROCMVERSION}-complete AS base-amd64

FROM --platform=linux/arm64 almalinux:9.7 AS base-arm64

FROM ubuntu:24.04
```

## Проверка актуальных версий

Для проверки актуальных версий используйте:

```bash
# Ubuntu
curl -s "https://registry.hub.docker.com/v2/repositories/library/ubuntu/tags?page_size=100" | grep -oE '"name":"[0-9]+\.[0-9]+"' | sort -V -r | head -5

# AlmaLinux
curl -s "https://registry.hub.docker.com/v2/repositories/library/almalinux/tags?page_size=100" | grep -oE '"name":"(9|10)(\.[0-9]+)?"' | sort -u

# ROCm
curl -s "https://hub.docker.com/v2/repositories/rocm/dev-almalinux-8/tags?page_size=50" | grep -oE '"name":"[^"]*complete"' | head -10

# JetPack (через NGC Catalog)
# https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-jetpack
```

## Примечания

1. **Совместимость:** При обновлении базовых образов необходимо проверить совместимость с:
   - Версиями CUDA
   - Версиями ROCm
   - Системными библиотеками
   - Компиляторами (gcc, clang)

2. **Тестирование:** Всегда тестируйте сборку после обновления базовых образов

3. **Воспроизводимость:** Используйте конкретные версии тегов вместо `latest` для стабильных сборок

4. **Безопасность:** Регулярно проверяйте обновления безопасности для используемых образов
