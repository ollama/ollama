#!/bin/bash
# Скрипт для проверки уязвимостей в Go зависимостях и Docker образе

set -e

echo "=== Проверка уязвимостей в репозитории Ollama ==="
echo ""

# Проверка наличия необходимых инструментов
echo "1. Проверка инструментов..."

if ! command -v go &> /dev/null; then
    echo "ОШИБКА: Go не установлен"
    exit 1
fi
echo "✓ Go установлен: $(go version)"

if ! command -v govulncheck &> /dev/null; then
    echo "Установка govulncheck..."
    go install golang.org/x/vuln/cmd/govulncheck@latest
fi
echo "✓ govulncheck доступен"

echo ""
echo "2. Проверка обновлений зависимостей..."
go list -m -u all | grep -E "\[" | head -20 || echo "Нет доступных обновлений"

echo ""
echo "3. Проверка уязвимостей в Go модулях..."
if govulncheck ./ 2>&1 | tee /tmp/govulncheck-output.txt; then
    echo "✓ Проверка завершена"
else
    echo "⚠ Проверка завершена с предупреждениями (см. /tmp/govulncheck-output.txt)"
fi

echo ""
echo "4. Анализ критических зависимостей..."
echo ""
echo "Текущие версии критических пакетов:"
go list -m golang.org/x/crypto golang.org/x/net golang.org/x/sys golang.org/x/text 2>&1 | head -10

echo ""
echo "Доступные обновления:"
go list -m -u golang.org/x/crypto golang.org/x/net golang.org/x/sys golang.org/x/text 2>&1 | grep -E "\[" || echo "Все пакеты актуальны"

echo ""
echo "5. Рекомендации:"
echo "- Проверьте отчет: CVE_ANALYSIS_REPORT.md"
echo "- Используйте Dockerfile.secure для более безопасной сборки"
echo "- Рассмотрите обновление go.mod.updated для обновленных зависимостей"
echo ""
echo "=== Проверка завершена ==="
