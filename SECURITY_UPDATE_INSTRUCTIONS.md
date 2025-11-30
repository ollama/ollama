# Инструкции по обновлению безопасности

## Обзор

В репозитории были выявлены потенциальные уязвимости безопасности в зависимостях и базовых образах Docker. Созданы обновленные версии файлов для устранения этих проблем.

## Созданные файлы

### 1. `CVE_ANALYSIS_REPORT.md`
Подробный отчет об анализе уязвимостей с описанием всех выявленных проблем и рекомендаций.

### 2. `Dockerfile.secure`
Обновленная версия Dockerfile с улучшениями безопасности:
- Добавлены обновления безопасности системных пакетов (`dnf update -y --security`, `apt-get upgrade -y`)
- Добавлена очистка временных файлов
- Добавлен non-root пользователь для запуска контейнера
- Обновлены сертификаты CA
- Добавлен `go mod tidy` для обновления зависимостей

### 3. `go.mod.updated`
Обновленная версия go.mod с новыми версиями критических зависимостей:
- `golang.org/x/crypto`: v0.36.0 → v0.45.0
- `golang.org/x/net`: v0.38.0 → v0.47.0
- `golang.org/x/sys`: v0.36.0 → v0.38.0
- `golang.org/x/text`: v0.23.0 → v0.31.0
- `github.com/containerd/console`: v1.0.3 → v1.0.5
- `github.com/gin-gonic/gin`: v1.10.0 → v1.11.0

### 4. `check-vulnerabilities.sh`
Скрипт для автоматической проверки уязвимостей в проекте.

## Как использовать обновления

### Вариант 1: Постепенное обновление (рекомендуется)

1. **Обновление Go зависимостей:**
   ```bash
   # Создайте резервную копию текущего go.mod
   cp go.mod go.mod.backup
   
   # Обновите критические пакеты по одному
   go get -u golang.org/x/crypto@latest
   go get -u golang.org/x/net@latest
   go get -u golang.org/x/sys@latest
   go get -u golang.org/x/text@latest
   
   # Обновите другие зависимости
   go get -u github.com/containerd/console@latest
   go get -u github.com/gin-gonic/gin@latest
   
   # Проверьте, что все работает
   go mod tidy
   go test ./...
   ```

2. **Обновление Dockerfile:**
   ```bash
   # Создайте резервную копию
   cp Dockerfile Dockerfile.backup
   
   # Скопируйте обновленную версию
   cp Dockerfile.secure Dockerfile
   
   # Или вручную добавьте обновления безопасности в текущий Dockerfile:
   # - Добавьте `dnf update -y --security` после установки пакетов
   # - Добавьте `apt-get upgrade -y` для Ubuntu образов
   # - Добавьте non-root пользователя в финальный образ
   ```

### Вариант 2: Полное обновление

1. **Использование обновленного go.mod:**
   ```bash
   cp go.mod.updated go.mod
   go mod download
   go mod verify
   go test ./...
   ```

2. **Использование обновленного Dockerfile:**
   ```bash
   cp Dockerfile.secure Dockerfile
   # Протестируйте сборку
   docker build -t ollama:secure .
   ```

## Проверка уязвимостей

### Автоматическая проверка:
```bash
./check-vulnerabilities.sh
```

### Ручная проверка:
```bash
# Установите govulncheck если еще не установлен
go install golang.org/x/vuln/cmd/govulncheck@latest

# Проверьте уязвимости
govulncheck ./...

# Проверьте обновления зависимостей
go list -m -u all
```

## Рекомендации по безопасности

### Немедленные действия:
1. ✅ Обновить критические пакеты `golang.org/x/*`
2. ✅ Добавить обновления безопасности в Dockerfile
3. ✅ Добавить non-root пользователя в Docker образ
4. ✅ Регулярно проверять уязвимости

### Долгосрочные действия:
1. Настроить автоматическое сканирование в CI/CD:
   ```yaml
   # Пример для GitHub Actions
   - name: Check vulnerabilities
     run: |
       go install golang.org/x/vuln/cmd/govulncheck@latest
       govulncheck ./...
   ```

2. Использовать Dependabot для автоматических обновлений зависимостей

3. Регулярно обновлять базовые образы Docker

4. Использовать сканеры уязвимостей контейнеров (trivy, grype, snyk)

## Тестирование после обновления

После применения обновлений обязательно выполните:

```bash
# Проверка компиляции
go build ./...

# Запуск тестов
go test ./...

# Проверка сборки Docker образа
docker build -t ollama:test .

# Проверка работы контейнера
docker run --rm ollama:test --version
```

## Откат изменений

Если возникли проблемы после обновления:

```bash
# Восстановление go.mod
cp go.mod.backup go.mod
go mod download

# Восстановление Dockerfile
cp Dockerfile.backup Dockerfile
```

## Дополнительные ресурсы

- [Go Vulnerability Database](https://pkg.go.dev/vuln)
- [Docker Security Best Practices](https://docs.docker.com/engine/security/)
- [OWASP Dependency Check](https://owasp.org/www-project-dependency-check/)

## Контакты

При обнаружении критических уязвимостей немедленно сообщите команде безопасности проекта.
