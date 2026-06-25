# Ollama for macOS and Windows

## Download

- [macOS](https://github.com/ollama/app/releases/download/latest/Ollama.dmg)
- [Windows](https://github.com/ollama/app/releases/download/latest/OllamaSetup.exe)

## Development

### Desktop App

```bash
go generate ./... &&
go run ./cmd/app
```

### UI Development

#### Setup

Install required tools:

```bash
go install github.com/tkrajina/typescriptify-golang-structs/tscriptify@latest
```

#### Develop UI (Development Mode)

1. Start the React development server (with hot-reload):

```bash
cd ui/app
npm install
npm run dev
```

2. In a separate terminal, run the Ollama app with the `-dev` flag:

```bash
go generate ./... &&
OLLAMA_DEBUG=1 go run ./cmd/app -dev
```

The `-dev` flag enables:

- Loading the UI from the Vite dev server at http://localhost:5173
- Fixed UI server port at http://127.0.0.1:3001 for API requests
- CORS headers for cross-origin requests
- Hot-reload support for UI development

### Upgrade Testing

These commands perform a real app upgrade. Use a test machine or a throwaway
install.

Build or copy `OllamaSetup.exe` into `dist\`, then build a local-test app:

```powershell
go generate ./...
go build -tags updater_localtest -trimpath `
  -ldflags "-H windowsgui -X=github.com/ollama/ollama/app/version.Version=0.0.0-localtest" `
  -o .\dist\windows-ollama-app-updater-localtest.exe .\app\cmd\app
```

For unsigned local `install.ps1` testing:

```powershell
go build -tags "updater_localtest updater_unsigned" -trimpath `
  -ldflags "-H windowsgui -X=github.com/ollama/ollama/app/version.Version=0.0.0-localtest" `
  -o .\dist\windows-ollama-app-updater-localtest.exe .\app\cmd\app
```

Start the local update server from the repository root:

```powershell
python .\scripts\tests\update-server.py `
  --port 8765 `
  --version 0.0.1-localtest
```

To test without ETags, add `--omit-etags`.

In another PowerShell window:

```powershell
Remove-Item "$env:LOCALAPPDATA\Ollama\updates_v2" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item "$env:LOCALAPPDATA\Ollama\install_cache" -Recurse -Force -ErrorAction SilentlyContinue
$env:OLLAMA_DEBUG = "1"
$env:OLLAMA_TEST_UPDATE_URL = "http://127.0.0.1:8765/api/update"
.\dist\windows-ollama-app-updater-localtest.exe
Get-Content "$env:LOCALAPPDATA\Ollama\app.log" -Wait
```

For the startup upgrade path:

```powershell
$env:OLLAMA_DEBUG = "1"
$env:OLLAMA_TEST_UPDATE_URL = "http://127.0.0.1:8765/api/update"
.\dist\windows-ollama-app-updater-localtest.exe
Get-Content "$env:LOCALAPPDATA\Ollama\app.log" -Wait

Get-Process "Ollama app" -ErrorAction SilentlyContinue | Stop-Process
.\dist\windows-ollama-app-updater-localtest.exe --hide
Get-Content "$env:LOCALAPPDATA\Ollama\app.log" -Wait
```

## Build


### Windows

- https://jrsoftware.org/isinfo.php


**Dependencies** - either build a local copy of ollama, or use a github release
```powershell
# Local dependencies
.\scripts\deps_local.ps1

# Release dependencies
.\scripts\deps_release.ps1 0.6.8
```

**Build**
```powershell
.\scripts\build_windows.ps1
```

### macOS

CI builds with Xcode 14.1 for OS compatibility prior to v13.  If you want to manually build v11+ support, you can download the older Xcode [here](https://developer.apple.com/services-account/download?path=/Developer_Tools/Xcode_14.1/Xcode_14.1.xip), extract, then `mv ./Xcode.app /Applications/Xcode_14.1.0.app` then activate with:

```
export CGO_CFLAGS="-O3 -mmacosx-version-min=12.0"
export CGO_CXXFLAGS="-O3 -mmacosx-version-min=12.0"
export CGO_LDFLAGS="-mmacosx-version-min=12.0"
export SDKROOT=/Applications/Xcode_14.1.0.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk
export DEVELOPER_DIR=/Applications/Xcode_14.1.0.app/Contents/Developer
```

**Dependencies** - either build a local copy of Ollama, or use a GitHub release:
```sh
# Local dependencies
./scripts/deps_local.sh

# Release dependencies
./scripts/deps_release.sh 0.6.8
```

**Build**
```sh
./scripts/build_darwin.sh
```
