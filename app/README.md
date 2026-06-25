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

The local updater endpoint override is compiled only when the app is built with
the `updater_localtest` tag. Official builds do not include this override.
The override is shared by the Windows and macOS app; the localhost server below
serves the current Windows `install.ps1` upgrade flow.
To intentionally run an unsigned `install.ps1` during local testing, also add
the `updater_unsigned` tag. Official builds must never use that tag.

These tests install and upgrade Ollama for real. Run them only on a machine
where it is OK to replace the current Ollama install.

Build a local-test app:

```powershell
go generate ./...
go build -tags updater_localtest -trimpath `
  -ldflags "-H windowsgui -X=github.com/ollama/ollama/app/version.Version=0.0.0-localtest" `
  -o .\dist\windows-ollama-app-updater-localtest.exe .\app\cmd\app
```

For the unsigned-script success path only, build with both tags:

```powershell
go build -tags "updater_localtest updater_unsigned" -trimpath `
  -ldflags "-H windowsgui -X=github.com/ollama/ollama/app/version.Version=0.0.0-localtest" `
  -o .\dist\windows-ollama-app-updater-localtest.exe .\app\cmd\app
```

Start by testing that the app rejects an unsigned `install.ps1`. This serves a
patched test copy of the script from localhost so the script would download the
local installer if it were trusted.

```powershell
python .\scripts\tests\update-server.py `
  --port 8765 `
  --version 0.0.1-localtest `
  --installer .\dist\OllamaSetup.exe `
  --install-script .\scripts\install.ps1 `
  --patch-install-script
```

To exercise the installer-cache fallback for a server that omits installer
ETags, add `--omit-installer-etag` to the server command.

In another PowerShell window:

```powershell
Remove-Item "$env:LOCALAPPDATA\Ollama\updates_v2" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item "$env:LOCALAPPDATA\Ollama\install_cache" -Recurse -Force -ErrorAction SilentlyContinue
$env:OLLAMA_DEBUG = "1"
.\dist\windows-ollama-app-updater-localtest.exe --update-check-url http://127.0.0.1:8765/api/update
Get-Content "$env:LOCALAPPDATA\Ollama\app.log" -Wait
```

The log should show `install.ps1 signature verification failed`, and no upgrade
should be staged.

For the signed success path, generate the patched test script once, sign that
generated file, then serve it without `--patch-install-script` so its signature
is preserved.

```powershell
python .\scripts\tests\update-server.py `
  --port 8765 `
  --version 0.0.1-localtest `
  --installer .\dist\OllamaSetup.exe `
  --install-script .\scripts\install.ps1 `
  --patch-install-script `
  --output-install-script .\.cache\update-server\install.ps1 `
  --prepare-only

# Sign .\.cache\update-server\install.ps1 using the same Authenticode signing
# setup used by .\scripts\build_windows.ps1 sign. The installer must also be signed.

python .\scripts\tests\update-server.py `
  --port 8765 `
  --version 0.0.1-localtest `
  --installer .\dist\OllamaSetup.exe `
  --install-script .\.cache\update-server\install.ps1
```

Then run the local-test app once to cache the update, quit it after the log
shows the cache phase completed, and start it hidden to exercise startup
upgrade:

```powershell
$env:OLLAMA_DEBUG = "1"
.\dist\windows-ollama-app-updater-localtest.exe --update-check-url http://127.0.0.1:8765/api/update
Get-Content "$env:LOCALAPPDATA\Ollama\app.log" -Wait

Get-Process "Ollama app" -ErrorAction SilentlyContinue | Stop-Process
.\dist\windows-ollama-app-updater-localtest.exe --update-check-url http://127.0.0.1:8765/api/update --hide
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
