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

## Build


### Windows

Install these prerequisites before building the Windows desktop app:

- [Go](https://go.dev/dl/)
- [Node.js 20](https://nodejs.org/en/download)
- [CMake 3.24 or newer](https://cmake.org/download/)
- [Visual Studio C++ Build Tools](https://visualstudio.microsoft.com/downloads/),
  including the Desktop development with C++ workload and a Windows SDK
- [MSYS2](https://www.msys2.org/docs/installer/) with its Clang64 toolchain,
  including a C++20 compiler, libc++, and `windres`
- [Windows App Runtime 1.8](https://learn.microsoft.com/windows/apps/windows-app-sdk/downloads)
- [Inno Setup](https://jrsoftware.org/isinfo.php) when building the installer


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
