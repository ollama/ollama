# Ollama for macOS and Windows

## Download

- [macOS](https://ollama.com/download/Ollama.dmg)
- [Windows](https://ollama.com/download/OllamaSetup.exe)

## Development

The app runs in the menu bar or notification area and has a native settings
window: AppKit on macOS (`cmd/app/*_darwin.m`) and WinUI 3 on Windows
(`cmd/app/settings_windows.*`). Settings logic lives in Go
(`cmd/app/settings.go`) so both platforms share it.

The Windows window is XAML markup that `app/winui` loads at run time, calling
WinUI from Go without cgo or a XAML compiler. It needs the
[Windows App Runtime](https://learn.microsoft.com/windows/apps/windows-app-sdk/downloads)
1.8, and on Windows 10 the runtime's bootstrapper DLL next to the app, which
the installer includes. Building the app on Windows still needs a C compiler,
such as [llvm-mingw](https://github.com/mstorsjo/llvm-mingw), for the SQLite
database.

Run it from the repository root, which is where it finds its icons during
development:

```bash
go run ./app/cmd/app --dev
```

`--dev` keeps the app from stopping other running `ollama` servers, and lets
`OLLAMA_APP_DB_PATH` point the app at a separate database.

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
