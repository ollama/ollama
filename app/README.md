# Ollama for macOS and Windows

## Download

- [macOS](https://ollama.com/download/Ollama.dmg)
- [Windows](https://ollama.com/download/OllamaSetup.exe)

## Development

The app runs in the menu bar or notification area and has a native settings
window: AppKit on macOS (`cmd/app/*_darwin.m`) and WinUI 3 on Windows
(`cmd/app/settings_windows.*`). Settings logic lives in Go
(`cmd/app/settings.go`) so both platforms share it.

The app runs the `ollama` on your `PATH`, or one you built into `dist/`, so
install Ollama or build it first. With `--dev` it leaves a running Ollama
alone and uses it, and reads its database from `OLLAMA_APP_DB_PATH` if you set
it. A new path starts the app fresh, as if it had never been opened.

### macOS

You need Go and the Xcode Command Line Tools (`xcode-select --install`). Run
the app from the repository root, which is where it finds its icons:

```bash
go run ./app/cmd/app --dev
```

### Windows

You need:

- Go, and a C compiler on your `PATH` for the SQLite database, such as
  [llvm-mingw](https://github.com/mstorsjo/llvm-mingw), which CI uses.
- The [Windows App Runtime](https://learn.microsoft.com/windows/apps/windows-app-sdk/downloads)
  1.8. If it's missing, the app offers to download it when you open Settings.

Run the app from the repository root:

```powershell
go run ./app/cmd/app --dev
```

On Windows 10 the app also needs the runtime's bootstrapper,
`Microsoft.WindowsAppRuntime.Bootstrap.dll`, next to its `.exe`. The installer
includes it, but `go run` can't, so build the app instead. The build script
extracts the DLL into `dist\winappsdk` (use `arm64` in place of `amd64` on ARM):

```powershell
.\scripts\build_windows.ps1 app
Copy-Item dist\winappsdk\amd64\Microsoft.WindowsAppRuntime.Bootstrap.dll dist\
.\dist\windows-ollama-app-amd64.exe --dev
```

The Windows window is XAML markup that `app/winui` loads at run time, calling
WinUI from Go without cgo or a XAML compiler.

## Build

### macOS

```sh
./scripts/build_darwin.sh
```

This builds Ollama and the app into `dist/Ollama.app` and `dist/Ollama.dmg`.
Set `APPLE_IDENTITY` to sign them; unsigned builds can't be used to test
upgrades.

### Windows

The installer needs [Inno Setup](https://jrsoftware.org/isinfo.php).

```powershell
.\scripts\build_windows.ps1
```

This builds Ollama, the app, and `OllamaSetup.exe` into `dist\`. To build only
some of it, name the steps, for example `.\scripts\build_windows.ps1 ollama app`.
