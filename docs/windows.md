# Ollama Windows

Welcome to Ollama for Windows.

No more WSL required!

Ollama now runs as a native Windows application, including NVIDIA and AMD Radeon GPU support.
After installing Ollama for Windows, Ollama will run in the background and
the `ollama` command line is available in `cmd`, `powershell` or your favorite
terminal application. As usual the Ollama [api](./api.md) will be served on
`http://localhost:11434`.

## System Requirements

* Windows 10 22H2 or newer, Home or Pro
* NVIDIA 452.39 or newer Drivers if you have an NVIDIA card
* AMD Radeon Driver https://www.amd.com/en/support if you have a Radeon card

Ollama uses unicode characters for progress indication, which may render as unknown squares in some older terminal fonts in Windows 10. If you see this, try changing your terminal font settings.

## Filesystem Requirements

The Ollama install does not require Administrator, and installs in your home directory by default.  You'll need at least 4GB of space for the binary install.  Once you've installed Ollama, you'll need additional space for storing the Large Language models, which can be tens to hundreds of GB in size.  If your home directory doesn't have enough space, you can change where the binaries are installed, and where the models are stored.

### Changing Install Location

To install the Ollama application in a location different than your home directory, start the installer with the following flag

```powershell
OllamaSetup.exe /DIR="d:\some\location"
```

### Changing Model Location

To change where Ollama stores the downloaded models instead of using your home directory, set the environment variable `OLLAMA_MODELS` in your user account.

1. Start the Settings (Windows 11) or Control Panel (Windows 10) application and search for _environment variables_.

2. Click on _Edit environment variables for your account_.

3. Edit or create a new variable for your user account for `OLLAMA_MODELS` where you want the models stored

4. Click OK/Apply to save.

If Ollama is already running, Quit the tray application and relaunch it from the Start menu, or a new terminal started after you saved the environment variables.

## API Access

Here's a quick example showing API access from `powershell`

```powershell
(Invoke-WebRequest -method POST -Body '{"model":"llama3.2", "prompt":"Why is the sky blue?", "stream": false}' -uri http://localhost:11434/api/generate ).Content | ConvertFrom-json
```

## Troubleshooting

Ollama on Windows stores files in a few different locations.  You can view them in
the explorer window by hitting `<Ctrl>+R` and type in:
- `explorer %LOCALAPPDATA%\Ollama` contains logs, and downloaded updates
    - *app.log* contains most resent logs from the GUI application
    - *server.log* contains the most recent server logs
    - *upgrade.log* contains log output for upgrades
- `explorer %LOCALAPPDATA%\Programs\Ollama` contains the binaries (The installer adds this to your user PATH)
- `explorer %HOMEPATH%\.ollama` contains models and configuration

## Uninstall

The Ollama Windows installer registers an Uninstaller application.  Under `Add or remove programs` in Windows Settings, you can uninstall Ollama.

> [!NOTE]
> If you have [changed the OLLAMA_MODELS location](#changing-model-location), the installer will not remove your downloaded models


## Standalone CLI

The easiest way to install Ollama on Windows is to use the `OllamaSetup.exe`
installer. It installs in your account without requiring Administrator rights.
We update Ollama regularly to support the latest models, and this installer will
help you keep up to date.

If you'd like to install or integrate Ollama as a service, a standalone
`ollama-windows-amd64.zip` zip file is available containing only the Ollama CLI
and GPU library dependencies for Nvidia.  If you have an AMD GPU, also download
and extract the additional ROCm package `ollama-windows-amd64-rocm.zip` into the
same directory.  This allows for embedding Ollama in existing applications, or
running it as a system service via `ollama serve` with tools such as
[NSSM](https://nssm.cc/). 

> [!NOTE]  
> If you are upgrading from a prior version, you should remove the old directories first.
