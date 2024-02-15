# Ollama Windows Preview

Welcome to the Ollama Windows preview.

No more WSL required!

Ollama now runs as a native Windows application, including NVIDIA GPU support.
After installing Ollama Windows Preview, Ollama will run in the background and
the `ollama` command line is available in `cmd`, `powershell` or your favorite
terminal application. As usual the Ollama [api](./api.md) will be served on
`http://localhost:11434`.

As this is a preview release, you should expect a few bugs here and there.  If
you run into a problem you can reach out on
[Discord](https://discord.gg/ollama), or file an 
[issue](https://github.com/ollama/ollama/issues).
Logs will often be helpful in dianosing the problem (see
[Troubleshooting](#troubleshooting) below)

## System Requirements

* Windows 10 or newer, Home or Pro
* NVIDIA 452.39 or newer Drivers if you have an NVIDIA card

## API Access

Here's a quick example showing API access from `powershell`
```powershell
(Invoke-WebRequest -method POST -Body '{"model":"llama2", "prompt":"Why is the sky blue?", "stream": false}' -uri http://localhost:11434/api/generate ).Content | ConvertFrom-json
```

## Troubleshooting

While we're in preview, `OLLAMA_DEBUG` is always enabled, which adds
a "view logs" menu item to the app, and increses logging for the GUI app and
server.

Ollama on Windows stores files in a few different locations.  You can view them in
the explorer window by hitting `<cmd>+R` and type in:
- `explorer %LOCALAPPDATA%\Ollama` contains logs, and downloaded updates
    - *app.log* contains logs from the GUI application
    - *server.log* contains the server logs
    - *upgrade.log* contains log output for upgrades
- `explorer %LOCALAPPDATA%\Programs\Ollama` contains the binaries (The installer adds this to your user PATH)
- `explorer %HOMEPATH%\.ollama` contains models and configuration
- `explorer %TEMP%` contains temporary executable files in one or more `ollama*` directories
