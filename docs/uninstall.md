# Uninstall Ollama

This guide explains how to completely remove Ollama from your system.

> [!WARNING]
> **Model Data**: Removing the model directories (`.ollama`) will permanently delete all downloaded models and cached data. The uninstall scripts below will prompt you before removing model data.
>
> **Package Manager Note**: These instructions only apply to installations done via the official `curl`/PowerShell install scripts or manual downloads. If you installed Ollama via a package manager (Homebrew, apt, Docker, etc.), use that package manager's uninstall mechanism instead.

---

## macOS

### Manual Uninstall

1. Stop any running Ollama processes:

```bash
pkill -f ollama 2>/dev/null || true
```

2. Remove the symlink:

```bash
sudo rm -f /usr/local/bin/ollama
```

3. Remove the application bundle:

```bash
sudo rm -rf /Applications/Ollama.app
```

4. (Optional) Remove downloaded models and cache:

```bash
rm -rf ~/.ollama
```

### Automated Uninstall

Run the [uninstall-mac.sh](../scripts/uninstall/uninstall-mac.sh) script:

```bash
chmod +x scripts/uninstall/uninstall-mac.sh
./scripts/uninstall/uninstall-mac.sh
```

The script will:
- Stop running Ollama processes
- Remove the binary symlink and application bundle
- Optionally remove model data (with confirmation prompt)
- Support `--dry-run` to preview changes

---

## Linux

### Manual Uninstall

1. Stop and disable the systemd service:

```bash
sudo systemctl stop ollama
sudo systemctl disable ollama
```

2. Remove the systemd service file and reload:

```bash
sudo rm -f /etc/systemd/system/ollama.service
sudo systemctl daemon-reload
```

3. Remove the binary and symlink:

```bash
sudo rm -f "$(which ollama)"
```

4. Remove library files (if present):

```bash
sudo rm -rf /usr/local/lib/ollama
```

5. Remove the dedicated system user and group:

```bash
sudo userdel -r ollama 2>/dev/null || true
```

6. (Optional) Remove model data:

```bash
sudo rm -rf /usr/share/ollama
rm -rf ~/.ollama
```

### Automated Uninstall

Run the [uninstall-linux.sh](../scripts/uninstall/uninstall-linux.sh) script:

```bash
chmod +x scripts/uninstall/uninstall-linux.sh
sudo ./scripts/uninstall/uninstall-linux.sh
```

The script will:
- Stop and disable the systemd service
- Remove the binary, symlinks, and library files
- Remove the systemd service file
- Remove the dedicated `ollama` system user
- Optionally remove model data (with confirmation prompt)
- Support `--dry-run` to preview changes

---

## Windows

### Manual Uninstall

1. Stop running Ollama processes (PowerShell as Administrator):

```powershell
Stop-Process -Name "ollama" -Force -ErrorAction SilentlyContinue
Stop-Process -Name "ollama app" -Force -ErrorAction SilentlyContinue
```

2. Run the official Inno Setup uninstaller:

```powershell
$UninstallExe = "$env:LOCALAPPDATA\Programs\Ollama\unins000.exe"
if (Test-Path $UninstallExe) {
    Start-Process -FilePath $UninstallExe -ArgumentList "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART" -Wait
}
```

3. Remove orphaned application folders:

```powershell
Remove-Item -Recurse -Force "$env:LOCALAPPDATA\Programs\Ollama" -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force "$env:LOCALAPPDATA\Ollama" -ErrorAction SilentlyContinue
```

4. (Optional) Remove downloaded models:

```powershell
Remove-Item -Recurse -Force "$env:USERPROFILE\.ollama" -ErrorAction SilentlyContinue
```

### Automated Uninstall

Run the [uninstall-windows.ps1](../scripts/uninstall/uninstall-windows.ps1) script in PowerShell as Administrator:

```powershell
.\scripts\uninstall\uninstall-windows.ps1
```

The script will:
- Stop running Ollama processes
- Invoke the official Inno Setup uninstaller
- Clean up orphaned application folders
- Optionally remove model data (with confirmation prompt)
- Support `-DryRun` to preview changes

---

## Verification

After uninstalling, verify that Ollama has been removed:

**macOS/Linux:**

```bash
if ! command -v ollama &>/dev/null; then
    echo "Ollama has been removed from your system."
else
    echo "Ollama is still installed."
fi
```

**Windows (PowerShell):**

```powershell
if (Get-Command ollama -ErrorAction SilentlyContinue) {
    Write-Host "Ollama is still installed."
} else {
    Write-Host "Ollama has been removed from your system."
}
```
