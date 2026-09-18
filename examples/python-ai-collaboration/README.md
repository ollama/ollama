# Python AI collaboration

This example asks Claude to privately compare implementation approaches and
produce a complete solution, then sends the result through one or more OpenAI
quality-gate rounds. Review stops early when OpenAI returns exactly `PASS`.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:ANTHROPIC_API_KEY = "your-anthropic-key"
$env:OPENAI_API_KEY = "your-openai-key"
```

## Run

```powershell
python client.py --task "Build a CSV folder monitor with Telegram alerts."
```

Use `--task-file task.txt` for a longer prompt, `--rounds` to control review
iterations, and `--output result.md` to save the final response. Model names can
be overridden with `--claude-model`, `--openai-model`, `ANTHROPIC_MODEL`, or
`OPENAI_MODEL`.

Run the included Windows AI deployment example:

```powershell
python client.py `
  --task-file tasks\windows-ai-deployment.txt `
  --rounds 3 `
  --output deployment-result.md
```

The reviewed PowerShell result is also checked in at
`results\windows-ai-deployment.ps1`. It requires an explicit branch:

```powershell
.\results\windows-ai-deployment.ps1 -Branch main
```

Validate the hardware and destination without changing the SSD:

```powershell
.\results\windows-ai-deployment.ps1 -Branch main -ValidateOnly
```

Portable environment scripts under `results\portable-ai` configure all model,
Python, package, and temporary storage beneath the external drive.

- Copy the PowerShell scripts (`*.ps1`) to `H:\ollama\env\scripts\`
- Copy the command wrappers (`Launch-Portable-AI.cmd`, `Resume-Portable-AI.cmd`,
  `Stop-And-Eject.cmd`, `Run-Qwen-Test.cmd`, `Open-Portable-AI-Shell.cmd`) to
  `H:\ollama\`

Then run:

```powershell
.\Start-Portable-Ollama.ps1 -Background
.\Test-Qwen.ps1
```

Before ejecting the drive, run `Stop-Portable-Ollama.ps1`. After reconnecting
it on a restored Windows host, double-click `Resume-Portable-AI.cmd`.

If you want automatic driver restore, pre-place the NVIDIA installer at:
`H:\ollama\env\drivers\616.64-desktop-win10-win11-64bit-international-dch-whql.exe`
and run `Resume-Portable-AI.cmd` (or `Resume-Portable-AI.ps1 -InstallDriver`).
Without that file, resume still works when the installed driver is already
compatible.

Run the tests without making API calls:

```powershell
python -m unittest -v
```
