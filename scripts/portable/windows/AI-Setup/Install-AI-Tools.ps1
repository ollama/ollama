Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$usbRoot = $PSScriptRoot
Set-Location $usbRoot

Write-Host '[1/4] 檢查資料夾...'
$dirs = @('input_media', 'output_result', 'python_embed', 'models', 'whisper_models', 'ollama')
foreach ($dir in $dirs) {
  New-Item -Path (Join-Path $usbRoot $dir) -ItemType Directory -Force | Out-Null
}

$env:OLLAMA_MODELS = Join-Path $usbRoot 'models'
$env:HF_HOME = Join-Path $usbRoot 'whisper_models'
$env:XDG_CACHE_HOME = Join-Path $usbRoot 'whisper_models'
[Environment]::SetEnvironmentVariable('OLLAMA_MODELS', $env:OLLAMA_MODELS, 'User')

function Ensure-EmbeddedPython {
  $pythonExe = Join-Path $usbRoot 'python_embed\python.exe'
  if (Test-Path $pythonExe) {
    Write-Host '[2/4] 已存在可攜式 Python，跳過下載。'
    return
  }

  Write-Host '[2/4] 下載可攜式 Python 3.10...'
  $zipFile = Join-Path $usbRoot 'python_embed.zip'
  Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.10.11/python-3.10.11-embed-amd64.zip' -OutFile $zipFile
  Expand-Archive -Path $zipFile -DestinationPath (Join-Path $usbRoot 'python_embed') -Force
  Remove-Item $zipFile -Force

  $pthFile = Join-Path $usbRoot 'python_embed\python310._pth'
  if (Test-Path $pthFile) {
    (Get-Content $pthFile) -replace '#import site', 'import site' | Set-Content -Encoding ascii $pthFile
  }
}

function Ensure-PipAndPythonDeps {
  $pythonExe = Join-Path $usbRoot 'python_embed\python.exe'
  $pipExe = Join-Path $usbRoot 'python_embed\Scripts\pip.exe'

  if (-not (Test-Path $pipExe)) {
    $getPip = Join-Path $usbRoot 'python_embed\get-pip.py'
    Write-Host '[3/4] 安裝 pip...'
    Invoke-WebRequest -Uri 'https://bootstrap.pypa.io/get-pip.py' -OutFile $getPip
    & $pythonExe $getPip --no-warn-script-location
    Remove-Item $getPip -Force
  }

  Write-Host '[3/4] 安裝 AI Python 依賴...'
  & $pythonExe -m pip install --no-warn-script-location requests tqdm openai-whisper
  try {
    & $pythonExe -m pip install --no-warn-script-location torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  } catch {
    Write-Host 'CUDA 版本安裝失敗，改安裝 CPU 版本 PyTorch...'
    & $pythonExe -m pip install --no-warn-script-location torch torchvision torchaudio
  }
}

function Ensure-Ollama {
  Write-Host '[4/4] 檢查 Ollama...'
  $existing = Get-Command ollama -ErrorAction SilentlyContinue
  if ($existing) {
    Write-Host '已偵測到系統 Ollama。'
    return
  }

  $setup = Join-Path $usbRoot 'ollama\OllamaSetup.exe'
  if (Test-Path $setup) {
    Write-Host '使用隨身碟中的 OllamaSetup.exe 安裝...'
    Start-Process -FilePath $setup -ArgumentList '/VERYSILENT /NORESTART /SUPPRESSMSGBOXES' -Wait
    return
  }

  Write-Host '下載官方 Windows 安裝腳本安裝 Ollama...'
  Invoke-Expression (Invoke-WebRequest -UseBasicParsing 'https://ollama.com/install.ps1').Content
}

function Write-AICoreScript {
  $script = @'
# -*- coding: utf-8 -*-
import os
import torch
import whisper
import requests

USB_ROOT = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(USB_ROOT, "input_media")
OUTPUT_DIR = os.path.join(USB_ROOT, "output_result")
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3"

os.environ["HF_HOME"] = os.path.join(USB_ROOT, "whisper_models")
os.environ["XDG_CACHE_HOME"] = os.path.join(USB_ROOT, "whisper_models")

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🎬 啟動本地 Whisper... 硬體加速偵測: {device.upper()}")
if device == "cpu":
    print("⚠️ 警告：這台電腦未偵測到 NVIDIA 顯卡加速，將使用 CPU 慢速運算。")

try:
    model = whisper.load_model("large-v3", device=device)
except Exception as e:
    print(f"模型載入失敗，正在改用 base。錯誤: {e}")
    model = whisper.load_model("base", device=device)

files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith((".mp4", ".mp3", ".m4a", ".wav", ".mkv"))]
if not files:
    print("\n[等待中] 請將影音檔案放入 input_media 後重新執行。")
else:
    for file_name in files:
        file_path = os.path.join(INPUT_DIR, file_name)
        base_name = os.path.splitext(file_name)[0]
        print(f"\n🚀 正在處理: {file_name}")
        try:
            result = model.transcribe(file_path)
            raw_text = result["text"]
            with open(os.path.join(OUTPUT_DIR, f"{base_name}_原始逐字稿.txt"), "w", encoding="utf-8") as f:
                f.write(raw_text)
            payload = {
                "model": OLLAMA_MODEL,
                "prompt": f"請將以下文本翻譯並潤飾為流暢的繁體中文（台灣商務口吻）：\n\n{raw_text}",
                "stream": False,
            }
            res = requests.post(OLLAMA_API_URL, json=payload, timeout=120)
            res.raise_for_status()
            trans_text = res.json().get("response", "").strip()
        except Exception as e:
            trans_text = f"Ollama 本地端未啟動或連線失敗，僅保留逐字稿。錯誤原因: {e}"
        with open(os.path.join(OUTPUT_DIR, f"{base_name}_最終繁中翻譯.txt"), "w", encoding="utf-8") as f:
            f.write(trans_text)
        os.rename(file_path, os.path.join(INPUT_DIR, f"processed_{file_name}"))
    print("\n🎉 隨身碟內所有影音檔案已全自動處理完畢！")
'@

  Set-Content -Path (Join-Path $usbRoot 'ai_core.py') -Value $script -Encoding utf8
}

Ensure-EmbeddedPython
Ensure-PipAndPythonDeps
Ensure-Ollama
Write-AICoreScript

& (Join-Path $usbRoot 'Pull-Portable-Models.ps1')
& (Join-Path $usbRoot 'Create-Desktop-Shortcuts.ps1')

Write-Host '初始化完成。'
