Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$usbRoot = $PSScriptRoot
Set-Location $usbRoot

# Security requirements:
# 1) Set PYTHON_EMBED_SHA256 to the official SHA256 of python-3.10.11-embed-amd64.zip
# 2) Set GET_PIP_SHA256 to the official SHA256 of get-pip.py
# Example (PowerShell): $env:PYTHON_EMBED_SHA256='...'; $env:GET_PIP_SHA256='...'

Write-Host '[1/4] 檢查資料夾...'
$dirs = @('input_media', 'output_result', 'python_embed', 'models', 'whisper_models', 'ollama')
foreach ($dir in $dirs) {
  New-Item -Path (Join-Path $usbRoot $dir) -ItemType Directory -Force | Out-Null
}

$env:OLLAMA_MODELS = Join-Path $usbRoot 'models'
$env:HF_HOME = Join-Path $usbRoot 'whisper_models'
$env:XDG_CACHE_HOME = Join-Path $usbRoot 'whisper_models'
[Environment]::SetEnvironmentVariable('OLLAMA_MODELS', $env:OLLAMA_MODELS, 'User')

function Invoke-DownloadFile {
  param(
    [Parameter(Mandatory = $true)][string]$Url,
    [Parameter(Mandatory = $true)][string]$OutFile,
    [string]$ExpectedSha256 = ''
  )

  [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
  try {
    Invoke-WebRequest -Uri $Url -OutFile $OutFile
  } catch {
    throw "下載失敗: $Url，錯誤: $($_.Exception.Message)"
  }
  if ($ExpectedSha256) {
    $actual = (Get-FileHash -Path $OutFile -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $ExpectedSha256.ToLowerInvariant()) {
      throw "檔案雜湊驗證失敗: $OutFile"
    }
  }
}

function Ensure-EmbeddedPython {
  $pythonExe = Join-Path $usbRoot 'python_embed\python.exe'
  if (Test-Path $pythonExe) {
    Write-Host '[2/4] 已存在可攜式 Python，跳過下載。'
    return
  }

  Write-Host '[2/4] 下載可攜式 Python 3.10...'
  $zipFile = Join-Path $usbRoot 'python_embed.zip'
  if (-not $env:PYTHON_EMBED_SHA256) {
    throw "未提供 PYTHON_EMBED_SHA256。請先執行：`$env:PYTHON_EMBED_SHA256='官方 SHA256 值'"
  }
  Invoke-DownloadFile `
    -Url 'https://www.python.org/ftp/python/3.10.11/python-3.10.11-embed-amd64.zip' `
    -OutFile $zipFile `
    -ExpectedSha256 $env:PYTHON_EMBED_SHA256
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
    if (-not $env:GET_PIP_SHA256) {
      throw "未提供 GET_PIP_SHA256。請先執行：`$env:GET_PIP_SHA256='官方 SHA256 值'"
    }
    Invoke-DownloadFile -Url 'https://bootstrap.pypa.io/get-pip.py' -OutFile $getPip -ExpectedSha256 $env:GET_PIP_SHA256
    & $pythonExe $getPip --no-warn-script-location
    Remove-Item $getPip -Force
  }

  Write-Host '[3/4] 安裝 AI Python 依賴...'
  & $pythonExe -m pip install --no-warn-script-location requests tqdm openai-whisper
  $hasNvidia = [bool](Get-Command nvidia-smi -ErrorAction SilentlyContinue)
  if ($hasNvidia) {
    try {
      & $pythonExe -m pip install --no-warn-script-location torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
      return
    } catch {
      Write-Host 'CUDA 版本安裝失敗，改安裝 CPU 版本 PyTorch...'
    }
  } else {
    Write-Host '未偵測到 NVIDIA 環境，安裝 CPU 版本 PyTorch...'
  }

  & $pythonExe -m pip install --no-warn-script-location torch torchvision torchaudio
}

function Test-OllamaSetupSignature {
  param([Parameter(Mandatory = $true)][string]$FilePath)
  $sig = Get-AuthenticodeSignature -FilePath $FilePath
  if ($sig.Status -ne 'Valid') {
    return $false
  }
  return $sig.SignerCertificate.Subject -match '(^|, )O=Ollama Inc\.(,|$)'
}

function Ensure-Ollama {
  Write-Host '[4/4] 檢查 Ollama...'
  $existing = Get-Command ollama -ErrorAction SilentlyContinue
  if ($existing) {
    Write-Host '已偵測到系統 Ollama。'
    return
  }

  $setup = Join-Path $usbRoot 'ollama\OllamaSetup.exe'
  if (-not (Test-Path $setup)) {
    Write-Host '下載官方 Ollama 安裝程式...'
    Invoke-DownloadFile -Url 'https://ollama.com/download/OllamaSetup.exe' -OutFile $setup
  }

  if (-not (Test-OllamaSetupSignature -FilePath $setup)) {
    throw 'Ollama 安裝程式簽章驗證失敗。'
  }

  Write-Host '安裝 Ollama...'
  $installLog = Join-Path $usbRoot 'ollama\install.log'
  $proc = Start-Process -FilePath $setup -ArgumentList "/VERYSILENT /NORESTART /SUPPRESSMSGBOXES /LOG=`"$installLog`"" -Wait -PassThru
  if ($proc.ExitCode -ne 0) {
    throw "Ollama 安裝失敗，exit code: $($proc.ExitCode)"
  }
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
# 可透過環境變數覆寫：
#   WHISPER_MODEL: 例如 base/small/medium/large-v3
#   OLLAMA_TIMEOUT_SECONDS: Ollama API timeout 秒數
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "base")
OLLAMA_TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT_SECONDS", "180"))
MAX_PROMPT_LENGTH = 20000  # 超過會被截斷，避免單次請求過長導致本地 API 超時或記憶體壓力

os.environ["HF_HOME"] = os.path.join(USB_ROOT, "whisper_models")
os.environ["XDG_CACHE_HOME"] = os.path.join(USB_ROOT, "whisper_models")

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
PROCESSED_DIR = os.path.join(INPUT_DIR, "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🎬 啟動本地 Whisper... 硬體加速偵測: {device.upper()}")
if device == "cpu":
    print("⚠️ 警告：這台電腦未偵測到 NVIDIA 顯卡加速，將使用 CPU 慢速運算。")

try:
    model = whisper.load_model(WHISPER_MODEL, device=device)
except Exception as e:
    print(f"模型 {WHISPER_MODEL} 載入失敗，正在改用 base。錯誤: {e}")
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
            safe_text = raw_text.replace("\x00", "")[:MAX_PROMPT_LENGTH]
            with open(os.path.join(OUTPUT_DIR, f"{base_name}_原始逐字稿.txt"), "w", encoding="utf-8") as f:
                f.write(raw_text)
            payload = {
                "model": OLLAMA_MODEL,
                "prompt": f"請將以下文本翻譯並潤飾為流暢的繁體中文（台灣商務口吻）：\n\n{safe_text}",
                "stream": False,
            }
            res = requests.post(OLLAMA_API_URL, json=payload, timeout=OLLAMA_TIMEOUT)
            res.raise_for_status()
            trans_text = res.json().get("response", "").strip()
        except Exception as e:
            err = str(e)
            if "timed out" in err.lower():
                trans_text = f"Ollama 請求逾時（timeout={OLLAMA_TIMEOUT}s），僅保留逐字稿。錯誤: {err}"
            elif "connection" in err.lower() or "refused" in err.lower():
                trans_text = f"Ollama 連線失敗，請確認 ollama serve 已啟動。錯誤: {err}"
            else:
                trans_text = f"Ollama API 呼叫失敗，僅保留逐字稿。錯誤: {err}"
        with open(os.path.join(OUTPUT_DIR, f"{base_name}_最終繁中翻譯.txt"), "w", encoding="utf-8") as f:
            f.write(trans_text)
        os.rename(file_path, os.path.join(PROCESSED_DIR, file_name))
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
