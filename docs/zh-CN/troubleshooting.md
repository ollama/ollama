# Ollama 故障排除指南 (Troubleshooting Guide)

本指南提供常見問題的解決方案，幫助您解決使用 Ollama 時遇到的問題。

## 安裝問題

### 問題 1: "ollama: command not found"

**症狀**：在終端輸入 `ollama` 命令後顯示找不到命令。

**可能原因**：
- Ollama 未正確安裝
- Ollama 未添加到系統 PATH

**解決方案**：

#### Linux/macOS
```bash
# 檢查 Ollama 是否已安裝
which ollama

# 如果找不到，將 Ollama 添加到 PATH
export PATH=$PATH:/usr/local/bin

# 永久添加到 PATH（bash）
echo 'export PATH=$PATH:/usr/local/bin' >> ~/.bashrc
source ~/.bashrc

# 永久添加到 PATH（zsh）
echo 'export PATH=$PATH:/usr/local/bin' >> ~/.zshrc
source ~/.zshrc
```

#### Windows
```powershell
# 檢查 Ollama 是否在 PATH 中
$env:PATH

# 添加 Ollama 到 PATH
$env:PATH += ";C:\Program Files\Ollama"

# 永久添加（需要管理員權限）
[Environment]::SetEnvironmentVariable("PATH", $env:PATH + ";C:\Program Files\Ollama", "Machine")
```

### 問題 2: 安裝腳本失敗（Linux）

**症狀**：運行 `curl -fsSL https://ollama.com/install.sh | sh` 失敗。

**解決方案**：

```bash
# 檢查網絡連接
ping -c 3 ollama.com

# 手動下載並檢查腳本
curl -fsSL https://ollama.com/install.sh -o install.sh
cat install.sh  # 檢查內容
bash install.sh

# 如果需要，使用代理
export http_proxy=http://proxy:port
export https_proxy=http://proxy:port
curl -fsSL https://ollama.com/install.sh | sh
```

### 問題 3: macOS 安全警告

**症狀**：macOS 阻止 Ollama 運行，顯示"無法驗證開發者"。

**解決方案**：

1. 打開"系統偏好設置" > "安全性與隱私"
2. 點擊"通用"標籤
3. 點擊"仍要打開" Ollama
4. 或在終端運行：
   ```bash
   xattr -d com.apple.quarantine /Applications/Ollama.app
   ```

### 問題 4: Windows Defender 警告

**症狀**：Windows Defender 阻止 Ollama 安裝或運行。

**解決方案**：

1. 打開 Windows 安全中心
2. 點擊"病毒和威脅防護"
3. 點擊"管理設置"
4. 將 Ollama 添加到排除項
5. 或在 PowerShell（管理員）中運行：
   ```powershell
   Add-MpPreference -ExclusionPath "C:\Program Files\Ollama"
   ```

## 服務運行問題

### 問題 5: "connection refused" 或無法連接到服務

**症狀**：運行 `ollama run` 或 API 調用時顯示連接被拒絕。

**可能原因**：
- Ollama 服務未啟動
- 端口被占用
- 防火牆阻止連接

**解決方案**：

```bash
# 檢查 Ollama 服務是否運行
# Linux/macOS
ps aux | grep ollama

# Windows
tasklist | findstr ollama

# 啟動服務
ollama serve

# 檢查端口 11434 是否被占用
# Linux/macOS
lsof -i :11434
netstat -an | grep 11434

# Windows
netstat -ano | findstr :11434

# 如果端口被占用，終止進程
# Linux/macOS
kill -9 <PID>

# Windows
taskkill /PID <PID> /F

# 使用其他端口
OLLAMA_HOST=0.0.0.0:11435 ollama serve
```

### 問題 6: 服務啟動後立即崩潰

**症狀**：運行 `ollama serve` 後服務立即退出。

**解決方案**：

```bash
# 查看詳細日誌
ollama serve --verbose

# 檢查日誌文件（如果有）
# Linux
journalctl -u ollama
tail -f /var/log/ollama.log

# macOS
tail -f ~/Library/Logs/Ollama/server.log

# Windows
Get-Content -Path "$env:LOCALAPPDATA\Ollama\logs\server.log" -Wait

# 檢查是否有舊進程
# 終止所有 Ollama 進程
pkill -9 ollama  # Linux/macOS
taskkill /F /IM ollama.exe  # Windows
```

## 模型下載問題

### 問題 7: 模型下載速度很慢

**症狀**：運行 `ollama pull` 或 `ollama run` 時下載速度極慢。

**解決方案**：

```bash
# 使用代理
export http_proxy=http://proxy:port
export https_proxy=http://proxy:port
ollama pull gemma3

# 檢查網絡速度
curl -o /dev/null http://speedtest.net/example.bin

# 稍後重試
# Ollama 支持斷點續傳

# 使用鏡像（如果可用）
# 設置環境變數
export OLLAMA_MODELS=/path/to/models
```

### 問題 8: 下載失敗或中斷

**症狀**：模型下載過程中出錯。

**解決方案**：

```bash
# 清理部分下載的文件
rm -rf ~/.ollama/models/blobs/sha256-partial-*

# 重新下載
ollama pull gemma3

# 檢查磁碟空間
df -h ~/.ollama

# 如果磁碟空間不足，清理舊模型
ollama list
ollama rm <model-name>
```

### 問題 9: "model not found" 錯誤

**症狀**：運行模型時顯示找不到模型。

**解決方案**：

```bash
# 列出已安裝的模型
ollama list

# 確認模型名稱和標籤
# 正確: ollama run gemma3:1b
# 錯誤: ollama run gemma3-1b

# 重新拉取模型
ollama pull gemma3:1b

# 檢查模型存儲目錄
ls -lh ~/.ollama/models/
```

## 記憶體和性能問題

### 問題 10: 記憶體不足 (Out of Memory)

**症狀**：運行模型時系統變慢或崩潰，顯示 OOM 錯誤。

**可能原因**：
- 系統 RAM 不足
- 同時運行多個大模型
- 其他程序占用記憶體

**解決方案**：

```bash
# 使用較小的模型
ollama run gemma3:1b  # 而不是 gemma3:12b

# 停止其他正在運行的模型
ollama ps
ollama stop <model-name>

# 關閉其他占用記憶體的應用程式

# 檢查系統記憶體使用情況
# Linux
free -h
top

# macOS
vm_stat
top

# Windows
tasklist
Get-Process | Sort-Object -Property WS -Descending | Select-Object -First 10

# 增加交換空間（Linux）
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### 問題 11: 推理速度很慢

**症狀**：模型生成響應的速度很慢。

**解決方案**：

```bash
# 使用較小的模型
ollama run gemma3:1b

# 確保 GPU 加速可用
# NVIDIA
nvidia-smi

# AMD
rocm-smi

# 檢查 GPU 是否被 Ollama 使用
# 在 ollama serve 的輸出中查看

# 減少上下文長度
ollama run gemma3 --ctx-size 2048

# 調整參數
# 創建 Modelfile
cat > Modelfile << 'EOF'
FROM gemma3
PARAMETER num_ctx 2048
PARAMETER num_gpu 1
EOF

ollama create fast-gemma -f Modelfile
ollama run fast-gemma
```

### 問題 12: GPU 未被檢測或使用

**症狀**：有 GPU 但 Ollama 只使用 CPU。

**解決方案**：

#### NVIDIA GPU (CUDA)
```bash
# 檢查 NVIDIA 驅動
nvidia-smi

# 檢查 CUDA 版本
nvcc --version

# 重新安裝 CUDA（如果需要）
# 訪問 https://developer.nvidia.com/cuda-downloads

# 驗證 Ollama 檢測到 GPU
ollama serve --verbose 2>&1 | grep -i cuda
```

#### AMD GPU (ROCm)
```bash
# 檢查 ROCm
rocm-smi

# 檢查 ROCm 版本
rocminfo

# 設置環境變數
export HSA_OVERRIDE_GFX_VERSION=10.3.0  # 根據您的 GPU 調整

# 重啟服務
ollama serve
```

#### Intel/AMD GPU (Vulkan)
```bash
# 檢查 Vulkan
vulkaninfo

# 設置環境變數
export VULKAN_SDK=/path/to/vulkan/sdk

# 重新構建 Ollama（如果從源碼安裝）
cmake -B build
cmake --build build
```

## API 和集成問題

### 問題 13: Python 集成失敗

**症狀**：使用 Python 庫時出錯。

**解決方案**：

```bash
# 更新 Python 庫
pip install --upgrade ollama

# 驗證安裝
python -c "import ollama; print(ollama.__version__)"

# 測試連接
python << 'EOF'
from ollama import chat
try:
    response = chat(model='gemma3:1b', messages=[{'role': 'user', 'content': 'hi'}])
    print("Success:", response['message']['content'])
except Exception as e:
    print("Error:", e)
EOF
```

### 問題 14: API 返回錯誤

**症狀**：使用 REST API 時返回錯誤。

**解決方案**：

```bash
# 檢查 API 端點
curl -v http://localhost:11434/api/version

# 檢查請求格式
curl http://localhost:11434/api/generate -d '{
  "model": "gemma3:1b",
  "prompt": "test"
}' -H "Content-Type: application/json"

# 啟用詳細日誌
OLLAMA_DEBUG=1 ollama serve

# 查看完整錯誤消息
curl -v http://localhost:11434/api/chat -d '{
  "model": "gemma3:1b",
  "messages": [{"role": "user", "content": "test"}]
}'
```

## Docker 相關問題

### 問題 15: Docker 容器無法啟動

**症狀**：Docker 容器啟動失敗。

**解決方案**：

```bash
# 檢查 Docker 日誌
docker logs ollama

# 檢查容器狀態
docker ps -a

# 重新創建容器
docker rm ollama
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama

# 使用 GPU（NVIDIA）
docker run -d --gpus=all -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama

# 檢查卷掛載
docker volume inspect ollama
```

### 問題 16: Docker 容器中無法訪問 GPU

**症狀**：Docker 容器中 GPU 不可用。

**解決方案**：

```bash
# 確保安裝了 nvidia-docker2
sudo apt-get install nvidia-docker2
sudo systemctl restart docker

# 使用正確的 GPU 標誌
docker run -d --gpus=all -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama

# 驗證 GPU 在容器中可用
docker exec -it ollama nvidia-smi
```

## 其他問題

### 問題 17: 模型輸出亂碼或質量差

**症狀**：模型生成的文本質量不佳或出現亂碼。

**解決方案**：

```bash
# 調整溫度參數
ollama run gemma3 --temperature 0.7

# 使用更大的模型
ollama run gemma3:4b

# 改進提示詞
ollama run gemma3 "請用中文回答：..."

# 創建自定義模型
cat > Modelfile << 'EOF'
FROM gemma3
PARAMETER temperature 0.8
PARAMETER top_p 0.9
SYSTEM "你是一個專業的助手，請用清晰準確的中文回答問題。"
EOF

ollama create my-model -f Modelfile
ollama run my-model
```

### 問題 18: 無法訪問 Ollama 網頁界面

**症狀**：期望有網頁界面但找不到。

**說明**：Ollama 本身不提供網頁界面，但您可以使用第三方界面。

**解決方案**：

```bash
# 使用 Open WebUI
docker run -d -p 3000:8080 \
  --add-host=host.docker.internal:host-gateway \
  -v open-webui:/app/backend/data \
  --name open-webui \
  --restart always \
  ghcr.io/open-webui/open-webui:main

# 訪問 http://localhost:3000

# 或使用其他界面
# 參見 README.md 中的 Community Integrations 部分
```

### 問題 19: 更新後出現問題

**症狀**：更新 Ollama 後出現新問題。

**解決方案**：

```bash
# 清理緩存
rm -rf ~/.ollama/cache

# 重新構建（如果從源碼安裝）
go clean -cache
cmake -B build
cmake --build build

# 回滾到舊版本（如果需要）
# Linux
curl -fsSL https://ollama.com/install.sh | sh -s -- --version 0.x.x

# 重新下載模型
ollama pull gemma3
```

## 獲取更多幫助

如果上述解決方案都無法解決您的問題：

1. **查看官方文檔**
   - [GitHub 倉庫](https://github.com/ollama/ollama)
   - [英文故障排除文檔](../troubleshooting.md)

2. **搜索已知問題**
   - [GitHub Issues](https://github.com/ollama/ollama/issues)
   - 使用關鍵詞搜索類似問題

3. **提問和尋求幫助**
   - [Discord 社群](https://discord.gg/ollama)
   - [Reddit r/ollama](https://reddit.com/r/ollama)

4. **提交問題報告**
   - 訪問 [GitHub Issues](https://github.com/ollama/ollama/issues/new)
   - 包含以下信息：
     - 作業系統和版本
     - Ollama 版本
     - 完整錯誤消息
     - 復現步驟
     - 系統日誌

## 診斷信息收集

提交問題報告時，請包含以下診斷信息：

```bash
# 創建診斷報告
cat > diagnostic.txt << 'EOF'
=== System Information ===
EOF

# 作業系統
uname -a >> diagnostic.txt
cat /etc/os-release >> diagnostic.txt 2>&1

# Ollama 版本
echo "=== Ollama Version ===" >> diagnostic.txt
ollama --version >> diagnostic.txt 2>&1

# 已安裝的模型
echo "=== Installed Models ===" >> diagnostic.txt
ollama list >> diagnostic.txt 2>&1

# 運行中的進程
echo "=== Running Processes ===" >> diagnostic.txt
ps aux | grep ollama >> diagnostic.txt 2>&1

# 網絡連接
echo "=== Network ===" >> diagnostic.txt
curl -v http://localhost:11434/api/version >> diagnostic.txt 2>&1

# GPU 信息
echo "=== GPU Information ===" >> diagnostic.txt
nvidia-smi >> diagnostic.txt 2>&1
rocm-smi >> diagnostic.txt 2>&1

# 系統資源
echo "=== System Resources ===" >> diagnostic.txt
free -h >> diagnostic.txt 2>&1
df -h >> diagnostic.txt 2>&1

cat diagnostic.txt
```

---

最後更新：2025-12-30
