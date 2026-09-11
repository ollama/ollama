# Ollama 本地安裝指南

歡迎使用 Ollama！本指南將幫助您在本地計算機上成功安裝和使用 Ollama。

## 系統要求

在開始安裝之前，請確保您的系統滿足以下要求：

### 硬體要求
- **記憶體 (RAM)**：
  - 至少 8 GB RAM 可運行 7B 模型
  - 至少 16 GB RAM 可運行 13B 模型
  - 至少 32 GB RAM 可運行 33B 模型
- **磁碟空間**：根據模型大小，至少需要 10-50 GB 的可用空間

### 作業系統支援
- macOS (Intel 或 Apple Silicon)
- Windows 10/11 (x64 或 ARM64)
- Linux (大多數發行版)

## 快速安裝

### macOS

1. **下載安裝程式**
   ```bash
   # 直接下載 DMG 檔案
   curl -L https://ollama.com/download/Ollama.dmg -o Ollama.dmg
   ```
   或者訪問 [官方下載頁面](https://ollama.com/download/Ollama.dmg)

2. **安裝**
   - 打開下載的 `Ollama.dmg` 檔案
   - 將 Ollama 拖曳到應用程式資料夾
   - 啟動 Ollama

3. **驗證安裝**
   ```bash
   ollama --version
   ```

### Windows

1. **下載安裝程式**
   ```powershell
   # 訪問官方下載頁面
   # https://ollama.com/download/OllamaSetup.exe
   ```
   或者直接下載 [OllamaSetup.exe](https://ollama.com/download/OllamaSetup.exe)

2. **安裝**
   - 執行下載的 `OllamaSetup.exe`
   - 按照安裝嚮導的指示進行操作
   - 完成安裝

3. **驗證安裝**
   ```powershell
   ollama --version
   ```

### Linux

1. **一鍵安裝**
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ```

2. **驗證安裝**
   ```bash
   ollama --version
   ```

3. **啟動服務**
   ```bash
   # 啟動 Ollama 服務
   ollama serve
   ```

## 從源碼構建（開發者選項）

如果您想從源碼構建 Ollama，請按照以下步驟操作：

### 前置要求

1. **安裝 Go 語言**（版本 1.22 或更高）
   - 訪問 [Go 官方網站](https://go.dev/doc/install) 下載並安裝

2. **安裝 C/C++ 編譯器**
   - **macOS**: Xcode Command Line Tools
     ```bash
     xcode-select --install
     ```
   - **Windows**: [TDM-GCC](https://github.com/jmeubank/tdm-gcc/releases/latest) (amd64) 或 [llvm-mingw](https://github.com/mstorsjo/llvm-mingw) (arm64)
   - **Linux**: GCC 或 Clang
     ```bash
     # Ubuntu/Debian
     sudo apt install build-essential
     
     # Fedora/CentOS
     sudo dnf install gcc gcc-c++
     ```

3. **安裝 CMake**（macOS Intel 和 Windows 需要）
   ```bash
   # macOS
   brew install cmake
   
   # Ubuntu/Debian
   sudo apt install cmake
   
   # Fedora/CentOS
   sudo dnf install cmake
   ```

### 構建步驟

1. **克隆儲存庫**
   ```bash
   git clone https://github.com/ollama/ollama.git
   cd ollama
   ```

2. **構建專案**

   **Linux / macOS (Apple Silicon)**
   ```bash
   go run . serve
   ```

   **macOS (Intel) / Windows**
   ```bash
   cmake -B build
   cmake --build build
   go run . serve
   ```

3. **運行本地構建**
   ```bash
   # 啟動服務
   ./ollama serve
   
   # 在另一個終端運行模型
   ./ollama run gemma3
   ```

## GPU 加速支援（可選）

### NVIDIA GPU (CUDA)
```bash
# 安裝 CUDA SDK
# 訪問 https://developer.nvidia.com/cuda-downloads
# CMake 會自動檢測並啟用 CUDA 支援
```

### AMD GPU (ROCm)
```bash
# 安裝 ROCm
# 訪問 https://rocm.docs.amd.com/en/latest/
```

### Intel/AMD GPU (Vulkan)
```bash
# 安裝 Vulkan SDK
# 訪問 https://vulkan.lunarg.com/sdk/home

# 設置環境變數（Windows 示例）
set VULKAN_SDK=C:\VulkanSDK\<version>
```

## 使用 Docker 安裝

如果您熟悉 Docker，可以使用官方 Docker 映像：

```bash
# 拉取並運行 Ollama
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama

# 在容器中運行模型
docker exec -it ollama ollama run gemma3
```

## 驗證安裝

安裝完成後，您可以通過以下步驟驗證 Ollama 是否正常工作：

### 使用自動驗證腳本

我們提供了一個自動驗證腳本來檢查您的安裝：

```bash
# 在 Ollama 源碼目錄中運行
bash scripts/verify-installation.sh
```

此腳本將檢查：
- Ollama 是否已安裝
- 服務是否正在運行
- 系統記憶體是否充足
- 磁碟空間是否足夠
- GPU 支援（如果有）

### 手動驗證步驟

#### 1. 檢查版本
```bash
ollama --version
```

#### 2. 啟動服務（如果尚未運行）
```bash
ollama serve
```

#### 3. 下載並運行您的第一個模型
```bash
# 運行 Gemma 3 模型（約 3.3GB）
ollama run gemma3

# 或運行更小的模型（約 815MB）
ollama run gemma3:1b
```

#### 4. 測試 API
```bash
curl http://localhost:11434/api/generate -d '{
  "model": "gemma3",
  "prompt":"為什麼天空是藍色的？"
}'
```

## 常見問題排除

### 問題 1: "ollama: command not found"

**解決方案**：
```bash
# 檢查 Ollama 是否已添加到 PATH
# Linux/macOS
export PATH=$PATH:/usr/local/bin

# Windows (PowerShell)
$env:PATH += ";C:\Program Files\Ollama"
```

### 問題 2: 記憶體不足

**解決方案**：
- 使用較小的模型（如 `gemma3:1b` 而不是 `gemma3:12b`）
- 關閉其他佔用記憶體的應用程式
- 增加系統交換空間

### 問題 3: 連接錯誤

**解決方案**：
```bash
# 確保 Ollama 服務正在運行
ollama serve

# 檢查端口 11434 是否被佔用
# Linux/macOS
lsof -i :11434

# Windows
netstat -ano | findstr :11434
```

### 問題 4: 下載速度慢

**解決方案**：
- 使用國內鏡像或代理
- 稍後重試
- 檢查網絡連接

### 問題 5: GPU 未被檢測

**解決方案**：
```bash
# 檢查 GPU 驅動程式是否已安裝
nvidia-smi  # NVIDIA GPU
rocm-smi    # AMD GPU

# 確保在構建時正確配置了 GPU 支援
```

## 下一步

安裝成功後，您可以：

1. **探索可用模型**：訪問 [Ollama 模型庫](https://ollama.com/library)
2. **學習 Modelfile**：閱讀 [Modelfile 文檔](https://github.com/ollama/ollama/blob/main/docs/modelfile.md)
3. **使用 API**：查看 [API 文檔](https://github.com/ollama/ollama/blob/main/docs/api.md)
4. **加入社群**：
   - [Discord](https://discord.gg/ollama)
   - [Reddit](https://reddit.com/r/ollama)

## 獲取幫助

如果您遇到任何問題：

1. 查看 [故障排除文檔](https://github.com/ollama/ollama/blob/main/docs/troubleshooting.md)
2. 搜索 [GitHub Issues](https://github.com/ollama/ollama/issues)
3. 在 [Discord](https://discord.gg/ollama) 上尋求幫助
4. 提交新的 [Issue](https://github.com/ollama/ollama/issues/new)

祝您使用愉快！🎉
