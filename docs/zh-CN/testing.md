# Ollama 安裝測試指南 (Installation Test Guide)

本文檔提供完整的測試流程，幫助您驗證 Ollama 是否正確安裝並能正常使用。

## 前置測試清單

在安裝 Ollama 之前，請先確認以下事項：

### 1. 系統要求檢查

```bash
# 檢查作業系統版本
# Check operating system version

# Linux
cat /etc/os-release

# macOS
sw_vers

# Windows (PowerShell)
systeminfo | findstr /B /C:"OS Name" /C:"OS Version"
```

### 2. 磁碟空間檢查

```bash
# Linux/macOS
df -h ~

# Windows (PowerShell)
Get-PSDrive C
```

建議至少有 20GB 可用空間用於存儲模型。

### 3. 記憶體檢查

```bash
# Linux
free -h

# macOS
sysctl hw.memsize

# Windows (PowerShell)
Get-CimInstance Win32_ComputerSystem | Select-Object TotalPhysicalMemory
```

## 安裝測試流程

### 步驟 1: 安裝 Ollama

根據您的作業系統，按照 [安裝指南](./install.md) 進行安裝。

### 步驟 2: 運行驗證腳本

```bash
cd ollama
bash scripts/verify-installation.sh
```

預期輸出應包含：
- ✓ Ollama 已安裝
- ✓ 記憶體充足
- 磁碟空間信息
- GPU 檢測結果（如果有）

### 步驟 3: 測試基本命令

```bash
# 檢查版本
ollama --version

# 查看幫助信息
ollama --help

# 列出可用命令
ollama
```

### 步驟 4: 啟動服務

在一個終端窗口中運行：

```bash
ollama serve
```

預期輸出類似：
```
Ollama server starting...
Listening on 127.0.0.1:11434
```

保持此終端開啟。

### 步驟 5: 測試 API 連接

在另一個終端窗口中運行：

```bash
curl http://localhost:11434/api/version
```

預期輸出：
```json
{"version":"x.x.x"}
```

### 步驟 6: 下載並運行模型

#### 測試小型模型（推薦首次測試）

```bash
# 下載並運行 gemma3:1b（約 815MB）
ollama run gemma3:1b
```

#### 測試對話功能

模型載入後，嘗試以下測試：

```
>>> 你好！
[等待模型響應]

>>> 請寫一個 Hello World 程序
[等待模型響應]

>>> /bye
[退出]
```

### 步驟 7: 測試 API 調用

```bash
# 測試生成 API
curl http://localhost:11434/api/generate -d '{
  "model": "gemma3:1b",
  "prompt": "你好，請介紹一下自己",
  "stream": false
}'

# 測試聊天 API
curl http://localhost:11434/api/chat -d '{
  "model": "gemma3:1b",
  "messages": [
    {
      "role": "user",
      "content": "什麼是機器學習？"
    }
  ],
  "stream": false
}'
```

### 步驟 8: 測試模型管理

```bash
# 列出已安裝的模型
ollama list

# 查看模型信息
ollama show gemma3:1b

# 查看運行中的模型
ollama ps

# 停止模型
ollama stop gemma3:1b
```

## 進階測試

### 測試 Python 集成

```bash
# 安裝 Python 庫
pip install ollama

# 創建測試腳本
cat > test_ollama.py << 'EOF'
from ollama import chat

try:
    response = chat(model='gemma3:1b', messages=[
        {
            'role': 'user',
            'content': 'Hello! Can you respond in one sentence?',
        },
    ])
    print("✓ Python integration successful!")
    print(f"Response: {response['message']['content']}")
except Exception as e:
    print(f"✗ Python integration failed: {e}")
EOF

# 運行測試
python test_ollama.py
```

### 測試 JavaScript 集成

```bash
# 創建測試項目
mkdir test-ollama-js
cd test-ollama-js
npm init -y
npm install ollama

# 創建測試腳本
cat > test.js << 'EOF'
import ollama from 'ollama';

async function test() {
  try {
    const response = await ollama.chat({
      model: 'gemma3:1b',
      messages: [{ role: 'user', content: 'Hello! Can you respond in one sentence?' }],
    });
    console.log('✓ JavaScript integration successful!');
    console.log('Response:', response.message.content);
  } catch (error) {
    console.log('✗ JavaScript integration failed:', error);
  }
}

test();
EOF

# 運行測試
node test.js
```

### 測試自定義模型

```bash
# 創建 Modelfile
cat > Modelfile << 'EOF'
FROM gemma3:1b

PARAMETER temperature 0.8

SYSTEM """
你是一個友善的中文助手。
"""
EOF

# 創建自定義模型
ollama create test-model -f Modelfile

# 測試自定義模型
ollama run test-model "你好！"

# 清理測試模型
ollama rm test-model
```

## 性能測試

### 測試響應時間

```bash
# 創建性能測試腳本
cat > perf_test.sh << 'EOF'
#!/bin/bash
echo "Testing Ollama response time..."

start_time=$(date +%s)
ollama run gemma3:1b "Say hello" --verbose
end_time=$(date +%s)

elapsed=$((end_time - start_time))
echo "Total time: ${elapsed} seconds"
EOF

chmod +x perf_test.sh
./perf_test.sh
```

### 測試記憶體使用

```bash
# Linux
watch -n 1 'ps aux | grep ollama'

# macOS
while true; do ps aux | grep ollama; sleep 1; done
```

## 故障排除測試

### 測試端口占用

```bash
# 檢查 11434 端口是否可用
# Linux/macOS
lsof -i :11434

# Windows
netstat -ano | findstr :11434
```

### 測試網絡連接

```bash
# 測試本地連接
curl -v http://localhost:11434/api/version

# 測試從其他機器連接（如果需要）
curl -v http://your-ip:11434/api/version
```

### 測試日誌輸出

```bash
# 查看 Ollama 服務日誌
# 在運行 ollama serve 的終端查看輸出

# 或者重定向到文件
ollama serve > ollama.log 2>&1 &
tail -f ollama.log
```

## 測試結果評估

### 成功標準

安裝被認為成功，如果：

- [ ] Ollama 命令可用
- [ ] 服務可以啟動
- [ ] API 端點可訪問
- [ ] 至少一個模型可以成功運行
- [ ] 模型可以生成有意義的響應
- [ ] API 調用返回預期結果
- [ ] 模型管理命令工作正常

### 性能基準

參考性能（gemma3:1b 在現代硬件上）：

- 首次載入時間：< 30 秒
- 後續載入時間：< 5 秒
- 單次推理時間：< 3 秒（短提示）
- 記憶體佔用：< 2 GB

注意：實際性能取決於硬件配置。

## 清理測試環境

完成測試後，如果需要清理：

```bash
# 停止服務
pkill ollama

# 刪除測試模型
ollama rm gemma3:1b

# 刪除所有模型（謹慎使用）
# ollama list | awk 'NR>1 {print $1}' | xargs -I {} ollama rm {}

# 清理 Python 測試文件
rm -f test_ollama.py

# 清理 JavaScript 測試目錄
rm -rf test-ollama-js
```

## 報告問題

如果測試失敗，請收集以下信息：

1. 作業系統和版本
2. Ollama 版本 (`ollama --version`)
3. 錯誤消息的完整輸出
4. 系統日誌（如果可用）
5. 硬件配置（CPU、RAM、GPU）

然後在 [GitHub Issues](https://github.com/ollama/ollama/issues) 提交問題報告。

## 額外資源

- [完整安裝指南](./install.md)
- [快速入門指南](./quickstart.md)
- [API 文檔](../api.md)
- [故障排除指南](./troubleshooting.md)

---

最後更新：2025-12-30
