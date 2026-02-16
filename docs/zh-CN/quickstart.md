# Ollama 快速入門指南

本指南將幫助您快速開始使用 Ollama 運行本地大型語言模型。

## 第一步：安裝 Ollama

### macOS
```bash
# 下載並安裝
curl -L https://ollama.com/download/Ollama.dmg -o Ollama.dmg
# 或訪問 https://ollama.com/download
```

### Windows
下載 [OllamaSetup.exe](https://ollama.com/download/OllamaSetup.exe) 並運行安裝程式

### Linux
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

## 第二步：運行您的第一個模型

### 使用命令行

打開終端並運行：

```bash
ollama run gemma3
```

這將下載並啟動 Gemma 3 模型（約 3.3GB）。首次運行可能需要幾分鐘來下載模型。

### 開始對話

模型啟動後，您可以直接開始對話：

```
>>> 你好！請介紹一下自己
我是一個AI助手，基於Gemma 3模型...

>>> 請幫我寫一首關於春天的詩
春風拂面暖陽照...
```

### 退出對話

輸入 `/bye` 或按 `Ctrl+D` 退出。

## 第三步：探索更多模型

### 查看可用模型

訪問 [Ollama 模型庫](https://ollama.com/library) 查看所有可用模型。

### 推薦模型

> **注意**：沒有標籤的模型名稱（如 `gemma3`）會自動使用默認版本（通常是最平衡的版本）。

| 模型名稱 | 參數量 | 大小 | 下載命令 | 適用場景 |
|---------|--------|------|----------|---------|
| Gemma 3 | 1B | 815MB | `ollama run gemma3:1b` | 輕量級對話 |
| Gemma 3 | 4B | 3.3GB | `ollama run gemma3` | 通用對話（默認） |
| Llama 3.2 | 1B | 1.3GB | `ollama run llama3.2:1b` | 快速響應 |
| Llama 3.2 | 3B | 2.0GB | `ollama run llama3.2` | 平衡性能（默認） |
| QwQ | 32B | 20GB | `ollama run qwq` | 複雜推理 |
| DeepSeek-R1 | 7B | 4.7GB | `ollama run deepseek-r1` | 代碼生成 |
| CodeLlama | 7B | 3.8GB | `ollama run codellama` | 編程助手 |
| Llava | 7B | 4.5GB | `ollama run llava` | 圖像理解 |

### 運行不同的模型

```bash
# 運行較小的模型（更快但能力較弱）
ollama run gemma3:1b

# 運行代碼專用模型
ollama run codellama

# 運行支持圖像的多模態模型
ollama run llava
```

## 第四步：常用命令

### 模型管理

```bash
# 列出已下載的模型
ollama list

# 拉取（下載）模型但不運行
ollama pull llama3.2

# 刪除模型以節省空間
ollama rm gemma3

# 複製模型
ollama cp gemma3 my-model

# 查看模型信息
ollama show gemma3
```

### 查看運行狀態

```bash
# 查看當前運行的模型
ollama ps

# 停止運行中的模型
ollama stop gemma3
```

### 啟動服務

```bash
# 手動啟動 Ollama 服務
ollama serve
```

## 第五步：使用 API

Ollama 提供 REST API，可以在您的應用程式中使用。

### 生成響應

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "gemma3",
  "prompt": "為什麼天空是藍色的？"
}'
```

### 對話模式

```bash
curl http://localhost:11434/api/chat -d '{
  "model": "gemma3",
  "messages": [
    {
      "role": "user",
      "content": "你好！"
    }
  ]
}'
```

### 使用 Python

```python
# 安裝 Python 庫
pip install ollama

# 使用 Python 與 Ollama 對話
from ollama import chat

response = chat(model='gemma3', messages=[
  {
    'role': 'user',
    'content': '你好！請介紹一下自己',
  },
])
print(response['message']['content'])
```

### 使用 JavaScript

```bash
# 安裝 JavaScript 庫
npm install ollama
```

```javascript
import ollama from 'ollama'

const response = await ollama.chat({
  model: 'gemma3',
  messages: [{ role: 'user', content: '你好！' }],
})
console.log(response.message.content)
```

## 第六步：創建自定義模型

### 使用 Modelfile

創建一個名為 `Modelfile` 的檔案：

```
FROM gemma3

# 設置溫度（較高值更有創意，較低值更連貫）
PARAMETER temperature 1

# 設置系統提示
SYSTEM """
你是一個專業的中文助手，善於用簡潔清晰的語言回答問題。
"""
```

### 創建並運行模型

```bash
# 創建自定義模型
ollama create my-chinese-assistant -f ./Modelfile

# 運行自定義模型
ollama run my-chinese-assistant
```

## 第七步：多行輸入

如果需要輸入多行文本，使用三引號：

```
>>> """請幫我總結以下內容：
... Ollama 是一個開源的大型語言模型運行工具
... 它支持在本地運行各種 AI 模型
... 無需連接互聯網即可使用
... """
```

## 第八步：處理圖像（多模態模型）

使用支持圖像的模型（如 Llava）：

```bash
ollama run llava "這張圖片裡有什麼？ /path/to/image.jpg"
```

## 進階技巧

### 1. 使用命令行參數傳遞提示

```bash
ollama run gemma3 "總結這個文件的內容：$(cat README.md)"
```

### 2. 生成嵌入向量

```bash
ollama run embeddinggemma "要嵌入的文本"

# 或使用管道
echo "要嵌入的文本" | ollama run embeddinggemma
```

### 3. 導入 GGUF 模型

```bash
# 創建 Modelfile
echo "FROM ./my-model.gguf" > Modelfile

# 創建並運行
ollama create my-model -f Modelfile
ollama run my-model
```

### 4. 從 Safetensors 導入

參考 [導入指南](https://docs.ollama.com/import) 了解詳細步驟。

## 性能優化建議

### 記憶體要求
- **7B 模型**：至少 8 GB RAM
- **13B 模型**：至少 16 GB RAM
- **33B 模型**：至少 32 GB RAM

### 選擇合適的模型
- 如果記憶體有限，選擇較小的模型（1B-3B）
- 如果需要更好的性能，選擇較大的模型（7B+）
- 對於特定任務，選擇專門的模型（如 CodeLlama 用於編程）

### GPU 加速
- NVIDIA GPU：確保安裝了 CUDA
- AMD GPU：確保安裝了 ROCm
- Intel/AMD GPU：確保安裝了 Vulkan SDK

## 常見問題

### 1. 模型下載很慢怎麼辦？
- 檢查網絡連接
- 稍後重試
- 考慮使用代理或 VPN

### 2. 如何節省磁碟空間？
```bash
# 刪除不需要的模型
ollama rm model-name

# 只保留必要的模型
ollama list
```

### 3. 如何更新模型？
```bash
# 重新拉取模型（只下載差異部分）
ollama pull gemma3
```

### 4. 如何在後台運行服務？
```bash
# Linux/macOS
nohup ollama serve &

# Windows
# Ollama 會自動作為服務運行
```

## 獲取更多幫助

- **完整安裝指南**：[docs/zh-CN/install.md](./install.md)
- **API 文檔**：[docs/api.md](../api.md)
- **Modelfile 文檔**：[docs/modelfile.md](../modelfile.md)
- **故障排除**：[docs/troubleshooting.md](../troubleshooting.md)
- **社群支援**：
  - [Discord](https://discord.gg/ollama)
  - [Reddit](https://reddit.com/r/ollama)
  - [GitHub Issues](https://github.com/ollama/ollama/issues)

## 下一步

現在您已經掌握了 Ollama 的基本使用，可以：

1. 探索更多模型和應用場景
2. 將 Ollama 集成到您的應用程式中
3. 創建自定義模型以滿足特定需求
4. 加入社群分享經驗和獲取幫助

祝您使用愉快！🚀
