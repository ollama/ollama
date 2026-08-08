# Ollama 繁體中文快速入門指南

Ollama 是一個讓您能在本地端輕鬆執行、管理與擴充大型語言模型（LLM）的開源架構。不論是在 macOS、Windows 還是 Linux 上，Ollama 都提供了極佳的效能、極簡的命令列工具（CLI）、豐富的模型庫以及標準 REST API，協助開發者快速建構生成式 AI 應用。

---

## 目錄

1. [系統安裝](#系統安裝)
   - [macOS](#macos)
   - [Windows](#windows)
   - [Linux](#linux)
   - [Docker 容器部署](#docker-容器部署)
2. [基本操作與 CLI 命令](#基本操作與-cli-命令)
   - [互動式選單](#互動式選單)
   - [下載與執行模型](#下載與執行模型)
   - [多行輸入與多模態圖片分析](#多行輸入與多模態圖片分析)
   - [常用管理指令表](#常用管理指令表)
3. [工具與 Agent 整合 (ollama launch)](#工具與-agent-整合-ollama-launch)
4. [開發者 API 與 SDK](#開發者-api-與-sdk)
   - [REST API](#rest-api)
   - [Python SDK](#python-sdk)
   - [JavaScript / TypeScript SDK](#javascript--typescript-sdk)
5. [自訂模型 (Modelfile)](#自訂模型-modelfile)
6. [進階環境設定與服務管理](#進階環境設定與服務管理)

---

## 系統安裝

### macOS

可以使用官方 Shell 安裝腳本：

```shell
curl -fsSL https://ollama.com/install.sh | sh
```

或是直接造訪 [Ollama 下載頁面](https://ollama.com/download/Ollama.dmg) 下載 `.dmg` 安裝檔安裝。

### Windows

在 PowerShell 中執行以下命令進行安裝：

```shell
irm https://ollama.com/install.ps1 | iex
```

或是手動下載 [OllamaSetup.exe](https://ollama.com/download/OllamaSetup.exe) 安裝檔。

### Linux

在 Linux 終端機中執行：

```shell
curl -fsSL https://ollama.com/install.sh | sh
```

如需進行手動安裝或詳細設定，請參考 [Linux 手動安裝指南](https://docs.ollama.com/linux#manual-install)。

### Docker 容器部署

Ollama 官方在 Docker Hub 上提供映像檔 `ollama/ollama`：

```shell
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
```

若需啟用 GPU 加速，請依據宿主機環境安裝 NVIDIA Container Toolkit 並加上 `--gpus=all` 參數。

---

## 基本操作與 CLI 命令

### 互動式選單

安裝完成後，直接在終端機中輸入 `ollama`：

```shell
ollama
```

系統將開啟主互動選單，協助您快速選取模型執行對話，或是配置外部工具整合。

### 下載與執行模型

使用 `ollama run` 命令可以直接下載並執行指定模型（例如 Gemma 4）：

```shell
ollama run gemma4
```

若模型不在本地端，Ollama 會自動從模型庫拉取（Pull）該模型並開始對話。

若想執行雲端端點模型，可以加上 `:cloud` 標籤：

```shell
ollama run gemma4:cloud
```

若只需下載模型而不立即對話，可使用 `ollama pull`：

```shell
ollama pull gemma4
```

### 多行輸入與多模態圖片分析

#### 多行文字輸入

在 `ollama run` 對話介面中，如果想輸入多行文字，可以使用 `"""` 引號包裹：

```text
>>> """你好！
... 請幫我撰寫一段 Python 的快速排序演算法。
... 感謝！
... """
```

#### 多模態模型分析圖片

使用支援視覺分析的多模態模型時，可在提示詞中直接傳入圖片的路徑：

```shell
ollama run gemma4 "請描述這張圖片的內容：/Users/username/Desktop/example.png"
```

退出對話輸入介面請輸入：

```shell
/bye
```

### 常用管理指令表

| 指令 | 說明 | 範例 |
| --- | --- | --- |
| `ollama ls` | 列出本地已下載的模型 | `ollama ls` |
| `ollama ps` | 列出目前正在記憶體中執行的模型 | `ollama ps` |
| `ollama stop <模型名稱>` | 停止指定正在執行的模型 | `ollama stop gemma4` |
| `ollama rm <模型名稱>` | 刪除本地模型 | `ollama rm gemma4` |
| `ollama create` | 根據 `Modelfile` 建立自訂模型 | `ollama create my-model -f Modelfile` |
| `ollama serve` | 手動啟動 Ollama 伺服器服務 | `ollama serve` |

---

## 工具與 Agent 整合 (ollama launch)

Ollama 提供全新的 `ollama launch` 指令，可自動配置並啟動支援的第三方 AI Agent 與程式碼編輯器：

### 啟動互動式選單

```shell
ollama launch
```

### 快速對接特定工具

- **Claude Code**：
  ```shell
  ollama launch claude
  ```
- **OpenClaw**（將 Ollama 接入 WhatsApp、Telegram、Discord 等社群）：
  ```shell
  ollama launch openclaw
  ```
- **OpenCode**：
  ```shell
  ollama launch opencode
  ```
- **Codex**：
  ```shell
  ollama launch codex
  ```

### 指定模型啟動

```shell
ollama launch claude --model qwen3.5
```

---

## 開發者 API 與 SDK

Ollama 預設會在 `http://localhost:11434` 啟動 HTTP 服務。

### REST API

透過 `POST /api/chat` 發送對話請求：

```bash
curl http://localhost:11434/api/chat -d '{
  "model": "gemma4",
  "messages": [
    {
      "role": "user",
      "content": "為什麼天空是藍色的？"
    }
  ],
  "stream": false
}'
```

### Python SDK

安裝官方 Python 套件：

```bash
pip install ollama
```

撰寫 Python 程式碼：

```python
from ollama import chat

response = chat(
    model='gemma4',
    messages=[
        {
            'role': 'user',
            'content': '請用一句話介紹 Ollama。',
        },
    ],
)

print(response.message.content)
```

### JavaScript / TypeScript SDK

安裝官方 Node.js 套件：

```bash
npm install ollama
```

撰寫 JavaScript 程式碼：

```javascript
import ollama from 'ollama';

const response = await ollama.chat({
  model: 'gemma4',
  messages: [{ role: 'user', content: '請用一句話介紹 Ollama。' }],
});

console.log(response.message.content);
```

---

## 自訂模型 (Modelfile)

您可以使用 `Modelfile` 建立專屬的模型微調配置。

1. 建立名為 `Modelfile` 的檔案：

```dockerfile
FROM gemma4

# 設定系統提示詞 (System Prompt)
SYSTEM """你是一名專業且有禮貌的繁體中文 AI 助手。"""

# 設定模型參數 (Temperature 溫度)
PARAMETER temperature 0.7
```

2. 執行建構指令：

```shell
ollama create custom-gemma4 -f Modelfile
```

3. 執行自訂模型：

```shell
ollama run custom-gemma4
```

---

## 進階環境設定與服務管理

常用的 Ollama 環境變數（可於系統或 `.bashrc` / `.zshrc` 中設定）：

- `OLLAMA_HOST`：設定服務監聽的 IP 與連接埠（例如 `0.0.0.0:11434` 允許外部網路連線）。
- `OLLAMA_MODELS`：自訂模型存放路徑（預設在 macOS 為 `~/.ollama/models`，Windows 為 `C:\Users\<user>\.ollama\models`）。
- `OLLAMA_KEEP_ALIVE`：設定模型在記憶體中保留閒置的時間（例如 `5m` 或 `-1` 永久保留）。

參閱官方文件以取得更多資源：
- [CLI 參考手冊](https://docs.ollama.com/cli)
- [API 參考手冊](https://docs.ollama.com/api)
- [Modelfile 指南](https://docs.ollama.com/modelfile)
