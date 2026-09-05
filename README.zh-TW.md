<p align="center">
  <a href="https://ollama.com">
    <img src="https://github.com/ollama/ollama/assets/3325447/0d0b44e2-8f4a-4e99-9b52-a5c1c741c8f7" alt="ollama" width="200"/>
  </a>
</p>

# Ollama

開始使用開放模型進行開發。

## 下載安裝

### macOS

```shell
curl -fsSL https://ollama.com/install.sh | sh
```

或 [手動下載](https://ollama.com/download/Ollama.dmg)

### Windows

```shell
irm https://ollama.com/install.ps1 | iex
```

或 [手動下載](https://ollama.com/download/OllamaSetup.exe)

### Linux

```shell
curl -fsSL https://ollama.com/install.sh | sh
```

[手動安裝說明](https://docs.ollama.com/linux#manual-install)

### Docker

官方 [Ollama Docker 映像檔](https://hub.docker.com/r/ollama/ollama) `ollama/ollama` 已在 Docker Hub 上提供。

### 官方 SDK 函式庫

- [ollama-python](https://github.com/ollama/ollama-python)
- [ollama-js](https://github.com/ollama/ollama-js)

### 社群頻道

- [Discord](https://discord.gg/ollama)
- [𝕏 (Twitter)](https://x.com/ollama)
- [Reddit](https://reddit.com/r/ollama)

## 快速開始

```
ollama
```

執行後系統將引導您執行模型，或將 Ollama 連接到您現有的 Agent 或應用程式，例如 `Claude Code`、`OpenClaw`、`OpenCode`、`Codex`、`Copilot` 等。

### 程式編碼 (Coding)

若要啟動特定的整合工具：

```
ollama launch claude
```

支援的整合工具包含 [Claude Code](https://docs.ollama.com/integrations/claude-code)、[Codex](https://docs.ollama.com/integrations/codex)、[Copilot CLI](https://docs.ollama.com/integrations/copilot-cli)、[Droid](https://docs.ollama.com/integrations/droid) 以及 [OpenCode](https://docs.ollama.com/integrations/opencode)。

### AI 助理 (AI assistant)

使用 [OpenClaw](https://docs.ollama.com/integrations/openclaw) 將 Ollama 轉換為跨 WhatsApp、Telegram、Slack、Discord 等平台的個人 AI 助理：

```
ollama launch openclaw
```

### 與模型對話 (Chat with a model)

執行並與 [Gemma 4](https://ollama.com/library/gemma4) 對話：

```
ollama run gemma4
```

請造訪 [ollama.com/library](https://ollama.com/library) 查看完整模型清單。

參閱 [快速入門指南](https://docs.ollama.com/quickstart) 以取得更多細節。

## REST API

Ollama 提供 REST API 用於執行與管理模型。

```
curl http://localhost:11434/api/chat -d '{
  "model": "gemma4",
  "messages": [{
    "role": "user",
    "content": "Why is the sky blue?"
  }],
  "stream": false
}'
```

請參閱 [API 文件](https://docs.ollama.com/api) 以了解所有 API 端點。

### Python

```
pip install ollama
```

```python
from ollama import chat

response = chat(model='gemma4', messages=[
  {
    'role': 'user',
    'content': 'Why is the sky blue?',
  },
])
print(response.message.content)
```

### JavaScript

```
npm i ollama
```

```javascript
import ollama from "ollama";

const response = await ollama.chat({
  model: "gemma4",
  messages: [{ role: "user", content: "Why is the sky blue?" }],
});
console.log(response.message.content);
```

## 支援的後端引擎

- 由 Georgi Gerganov 發起的 [llama.cpp](https://github.com/ggml-org/llama.cpp) 專案。

## 官方文件

- [CLI 參考手冊](https://docs.ollama.com/cli)
- [REST API 參考手冊](https://docs.ollama.com/api)
- [匯入模型指南](https://docs.ollama.com/import)
- [Modelfile 參考手冊](https://docs.ollama.com/modelfile)
- [從原始碼編譯](https://github.com/ollama/ollama/blob/main/docs/development.md)

## 社群整合工具

> 想將您的專案加入清單？歡迎提交 Pull Request。

### 對話介面 (Chat Interfaces)

#### Web 網頁介面

- [Open WebUI](https://github.com/open-webui/open-webui) - 可擴充、可自建自託管的 AI 介面
- [Onyx](https://github.com/onyx-dot-app/onyx) - 連結企業知識庫的 AI 工作空間
- [LibreChat](https://github.com/danny-avila/LibreChat) - 支援多提供者的增強版 ChatGPT 複製專案
- [Lobe Chat](https://github.com/lobehub/lobe-chat) - 具有外掛生態系統的現代化對話框架（[文件](https://lobehub.com/docs/self-hosting/examples/ollama)）
- [NextChat](https://github.com/ChatGPTNextWeb/ChatGPT-Next-Web) - 跨平台 ChatGPT UI（[文件](https://docs.nextchat.dev/models/ollama)）
- [Perplexica](https://github.com/ItzCrazyKns/Perplexica) - AI 驅動的搜尋引擎，開源版 Perplexity 替代方案
- [big-AGI](https://github.com/enricoros/big-AGI) - 專為專業人士打造的 AI 套件
- [Lollms WebUI](https://github.com/ParisNeo/lollms-webui) - 多模型 Web 介面
- [ChatOllama](https://github.com/sugarforever/chat-ollama) - 整合知識庫的聊天機器人
- [Bionic GPT](https://github.com/bionic-gpt/bionic-gpt) - 企業地端自建 AI 平台
- [Chatbot UI](https://github.com/ivanfioravanti/chatbot-ollama) - ChatGPT 風格的 Web 介面
- [Hollama](https://github.com/fmaclen/hollama) - 極簡風格 Web 介面
- [Chatbox](https://github.com/Bin-Huang/Chatbox) - 桌面與 Web AI 客戶端
- [chat](https://github.com/swuecho/chat) - 專為團隊設計的對話 Web 應用程式
- [Ollama RAG Chatbot](https://github.com/datvodinh/rag-chatbot.git) - 使用 RAG 技術與多個 PDF 文件對話
- [Tkinter-based client](https://github.com/chyok/ollama-gui) - Python 桌面端客戶端

#### Desktop 桌面端

- [Dify.AI](https://github.com/langgenius/dify) - LLM 應用開發平台
- [AnythingLLM](https://github.com/Mintplex-Labs/anything-llm) - 適用於 Mac、Windows 與 Linux 的全功能 AI 應用程式
- [Maid](https://github.com/Mobile-Artificial-Intelligence/maid) - 跨平台行動與桌面端客戶端
- [Witsy](https://github.com/nbonamy/witsy) - 適用於 Mac、Windows 與 Linux 的 AI 桌面應用程式
- [Cherry Studio](https://github.com/kangfenmao/cherry-studio) - 支援多模型的桌面端客戶端
- [Ollama App](https://github.com/JHubi1/ollama-app) - 適用於桌面與行動裝置的多平台客戶端
- [PyGPT](https://github.com/szczyglis-dev/py-gpt) - 適用於 Linux、Windows 與 Mac 的 AI 桌面助理
- [Alpaca](https://github.com/Jeffser/Alpaca) - 適用於 Linux 與 macOS 的 GTK4 客戶端
- [SwiftChat](https://github.com/aws-samples/swift-chat) - 包含 iOS、Android 與 Apple Vision Pro 的跨平台應用
- [Enchanted](https://github.com/AugustDev/enchanted) - 原生 macOS 與 iOS 客戶端
- [RWKV-Runner](https://github.com/josStorer/RWKV-Runner) - 多模型桌面端執行器
- [Ollama Grid Search](https://github.com/dezoito/ollama-grid-search) - 評估與比較多個模型
- [macai](https://github.com/Renset/macai) - 適用於 Ollama 與 ChatGPT 的 macOS 客戶端
- [AI Studio](https://github.com/MindWorkAI/AI-Studio) - 支援多模型提供者的桌面 IDE
- [Reins](https://github.com/ibrahimcetin/reins) - 支援參數微調與推理模型的介面
- [ConfiChat](https://github.com/1runeberg/confichat) - 著重隱私保護並支援可選加密的客戶端
- [LLocal.in](https://github.com/kartikm7/llocal) - Electron 桌面端客戶端
- [MindMac](https://mindmac.app) - 適用於 Mac 的 AI 對話客戶端
- [Msty](https://msty.app) - 多模型桌面端客戶端
- [BoltAI for Mac](https://boltai.com) - 適用於 Mac 的 AI 對話客戶端
- [IntelliBar](https://intellibar.app/) - 適用於 macOS 的 AI 驅動助理
- [Kerlig AI](https://www.kerlig.com/) - 適用於 macOS 的 AI 寫作助理
- [Hillnote](https://hillnote.com) - Markdown 優先的 AI 工作空間
- [Perfect Memory AI](https://www.perfectmemory.ai/) - 結合螢幕與會議紀錄的個人化生產力 AI

#### Mobile 行動端

- [Ollama Android Chat](https://github.com/sunshine0523/OllamaServer) - Android 上的一鍵 Ollama 對話
> 上述列出的 SwiftChat、Enchanted、Maid、Ollama App、Reins 與 ConfiChat 同樣支援行動平台。

### 程式碼編輯器與開發工具 (Code Editors & Development)

- [Cline](https://github.com/cline/cline) - 支援多檔案與全專案開發的 VS Code 擴充功能
- [Continue](https://github.com/continuedev/continue) - 適用於任何 IDE 的開源 AI 程式碼助理
- [Void](https://github.com/voideditor/void) - 開源 AI 程式碼編輯器，Cursor 的替代方案
- [Copilot for Obsidian](https://github.com/logancyang/obsidian-copilot) - Obsidian 的 AI 助理
- [twinny](https://github.com/rjmacarthy/twinny) - Copilot 與 Copilot Chat 的開源替代方案
- [gptel Emacs client](https://github.com/karthink/gptel) - Emacs 的 LLM 客戶端
- [Ollama Copilot](https://github.com/bernardo-bruning/ollama-copilot) - 將 Ollama 當作 GitHub Copilot 使用
- [Obsidian Local GPT](https://github.com/pfrankov/obsidian-local-gpt) - Obsidian 的本地 AI 外掛
- [Ellama Emacs client](https://github.com/s-kostyaev/ellama) - Emacs 的 LLM 工具
- [orbiton](https://github.com/xyproto/orbiton) - 免設定且內建 Ollama Tab 自動補全的文字編輯器
- [AI ST Completion](https://github.com/yaroslavyaroslav/OpenAI-sublime-text) - Sublime Text 4 AI 助理
- [VT Code](https://github.com/vinhnx/vtcode) - 基於 Tree-sitter 與 Rust 的終端機編碼 Agent
- [QodeAssist](https://github.com/Palm1r/QodeAssist) - Qt Creator 的 AI 編碼助理
- [AI Toolkit for VS Code](https://aka.ms/ai-tooklit/ollama-docs) - 微軟官方的 VS Code 擴充功能
- [Open Interpreter](https://docs.openinterpreter.com/language-model-setup/local-models/ollama) - 電腦的自然語言操作介面

### 函式庫與 SDK (Libraries & SDKs)

- [LiteLLM](https://github.com/BerriAI/litellm) - 整合 100+ LLM 提供者的統一 API
- [Semantic Kernel](https://github.com/microsoft/semantic-kernel/tree/main/python/semantic_kernel/connectors/ai/ollama) - 微軟 AI 編排 SDK
- [LangChain4j](https://github.com/langchain4j/langchain4j) - Java 版 LangChain（[範例](https://github.com/langchain4j/langchain4j-examples/tree/main/ollama-examples/src/main/java)）
- [LangChainGo](https://github.com/tmc/langchaingo/) - Go 版 LangChain（[範例](https://github.com/tmc/langchaingo/tree/main/examples/ollama-completion-example)）
- [Spring AI](https://github.com/spring-projects/spring-ai) - Spring 框架 AI 支援（[文件](https://docs.spring.io/spring-ai/reference/api/chat/ollama-chat.html)）
- [LangChain](https://python.langchain.com/docs/integrations/chat/ollama/) 與 [LangChain.js](https://js.langchain.com/docs/integrations/chat/ollama/)（包含 [範例](https://js.langchain.com/docs/tutorials/local_rag/)）
- [Ollama for Ruby](https://github.com/crmne/ruby_llm) - Ruby LLM 函式庫
- [any-llm](https://github.com/mozilla-ai/any-llm) - Mozilla 推出的統一 LLM 介面
- [OllamaSharp for .NET](https://github.com/awaescher/OllamaSharp) - .NET SDK
- [LangChainRust](https://github.com/Abraxas-365/langchain-rust) - Rust 版 LangChain（[範例](https://github.com/Abraxas-365/langchain-rust/blob/main/examples/llm_ollama.rs)）
- [Agents-Flex for Java](https://github.com/agents-flex/agents-flex) - Java Agent 框架（[範例](https://github.com/agents-flex/agents-flex/tree/main/agents-flex-llm/agents-flex-llm-ollama/src/test/java/com/agentsflex/llm/ollama)）
- [Elixir LangChain](https://github.com/brainlid/langchain) - Elixir 版 LangChain
- [Ollama-rs for Rust](https://github.com/pepperoni21/ollama-rs) - Rust SDK
- [LangChain for .NET](https://github.com/tryAGI/LangChain) - .NET 版 LangChain（[範例](https://github.com/tryAGI/LangChain/blob/main/examples/LangChain.Samples.OpenAI/Program.cs)）
- [chromem-go](https://github.com/philippgille/chromem-go) - 支援 Ollama 嵌入向量的 Go 向量資料庫（[範例](https://github.com/philippgille/chromem-go/tree/v0.5.0/examples/rag-wikipedia-ollama)）
- [LangChainDart](https://github.com/davidmigloz/langchain_dart) - Dart 版 LangChain
- [LlmTornado](https://github.com/lofcz/llmtornado) - 支援多種推論 API 的 C# 統一介面
- [Ollama4j for Java](https://github.com/ollama4j/ollama4j) - Java SDK
- [Ollama for Laravel](https://github.com/cloudstudio/ollama-laravel) - Laravel 整合套件
- [Ollama for Swift](https://github.com/mattt/ollama-swift) - Swift SDK
- [LlamaIndex](https://docs.llamaindex.ai/en/stable/examples/llm/ollama/) 與 [LlamaIndexTS](https://ts.llamaindex.ai/modules/llms/available_llms/ollama) - LLM 應用的資料框架
- [Haystack](https://github.com/deepset-ai/haystack-integrations/blob/main/integrations/ollama.md) - AI 管線框架
- [Firebase Genkit](https://firebase.google.com/docs/genkit/plugins/ollama) - Google AI 開發框架
- [Ollama-hpp for C++](https://github.com/jmont-dev/ollama-hpp) - C++ SDK
- [PromptingTools.jl](https://github.com/svilupp/PromptingTools.jl) - Julia LLM 工具包（[範例](https://svilupp.github.io/PromptingTools.jl/dev/examples/working_with_ollama)）
- [Ollama for R - rollama](https://github.com/JBGruber/rollama) - R SDK
- [Portkey](https://portkey.ai/docs/welcome/integration-guides/ollama) - AI 網關 (Gateway)
- [Testcontainers](https://testcontainers.com/modules/ollama/) - 基於容器的測試工具
- [LLPhant](https://github.com/theodo-group/LLPhant?tab=readme-ov-file#ollama) - PHP AI 框架

### 框架與 Agent (Frameworks & Agents)

- [AutoGPT](https://github.com/Significant-Gravitas/AutoGPT/blob/master/docs/content/platform/ollama.md) - 自主 AI Agent 平台
- [crewAI](https://github.com/crewAIInc/crewAI) - 多 Agent 編排框架
- [Strands Agents](https://github.com/strands-agents/sdk-python) - AWS 推出的模型驅動 Agent 建構工具
- [Cheshire Cat](https://github.com/cheshire-cat-ai/core) - AI 助理框架
- [any-agent](https://github.com/mozilla-ai/any-agent) - Mozilla 推出的統一 Agent 框架介面
- [Stakpak](https://github.com/stakpak/agent) - 開源 DevOps Agent
- [Hexabot](https://github.com/hexastack/hexabot) - 對話型 AI 建構工具
- [Neuro SAN](https://github.com/cognizant-ai-lab/neuro-san-studio) - 多 Agent 編排系統（[文件](https://github.com/cognizant-ai-lab/neuro-san-studio/blob/main/docs/user_guide.md#ollama)）

### RAG 與知識庫 (RAG & Knowledge Bases)

- [RAGFlow](https://github.com/infiniflow/ragflow) - 基於深度文件理解的 RAG 引擎
- [R2R](https://github.com/SciPhi-AI/R2R) - 開源 RAG 引擎
- [MaxKB](https://github.com/1Panel-dev/MaxKB/) - 開箱即用的 RAG 聊天機器人
- [Minima](https://github.com/dmayboroda/minima) - 地端或完全本地化的 RAG 系統
- [Chipper](https://github.com/TilmanGriesel/chipper) - 整合 Haystack RAG 的 AI 介面
- [ARGO](https://github.com/xark-argo/argo) - 適用於 Mac/Windows/Linux 的 RAG 與深度研究工具
- [Archyve](https://github.com/nickthecook/archyve) - 支援 RAG 的文件庫
- [Casibase](https://casibase.org) - 支援 RAG 與 SSO 的 AI 知識庫
- [BrainSoup](https://www.nurgo-software.com/products/brainsoup) - 內建 RAG 與多 Agent 自動化的原生客戶端

### 機器人與通訊軟體 (Bots & Messaging)

- [LangBot](https://github.com/RockChinQ/LangBot) - 支援 Agent 與 RAG 的多平台訊息機器人
- [AstrBot](https://github.com/Soulter/AstrBot/) - 支援 RAG 與外掛的外掛式多平台聊天機器人
- [Discord-Ollama Chat Bot](https://github.com/kevinthedang/discord-ollama) - TypeScript 版 Discord 機器人
- [Ollama Telegram Bot](https://github.com/ruecat/ollama-telegram) - Telegram 機器人
- [LLM Telegram Bot](https://github.com/innightwolfsleep/llm_telegram_bot) - 用於角色扮演的 Telegram 機器人

### 終端機與 CLI (Terminal & CLI)

- [aichat](https://github.com/sigoden/aichat) - 全功能 LLM CLI，整合 Shell 助手、RAG 與 AI 工具
- [oterm](https://github.com/ggozad/oterm) - Ollama 終端機客戶端
- [gollama](https://github.com/sammcj/gollama) - 基於 Go 的 Ollama 模型管理工具
- [tlm](https://github.com/yusufcanb/tlm) - 本地 Shell Copilot
- [tenere](https://github.com/pythops/tenere) - LLM 終端機使用者介面 (TUI)
- [ParLlama](https://github.com/paulrobello/parllama) - Ollama 終端機使用者介面 (TUI)
- [llm-ollama](https://github.com/taketwo/llm-ollama) - [Datasette LLM CLI](https://llm.datasette.io/en/stable/) 外掛
- [ShellOracle](https://github.com/djcopley/ShellOracle) - Shell 指令建議工具
- [LLM-X](https://github.com/mrdjohnson/llm-x) - 適用於 LLM 的漸進式 Web 應用程式 (PWA)
- [cmdh](https://github.com/pgibler/cmdh) - 自然語言轉換為 Shell 指令
- [VT](https://github.com/vinhnx/vt.ai) - 極簡多模態 AI 對話應用

### 生產力與應用程式 (Productivity & Apps)

- [AppFlowy](https://github.com/AppFlowy-IO/AppFlowy) - AI 協作工作空間，可自建託管的 Notion 替代方案
- [Screenpipe](https://github.com/mediar-ai/screenpipe) - 24/7 螢幕與麥克風錄製及 AI 搜尋工具
- [Vibe](https://github.com/thewh1teagle/vibe) - 會議錄音轉寫與分析
- [Page Assist](https://github.com/n4ze3m/page-assist) - AI 網頁瀏覽 Chrome 擴充功能
- [NativeMind](https://github.com/NativeMindBrowser/NativeMindExtension) - 私密且完全在地執行的瀏覽器 AI 助理
- [Ollama Fortress](https://github.com/ParisNeo/ollama_proxy_server) - Ollama 的安全代理伺服器
- [1Panel](https://github.com/1Panel-dev/1Panel/) - 基於 Web 的 Linux 伺服器管理面板
- [Writeopia](https://github.com/Writeopia/Writeopia) - 整合 Ollama 的文字編輯器
- [QA-Pilot](https://github.com/reid41/QA-Pilot) - GitHub 程式碼儲存庫理解工具
- [Raycast extension](https://github.com/MassimilianoPasquini97/raycast_ollama) - Raycast 中的 Ollama 擴充功能
- [Painting Droid](https://github.com/mateuszmigas/painting-droid) - 整合 AI 的繪圖應用程式
- [Serene Pub](https://github.com/doolijb/serene-pub) - AI 角色扮演應用程式
- [Mayan EDMS](https://gitlab.com/mayan-edms/mayan-edms) - 支援 Ollama 工作流程的文件管理系統
- [TagSpaces](https://www.tagspaces.org) - 支援 [AI 標籤](https://docs.tagspaces.org/ai/) 的檔案管理工具

### 可觀測性與監控 (Observability & Monitoring)

- [Opik](https://www.comet.com/docs/opik/cookbook/ollama) - 除錯、評估與監控 LLM 應用程式
- [OpenLIT](https://github.com/openlit/openlit) - 適用於 Ollama 與 GPU 的 OpenTelemetry 原生監控工具
- [Lunary](https://lunary.ai/docs/integrations/ollama) - 具備分析與 PII 遮蔽功能的 LLM 可觀測性工具
- [Langfuse](https://langfuse.com/docs/integrations/ollama) - 開源 LLM 可觀測性平台
- [HoneyHive](https://docs.honeyhive.ai/integrations/ollama) - Agent 的 AI 可觀測性與評估工具
- [MLflow Tracing](https://mlflow.org/docs/latest/llms/tracing/index.html#automatic-tracing) - 開源 LLM 自動追蹤可觀測性

### 資料庫與嵌入向量 (Database & Embeddings)

- [pgai](https://github.com/timescale/pgai) - 將 PostgreSQL 當作向量資料庫使用（[指南](https://github.com/timescale/pgai/blob/main/docs/vectorizer-quick-start.md)）
- [MindsDB](https://github.com/mindsdb/mindsdb/blob/staging/mindsdb/integrations/handlers/ollama_handler/README.md) - 將 Ollama 連結至 200+ 個資料平台
- [chromem-go](https://github.com/philippgille/chromem-go/blob/v0.5.0/embed_ollama.go) - Go 語言可嵌入式向量資料庫（[範例](https://github.com/philippgille/chromem-go/tree/v0.5.0/examples/rag-wikipedia-ollama)）
- [Kangaroo](https://github.com/dbkangaroo/kangaroo) - AI 驅動的 SQL 客戶端

### 基礎設施與部署 (Infrastructure & Deployment)

#### 雲端平台 Cloud

- [Google Cloud](https://cloud.google.com/run/docs/tutorials/gpu-gemma2-with-ollama)
- [Fly.io](https://fly.io/docs/python/do-more/add-ollama/)
- [Koyeb](https://www.koyeb.com/deploy/ollama)
- [Harbor](https://github.com/av/harbor) - 容器化 LLM 工具包，預設後端為 Ollama

#### 套件管理工具 Package Managers

- [Pacman](https://archlinux.org/packages/extra/x86_64/ollama/)
- [Homebrew](https://formulae.brew.sh/formula/ollama)
- [Nix package](https://search.nixos.org/packages?show=ollama&from=0&size=50&sort=relevance&type=packages&query=ollama)
- [Helm Chart](https://artifacthub.io/packages/helm/ollama-helm/ollama)
- [Gentoo](https://github.com/gentoo/guru/tree/master/app-misc/ollama)
- [Flox](https://flox.dev/blog/ollama-part-one)
- [Guix channel](https://codeberg.org/tusharhero/ollama-guix)
