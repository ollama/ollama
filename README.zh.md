<!--
IMPORTANT: This file is a localized version of README.md. 
When updating README.md, please ensure that the corresponding changes are also applied to this file to maintain parity.
-->

<p align="center">
  <a href="https://ollama.com">
    <img src="https://github.com/ollama/ollama/assets/3325447/0d0b44e2-8f4a-4e99-9b52-a5c1c741c8f7" alt="ollama" width="200"/>
  </a>
</p>

<p align="center">
  <a href="README.md">English</a> ·
  <strong>中文</strong>
</p>

# Ollama

开始使用开源模型进行构建。

## 下载 (Download)

### macOS

```shell
curl -fsSL https://ollama.com/install.sh | sh
```

或者[手动下载](https://ollama.com/download/Ollama.dmg)

### Windows

```shell
irm https://ollama.com/install.ps1 | iex
```

或者[手动下载](https://ollama.com/download/OllamaSetup.exe)

### Linux

```shell
curl -fsSL https://ollama.com/install.sh | sh
```

[手动安装指南](https://docs.ollama.com/linux#manual-install)

### Docker

官方的 [Ollama Docker 镜像](https://hub.docker.com/r/ollama/ollama) `ollama/ollama` 已在 Docker Hub 上提供。

### 代码库 (Libraries)

- [ollama-python](https://github.com/ollama/ollama-python)
- [ollama-js](https://github.com/ollama/ollama-js)

### 社区 (Community)

- [Discord](https://discord.gg/ollama)
- [𝕏 (Twitter)](https://x.com/ollama)
- [Reddit](https://reddit.com/r/ollama)

## 快速开始 (Get started)

```
ollama
```

系统会提示你运行一个模型，或者将 Ollama 连接到你现有的 Agent 或应用程序，例如 `claude`、`codex`、`openclaw` 等等。

### 编程 (Coding)

要启动特定的集成：

```
ollama launch claude
```

支持的集成包括 [Claude Code](https://docs.ollama.com/integrations/claude-code)、[Codex](https://docs.ollama.com/integrations/codex)、[Droid](https://docs.ollama.com/integrations/droid) 以及 [OpenCode](https://docs.ollama.com/integrations/opencode)。

### AI 助手 (AI assistant)

使用 [OpenClaw](https://docs.ollama.com/integrations/openclaw) 将 Ollama 变成一个跨越 WhatsApp、Telegram、Slack、Discord 等多个渠道的个人 AI 助手：

```
ollama launch openclaw
```

### 与模型聊天 (Chat with a model)

运行并与 [Gemma 3](https://ollama.com/library/gemma3) 聊天：

```
ollama run gemma3
```

查看 [ollama.com/library](https://ollama.com/library) 获取完整列表。

更多详细信息请参阅[快速入门指南](https://docs.ollama.com/quickstart)。

## REST API

Ollama 提供了一个用于运行和管理模型的 REST API。

```
curl http://localhost:11434/api/chat -d '{
  "model": "gemma3",
  "messages": [{
    "role": "user",
    "content": "Why is the sky blue?"
  }],
  "stream": false
}'
```

查看 [API 文档](https://docs.ollama.com/api) 获取所有的接口端点。

### Python

```
pip install ollama
```

```python
from ollama import chat

response = chat(model='gemma3', messages=[
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
  model: "gemma3",
  messages: [{ role: "user", content: "Why is the sky blue?" }],
});
console.log(response.message.content);
```

## 支持的后端 (Supported backends)

- 由 Georgi Gerganov 创立的 [llama.cpp](https://github.com/ggml-org/llama.cpp) 项目。

## 文档 (Documentation)

- [CLI 参考手册](https://docs.ollama.com/cli)
- [REST API 参考手册](https://docs.ollama.com/api)
- [导入模型](https://docs.ollama.com/import)
- [Modelfile 参考手册](https://docs.ollama.com/modelfile)
- [从源码构建](https://github.com/ollama/ollama/blob/main/docs/development.md)

## 社区集成 (Community Integrations)

> 想要添加你的项目？请提交 Pull Request。

### 聊天界面 (Chat Interfaces)

#### Web端 (Web)

- [Open WebUI](https://github.com/open-webui/open-webui) - 可扩展的、可自托管的 AI 界面
- [Onyx](https://github.com/onyx-dot-app/onyx) - 互联的 AI 工作空间
- [LibreChat](https://github.com/danny-avila/LibreChat) - 支持多提供商的增强版 ChatGPT 克隆
- [Lobe Chat](https://github.com/lobehub/lobe-chat) - 带有插件生态的现代聊天框架 ([文档](https://lobehub.com/docs/self-hosting/examples/ollama))
- [NextChat](https://github.com/ChatGPTNextWeb/ChatGPT-Next-Web) - 跨平台 ChatGPT UI ([文档](https://docs.nextchat.dev/models/ollama))
- [Perplexica](https://github.com/ItzCrazyKns/Perplexica) - AI 驱动的搜索引擎，开源的 Perplexity 替代品
- [big-AGI](https://github.com/enricoros/big-AGI) - 面向专业人士的 AI 套件
- [Lollms WebUI](https://github.com/ParisNeo/lollms-webui) - 多模型 Web 界面
- [ChatOllama](https://github.com/sugarforever/chat-ollama) - 带有知识库的聊天机器人
- [Bionic GPT](https://github.com/bionic-gpt/bionic-gpt) - 本地部署的 AI 平台
- [Chatbot UI](https://github.com/ivanfioravanti/chatbot-ollama) - ChatGPT 风格的 Web 界面
- [Hollama](https://github.com/fmaclen/hollama) - 极简的 Web 界面
- [Chatbox](https://github.com/Bin-Huang/Chatbox) - 桌面和 Web AI 客户端
- [chat](https://github.com/swuecho/chat) - 团队专用的聊天 Web 应用
- [Ollama RAG Chatbot](https://github.com/datvodinh/rag-chatbot.git) - 基于 RAG 与多个 PDF 聊天
- [Tkinter-based client](https://github.com/chyok/ollama-gui) - Python 桌面客户端

#### 桌面端 (Desktop)

- [Dify.AI](https://github.com/langgenius/dify) - LLM 应用开发平台
- [AnythingLLM](https://github.com/Mintplex-Labs/anything-llm) - 适用于 Mac、Windows 和 Linux 的全能 AI 应用
- [Maid](https://github.com/Mobile-Artificial-Intelligence/maid) - 跨平台的移动和桌面客户端
- [Witsy](https://github.com/nbonamy/witsy) - 适用于 Mac、Windows 和 Linux 的 AI 桌面应用
- [Cherry Studio](https://github.com/kangfenmao/cherry-studio) - 多提供商桌面客户端
- [Ollama App](https://github.com/JHubi1/ollama-app) - 桌面和移动的多平台客户端
- [PyGPT](https://github.com/szczyglis-dev/py-gpt) - 适用于 Linux、Windows 和 Mac 的 AI 桌面助手
- [Alpaca](https://github.com/Jeffser/Alpaca) - 适用于 Linux 和 macOS 的 GTK4 客户端
- [SwiftChat](https://github.com/aws-samples/swift-chat) - 跨平台（包括 iOS、Android 和 Apple Vision Pro）
- [Enchanted](https://github.com/AugustDev/enchanted) - 原生 macOS 和 iOS 客户端
- [RWKV-Runner](https://github.com/josStorer/RWKV-Runner) - 多模型桌面运行器
- [Ollama Grid Search](https://github.com/dezoito/ollama-grid-search) - 评估和比较模型
- [macai](https://github.com/Renset/macai) - 适用于 Ollama 和 ChatGPT 的 macOS 客户端
- [AI Studio](https://github.com/MindWorkAI/AI-Studio) - 多提供商桌面 IDE
- [Reins](https://github.com/ibrahimcetin/reins) - 支持参数微调和推理模型
- [ConfiChat](https://github.com/1runeberg/confichat) - 注重隐私且支持可选加密
- [LLocal.in](https://github.com/kartikm7/llocal) - Electron 桌面客户端
- [MindMac](https://mindmac.app) - Mac 专属 AI 聊天客户端
- [Msty](https://msty.app) - 多模型桌面客户端
- [BoltAI for Mac](https://boltai.com) - Mac 专属 AI 聊天客户端
- [IntelliBar](https://intellibar.app/) - macOS 上的 AI 驱动助手
- [Kerlig AI](https://www.kerlig.com/) - macOS 上的 AI 写作助手
- [Hillnote](https://hillnote.com) - Markdown 优先的 AI 工作空间
- [Perfect Memory AI](https://www.perfectmemory.ai/) - 根据屏幕和会议历史提供个性化服务的生产力 AI

#### 移动端 (Mobile)

- [Ollama Android Chat](https://github.com/sunshine0523/OllamaServer) - Android 上的一键部署 Ollama

> 上方列出的 SwiftChat、Enchanted、Maid、Ollama App、Reins 以及 ConfiChat 也同样支持移动平台。

### 代码编辑器与开发 (Code Editors & Development)

- [Cline](https://github.com/cline/cline) - 适用于多文件/全代码库编码的 VS Code 扩展
- [Continue](https://github.com/continuedev/continue) - 适用于任何 IDE 的开源 AI 代码助手
- [Void](https://github.com/voideditor/void) - 开源的 AI 代码编辑器，Cursor 的替代品
- [Copilot for Obsidian](https://github.com/logancyang/obsidian-copilot) - Obsidian 的 AI 助手
- [twinny](https://github.com/rjmacarthy/twinny) - Copilot 及 Copilot chat 的替代品
- [gptel Emacs client](https://github.com/karthink/gptel) - Emacs 的 LLM 客户端
- [Ollama Copilot](https://github.com/bernardo-bruning/ollama-copilot) - 将 Ollama 作为 GitHub Copilot 使用
- [Obsidian Local GPT](https://github.com/pfrankov/obsidian-local-gpt) - Obsidian 的本地 AI
- [Ellama Emacs client](https://github.com/s-kostyaev/ellama) - Emacs 的 LLM 工具
- [orbiton](https://github.com/xyproto/orbiton) - 带有 Ollama Tab 补全的免配置文本编辑器
- [AI ST Completion](https://github.com/yaroslavyaroslav/OpenAI-sublime-text) - Sublime Text 4 的 AI 助手
- [VT Code](https://github.com/vinhnx/vtcode) - 基于 Rust 和 Tree-sitter 的终端编码代理
- [QodeAssist](https://github.com/Palm1r/QodeAssist) - Qt Creator 的 AI 编码助手
- [AI Toolkit for VS Code](https://aka.ms/ai-tooklit/ollama-docs) - 微软官方的 VS Code 扩展
- [Open Interpreter](https://docs.openinterpreter.com/language-model-setup/local-models/ollama) - 计算机的自然语言接口

### 代码库与 SDK (Libraries & SDKs)

- [LiteLLM](https://github.com/BerriAI/litellm) - 支持 100+ LLM 提供商的统一 API
- [Semantic Kernel](https://github.com/microsoft/semantic-kernel/tree/main/python/semantic_kernel/connectors/ai/ollama) - 微软的 AI 编排 SDK
- [LangChain4j](https://github.com/langchain4j/langchain4j) - Java 版 LangChain ([示例](https://github.com/langchain4j/langchain4j-examples/tree/main/ollama-examples/src/main/java))
- [LangChainGo](https://github.com/tmc/langchaingo/) - Go 版 LangChain ([示例](https://github.com/tmc/langchaingo/tree/main/examples/ollama-completion-example))
- [Spring AI](https://github.com/spring-projects/spring-ai) - Spring 框架的 AI 支持 ([文档](https://docs.spring.io/spring-ai/reference/api/chat/ollama-chat.html))
- [LangChain](https://python.langchain.com/docs/integrations/chat/ollama/) 与 [LangChain.js](https://js.langchain.com/docs/integrations/chat/ollama/) 及其 [示例](https://js.langchain.com/docs/tutorials/local_rag/)
- [Ollama for Ruby](https://github.com/crmne/ruby_llm) - Ruby LLM 库
- [any-llm](https://github.com/mozilla-ai/any-llm) - Mozilla 的统一 LLM 接口
- [OllamaSharp for .NET](https://github.com/awaescher/OllamaSharp) - .NET SDK
- [LangChainRust](https://github.com/Abraxas-365/langchain-rust) - Rust 版 LangChain ([示例](https://github.com/Abraxas-365/langchain-rust/blob/main/examples/llm_ollama.rs))
- [Agents-Flex for Java](https://github.com/agents-flex/agents-flex) - Java 的 Agent 框架 ([示例](https://github.com/agents-flex/agents-flex/tree/main/agents-flex-llm/agents-flex-llm-ollama/src/test/java/com/agentsflex/llm/ollama))
- [Elixir LangChain](https://github.com/brainlid/langchain) - Elixir 版 LangChain
- [Ollama-rs for Rust](https://github.com/pepperoni21/ollama-rs) - Rust SDK
- [LangChain for .NET](https://github.com/tryAGI/LangChain) - .NET 版 LangChain ([示例](https://github.com/tryAGI/LangChain/blob/main/examples/LangChain.Samples.OpenAI/Program.cs))
- [chromem-go](https://github.com/philippgille/chromem-go) - 带有 Ollama 嵌入的 Go 向量数据库 ([示例](https://github.com/philippgille/chromem-go/tree/v0.5.0/examples/rag-wikipedia-ollama))
- [LangChainDart](https://github.com/davidmigloz/langchain_dart) - Dart 版 LangChain
- [LlmTornado](https://github.com/lofcz/llmtornado) - 用于多个推理 API 的统一 C# 接口
- [Ollama4j for Java](https://github.com/ollama4j/ollama4j) - Java SDK
- [Ollama for Laravel](https://github.com/cloudstudio/ollama-laravel) - Laravel 集成
- [Ollama for Swift](https://github.com/mattt/ollama-swift) - Swift SDK
- [LlamaIndex](https://docs.llamaindex.ai/en/stable/examples/llm/ollama/) 和 [LlamaIndexTS](https://ts.llamaindex.ai/modules/llms/available_llms/ollama) - 用于 LLM 应用的数据框架
- [Haystack](https://github.com/deepset-ai/haystack-integrations/blob/main/integrations/ollama.md) - AI 管道框架
- [Firebase Genkit](https://firebase.google.com/docs/genkit/plugins/ollama) - Google 的 AI 框架
- [Ollama-hpp for C++](https://github.com/jmont-dev/ollama-hpp) - C++ SDK
- [PromptingTools.jl](https://github.com/svilupp/PromptingTools.jl) - Julia 的 LLM 工具包 ([示例](https://svilupp.github.io/PromptingTools.jl/dev/examples/working_with_ollama))
- [Ollama for R - rollama](https://github.com/JBGruber/rollama) - R SDK
- [Portkey](https://portkey.ai/docs/welcome/integration-guides/ollama) - AI 网关
- [Testcontainers](https://testcontainers.com/modules/ollama/) - 基于容器的测试
- [LLPhant](https://github.com/theodo-group/LLPhant?tab=readme-ov-file#ollama) - PHP 的 AI 框架

### 框架与代理 (Frameworks & Agents)

- [AutoGPT](https://github.com/Significant-Gravitas/AutoGPT/blob/master/docs/content/platform/ollama.md) - 自主 AI Agent 平台
- [crewAI](https://github.com/crewAIInc/crewAI) - 多 Agent 编排框架
- [Strands Agents](https://github.com/strands-agents/sdk-python) - AWS 驱动的模型 Agent 构建
- [Cheshire Cat](https://github.com/cheshire-cat-ai/core) - AI 助手框架
- [any-agent](https://github.com/mozilla-ai/any-agent) - Mozilla 的统一 Agent 框架接口
- [Stakpak](https://github.com/stakpak/agent) - 开源 DevOps Agent
- [Hexabot](https://github.com/hexastack/hexabot) - 对话式 AI 构建器
- [Neuro SAN](https://github.com/cognizant-ai-lab/neuro-san-studio) - 多 Agent 编排 ([文档](https://github.com/cognizant-ai-lab/neuro-san-studio/blob/main/docs/user_guide.md#ollama))

### RAG 与知识库 (RAG & Knowledge Bases)

- [RAGFlow](https://github.com/infiniflow/ragflow) - 基于深度文档理解的 RAG 引擎
- [R2R](https://github.com/SciPhi-AI/R2R) - 开源 RAG 引擎
- [MaxKB](https://github.com/1Panel-dev/MaxKB/) - 开箱即用的 RAG 聊天机器人
- [Minima](https://github.com/dmayboroda/minima) - 内部部署或完全本地化的 RAG
- [Chipper](https://github.com/TilmanGriesel/chipper) - 采用 Haystack RAG 的 AI 界面
- [ARGO](https://github.com/xark-argo/argo) - 适用于 Mac/Windows/Linux 的 RAG 与深度研究工具
- [Archyve](https://github.com/nickthecook/archyve) - 赋能 RAG 的文档库
- [Casibase](https://casibase.org) - 带有 RAG 和 SSO 的 AI 知识库
- [BrainSoup](https://www.nurgo-software.com/products/brainsoup) - 带有 RAG 和多 Agent 自动化的原生客户端

### 机器人与消息通讯 (Bots & Messaging)

- [LangBot](https://github.com/RockChinQ/LangBot) - 包含 Agent 和 RAG 的多平台消息机器人
- [AstrBot](https://github.com/Soulter/AstrBot/) - 带有 RAG 和插件的多平台聊天机器人
- [Discord-Ollama Chat Bot](https://github.com/kevinthedang/discord-ollama) - TypeScript 编写的 Discord 机器人
- [Ollama Telegram Bot](https://github.com/ruecat/ollama-telegram) - Telegram 机器人
- [LLM Telegram Bot](https://github.com/innightwolfsleep/llm_telegram_bot) - 用于角色扮演的 Telegram 机器人

### 终端与命令行 (Terminal & CLI)

- [aichat](https://github.com/sigoden/aichat) - 一站式 LLM CLI（包含 Shell 助手、RAG 及 AI 工具）
- [oterm](https://github.com/ggozad/oterm) - Ollama 的终端客户端
- [gollama](https://github.com/sammcj/gollama) - 基于 Go 的 Ollama 模型管理器
- [tlm](https://github.com/yusufcanb/tlm) - 本地 shell 的副驾驶
- [tenere](https://github.com/pythops/tenere) - LLM 的 TUI（终端用户界面）
- [ParLlama](https://github.com/paulrobello/parllama) - Ollama 的 TUI
- [llm-ollama](https://github.com/taketwo/llm-ollama) - [Datasette LLM CLI](https://llm.datasette.io/en/stable/) 的插件
- [ShellOracle](https://github.com/djcopley/ShellOracle) - Shell 命令建议工具
- [LLM-X](https://github.com/mrdjohnson/llm-x) - 面向 LLM 的渐进式 Web 应用 (PWA)
- [cmdh](https://github.com/pgibler/cmdh) - 将自然语言转化为 shell 命令
- [VT](https://github.com/vinhnx/vt.ai) - 极简的多模态 AI 聊天应用

### 生产力应用 (Productivity & Apps)

- [AppFlowy](https://github.com/AppFlowy-IO/AppFlowy) - AI 协作工作空间，可自托管的 Notion 替代品
- [Screenpipe](https://github.com/mediar-ai/screenpipe) - 全天候的屏幕与麦克风录制，配合 AI 驱动搜索
- [Vibe](https://github.com/thewh1teagle/vibe) - 会议的转写与分析
- [Page Assist](https://github.com/n4ze3m/page-assist) - 赋能 AI 浏览的 Chrome 扩展
- [NativeMind](https://github.com/NativeMindBrowser/NativeMindExtension) - 私有的、设备端的浏览器 AI 助手
- [Ollama Fortress](https://github.com/ParisNeo/ollama_proxy_server) - Ollama 的安全代理
- [1Panel](https://github.com/1Panel-dev/1Panel/) - 基于 Web 的 Linux 服务器管理
- [Writeopia](https://github.com/Writeopia/Writeopia) - 集成 Ollama 的文本编辑器
- [QA-Pilot](https://github.com/reid41/QA-Pilot) - GitHub 代码库理解工具
- [Raycast extension](https://github.com/MassimilianoPasquini97/raycast_ollama) - Raycast 中的 Ollama 插件
- [Painting Droid](https://github.com/mateuszmigas/painting-droid) - 集成了 AI 的绘画应用
- [Serene Pub](https://github.com/doolijb/serene-pub) - AI 角色扮演应用
- [Mayan EDMS](https://gitlab.com/mayan-edms/mayan-edms) - 具有 Ollama 工作流的文档管理系统
- [TagSpaces](https://www.tagspaces.org) - 具备 [AI 标签功能](https://docs.tagspaces.org/ai/) 的文件管理系统

### 观测与监控 (Observability & Monitoring)

- [Opik](https://www.comet.com/docs/opik/cookbook/ollama) - 调试、评估并监控 LLM 应用
- [OpenLIT](https://github.com/openlit/openlit) - 针对 Ollama 与 GPU 的原生 OpenTelemetry 监控
- [Lunary](https://lunary.ai/docs/integrations/ollama) - 带有分析和 PII 掩码的 LLM 可观测性工具
- [Langfuse](https://langfuse.com/docs/integrations/ollama) - 开源的 LLM 可观测平台
- [HoneyHive](https://docs.honeyhive.ai/integrations/ollama) - 针对 Agent 的 AI 可观测性与评估
- [MLflow Tracing](https://mlflow.org/docs/latest/llms/tracing/index.html#automatic-tracing) - 开源的 LLM 可观测追踪

### 数据库与嵌入向量 (Database & Embeddings)

- [pgai](https://github.com/timescale/pgai) - 将 PostgreSQL 作为向量数据库 ([指南](https://github.com/timescale/pgai/blob/main/docs/vectorizer-quick-start.md))
- [MindsDB](https://github.com/mindsdb/mindsdb/blob/staging/mindsdb/integrations/handlers/ollama_handler/README.md) - 将 Ollama 连接至 200+ 个数据平台
- [chromem-go](https://github.com/philippgille/chromem-go/blob/v0.5.0/embed_ollama.go) - Go 的可嵌入式向量数据库 ([示例](https://github.com/philippgille/chromem-go/tree/v0.5.0/examples/rag-wikipedia-ollama))
- [Kangaroo](https://github.com/dbkangaroo/kangaroo) - AI 驱动的 SQL 客户端

### 基础设施与部署 (Infrastructure & Deployment)

#### 云端 (Cloud)

- [Google Cloud](https://cloud.google.com/run/docs/tutorials/gpu-gemma2-with-ollama)
- [Fly.io](https://fly.io/docs/python/do-more/add-ollama/)
- [Koyeb](https://www.koyeb.com/deploy/ollama)
- [Harbor](https://github.com/av/harbor) - 默认以 Ollama 作为后端的容器化 LLM 工具包

#### 包管理器 (Package Managers)

- [Pacman](https://archlinux.org/packages/extra/x86_64/ollama/)
- [Homebrew](https://formulae.brew.sh/formula/ollama)
- [Nix package](https://search.nixos.org/packages?show=ollama&from=0&size=50&sort=relevance&type=packages&query=ollama)
- [Helm Chart](https://artifacthub.io/packages/helm/ollama-helm/ollama)
- [Gentoo](https://github.com/gentoo/guru/tree/master/app-misc/ollama)
- [Flox](https://flox.dev/blog/ollama-part-one)
- [Guix channel](https://codeberg.org/tusharhero/ollama-guix)
