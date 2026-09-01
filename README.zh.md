<p align="center">
  <a href="https://ollama.com">
    <img src="https://github.com/ollama/ollama/assets/3325447/0d0b44e2-8f4a-4e99-9b52-a5c1c741c8f7" alt="ollama" width="200"/>
  </a>
</p>

# Ollama

开始使用开源模型进行构建。

---

## 下载安装

### macOS

```shell
curl -fsSL https://ollama.com/install.sh | sh
```

或[手动下载](https://ollama.com/download/Ollama.dmg)

### Windows

```shell
irm https://ollama.com/install.ps1 | iex
```

或[手动下载](https://ollama.com/download/OllamaSetup.exe)

### Linux

```shell
curl -fsSL https://ollama.com/install.sh | sh
```

[手动安装说明](https://docs.ollama.com/linux#manual-install)

### Docker

官方 [Ollama Docker 镜像](https://hub.docker.com/r/ollama/ollama) `ollama/ollama` 可在 Docker Hub 获取。

---

## 库

- [ollama-python](https://github.com/ollama/ollama-python)
- [ollama-js](https://github.com/ollama/ollama-js)

## 社区

- [Discord](https://discord.gg/ollama)
- [𝕏 (Twitter)](https://x.com/ollama)
- [Reddit](https://reddit.com/r/ollama)

---

## 快速开始

```
ollama
```

系统将提示你运行模型或将 Ollama 连接到现有的智能体或应用，如 `claude`、`codex`、`openclaw` 等。

### 编程助手

启动特定集成：

```
ollama launch claude
```

支持的集成包括 [Claude Code](https://docs.ollama.com/integrations/claude-code)、[Codex](https://docs.ollama.com/integrations/codex)、[Droid](https://docs.ollama.com/integrations/droid) 和 [OpenCode](https://docs.ollama.com/integrations/opencode)。

### AI 助手

使用 [OpenClaw](https://docs.ollama.com/integrations/openclaw) 将 Ollama 变成跨 WhatsApp、Telegram、Slack、Discord 等平台的个人 AI 助手：

```
ollama launch openclaw
```

### 与模型对话

运行并对话 [Gemma 3](https://ollama.com/library/gemma3)：

```
ollama run gemma3
```

查看 [ollama.com/library](https://ollama.com/library) 获取完整模型列表。

查看[快速入门指南](https://docs.ollama.com/quickstart)了解更多详情。

---

## REST API

Ollama 提供 REST API 用于运行和管理模型。

```
curl http://localhost:11434/api/chat -d '{
  "model": "gemma3",
  "messages": [{
    "role": "user",
    "content": "为什么天空是蓝色的？"
  }],
  "stream": false
}'
```

查看 [API 文档](https://docs.ollama.com/api) 了解所有接口。

### Python

```
pip install ollama
```

```python
from ollama import chat

response = chat(model='gemma3', messages=[
  {
    'role': 'user',
    'content': '为什么天空是蓝色的？',
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
  messages: [{ role: "user", content: "为什么天空是蓝色的？" }],
});
console.log(response.message.content);
```

---

## 支持的后端

- [llama.cpp](https://github.com/ggml-org/llama.cpp) 项目由 Georgi Gerganov 创立。

---

## 文档

- [CLI 参考](https://docs.ollama.com/cli)
- [REST API 参考](https://docs.ollama.com/api)
- [导入模型](https://docs.ollama.com/import)
- [Modelfile 参考](https://docs.ollama.com/modelfile)
- [从源码构建](https://github.com/ollama/ollama/blob/main/docs/development.md)

---

## 社区集成

> 想添加你的项目？提交一个 Pull Request。

### 聊天界面

#### Web

- [Open WebUI](https://github.com/open-webui/open-webui) - 可扩展的本地 AI 界面
- [Onyx](https://github.com/onyx-dot-app/onyx) - 互联 AI 工作空间
- [LibreChat](https://github.com/danny-avila/LibreChat) - 支持多提供商的增强版 ChatGPT 克隆
- [Lobe Chat](https://github.com/lobehub/lobe-chat) - 带插件生态系统的现代聊天框架 ([文档](https://lobehub.com/docs/self-hosting/examples/ollama))
- [NextChat](https://github.com/ChatGPTNextWeb/ChatGPT-Next-Web) - 跨平台 ChatGPT 界面 ([文档](https://docs.nextchat.dev/models/ollama))
- [Perplexica](https://github.com/ItzCrazyKns/Perplexica) - AI 驱动的搜索引擎，开源 Perplexity 替代品
- [big-AGI](https://github.com/enricoros/big-AGI) - 专业级 AI 套件
- [Lollms WebUI](https://github.com/ParisNeo/lollms-webui) - 多模型 Web 界面
- [ChatOllama](https://github.com/sugarforever/chat-ollama) - 支持知识库的聊天机器人
- [Bionic GPT](https://github.com/bionic-gpt/bionic-gpt) - 本地部署 AI 平台
- [Chatbot UI](https://github.com/ivanfioravanti/chatbot-ollama) - ChatGPT 风格 Web 界面
- [Hollama](https://github.com/fmaclen/hollama) - 极简 Web 界面
- [Chatbox](https://github.com/Bin-Huang/Chatbox) - 桌面和 Web AI 客户端
- [chat](https://github.com/swuecho/chat) - 团队聊天 Web 应用
- [Ollama RAG Chatbot](https://github.com/datvodinh/rag-chatbot.git) - 使用 RAG 与多个 PDF 对话
- [Tkinter-based client](https://github.com/chyok/ollama-gui) - Python 桌面客户端

#### 桌面端

- [Dify.AI](https://github.com/langgenius/dify) - LLM 应用开发平台
- [AnythingLLM](https://github.com/Mintplex-Labs/anything-llm) - Mac、Windows 和 Linux 一体化 AI 应用
- [Maid](https://github.com/Mobile-Artificial-Intelligence/maid) - 跨平台移动端和桌面端客户端
- [Witsy](https://github.com/nbonamy/witsy) - Mac、Windows 和 Linux AI 桌面应用
- [Cherry Studio](https://github.com/kangfenmao/cherry-studio) - 多提供商桌面客户端
- [Ollama App](https://github.com/JHubi1/ollama-app) - 跨平台桌面和移动端客户端
- [PyGPT](https://github.com/szczyglis-dev/py-gpt) - Linux、Windows 和 Mac AI 桌面助手
- [Alpaca](https://github.com/Jeffser/Alpaca) - Linux 和 macOS 的 GTK4 客户端
- [SwiftChat](https://github.com/aws-samples/swift-chat) - 跨平台，包括 iOS、Android 和 Apple Vision Pro
- [Enchanted](https://github.com/AugustDev/enchanted) - 原生 macOS 和 iOS 客户端
- [RWKV-Runner](https://github.com/josStorer/RWKV-Runner) - 多模型桌面运行器
- [Ollama Grid Search](https://github.com/dezoito/ollama-grid-search) - 评估和比较模型
- [macai](https://github.com/Renset/macai) - macOS Ollama 和 ChatGPT 客户端
- [AI Studio](https://github.com/MindWorkAI/AI-Studio) - 多提供商桌面 IDE
- [Reins](https://github.com/ibrahimcetin/reins) - 参数调优和推理模型支持
- [ConfiChat](https://github.com/1runeberg/confichat) - 注重隐私，可选加密
- [LLocal.in](https://github.com/kartikm7/llocal) - Electron 桌面客户端
- [MindMac](https://mindmac.app) - Mac AI 聊天客户端
- [Msty](https://msty.app) - 多模型桌面客户端
- [BoltAI for Mac](https://boltai.com) - Mac AI 聊天客户端
- [IntelliBar](https://intellibar.app/) - macOS AI 驱动助手
- [Kerlig AI](https://www.kerlig.com/) - macOS AI 写作助手
- [Hillnote](https://hillnote.com) - Markdown 优先 AI 工作空间
- [Perfect Memory AI](https://www.perfectmemory.ai/) - 基于屏幕和会议历史的个性化生产力 AI

#### 移动端

- [Ollama Android Chat](https://github.com/sunshine0523/OllamaServer) - Android 上一键运行 Ollama

> SwiftChat、Enchanted、Maid、Ollama App、Reins 和 ConfiChat 也支持移动平台。

### 代码编辑器和开发工具

- [Cline](https://github.com/cline/cline) - VS Code 扩展，支持多文件/整个仓库编码
- [Continue](https://github.com/continuedev/continue) - 任何 IDE 的开源 AI 代码助手
- [Void](https://github.com/voideditor/void) - 开源 AI 代码编辑器，Cursor 替代品
- [Copilot for Obsidian](https://github.com/logancyang/obsidian-copilot) - Obsidian AI 助手
- [twinny](https://github.com/rjmacarthy/twinny) - Copilot 和 Copilot 聊天替代品
- [gptel Emacs client](https://github.com/karthink/gptel) - Emacs LLM 客户端
- [Ollama Copilot](https://github.com/bernardo-bruning/ollama-copilot) - 将 Ollama 用作 GitHub Copilot
- [Obsidian Local GPT](https://github.com/pfrankov/obsidian-local-gpt) - Obsidian 本地 AI
- [Ellama Emacs client](https://github.com/s-kostyaev/ellama) - Emacs LLM 工具
- [orbiton](https://github.com/xyproto/orbiton) - 零配置文本编辑器，带 Ollama 自动补全
- [AI ST Completion](https://github.com/yaroslavyaroslav/OpenAI-sublime-text) - Sublime Text 4 AI 助手
- [VT Code](https://github.com/vinhnx/vtcode) - 基于 Rust 的终端编码助手，带 Tree-sitter
- [QodeAssist](https://github.com/Palm1r/QodeAssist) - Qt Creator AI 编码助手
- [AI Toolkit for VS Code](https://aka.ms/ai-tooklit/ollama-docs) - 微软官方 VS Code 扩展
- [Open Interpreter](https://docs.openinterpreter.com/language-model-setup/local-models/ollama) - 计算机自然语言接口

### 库和 SDK

- [LiteLLM](https://github.com/BerriAI/litellm) - 100+ LLM 提供商的统一 API
- [Semantic Kernel](https://github.com/microsoft/semantic-kernel/tree/main/python/semantic_kernel/connectors/ai/ollama) - 微软 AI 编排 SDK
- [LangChain4j](https://github.com/langchain4j/langchain4j) - Java 版 LangChain ([示例](https://github.com/langchain4j/langchain4j-examples/tree/main/ollama-examples/src/main/java))
- [LangChainGo](https://github.com/tmc/langchaingo/) - Go 版 LangChain
- [LangChainRust](https://github.com/Abraxas-365/langchain-rust) - Rust 版 LangChain
- [OllamaSharp](https://github.com/awaescher/OllamaSharp) - .NET 版 Ollama 客户端
- [Ollamanim](https://github.com/woodjobber/Ollamanim) - iOS/macOS Swift 封装
- [LangChain.rb](https://github.com/andreibondarev/langchainrb) - Ruby 版 LangChain
- [Ollama for Dart](https://github.com/breitburg/dart-ollama) - Dart/Flutter Ollama 客户端
- [Ollama for Kotlin](https://github.com/aj8gh/ollama-kotlin) - Kotlin Ollama 客户端
- [Ollama for PHP](https://github.com/Arkanius/ollama-pp) - PHP Ollama 客户端

### 其他工具

- [KIM](https://github.com/user-attachments/assets/495ad0bf) - 基于 React 的个人知识库管理器
- [Pinokio](https://pinokio.computer) - AI 应用浏览器
- [Instructor](https://github.com/instructor-ai/instructor) - 结构化 LLM 输出
- [Ollama Bubble Tea 示例](https://github.com/magicmonkey/ollama-bubbletea) - 终端 UI 示例
- [Ollama App](https://github.com/JHubi1/ollama-app) - 跨平台客户端
- [SimpleGPT](https://github.com/alexanderatallah/simplegpt) - 简约 Web UI
- [Ollama WebUI Lite](https://github.com/ollama-webui/ollama-webui-lite) - 轻量级 Web 界面
- [LocalGPT](https://github.com/PromtEngineer/localGPT) - 本地文档聊天
- [Ollama GUI](https://github.com/chyok/ollama-gui) - Python Tkinter GUI
- [Ollama With Voice](https://github.com/technovangelist/ollamawithvoice) - 语音交互界面

---

## 许可协议

MIT 许可协议
