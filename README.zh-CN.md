<div align="center">
 <img alt="ollama" height="200px" src="https://github.com/ollama/ollama/assets/3325447/0d0b44e2-8f4a-4e99-9b52-a5c1c741c8f7">
</div>
# Ollama

[![Discord](https://dcbadge.vercel.app/api/server/ollama?style=flat&compact=true)](https://discord.gg/ollama)

开始使用大型语言模型。

### macOS

[Download](https://ollama.com/download/Ollama-darwin.zip)

### Windows 预览版

[Download](https://ollama.com/download/OllamaSetup.exe)

### Linux

```
curl -fsSL https://ollama.com/install.sh | sh
```

[手动安装说明](https://github.com/ollama/ollama/blob/main/docs/linux.md)

### Docker

官方 [Ollama Docker](https://hub.docker.com/r/ollama/ollama) 镜像 `ollama/ollama` 可在 Docker Hub 上获取。

### 库

- ollama-python
- ollama-js

## 快速开始

要运行并与 Llama 3 聊天：

```
ollama run llama3
```

## 模型库

Ollama 支持在 ollama.com/library 上可获取的模型列表。

以下是一些可下载的示例模型：

| 模型               | 参数  | 大小  | 下载方式                       |
| ------------------ | ----- | ----- | ------------------------------ |
| Llama 3            | 8B    | 4.7GB | `ollama run llama3`            |
| Llama 3            | 70B   | 40GB  | `ollama run llama3:70b`        |
| Phi 3 Mini         | 3.8B  | 2.3GB | `ollama run phi3`              |
| Phi 3 Medium       | 14B   | 7.9GB | `ollama run phi3:medium`       |
| Gemma              | 2B    | 1.4GB | `ollama run gemma:2b`          |
| Gemma              | 7B    | 4.8GB | `ollama run gemma:7b`          |
| Mistral            | 7B    | 4.1GB | `ollama run mistral`           |
| Moondream 2        | 1.4B  | 829MB | `ollama run moondream`         |
| Neural Chat        | 7B    | 4.1GB | `ollama run neural-chat`       |
| Starling           | 7B    | 4.1GB | `ollama run starling-lm`       |
| Code Llama         | 7B    | 3.8GB | `ollama run codellama`         |
| Llama 2 Uncensored | 7B    | 3.8GB | `ollama run llama2-uncensored` |
| LLaVA              | 7B    | 4.5GB | `ollama run llava`             |
| Solar              | 10.7B | 6.1GB | `ollama run solar`             |

> 注意：运行 7B 模型至少需要 8 GB 的 RAM，运行 13B 模型需要 16 GB，运行 33B 模型需要 32 GB。

## 自定义模型

### 从 GGUF 导入

Ollama 支持在 Modelfile 中导入 GGUF 模型：

1. 创建一个名为 `Modelfile` 的文件，使用 `FROM` 指令指定要导入的模型的本地文件路径。

   ```
   FROM ./vicuna-33b.Q4_0.gguf
   ```

2. 在 Ollama 中创建模型

   ```
   ollama create example -f Modelfile
   ```

3. 运行模型

   ```
   ollama run example
   ```

### 从 PyTorch 或 Safetensors 导入

有关导入模型的[更多信息](docs/import.md)，请参阅导入指南。

### 自定义提示

可以使用提示来自定义 Ollama 库中的模型。例如，自定义 `llama3` 模型：

```
ollama pull llama3
```

创建 `Modelfile`：

```
FROM llama3

# 设置温度为1 [较高的值更具创造性，较低的值更加连贯]
PARAMETER temperature 1

# 设置系统消息
SYSTEM """
你是超级马里奥兄弟中的马里奥。只作为马里奥这个助手回答。
"""
```

接下来，创建并运行模型：

```
ollama create mario -f ./Modelfile
ollama run mario
>>> hi
Hello! It's your friend Mario.
```

更多示例请查看 示例 目录。更多关于使用 Modelfile 的信息，请参阅 [Modelfile](docs/modelfile.md) 文档。

## 命令行参考

### 创建模型

`ollama create` 用于从 Modelfile 创建模型。

```
ollama create mymodel -f ./Modelfile
```

### 拉取模型

```
ollama pull llama3
```

> 此命令也可以用来更新本地模型，只会拉取差异部分。

### 删除模型

```
ollama rm llama3
```

### 复制模型

```
ollama cp llama3 my-model
```

### 多行输入

对于多行输入，可以使用 `"""` 包裹文本：

```
>>> """Hello,
... world!
... """
I'm a basic program that prints the famous "Hello, world!" message to the console.
```

### 多模态模型

```
>>> What's in this image? /Users/jmorgan/Desktop/smile.png
这张图片中有一个黄色的笑脸表情，这很可能是图片的中心焦点。
```

### 将提示作为参数传递

```
$ ollama run llama3 "Summarize this file: $(cat README.md)"
Ollama 是一个轻量级、可扩展的框架，用于在本地机器上构建和运行语言模型。它提供了一个简单的 API 用于创建、运行和管理模型，以及一个预建模型库，这些模型可以在多种应用中轻松使用。
```

### 列出计算机上的模型

```
ollama list
```

### 启动 Ollama

使用 `ollama serve` 可以在不运行桌面应用程序的情况下启动 ollama。

## 构建

参阅 [开发者指南](https://github.com/ollama/ollama/blob/main/docs/development.md)

### 运行本地构建

接下来，启动服务器：

```
./ollama serve
```

最后，在另一个终端中运行模型：

```
./ollama run llama3
```

## REST API

Ollama 提供了一个 REST API 用于运行和管理模型。

### 生成响应

```
curl http://localhost:11434/api/generate -d '{
  "model": "llama3",
  "prompt":"为什么天空是蓝色的？"
}'
```

### 与模型聊天

```
curl http://localhost:11434/api/chat -d '{
  "model": "llama3",
  "messages": [
    { "role": "user", "content": "为什么天空是蓝色的？" }
  ]
}'
```

有关所有端点的更多信息，请参阅 [API 文档](./docs/api.md)。

## 社区集成

### Web & Desktop

- [Open WebUI](https://github.com/open-webui/open-webui)
- [Enchanted (macOS native)](https://github.com/AugustDev/enchanted)
- [Hollama](https://github.com/fmaclen/hollama)
- [Lollms-Webui](https://github.com/ParisNeo/lollms-webui)
- [LibreChat](https://github.com/danny-avila/LibreChat)
- [Bionic GPT](https://github.com/bionic-gpt/bionic-gpt)
- [HTML UI](https://github.com/rtcfirefly/ollama-ui)
- [Saddle](https://github.com/jikkuatwork/saddle)
- [Chatbot UI](https://github.com/ivanfioravanti/chatbot-ollama)
- [Chatbot UI v2](https://github.com/mckaywrigley/chatbot-ui)
- [Typescript UI](https://github.com/ollama-interface/Ollama-Gui?tab=readme-ov-file)
- [Minimalistic React UI for Ollama Models](https://github.com/richawo/minimal-llm-ui)
- [Ollamac](https://github.com/kevinhermawan/Ollamac)
- [big-AGI](https://github.com/enricoros/big-AGI/blob/main/docs/config-local-ollama.md)
- [Cheshire Cat assistant framework](https://github.com/cheshire-cat-ai/core)
- [Amica](https://github.com/semperai/amica)
- [chatd](https://github.com/BruceMacD/chatd)
- [Ollama-SwiftUI](https://github.com/kghandour/Ollama-SwiftUI)
- [Dify.AI](https://github.com/langgenius/dify)
- [MindMac](https://mindmac.app)
- [NextJS Web Interface for Ollama](https://github.com/jakobhoeg/nextjs-ollama-llm-ui)
- [Msty](https://msty.app)
- [Chatbox](https://github.com/Bin-Huang/Chatbox)
- [WinForm Ollama Copilot](https://github.com/tgraupmann/WinForm_Ollama_Copilot)
- [NextChat](https://github.com/ChatGPTNextWeb/ChatGPT-Next-Web) with [Get Started Doc](https://docs.nextchat.dev/models/ollama)
- [Alpaca WebUI](https://github.com/mmo80/alpaca-webui)
- [OllamaGUI](https://github.com/enoch1118/ollamaGUI)
- [OpenAOE](https://github.com/InternLM/OpenAOE)
- [Odin Runes](https://github.com/leonid20000/OdinRunes)
- [LLM-X](https://github.com/mrdjohnson/llm-x) (Progressive Web App)
- [AnythingLLM (Docker + MacOs/Windows/Linux native app)](https://github.com/Mintplex-Labs/anything-llm)
- [Ollama Basic Chat: Uses HyperDiv Reactive UI](https://github.com/rapidarchitect/ollama_basic_chat)
- [Ollama-chats RPG](https://github.com/drazdra/ollama-chats)
- [QA-Pilot](https://github.com/reid41/QA-Pilot) (Chat with Code Repository)
- [ChatOllama](https://github.com/sugarforever/chat-ollama) (Open Source Chatbot based on Ollama with Knowledge Bases)
- [CRAG Ollama Chat](https://github.com/Nagi-ovo/CRAG-Ollama-Chat) (Simple Web Search with Corrective RAG)
- [RAGFlow](https://github.com/infiniflow/ragflow) (Open-source Retrieval-Augmented Generation engine based on deep document understanding)
- [StreamDeploy](https://github.com/StreamDeploy-DevRel/streamdeploy-llm-app-scaffold) (LLM Application Scaffold)
- [chat](https://github.com/swuecho/chat) (chat web app for teams)
- [Lobe Chat](https://github.com/lobehub/lobe-chat) with [Integrating Doc](https://lobehub.com/docs/self-hosting/examples/ollama)
- [Ollama RAG Chatbot](https://github.com/datvodinh/rag-chatbot.git) (Local Chat with multiple PDFs using Ollama and RAG)
- [BrainSoup](https://www.nurgo-software.com/products/brainsoup) (Flexible native client with RAG & multi-agent automation)
- [macai](https://github.com/Renset/macai) (macOS client for Ollama, ChatGPT, and other compatible API back-ends)
- [Olpaka](https://github.com/Otacon/olpaka) (User-friendly Flutter Web App for Ollama)
- [OllamaSpring](https://github.com/CrazyNeil/OllamaSpring) (Ollama Client for macOS)
- [LLocal.in](https://github.com/kartikm7/llocal) (Easy to use Electron Desktop Client for Ollama)

### Terminal

- [oterm](https://github.com/ggozad/oterm)
- [Ellama Emacs client](https://github.com/s-kostyaev/ellama)
- [Emacs client](https://github.com/zweifisch/ollama)
- [gen.nvim](https://github.com/David-Kunz/gen.nvim)
- [ollama.nvim](https://github.com/nomnivore/ollama.nvim)
- [ollero.nvim](https://github.com/marco-souza/ollero.nvim)
- [ollama-chat.nvim](https://github.com/gerazov/ollama-chat.nvim)
- [ogpt.nvim](https://github.com/huynle/ogpt.nvim)
- [gptel Emacs client](https://github.com/karthink/gptel)
- [Oatmeal](https://github.com/dustinblackman/oatmeal)
- [cmdh](https://github.com/pgibler/cmdh)
- [ooo](https://github.com/npahlfer/ooo)
- [shell-pilot](https://github.com/reid41/shell-pilot)
- [tenere](https://github.com/pythops/tenere)
- [llm-ollama](https://github.com/taketwo/llm-ollama) for [Datasette's LLM CLI](https://llm.datasette.io/en/stable/).
- [typechat-cli](https://github.com/anaisbetts/typechat-cli)
- [ShellOracle](https://github.com/djcopley/ShellOracle)
- [tlm](https://github.com/yusufcanb/tlm)
- [podman-ollama](https://github.com/ericcurtin/podman-ollama)
- [gollama](https://github.com/sammcj/gollama)

### Database

- [MindsDB](https://github.com/mindsdb/mindsdb/blob/staging/mindsdb/integrations/handlers/ollama_handler/README.md)（连接 Ollama 模型与近 200 个数据平台和应用程序）
- [chromem-go](https://github.com/philippgille/chromem-go/blob/v0.5.0/embed_ollama.go) 和 [示例](https://github.com/philippgille/chromem-go/tree/v0.5.0/examples/rag-wikipedia-ollama)

### Package managers

- [Pacman](https://archlinux.org/packages/extra/x86_64/ollama/)（Arch Linux 的包管理器）
- [Helm Chart](https://artifacthub.io/packages/helm/ollama-helm/ollama)（用于 Kubernetes 的包管理）
- [Guix channel](https://codeberg.org/tusharhero/ollama-guix)（Guix 系统的软件包通道）

### Libraries

- [LangChain](https://python.langchain.com/docs/integrations/llms/ollama) and [LangChain.js](https://js.langchain.com/docs/modules/model_io/models/llms/integrations/ollama) with [example](https://js.langchain.com/docs/use_cases/question_answering/local_retrieval_qa)
- [LangChainGo](https://github.com/tmc/langchaingo/) with [example](https://github.com/tmc/langchaingo/tree/main/examples/ollama-completion-example)
- [LangChain4j](https://github.com/langchain4j/langchain4j) with [example](https://github.com/langchain4j/langchain4j-examples/tree/main/ollama-examples/src/main/java)
- [LangChainRust](https://github.com/Abraxas-365/langchain-rust) with [example](https://github.com/Abraxas-365/langchain-rust/blob/main/examples/llm_ollama.rs)
- [LlamaIndex](https://gpt-index.readthedocs.io/en/stable/examples/llm/ollama.html)
- [LiteLLM](https://github.com/BerriAI/litellm)
- [OllamaSharp for .NET](https://github.com/awaescher/OllamaSharp)
- [Ollama for Ruby](https://github.com/gbaptista/ollama-ai)
- [Ollama-rs for Rust](https://github.com/pepperoni21/ollama-rs)
- [Ollama-hpp for C++](https://github.com/jmont-dev/ollama-hpp)
- [Ollama4j for Java](https://github.com/amithkoujalgi/ollama4j)
- [ModelFusion Typescript Library](https://modelfusion.dev/integration/model-provider/ollama)
- [OllamaKit for Swift](https://github.com/kevinhermawan/OllamaKit)
- [Ollama for Dart](https://github.com/breitburg/dart-ollama)
- [Ollama for Laravel](https://github.com/cloudstudio/ollama-laravel)
- [LangChainDart](https://github.com/davidmigloz/langchain_dart)
- [Semantic Kernel - Python](https://github.com/microsoft/semantic-kernel/tree/main/python/semantic_kernel/connectors/ai/ollama)
- [Haystack](https://github.com/deepset-ai/haystack-integrations/blob/main/integrations/ollama.md)
- [Elixir LangChain](https://github.com/brainlid/langchain)
- [Ollama for R - rollama](https://github.com/JBGruber/rollama)
- [Ollama for R - ollama-r](https://github.com/hauselin/ollama-r)
- [Ollama-ex for Elixir](https://github.com/lebrunel/ollama-ex)
- [Ollama Connector for SAP ABAP](https://github.com/b-tocs/abap_btocs_ollama)
- [Testcontainers](https://testcontainers.com/modules/ollama/)
- [Portkey](https://portkey.ai/docs/welcome/integration-guides/ollama)
- [PromptingTools.jl](https://github.com/svilupp/PromptingTools.jl) with an [example](https://svilupp.github.io/PromptingTools.jl/dev/examples/working_with_ollama)
- [LlamaScript](https://github.com/Project-Llama/llamascript)

### Mobile

- [Enchanted](https://github.com/AugustDev/enchanted)
- [Maid](https://github.com/Mobile-Artificial-Intelligence/maid)

### Extensions & Plugins

- [Raycast extension](https://github.com/MassimilianoPasquini97/raycast_ollama)
- [Discollama](https://github.com/mxyng/discollama) (Discord bot inside the Ollama discord channel)
- [Continue](https://github.com/continuedev/continue)
- [Obsidian Ollama plugin](https://github.com/hinterdupfinger/obsidian-ollama)
- [Logseq Ollama plugin](https://github.com/omagdy7/ollama-logseq)
- [NotesOllama](https://github.com/andersrex/notesollama) (Apple Notes Ollama plugin)
- [Dagger Chatbot](https://github.com/samalba/dagger-chatbot)
- [Discord AI Bot](https://github.com/mekb-turtle/discord-ai-bot)
- [Ollama Telegram Bot](https://github.com/ruecat/ollama-telegram)
- [Hass Ollama Conversation](https://github.com/ej52/hass-ollama-conversation)
- [Rivet plugin](https://github.com/abrenneke/rivet-plugin-ollama)
- [Obsidian BMO Chatbot plugin](https://github.com/longy2k/obsidian-bmo-chatbot)
- [Cliobot](https://github.com/herval/cliobot) (Telegram bot with Ollama support)
- [Copilot for Obsidian plugin](https://github.com/logancyang/obsidian-copilot)
- [Obsidian Local GPT plugin](https://github.com/pfrankov/obsidian-local-gpt)
- [Open Interpreter](https://docs.openinterpreter.com/language-model-setup/local-models/ollama)
- [Llama Coder](https://github.com/ex3ndr/llama-coder) (Copilot alternative using Ollama)
- [Ollama Copilot](https://github.com/bernardo-bruning/ollama-copilot) (Proxy that allows you to use ollama as a copilot like Github copilot)
- [twinny](https://github.com/rjmacarthy/twinny) (Copilot and Copilot chat alternative using Ollama)
- [Wingman-AI](https://github.com/RussellCanfield/wingman-ai) (Copilot code and chat alternative using Ollama and HuggingFace)
- [Page Assist](https://github.com/n4ze3m/page-assist) (Chrome Extension)
- [AI Telegram Bot](https://github.com/tusharhero/aitelegrambot) (Telegram bot using Ollama in backend)
- [AI ST Completion](https://github.com/yaroslavyaroslav/OpenAI-sublime-text) (Sublime Text 4 AI assistant plugin with Ollama support)
- [Discord-Ollama Chat Bot](https://github.com/kevinthedang/discord-ollama) (Generalized TypeScript Discord Bot w/ Tuning Documentation)
- [Discord AI chat/moderation bot](https://github.com/rapmd73/Companion) Chat/moderation bot written in python. Uses Ollama to create personalities.
- [Headless Ollama](https://github.com/nischalj10/headless-ollama) (Scripts to automatically install ollama client & models on any OS for apps that depends on ollama server)

### Supported backends

- [llama.cpp](https://github.com/ggerganov/llama.cpp) project founded by Georgi Gerganov.

