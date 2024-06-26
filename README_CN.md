# Ollama

[![Discord](https://dcbadge.vercel.app/api/server/ollama?style=flat&compact=true)](https://discord.gg/ollama)

欢迎您使用Ollama,下面是关于启动并运行Ollama的文档。

### macOS

[下载](https://ollama.com/download/Ollama-darwin.zip)

### Windows preview

[下载](https://ollama.com/download/OllamaSetup.exe)

### Linux

```
curl -fsSL https://ollama.com/install.sh | sh
```

[手动安装说明](https://github.com/ollama/ollama/blob/main/docs/linux.md)

### Docker

官方的Docker Hub是 [Ollama Docker image](https://hub.docker.com/r/ollama/ollama) `ollama/ollama`

### Libraries

- [ollama-python](https://github.com/ollama/ollama-python)
- [ollama-js](https://github.com/ollama/ollama-js)

## 快速入门

用聊天的方式运行 [Llama 3](https://ollama.com/library/llama3)：

```
ollama run llama3
```

## 模型库

Ollama支持的模型可以在 [ollama.com/library](https://ollama.com/library "ollama model library") 查看模型列表

以下是一些可以下载的示例模型

| Model              | Parameters | Size  | Download                         |
| ------------------ | ---------- | ----- | -------------------------------- |
| Llama 3            | 8B         | 4.7GB | `ollama run llama3`            |
| Llama 3            | 70B        | 40GB  | `ollama run llama3:70b`        |
| Phi 3 Mini         | 3.8B       | 2.3GB | `ollama run phi3`              |
| Phi 3 Medium       | 14B        | 7.9GB | `ollama run phi3:medium`       |
| Gemma              | 2B         | 1.4GB | `ollama run gemma:2b`          |
| Gemma              | 7B         | 4.8GB | `ollama run gemma:7b`          |
| Mistral            | 7B         | 4.1GB | `ollama run mistral`           |
| Moondream 2        | 1.4B       | 829MB | `ollama run moondream`         |
| Neural Chat        | 7B         | 4.1GB | `ollama run neural-chat`       |
| Starling           | 7B         | 4.1GB | `ollama run starling-lm`       |
| Code Llama         | 7B         | 3.8GB | `ollama run codellama`         |
| Llama 2 Uncensored | 7B         | 3.8GB | `ollama run llama2-uncensored` |
| LLaVA              | 7B         | 4.5GB | `ollama run llava`             |
| Solar              | 10.7B      | 6.1GB | `ollama run solar`             |

> 注意：您应该至少有 8 GB 的 RAM 可用于运行 7B 型号，16 GB 可用于运行 13B 型号，32 GB 可用于运行 33B 型号。

## 自定义模型

### 从 GGUF 导入

Ollama 支持在 Modelfile 中导入 GGUF 模型:

1. 创建一个名为 `Modelfile` 的文件，其中包含一条指令，其中包含要导入的模型的本地文件路径

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

有关详细信息，请参阅有关导入模型[的指南](https://github.com/ollama/ollama/blob/main/docs/import.md)。

### 自定义 prompt

Ollama 库中的模型可以通过prompt进行自定义。例如，要自定义模型：`llama3`

```
ollama pull llama3
```

创建 `Modelfile`:

```
FROM llama3

# set the temperature to 1 [higher is more creative, lower is more coherent]
PARAMETER temperature 1

# set the system message
SYSTEM """
You are Mario from Super Mario Bros. Answer as Mario, the assistant, only.
"""
```

接下来，创建并运行模型:

```
ollama create mario -f ./Modelfile
ollama run mario
>>> hi
Hello! It's your friend Mario.
```

有关更多示例，请参阅 [examples](https://github.com/ollama/ollama/blob/main/examples) 目录。有关使用 Modelfile 的详细信息，请参阅 [Modelfile](https://github.com/ollama/ollama/blob/main/docs/modelfile.md) 文档

## CLI 参考

### 创建模型

`ollama create`用于从 Modelfile 创建模型

```
ollama create mymodel -f ./Modelfile
```

### 拉取模型

```
ollama pull llama3
```

> 此命令还可用于更新本地模型。只会拉取差异。

### 删除模型

```
ollama rm llama3
```

### 复制模型

```
ollama cp llama3 my-model
```

### Multiline input

对于多行输入，您可以使用以下命令换行文本：`"""`

```
>>> """Hello,
... world!
... """
I'm a basic program that prints the famous "Hello, world!" message to the console.
```

### Multimodal models

```
>>> What's in this image? /Users/jmorgan/Desktop/smile.png
The image features a yellow smiley face, which is likely the central focus of the picture.
```

### Pass the prompt as an argument

```
$ ollama run llama3 "Summarize this file: $(cat README.md)"
 Ollama is a lightweight, extensible framework for building and running language models on the local machine. It provides a simple API for creating, running, and managing models, as well as a library of pre-built models that can be easily used in a variety of applications.
```

### 显示模型信息

```
ollama show llama3
```

### 列出计算机上的型号

```
ollama list
```

### 启动Ollama

`ollama serve`当您想要在不运行桌面应用程序的情况下启动 Ollama 时使用

### Building

请参阅开发者指南 [developer guide](https://github.com/ollama/ollama/blob/main/docs/development.md)

### 在本地启动

接下来，启动服务器:

```
./ollama serve
```

最后，在单独的 shell 中，运行一个模型:

```
./ollama run llama3
```

## REST API

Ollama有一个用于运行和管理模型的 REST API.

### 返回结果

```
curl http://localhost:11434/api/generate -d '{
  "model": "llama3",
  "prompt":"Why is the sky blue?"
}'
```

### 和模型对话

```
curl http://localhost:11434/api/chat -d '{
  "model": "llama3",
  "messages": [
    { "role": "user", "content": "why is the sky blue?" }
  ]
}'
```

请参阅所有端点的 [API documentation](./docs/api.md)

## 社区资源汇总

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
- [llm-ollama](https://github.com/taketwo/llm-ollama) for [Datasette&#39;s LLM CLI](https://llm.datasette.io/en/stable/).
- [typechat-cli](https://github.com/anaisbetts/typechat-cli)
- [ShellOracle](https://github.com/djcopley/ShellOracle)
- [tlm](https://github.com/yusufcanb/tlm)
- [podman-ollama](https://github.com/ericcurtin/podman-ollama)
- [gollama](https://github.com/sammcj/gollama)

### Database

- [MindsDB](https://github.com/mindsdb/mindsdb/blob/staging/mindsdb/integrations/handlers/ollama_handler/README.md) (Connects Ollama models with nearly 200 data platforms and apps)
- [chromem-go](https://github.com/philippgille/chromem-go/blob/v0.5.0/embed_ollama.go) with [example](https://github.com/philippgille/chromem-go/tree/v0.5.0/examples/rag-wikipedia-ollama)

### Package managers

- [Pacman](https://archlinux.org/packages/extra/x86_64/ollama/)
- [Helm Chart](https://artifacthub.io/packages/helm/ollama-helm/ollama)
- [Guix channel](https://codeberg.org/tusharhero/ollama-guix)

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
