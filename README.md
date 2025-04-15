<div align="center">
  <a href="https://ollama.com">
    <img alt="ollama" height="200px" src="https://github.com/ollama/ollama/assets/3325447/0d0b44e2-8f4a-4e99-9b52-a5c1c741c8f7">
  </a>
</div>

# Ollama

Get up and running with large language models.

### macOS

[Download](https://ollama.com/download/Ollama-darwin.zip)

### Windows

[Download](https://ollama.com/download/OllamaSetup.exe)

### Linux

```shell
curl -fsSL https://ollama.com/install.sh | sh
```

[Manual install instructions](https://github.com/ollama/ollama/blob/main/docs/linux.md)

### Docker

The official [Ollama Docker image](https://hub.docker.com/r/ollama/ollama) `ollama/ollama` is available on Docker Hub.

### Libraries

- [ollama-python](https://github.com/ollama/ollama-python)
- [ollama-js](https://github.com/ollama/ollama-js)

### Community

- [Discord](https://discord.gg/ollama)
- [Reddit](https://reddit.com/r/ollama)

## Quickstart

To run and chat with [Llama 3.2](https://ollama.com/library/llama3.2):

```shell
ollama run llama3.2
```

## Model library

Ollama supports a list of models available on [ollama.com/library](https://ollama.com/library 'ollama model library')

Here are some example models that can be downloaded:

| Model              | Parameters | Size  | Download                         |
| ------------------ | ---------- | ----- | -------------------------------- |
| Gemma 3            | 1B         | 815MB | `ollama run gemma3:1b`           |
| Gemma 3            | 4B         | 3.3GB | `ollama run gemma3`              |
| Gemma 3            | 12B        | 8.1GB | `ollama run gemma3:12b`          |
| Gemma 3            | 27B        | 17GB  | `ollama run gemma3:27b`          |
| QwQ                | 32B        | 20GB  | `ollama run qwq`                 |
| DeepSeek-R1        | 7B         | 4.7GB | `ollama run deepseek-r1`         |
| DeepSeek-R1        | 671B       | 404GB | `ollama run deepseek-r1:671b`    |
| Llama 3.3          | 70B        | 43GB  | `ollama run llama3.3`            |
| Llama 3.2          | 3B         | 2.0GB | `ollama run llama3.2`            |
| Llama 3.2          | 1B         | 1.3GB | `ollama run llama3.2:1b`         |
| Llama 3.2 Vision   | 11B        | 7.9GB | `ollama run llama3.2-vision`     |
| Llama 3.2 Vision   | 90B        | 55GB  | `ollama run llama3.2-vision:90b` |
| Llama 3.1          | 8B         | 4.7GB | `ollama run llama3.1`            |
| Llama 3.1          | 405B       | 231GB | `ollama run llama3.1:405b`       |
| Phi 4              | 14B        | 9.1GB | `ollama run phi4`                |
| Phi 4 Mini         | 3.8B       | 2.5GB | `ollama run phi4-mini`           |
| Mistral            | 7B         | 4.1GB | `ollama run mistral`             |
| Moondream 2        | 1.4B       | 829MB | `ollama run moondream`           |
| Neural Chat        | 7B         | 4.1GB | `ollama run neural-chat`         |
| Starling           | 7B         | 4.1GB | `ollama run starling-lm`         |
| Code Llama         | 7B         | 3.8GB | `ollama run codellama`           |
| Llama 2 Uncensored | 7B         | 3.8GB | `ollama run llama2-uncensored`   |
| LLaVA              | 7B         | 4.5GB | `ollama run llava`               |
| Granite-3.2         | 8B         | 4.9GB | `ollama run granite3.2`          |

> [!NOTE]
> You should have at least 8 GB of RAM available to run the 7B models, 16 GB to run the 13B models, and 32 GB to run the 33B models.

## Customize a model

### Import from GGUF

Ollama supports importing GGUF models in the Modelfile:

1. Create a file named `Modelfile`, with a `FROM` instruction with the local filepath to the model you want to import.

   ```
   FROM ./vicuna-33b.Q4_0.gguf
   ```

2. Create the model in Ollama

   ```shell
   ollama create example -f Modelfile
   ```

3. Run the model

   ```shell
   ollama run example
   ```

### Import from Safetensors

See the [guide](docs/import.md) on importing models for more information.

### Customize a prompt

Models from the Ollama library can be customized with a prompt. For example, to customize the `llama3.2` model:

```shell
ollama pull llama3.2
```

Create a `Modelfile`:

```
FROM llama3.2

# set the temperature to 1 [higher is more creative, lower is more coherent]
PARAMETER temperature 1

# set the system message
SYSTEM """
You are Mario from Super Mario Bros. Answer as Mario, the assistant, only.
"""
```

Next, create and run the model:

```
ollama create mario -f ./Modelfile
ollama run mario
>>> hi
Hello! It's your friend Mario.
```

For more information on working with a Modelfile, see the [Modelfile](docs/modelfile.md) documentation.

## CLI Reference

### Create a model

`ollama create` is used to create a model from a Modelfile.

```shell
ollama create mymodel -f ./Modelfile
```

### Pull a model

```shell
ollama pull llama3.2
```

> This command can also be used to update a local model. Only the diff will be pulled.

### Remove a model

```shell
ollama rm llama3.2
```

### Copy a model

```shell
ollama cp llama3.2 my-model
```

### Multiline input

For multiline input, you can wrap text with `"""`:

```
>>> """Hello,
... world!
... """
I'm a basic program that prints the famous "Hello, world!" message to the console.
```

### Multimodal models

```
ollama run llava "What's in this image? /Users/jmorgan/Desktop/smile.png"
```

> **Output**: The image features a yellow smiley face, which is likely the central focus of the picture.

### Pass the prompt as an argument

```shell
ollama run llama3.2 "Summarize this file: $(cat README.md)"
```

> **Output**: Ollama is a lightweight, extensible framework for building and running language models on the local machine. It provides a simple API for creating, running, and managing models, as well as a library of pre-built models that can be easily used in a variety of applications.

### Show model information

```shell
ollama show llama3.2
```

### List models on your computer

```shell
ollama list
```

### List which models are currently loaded

```shell
ollama ps
```

### Stop a model which is currently running

```shell
ollama stop llama3.2
```

### Start Ollama

`ollama serve` is used when you want to start ollama without running the desktop application.

## Building

See the [developer guide](https://github.com/ollama/ollama/blob/main/docs/development.md)

### Running local builds

Next, start the server:

```shell
./ollama serve
```

Finally, in a separate shell, run a model:

```shell
./ollama run llama3.2
```

## REST API

Ollama has a REST API for running and managing models.

### Generate a response

```shell
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.2",
  "prompt":"Why is the sky blue?"
}'
```

### Chat with a model

```shell
curl http://localhost:11434/api/chat -d '{
  "model": "llama3.2",
  "messages": [
    { "role": "user", "content": "why is the sky blue?" }
  ]
}'
```

See the [API documentation](./docs/api.md) for all endpoints.

## Community Integrations

### Web & Desktop

- [Open WebUI](https://github.com/open-webui/open-webui)
- [SwiftChat (macOS with ReactNative)](https://github.com/aws-samples/swift-chat)
- [Enchanted (macOS native)](https://github.com/AugustDev/enchanted)
- [Hollama](https://github.com/fmaclen/hollama)
- [Lollms-Webui](https://github.com/ParisNeo/lollms-webui)
- [LibreChat](https://github.com/danny-avila/LibreChat)
- [Bionic GPT](https://github.com/bionic-gpt/bionic-gpt)
- [HTML UI](https://github.com/rtcfirefly/ollama-ui)
- [Saddle](https://github.com/jikkuatwork/saddle)
- [TagSpaces](https://www.tagspaces.org) (A platform for file based apps, [utilizing Ollama](https://docs.tagspaces.org/ai/) for the generation of tags and descriptions)
- [Chatbot UI](https://github.com/ivanfioravanti/chatbot-ollama)
- [Chatbot UI v2](https://github.com/mckaywrigley/chatbot-ui)
- [Typescript UI](https://github.com/ollama-interface/Ollama-Gui?tab=readme-ov-file)
- [Minimalistic React UI for Ollama Models](https://github.com/richawo/minimal-llm-ui)
- [Ollamac](https://github.com/kevinhermawan/Ollamac)
- [big-AGI](https://github.com/enricoros/big-AGI) 
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
- [IntelliBar](https://intellibar.app/) (AI-powered assistant for macOS)
- [QA-Pilot](https://github.com/reid41/QA-Pilot) (Interactive chat tool that can leverage Ollama models for rapid understanding and navigation of GitHub code repositories)
- [ChatOllama](https://github.com/sugarforever/chat-ollama) (Open Source Chatbot based on Ollama with Knowledge Bases)
- [CRAG Ollama Chat](https://github.com/Nagi-ovo/CRAG-Ollama-Chat) (Simple Web Search with Corrective RAG)
- [RAGFlow](https://github.com/infiniflow/ragflow) (Open-source Retrieval-Augmented Generation engine based on deep document understanding)
- [StreamDeploy](https://github.com/StreamDeploy-DevRel/streamdeploy-llm-app-scaffold) (LLM Application Scaffold)
- [chat](https://github.com/swuecho/chat) (chat web app for teams)
- [Lobe Chat](https://github.com/lobehub/lobe-chat) with [Integrating Doc](https://lobehub.com/docs/self-hosting/examples/ollama)
- [Ollama RAG Chatbot](https://github.com/datvodinh/rag-chatbot.git) (Local Chat with multiple PDFs using Ollama and RAG)
- [BrainSoup](https://www.nurgo-software.com/products/brainsoup) (Flexible native client with RAG & multi-agent automation)
- [macai](https://github.com/Renset/macai) (macOS client for Ollama, ChatGPT, and other compatible API back-ends)
- [RWKV-Runner](https://github.com/josStorer/RWKV-Runner) (RWKV offline LLM deployment tool, also usable as a client for ChatGPT and Ollama)
- [Ollama Grid Search](https://github.com/dezoito/ollama-grid-search) (app to evaluate and compare models)
- [Olpaka](https://github.com/Otacon/olpaka) (User-friendly Flutter Web App for Ollama)
- [Casibase](https://casibase.org) (An open source AI knowledge base and dialogue system combining the latest RAG, SSO, ollama support and multiple large language models.)
- [OllamaSpring](https://github.com/CrazyNeil/OllamaSpring) (Ollama Client for macOS)
- [LLocal.in](https://github.com/kartikm7/llocal) (Easy to use Electron Desktop Client for Ollama)
- [Shinkai Desktop](https://github.com/dcSpark/shinkai-apps) (Two click install Local AI using Ollama + Files + RAG)
- [AiLama](https://github.com/zeyoyt/ailama) (A Discord User App that allows you to interact with Ollama anywhere in discord )
- [Ollama with Google Mesop](https://github.com/rapidarchitect/ollama_mesop/) (Mesop Chat Client implementation with Ollama)
- [R2R](https://github.com/SciPhi-AI/R2R) (Open-source RAG engine)
- [Ollama-Kis](https://github.com/elearningshow/ollama-kis) (A simple easy to use GUI with sample custom LLM for Drivers Education)
- [OpenGPA](https://opengpa.org) (Open-source offline-first Enterprise Agentic Application)
- [Painting Droid](https://github.com/mateuszmigas/painting-droid) (Painting app with AI integrations)
- [Kerlig AI](https://www.kerlig.com/) (AI writing assistant for macOS)
- [AI Studio](https://github.com/MindWorkAI/AI-Studio)
- [Sidellama](https://github.com/gyopak/sidellama) (browser-based LLM client)
- [LLMStack](https://github.com/trypromptly/LLMStack) (No-code multi-agent framework to build LLM agents and workflows)
- [BoltAI for Mac](https://boltai.com) (AI Chat Client for Mac)
- [Harbor](https://github.com/av/harbor) (Containerized LLM Toolkit with Ollama as default backend)
- [PyGPT](https://github.com/szczyglis-dev/py-gpt) (AI desktop assistant for Linux, Windows and Mac)
- [Alpaca](https://github.com/Jeffser/Alpaca) (An Ollama client application for linux and macos made with GTK4 and Adwaita)
- [AutoGPT](https://github.com/Significant-Gravitas/AutoGPT/blob/master/docs/content/platform/ollama.md) (AutoGPT Ollama integration)
- [Go-CREW](https://www.jonathanhecl.com/go-crew/) (Powerful Offline RAG in Golang)
- [PartCAD](https://github.com/openvmp/partcad/) (CAD model generation with OpenSCAD and CadQuery)
- [Ollama4j Web UI](https://github.com/ollama4j/ollama4j-web-ui) - Java-based Web UI for Ollama built with Vaadin, Spring Boot and Ollama4j
- [PyOllaMx](https://github.com/kspviswa/pyOllaMx) - macOS application capable of chatting with both Ollama and Apple MLX models.
- [Cline](https://github.com/cline/cline) - Formerly known as Claude Dev is a VSCode extension for multi-file/whole-repo coding
- [Cherry Studio](https://github.com/kangfenmao/cherry-studio) (Desktop client with Ollama support)
- [ConfiChat](https://github.com/1runeberg/confichat) (Lightweight, standalone, multi-platform, and privacy focused LLM chat interface with optional encryption)
- [Archyve](https://github.com/nickthecook/archyve) (RAG-enabling document library)
- [crewAI with Mesop](https://github.com/rapidarchitect/ollama-crew-mesop) (Mesop Web Interface to run crewAI with Ollama)
- [Tkinter-based client](https://github.com/chyok/ollama-gui) (Python tkinter-based Client for Ollama)
- [LLMChat](https://github.com/trendy-design/llmchat) (Privacy focused, 100% local, intuitive all-in-one chat interface)
- [Local Multimodal AI Chat](https://github.com/Leon-Sander/Local-Multimodal-AI-Chat) (Ollama-based LLM Chat with support for multiple features, including PDF RAG, voice chat, image-based interactions, and integration with OpenAI.)
- [ARGO](https://github.com/xark-argo/argo) (Locally download and run Ollama and Huggingface models with RAG on Mac/Windows/Linux)
- [OrionChat](https://github.com/EliasPereirah/OrionChat) - OrionChat is a web interface for chatting with different AI providers
- [G1](https://github.com/bklieger-groq/g1) (Prototype of using prompting strategies to improve the LLM's reasoning through o1-like reasoning chains.)
- [Web management](https://github.com/lemonit-eric-mao/ollama-web-management) (Web management page)
- [Promptery](https://github.com/promptery/promptery) (desktop client for Ollama.)
- [Ollama App](https://github.com/JHubi1/ollama-app) (Modern and easy-to-use multi-platform client for Ollama)
- [chat-ollama](https://github.com/annilq/chat-ollama) (a React Native client for Ollama)
- [SpaceLlama](https://github.com/tcsenpai/spacellama) (Firefox and Chrome extension to quickly summarize web pages with ollama in a sidebar)
- [YouLama](https://github.com/tcsenpai/youlama) (Webapp to quickly summarize any YouTube video, supporting Invidious as well)
- [DualMind](https://github.com/tcsenpai/dualmind) (Experimental app allowing two models to talk to each other in the terminal or in a web interface)
- [ollamarama-matrix](https://github.com/h1ddenpr0cess20/ollamarama-matrix) (Ollama chatbot for the Matrix chat protocol)
- [ollama-chat-app](https://github.com/anan1213095357/ollama-chat-app) (Flutter-based chat app)
- [Perfect Memory AI](https://www.perfectmemory.ai/) (Productivity AI assists personalized by what you have seen on your screen, heard and said in the meetings)
- [Hexabot](https://github.com/hexastack/hexabot) (A conversational AI builder)
- [Reddit Rate](https://github.com/rapidarchitect/reddit_analyzer) (Search and Rate Reddit topics with a weighted summation)
- [OpenTalkGpt](https://github.com/adarshM84/OpenTalkGpt) (Chrome Extension to manage open-source models supported by Ollama, create custom models, and chat with models from a user-friendly UI)
- [VT](https://github.com/vinhnx/vt.ai) (A minimal multimodal AI chat app, with dynamic conversation routing. Supports local models via Ollama)
- [Nosia](https://github.com/nosia-ai/nosia) (Easy to install and use RAG platform based on Ollama)
- [Witsy](https://github.com/nbonamy/witsy) (An AI Desktop application available for Mac/Windows/Linux)
- [Abbey](https://github.com/US-Artificial-Intelligence/abbey) (A configurable AI interface server with notebooks, document storage, and YouTube support)
- [Minima](https://github.com/dmayboroda/minima) (RAG with on-premises or fully local workflow)
- [aidful-ollama-model-delete](https://github.com/AidfulAI/aidful-ollama-model-delete) (User interface for simplified model cleanup)
- [Perplexica](https://github.com/ItzCrazyKns/Perplexica) (An AI-powered search engine & an open-source alternative to Perplexity AI)
- [Ollama Chat WebUI for Docker ](https://github.com/oslook/ollama-webui) (Support for local docker deployment, lightweight ollama webui)
- [AI Toolkit for Visual Studio Code](https://aka.ms/ai-tooklit/ollama-docs) (Microsoft-official VSCode extension to chat, test, evaluate models with Ollama support, and use them in your AI applications.)
- [MinimalNextOllamaChat](https://github.com/anilkay/MinimalNextOllamaChat) (Minimal Web UI for Chat and Model Control)
- [Chipper](https://github.com/TilmanGriesel/chipper) AI interface for tinkerers (Ollama, Haystack RAG, Python)
- [ChibiChat](https://github.com/CosmicEventHorizon/ChibiChat) (Kotlin-based Android app to chat with Ollama and Koboldcpp API endpoints)
- [LocalLLM](https://github.com/qusaismael/localllm) (Minimal Web-App to run ollama models on it with a GUI)
- [Ollamazing](https://github.com/buiducnhat/ollamazing) (Web extension to run Ollama models)
- [OpenDeepResearcher-via-searxng](https://github.com/benhaotang/OpenDeepResearcher-via-searxng) (A Deep Research equivent endpoint with Ollama support for running locally)
- [AntSK](https://github.com/AIDotNet/AntSK) (Out-of-the-box & Adaptable RAG Chatbot)
- [MaxKB](https://github.com/1Panel-dev/MaxKB/) (Ready-to-use & flexible RAG Chatbot)
- [yla](https://github.com/danielekp/yla) (Web interface to freely interact with your customized models)
- [LangBot](https://github.com/RockChinQ/LangBot) (LLM-based instant messaging bots platform, with Agents, RAG features, supports multiple platforms)
- [1Panel](https://github.com/1Panel-dev/1Panel/) (Web-based Linux Server Management Tool)
- [AstrBot](https://github.com/Soulter/AstrBot/) (User-friendly LLM-based multi-platform chatbot with a WebUI, supporting RAG, LLM agents, and plugins integration)
- [Reins](https://github.com/ibrahimcetin/reins) (Easily tweak parameters, customize system prompts per chat, and enhance your AI experiments with reasoning model support.)
- [Ellama](https://github.com/zeozeozeo/ellama) (Friendly native app to chat with an Ollama instance)
- [screenpipe](https://github.com/mediar-ai/screenpipe) Build agents powered by your screen history
- [Ollamb](https://github.com/hengkysteen/ollamb) (Simple yet rich in features, cross-platform built with Flutter and designed for Ollama. Try the [web demo](https://hengkysteen.github.io/demo/ollamb/).)
- [Writeopia](https://github.com/Writeopia/Writeopia) (Text editor with integration with Ollama)

### Cloud

- [Google Cloud](https://cloud.google.com/run/docs/tutorials/gpu-gemma2-with-ollama)
- [Fly.io](https://fly.io/docs/python/do-more/add-ollama/)
- [Koyeb](https://www.koyeb.com/deploy/ollama)

### Terminal

- [oterm](https://github.com/ggozad/oterm)
- [Ellama Emacs client](https://github.com/s-kostyaev/ellama)
- [Emacs client](https://github.com/zweifisch/ollama)
- [neollama](https://github.com/paradoxical-dev/neollama) UI client for interacting with models from within Neovim
- [gen.nvim](https://github.com/David-Kunz/gen.nvim)
- [ollama.nvim](https://github.com/nomnivore/ollama.nvim)
- [ollero.nvim](https://github.com/marco-souza/ollero.nvim)
- [ollama-chat.nvim](https://github.com/gerazov/ollama-chat.nvim)
- [ogpt.nvim](https://github.com/huynle/ogpt.nvim)
- [gptel Emacs client](https://github.com/karthink/gptel)
- [Oatmeal](https://github.com/dustinblackman/oatmeal)
- [cmdh](https://github.com/pgibler/cmdh)
- [ooo](https://github.com/npahlfer/ooo)
- [shell-pilot](https://github.com/reid41/shell-pilot)(Interact with models via pure shell scripts on Linux or macOS)
- [tenere](https://github.com/pythops/tenere)
- [llm-ollama](https://github.com/taketwo/llm-ollama) for [Datasette's LLM CLI](https://llm.datasette.io/en/stable/).
- [typechat-cli](https://github.com/anaisbetts/typechat-cli)
- [ShellOracle](https://github.com/djcopley/ShellOracle)
- [tlm](https://github.com/yusufcanb/tlm)
- [podman-ollama](https://github.com/ericcurtin/podman-ollama)
- [gollama](https://github.com/sammcj/gollama)
- [ParLlama](https://github.com/paulrobello/parllama)
- [Ollama eBook Summary](https://github.com/cognitivetech/ollama-ebook-summary/)
- [Ollama Mixture of Experts (MOE) in 50 lines of code](https://github.com/rapidarchitect/ollama_moe)
- [vim-intelligence-bridge](https://github.com/pepo-ec/vim-intelligence-bridge) Simple interaction of "Ollama" with the Vim editor
- [x-cmd ollama](https://x-cmd.com/mod/ollama)
- [bb7](https://github.com/drunkwcodes/bb7)
- [SwollamaCLI](https://github.com/marcusziade/Swollama) bundled with the Swollama Swift package. [Demo](https://github.com/marcusziade/Swollama?tab=readme-ov-file#cli-usage)
- [aichat](https://github.com/sigoden/aichat) All-in-one LLM CLI tool featuring Shell Assistant, Chat-REPL, RAG, AI tools & agents, with access to OpenAI, Claude, Gemini, Ollama, Groq, and more.
- [PowershAI](https://github.com/rrg92/powershai) PowerShell module that brings AI to terminal on Windows, including support for Ollama
- [DeepShell](https://github.com/Abyss-c0re/deepshell) Your self-hosted AI assistant. Interactive Shell, Files and Folders analysis.
- [orbiton](https://github.com/xyproto/orbiton) Configuration-free text editor and IDE with support for tab completion with Ollama.
- [orca-cli](https://github.com/molbal/orca-cli) Ollama Registry CLI Application - Browse, pull and download models from Ollama Registry in your terminal.
- [GGUF-to-Ollama](https://github.com/jonathanhecl/gguf-to-ollama) - Importing GGUF to Ollama made easy (multiplatform)

### Apple Vision Pro

- [SwiftChat](https://github.com/aws-samples/swift-chat) (Cross-platform AI chat app supporting Apple Vision Pro via "Designed for iPad")
- [Enchanted](https://github.com/AugustDev/enchanted)

### Database

- [pgai](https://github.com/timescale/pgai) - PostgreSQL as a vector database (Create and search embeddings from Ollama models using pgvector)
   - [Get started guide](https://github.com/timescale/pgai/blob/main/docs/vectorizer-quick-start.md)
- [MindsDB](https://github.com/mindsdb/mindsdb/blob/staging/mindsdb/integrations/handlers/ollama_handler/README.md) (Connects Ollama models with nearly 200 data platforms and apps)
- [chromem-go](https://github.com/philippgille/chromem-go/blob/v0.5.0/embed_ollama.go) with [example](https://github.com/philippgille/chromem-go/tree/v0.5.0/examples/rag-wikipedia-ollama)
- [Kangaroo](https://github.com/dbkangaroo/kangaroo) (AI-powered SQL client and admin tool for popular databases)

### Package managers

- [Pacman](https://archlinux.org/packages/extra/x86_64/ollama/)
- [Gentoo](https://github.com/gentoo/guru/tree/master/app-misc/ollama)
- [Homebrew](https://formulae.brew.sh/formula/ollama)
- [Helm Chart](https://artifacthub.io/packages/helm/ollama-helm/ollama)
- [Guix channel](https://codeberg.org/tusharhero/ollama-guix)
- [Nix package](https://search.nixos.org/packages?show=ollama&from=0&size=50&sort=relevance&type=packages&query=ollama)
- [Flox](https://flox.dev/blog/ollama-part-one)

### Libraries

- [LangChain](https://python.langchain.com/docs/integrations/llms/ollama) and [LangChain.js](https://js.langchain.com/docs/integrations/chat/ollama/) with [example](https://js.langchain.com/docs/tutorials/local_rag/)
- [Firebase Genkit](https://firebase.google.com/docs/genkit/plugins/ollama)
- [crewAI](https://github.com/crewAIInc/crewAI)
- [Yacana](https://remembersoftwares.github.io/yacana/) (User-friendly multi-agent framework for brainstorming and executing predetermined flows with built-in tool integration)
- [Spring AI](https://github.com/spring-projects/spring-ai) with [reference](https://docs.spring.io/spring-ai/reference/api/chat/ollama-chat.html) and [example](https://github.com/tzolov/ollama-tools)
- [LangChainGo](https://github.com/tmc/langchaingo/) with [example](https://github.com/tmc/langchaingo/tree/main/examples/ollama-completion-example)
- [LangChain4j](https://github.com/langchain4j/langchain4j) with [example](https://github.com/langchain4j/langchain4j-examples/tree/main/ollama-examples/src/main/java)
- [LangChainRust](https://github.com/Abraxas-365/langchain-rust) with [example](https://github.com/Abraxas-365/langchain-rust/blob/main/examples/llm_ollama.rs)
- [LangChain for .NET](https://github.com/tryAGI/LangChain) with [example](https://github.com/tryAGI/LangChain/blob/main/examples/LangChain.Samples.OpenAI/Program.cs)
- [LLPhant](https://github.com/theodo-group/LLPhant?tab=readme-ov-file#ollama)
- [LlamaIndex](https://docs.llamaindex.ai/en/stable/examples/llm/ollama/) and [LlamaIndexTS](https://ts.llamaindex.ai/modules/llms/available_llms/ollama)
- [LiteLLM](https://github.com/BerriAI/litellm)
- [OllamaFarm for Go](https://github.com/presbrey/ollamafarm)
- [OllamaSharp for .NET](https://github.com/awaescher/OllamaSharp)
- [Ollama for Ruby](https://github.com/gbaptista/ollama-ai)
- [Ollama-rs for Rust](https://github.com/pepperoni21/ollama-rs)
- [Ollama-hpp for C++](https://github.com/jmont-dev/ollama-hpp)
- [Ollama4j for Java](https://github.com/ollama4j/ollama4j)
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
- [llm-axe](https://github.com/emirsahin1/llm-axe) (Python Toolkit for Building LLM Powered Apps)
- [Gollm](https://docs.gollm.co/examples/ollama-example)
- [Gollama for Golang](https://github.com/jonathanhecl/gollama)
- [Ollamaclient for Golang](https://github.com/xyproto/ollamaclient)
- [High-level function abstraction in Go](https://gitlab.com/tozd/go/fun)
- [Ollama PHP](https://github.com/ArdaGnsrn/ollama-php)
- [Agents-Flex for Java](https://github.com/agents-flex/agents-flex) with [example](https://github.com/agents-flex/agents-flex/tree/main/agents-flex-llm/agents-flex-llm-ollama/src/test/java/com/agentsflex/llm/ollama)
- [Parakeet](https://github.com/parakeet-nest/parakeet) is a GoLang library, made to simplify the development of small generative AI applications with Ollama.
- [Haverscript](https://github.com/andygill/haverscript) with [examples](https://github.com/andygill/haverscript/tree/main/examples)
- [Ollama for Swift](https://github.com/mattt/ollama-swift)
- [Swollama for Swift](https://github.com/marcusziade/Swollama) with [DocC](https://marcusziade.github.io/Swollama/documentation/swollama/)
- [GoLamify](https://github.com/prasad89/golamify)
- [Ollama for Haskell](https://github.com/tusharad/ollama-haskell)
- [multi-llm-ts](https://github.com/nbonamy/multi-llm-ts) (A Typescript/JavaScript library allowing access to different LLM in unified API)
- [LlmTornado](https://github.com/lofcz/llmtornado) (C# library providing a unified interface for major FOSS & Commercial inference APIs)
- [Ollama for Zig](https://github.com/dravenk/ollama-zig)
- [Abso](https://github.com/lunary-ai/abso) (OpenAI-compatible TypeScript SDK for any LLM provider)
- [Nichey](https://github.com/goodreasonai/nichey) is a Python package for generating custom wikis for your research topic
- [Ollama for D](https://github.com/kassane/ollama-d)

### Mobile

- [SwiftChat](https://github.com/aws-samples/swift-chat) (Lightning-fast Cross-platform AI chat app with native UI for Android, iOS and iPad)
- [Enchanted](https://github.com/AugustDev/enchanted)
- [Maid](https://github.com/Mobile-Artificial-Intelligence/maid)
- [Ollama App](https://github.com/JHubi1/ollama-app) (Modern and easy-to-use multi-platform client for Ollama)
- [ConfiChat](https://github.com/1runeberg/confichat) (Lightweight, standalone, multi-platform, and privacy focused LLM chat interface with optional encryption)
- [Ollama Android Chat](https://github.com/sunshine0523/OllamaServer) (No need for Termux, start the Ollama service with one click on an Android device)
- [Reins](https://github.com/ibrahimcetin/reins) (Easily tweak parameters, customize system prompts per chat, and enhance your AI experiments with reasoning model support.)

### Extensions & Plugins

- [Raycast extension](https://github.com/MassimilianoPasquini97/raycast_ollama)
- [Discollama](https://github.com/mxyng/discollama) (Discord bot inside the Ollama discord channel)
- [Continue](https://github.com/continuedev/continue)
- [Vibe](https://github.com/thewh1teagle/vibe) (Transcribe and analyze meetings with Ollama)
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
- [Wingman-AI](https://github.com/RussellCanfield/wingman-ai) (Copilot code and chat alternative using Ollama and Hugging Face)
- [Page Assist](https://github.com/n4ze3m/page-assist) (Chrome Extension)
- [Plasmoid Ollama Control](https://github.com/imoize/plasmoid-ollamacontrol) (KDE Plasma extension that allows you to quickly manage/control Ollama model)
- [AI Telegram Bot](https://github.com/tusharhero/aitelegrambot) (Telegram bot using Ollama in backend)
- [AI ST Completion](https://github.com/yaroslavyaroslav/OpenAI-sublime-text) (Sublime Text 4 AI assistant plugin with Ollama support)
- [Discord-Ollama Chat Bot](https://github.com/kevinthedang/discord-ollama) (Generalized TypeScript Discord Bot w/ Tuning Documentation)
- [ChatGPTBox: All in one browser extension](https://github.com/josStorer/chatGPTBox) with [Integrating Tutorial](https://github.com/josStorer/chatGPTBox/issues/616#issuecomment-1975186467)
- [Discord AI chat/moderation bot](https://github.com/rapmd73/Companion) Chat/moderation bot written in python. Uses Ollama to create personalities.
- [Headless Ollama](https://github.com/nischalj10/headless-ollama) (Scripts to automatically install ollama client & models on any OS for apps that depends on ollama server)
- [Terraform AWS Ollama & Open WebUI](https://github.com/xuyangbocn/terraform-aws-self-host-llm) (A Terraform module to deploy on AWS a ready-to-use Ollama service, together with its front end Open WebUI service.)
- [node-red-contrib-ollama](https://github.com/jakubburkiewicz/node-red-contrib-ollama)
- [Local AI Helper](https://github.com/ivostoykov/localAI) (Chrome and Firefox extensions that enable interactions with the active tab and customisable API endpoints. Includes secure storage for user prompts.)
- [vnc-lm](https://github.com/jake83741/vnc-lm) (Discord bot for messaging with LLMs through Ollama and LiteLLM. Seamlessly move between local and flagship models.)
- [LSP-AI](https://github.com/SilasMarvin/lsp-ai) (Open-source language server for AI-powered functionality)
- [QodeAssist](https://github.com/Palm1r/QodeAssist) (AI-powered coding assistant plugin for Qt Creator)
- [Obsidian Quiz Generator plugin](https://github.com/ECuiDev/obsidian-quiz-generator)
- [AI Summmary Helper plugin](https://github.com/philffm/ai-summary-helper)
- [TextCraft](https://github.com/suncloudsmoon/TextCraft) (Copilot in Word alternative using Ollama)
- [Alfred Ollama](https://github.com/zeitlings/alfred-ollama) (Alfred Workflow)
- [TextLLaMA](https://github.com/adarshM84/TextLLaMA) A Chrome Extension that helps you write emails, correct grammar, and translate into any language
- [Simple-Discord-AI](https://github.com/zyphixor/simple-discord-ai)
- [LLM Telegram Bot](https://github.com/innightwolfsleep/llm_telegram_bot) (telegram bot, primary for RP. Oobabooga-like buttons, [A1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui) API integration e.t.c)
- [mcp-llm](https://github.com/sammcj/mcp-llm) (MCP Server to allow LLMs to call other LLMs)

### Supported backends

- [llama.cpp](https://github.com/ggerganov/llama.cpp) project founded by Georgi Gerganov.

### Observability
- [Opik](https://www.comet.com/docs/opik/cookbook/ollama) is an open-source platform to debug, evaluate, and monitor your LLM applications, RAG systems, and agentic workflows with comprehensive tracing, automated evaluations, and production-ready dashboards. Opik supports native intergration to Ollama.
- [Lunary](https://lunary.ai/docs/integrations/ollama) is the leading open-source LLM observability platform. It provides a variety of enterprise-grade features such as real-time analytics, prompt templates management, PII masking, and comprehensive agent tracing.
- [OpenLIT](https://github.com/openlit/openlit) is an OpenTelemetry-native tool for monitoring Ollama Applications & GPUs using traces and metrics.
- [HoneyHive](https://docs.honeyhive.ai/integrations/ollama) is an AI observability and evaluation platform for AI agents. Use HoneyHive to evaluate agent performance, interrogate failures, and monitor quality in production.
- [Langfuse](https://langfuse.com/docs/integrations/ollama) is an open source LLM observability platform that enables teams to collaboratively monitor, evaluate and debug AI applications.
- [MLflow Tracing](https://mlflow.org/docs/latest/llms/tracing/index.html#automatic-tracing) is an open source LLM observability tool with a convenient API to log and visualize traces, making it easy to debug and evaluate GenAI applications.
