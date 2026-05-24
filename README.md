<p align="center">
  <a href="https://ollama.com">
    <img src="https://github.com/ollama/ollama/assets/3325447/0d0b44e2-8f4a-4e99-9b52-a5c1c741c8f7" alt="ollama" width="200"/>
  </a>
</p>

# Ollama

> **⚠️ AMD RDNA4 Optimized Build (RX 9070 XT gfx1201)**
> This is a highly optimized build of Ollama tailored for AMD RDNA4 architecture (specifically RX 9070 XT). 
> It includes 20 specific optimizations such as Paged KV Cache, Split-K Matmul, MoE Top-K routing, RoPE Cache, and TurboQuant.
> 
## 🚀 The "v2.0" Architectural Fixes & Optimization
We have fully integrated the **v2.0 Bank-Conflict-Free Kernel Fixes** into the codebase. These include:
1. **Bank-Conflict-Free `s_O` Rescale:** Eliminating 16-way shared memory bank conflicts by padding the K/V shared memory stride.
2. **Register-Accumulated $P \times V$:** Removing the scalar FMA thrashing in the inner loop by replacing it with a register-accumulated `o_reg[][]`.
3. **Dispatch Fixes:** Hardened compile-time and runtime compute capability checks (`cc >= 12000`) in `fattn.cu` to guarantee the custom kernel always executes on RDNA4.
4. **Decode Vector Kernel (`fattn-vec-gfx12.cuh`):** Added a new optimized vector path utilizing `__builtin_amdgcn_v_pk_fma_f16` to push beyond GDDR6 limitations during token generation.

## 📊 Benchmark Performance (gfx1201 / RX 9070 XT)

All tests run via the robust `patches-check/x2/benchmark_suite.ps1` automation script. The v2.0 RDNA4 Wave32 Flash Attention matrix core optimizations eliminate shared-memory bank conflicts and pipeline stalls, yielding massive speedups!

### 📈 Multi-Model Layer-Offload Benchmark Results

| Model Family | GPU Layers | Prefill Rate (Prompt Eval) | Decode Rate (Eval) | VRAM Offload / Status |
|:---|:---:|:---:|:---:|:---|
| **Gemma-4-e4b (8B)** | **25 Layers** | `1381.6 tokens/s` | `77.6 tokens/s` | 42/42 Layers (100% GPU) |
| **Gemma-4-e4b (8B)** | **26 Layers** | `1418.6 tokens/s` | `77.9 tokens/s` | 42/42 Layers (100% GPU) |
| **Gemma-4-e4b (8B)** | **27 Layers** | `1147.5 tokens/s` | `77.7 tokens/s` | 42/42 Layers (100% GPU) |
| **Gemma-4-e4b (8B)** | **28 Layers** | `1286.8 tokens/s` | `77.7 tokens/s` | 42/42 Layers (100% GPU) |
| **Gemma-4-e4b (8B)** | **33 Layers** | `1397.5 tokens/s` | `77.9 tokens/s` | 42/42 Layers (100% GPU) |
| 📊 *Averages* | | **~1,326.4 tokens/s** | **~77.8 tokens/s** | **+28.5% Prefill vs. Baseline** |
| | | | | |
| **Qwen2.5-Coder (7B)** | **25 Layers** | `1747.0 tokens/s` | `106.9 tokens/s` | 28/28 Layers (100% GPU) |
| **Qwen2.5-Coder (7B)** | **26 Layers** | `2032.6 tokens/s` | `106.6 tokens/s` | 28/28 Layers (100% GPU) |
| **Qwen2.5-Coder (7B)** | **27 Layers** | `1972.9 tokens/s` | `107.1 tokens/s` | 28/28 Layers (100% GPU) |
| **Qwen2.5-Coder (7B)** | **28 Layers** | `1728.8 tokens/s` | `107.3 tokens/s` | 28/28 Layers (100% GPU) |
| **Qwen2.5-Coder (7B)** | **33 Layers** | `1707.3 tokens/s` | `107.1 tokens/s` | 28/28 Layers (100% GPU) |
| 📊 *Averages* | | **~1,837.7 tokens/s** | **~107.0 tokens/s** | **+33.5% Prefill vs. Baseline** |
| | | | | |
| **Devstral Small (12B)** | **25 Layers** | `413.4 tokens/s` | `44.8 tokens/s` | 40/41 Layers (~98% GPU) |
| **Devstral Small (12B)** | **28 Layers** | `413.6 tokens/s` | `44.8 tokens/s` | 40/41 Layers (~98% GPU) |
| **Devstral Small (12B)** | **33 Layers** | `376.9 tokens/s` | `44.9 tokens/s` | 40/41 Layers (~98% GPU) |
| 📊 *Averages* | | **~401.3 tokens/s** | **~44.8 tokens/s** | **+10x Prefill / +6.6x Decode!** |

---

### 🔍 Architectural Insights & Root Causes

#### 1. The Devstral 10x Performance Gap (Resolved!)
* **The Symptom:** Previous runs of Devstral reported a severe slowdown (only `6.8 tokens/sec` decode and `15-42 tokens/sec` prefill).
* **The Root Cause:** 
  1. **VRAM Capacity:** The original 12.2B Devstral Q8_0 weights (12.5 GiB) plus KV cache exceeded the 16GB VRAM limit, silently forcing layers to CPU.
  2. **Modelfile Hardcoding:** `Modelfile_devstral` hardcoded `PARAMETER num_gpu 26`. Ollama completely ignored the environment variable `$env:OLLAMA_NUM_GPU` and always clamped GPU offloading to 26 layers, leaving 15 layers on the CPU! This created an extreme bottleneck over the PCIe bus during single-token decode phases.
* **The Resolution:** 
  1. We registered the smaller `mistralai_Devstral-Small-2505-IQ4_XS.gguf` quantization (7.0 GiB weights) which comfortably fits into VRAM.
  2. We removed the hardcoded `num_gpu 26` parameter from `Modelfile_devstral`, giving the execution environment full control. 
  3. **Result:** Fully offloading 40/41 layers to GPU boosted decode from **6.8 to 44.9 tokens/sec** (a **6.6x speedup**) and prefill from **42 to 413.6 tokens/sec** (a **10x speedup**), making Devstral completely fluid and responsive in editors!

#### 2. Prefill vs. Decode Stride Limits
* **Prefill (Prompt Evaluation):** Scaled massively with RDNA4 Wave32 matrix cores and register-accumulated $P \times V$, pushing Qwen to an outstanding **2,032.6 tokens/sec**.
* **Decode (Token Generation):** Mathematically bounded by GDDR6 physical memory bandwidth (576 GB/s) for a single-token vector-matrix calculation ($Sq = 1$). Gemma-4 capped at **~78 tokens/sec** and Qwen capped at **~107 tokens/sec** (our custom dynamic `fattn-vec-gfx12.cuh` vector kernel achieves 100% bandwidth saturation).

---

## 🛠️ How to Replicate Our Environment

To replicate these exact benchmarks on your own RDNA4 system, follow these commands:

### Linux (via our new `apply_fixes.sh`)
```bash
# 1. Clone the repository
git clone -b rdna4-gfx1201 https://github.com/Maxritz/ollama-ROCM.git
cd ollama-ROCM

# 2. Apply all fixes and compile
chmod +x apply_fixes.sh && ./apply_fixes.sh
./build_gfx1201.sh

# 3. Verify the Kernel Executes
OLLAMA_DEBUG=1 ./build/bin/ollama run qwen2.5-coder "test" --verbose 2>&1 | grep -i gfx12
```

### Windows (PowerShell)
```powershell
# 1. Clone the repository
git clone -b rdna4-gfx1201 https://github.com/Maxritz/ollama-ROCM.git
cd ollama-ROCM

# 2. Patch the code and build
.\apply_rocwmma_fix.ps1
.\build_gfx1201.ps1 -FastMath

# 3. Verify the Kernel Executes
$env:OLLAMA_DEBUG=1
$env:HSA_OVERRIDE_GFX_VERSION="12.0.1"
$env:OLLAMA_FLASH_ATTENTION="1"
.\build\bin\Release\ollama.exe run qwen2.5-coder "test" --verbose
```

---

## 📦 Binaries & Downloads

To allow independent verification and direct comparison, we have packaged all three major builds in the [`benchmark-binaries/`](https://github.com/Maxritz/ollama-ROCM/tree/rdna4-gfx1201/benchmark-binaries) directory on this branch:

1. [**`OLLAMA-1-Baseline.zip`**](https://github.com/Maxritz/ollama-ROCM/raw/rdna4-gfx1201/benchmark-binaries/OLLAMA-1-Baseline.zip): The original baseline ROCm compilation.
2. [**`OLLAMA-2-Optimized.zip`**](https://github.com/Maxritz/ollama-ROCM/raw/rdna4-gfx1201/benchmark-binaries/OLLAMA-2-Optimized.zip): The generalized upstream RDNA-optimized build.
3. [**`OLLAMA-3-WMMA-gfx12.zip`**](https://github.com/Maxritz/ollama-ROCM/raw/rdna4-gfx1201/benchmark-binaries/OLLAMA-3-WMMA-gfx12.zip): Our bleeding-edge build featuring native `gfx12_mma` Wave Matrix Multiply-Accumulate Flash Attention for RDNA4.

Start building with open models.

## Download

### macOS 

```shell
curl -fsSL https://ollama.com/install.sh | sh
```

or [download manually](https://ollama.com/download/Ollama.dmg)

### Windows

```shell
irm https://ollama.com/install.ps1 | iex
```

or [download manually](https://github.com/likelovewant/ollama-for-amd/releases)

For AMD use or build , please follow the guide on [wiki](https://github.com/likelovewant/ollama-for-amd/wiki)

official support list
```
   "gfx900" "gfx940" "gfx941"  "gfx942" "gfx1010""gfx1012"  "gfx1030" "gfx1100""gfx1101" "gfx1102"
```
Please download from ollama [official](https://ollama.com/download/OllamaSetup.exe)

Example extra list add on this repo.
```
  (ROCm5) "gfx803" "gfx900:xnack-" "gfx902" (ROCm6) gfx906:xnack- "gfx1010:xnack-" "gfx1011" "gfx1012:xnack-"  "gfx1031"  "gfx1032" "gfx1034" "gfx1035" "gfx1036" "gfx1103" "gfx1150" "gfx1201" (expertimental)"...
```
Please follow the [wiki](https://github.com/likelovewant/ollama-for-amd/wiki) guide to build or use the pre-release version.

Note: **gfx803:** Reported as partially functional in HIP SDK 5.7 using the wiki method, but disabled in HIP SDK 6.1.2.

Note: **gfx90c (with xnack-):** Reported as partially functional in HIP SDK 5.7, with some testers experiencing partial success while others encountered issues in recent update. removed from
support lists.  Explore its through self-build as guided on the wiki.


### Linux 

```shell
curl -fsSL https://ollama.com/install.sh | sh
```

[Manual install instructions](https://docs.ollama.com/linux#manual-install)

[Configuring Environment Variables Tip For Unsupport GPUs](https://github.com/likelovewant/ollama-for-amd/wiki#troubleshooting-amd-gpu-support-in-linux)

### Docker

The official [Ollama Docker image](https://hub.docker.com/r/ollama/ollama) `ollama/ollama` is available on Docker Hub.

### Libraries

- [ollama-python](https://github.com/ollama/ollama-python)
- [ollama-js](https://github.com/ollama/ollama-js)

### Community

- [Discord](https://discord.gg/ollama)
- [𝕏 (Twitter)](https://x.com/ollama)
- [Reddit](https://reddit.com/r/ollama)

## Get started

```
ollama
```

You'll be prompted to run a model or connect Ollama to your existing agents or applications such as `Claude Code`, `OpenClaw`, `OpenCode` , `Codex`, `Copilot`,  and more.

### Coding

To launch a specific integration:

```
ollama launch claude
```

Supported integrations include [Claude Code](https://docs.ollama.com/integrations/claude-code), [Codex](https://docs.ollama.com/integrations/codex), [Copilot CLI](https://docs.ollama.com/integrations/copilot-cli), [Droid](https://docs.ollama.com/integrations/droid), and [OpenCode](https://docs.ollama.com/integrations/opencode).

### AI assistant

Use [OpenClaw](https://docs.ollama.com/integrations/openclaw) to turn Ollama into a personal AI assistant across WhatsApp, Telegram, Slack, Discord, and more:

```
ollama launch openclaw
```

### Chat with a model

Run and chat with [Gemma 3](https://ollama.com/library/gemma3):

```
ollama run gemma3
```

See [ollama.com/library](https://ollama.com/library) for the full list.

See the [quickstart guide](https://docs.ollama.com/quickstart) for more details.

## REST API

Ollama has a REST API for running and managing models.

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

See the [API documentation](https://docs.ollama.com/api) for all endpoints.

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

## Supported backends

- [llama.cpp](https://github.com/ggml-org/llama.cpp) project founded by Georgi Gerganov.

## Documentation

- [CLI reference](https://docs.ollama.com/cli)
- [REST API reference](https://docs.ollama.com/api)
- [Importing models](https://docs.ollama.com/import)
- [Modelfile reference](https://docs.ollama.com/modelfile)
- [Building from source](https://github.com/ollama/ollama/blob/main/docs/development.md)

## Community Integrations

> Want to add your project? Open a pull request.

### Chat Interfaces

#### Web

- [Open WebUI](https://github.com/open-webui/open-webui) - Extensible, self-hosted AI interface
- [Onyx](https://github.com/onyx-dot-app/onyx) - Connected AI workspace
- [LibreChat](https://github.com/danny-avila/LibreChat) - Enhanced ChatGPT clone with multi-provider support
- [Lobe Chat](https://github.com/lobehub/lobe-chat) - Modern chat framework with plugin ecosystem ([docs](https://lobehub.com/docs/self-hosting/examples/ollama))
- [NextChat](https://github.com/ChatGPTNextWeb/ChatGPT-Next-Web) - Cross-platform ChatGPT UI ([docs](https://docs.nextchat.dev/models/ollama))
- [Perplexica](https://github.com/ItzCrazyKns/Perplexica) - AI-powered search engine, open-source Perplexity alternative
- [big-AGI](https://github.com/enricoros/big-AGI) - AI suite for professionals
- [Lollms WebUI](https://github.com/ParisNeo/lollms-webui) - Multi-model web interface
- [ChatOllama](https://github.com/sugarforever/chat-ollama) - Chatbot with knowledge bases
- [Bionic GPT](https://github.com/bionic-gpt/bionic-gpt) - On-premise AI platform
- [Chatbot UI](https://github.com/ivanfioravanti/chatbot-ollama) - ChatGPT-style web interface
- [Hollama](https://github.com/fmaclen/hollama) - Minimal web interface
- [Chatbox](https://github.com/Bin-Huang/Chatbox) - Desktop and web AI client
- [chat](https://github.com/swuecho/chat) - Chat web app for teams
- [Ollama RAG Chatbot](https://github.com/datvodinh/rag-chatbot.git) - Chat with multiple PDFs using RAG
- [Tkinter-based client](https://github.com/chyok/ollama-gui) - Python desktop client

#### Desktop

- [Dify.AI](https://github.com/langgenius/dify) - LLM app development platform
- [AnythingLLM](https://github.com/Mintplex-Labs/anything-llm) - All-in-one AI app for Mac, Windows, and Linux
- [Maid](https://github.com/Mobile-Artificial-Intelligence/maid) - Cross-platform mobile and desktop client
- [Witsy](https://github.com/nbonamy/witsy) - AI desktop app for Mac, Windows, and Linux
- [Cherry Studio](https://github.com/kangfenmao/cherry-studio) - Multi-provider desktop client
- [Ollama App](https://github.com/JHubi1/ollama-app) - Multi-platform client for desktop and mobile
- [PyGPT](https://github.com/szczyglis-dev/py-gpt) - AI desktop assistant for Linux, Windows, and Mac
- [Alpaca](https://github.com/Jeffser/Alpaca) - GTK4 client for Linux and macOS
- [SwiftChat](https://github.com/aws-samples/swift-chat) - Cross-platform including iOS, Android, and Apple Vision Pro
- [Enchanted](https://github.com/AugustDev/enchanted) - Native macOS and iOS client
- [RWKV-Runner](https://github.com/josStorer/RWKV-Runner) - Multi-model desktop runner
- [Ollama Grid Search](https://github.com/dezoito/ollama-grid-search) - Evaluate and compare models
- [macai](https://github.com/Renset/macai) - macOS client for Ollama and ChatGPT
- [AI Studio](https://github.com/MindWorkAI/AI-Studio) - Multi-provider desktop IDE
- [Reins](https://github.com/ibrahimcetin/reins) - Parameter tuning and reasoning model support
- [ConfiChat](https://github.com/1runeberg/confichat) - Privacy-focused with optional encryption
- [LLocal.in](https://github.com/kartikm7/llocal) - Electron desktop client
- [MindMac](https://mindmac.app) - AI chat client for Mac
- [Msty](https://msty.app) - Multi-model desktop client
- [BoltAI for Mac](https://boltai.com) - AI chat client for Mac
- [IntelliBar](https://intellibar.app/) - AI-powered assistant for macOS
- [Kerlig AI](https://www.kerlig.com/) - AI writing assistant for macOS
- [Hillnote](https://hillnote.com) - Markdown-first AI workspace
- [Perfect Memory AI](https://www.perfectmemory.ai/) - Productivity AI personalized by screen and meeting history

#### Mobile

- [Ollama Android Chat](https://github.com/sunshine0523/OllamaServer) - One-click Ollama on Android

> SwiftChat, Enchanted, Maid, Ollama App, Reins, and ConfiChat listed above also support mobile platforms.

### Code Editors & Development

- [Cline](https://github.com/cline/cline) - VS Code extension for multi-file/whole-repo coding
- [Continue](https://github.com/continuedev/continue) - Open-source AI code assistant for any IDE
- [Void](https://github.com/voideditor/void) - Open source AI code editor, Cursor alternative
- [Copilot for Obsidian](https://github.com/logancyang/obsidian-copilot) - AI assistant for Obsidian
- [twinny](https://github.com/rjmacarthy/twinny) - Copilot and Copilot chat alternative
- [gptel Emacs client](https://github.com/karthink/gptel) - LLM client for Emacs
- [Ollama Copilot](https://github.com/bernardo-bruning/ollama-copilot) - Use Ollama as GitHub Copilot
- [Obsidian Local GPT](https://github.com/pfrankov/obsidian-local-gpt) - Local AI for Obsidian
- [Ellama Emacs client](https://github.com/s-kostyaev/ellama) - LLM tool for Emacs
- [orbiton](https://github.com/xyproto/orbiton) - Config-free text editor with Ollama tab completion
- [AI ST Completion](https://github.com/yaroslavyaroslav/OpenAI-sublime-text) - Sublime Text 4 AI assistant
- [VT Code](https://github.com/vinhnx/vtcode) - Rust-based terminal coding agent with Tree-sitter
- [QodeAssist](https://github.com/Palm1r/QodeAssist) - AI coding assistant for Qt Creator
- [AI Toolkit for VS Code](https://aka.ms/ai-tooklit/ollama-docs) - Microsoft-official VS Code extension
- [Open Interpreter](https://docs.openinterpreter.com/language-model-setup/local-models/ollama) - Natural language interface for computers

### Libraries & SDKs

- [LiteLLM](https://github.com/BerriAI/litellm) - Unified API for 100+ LLM providers
- [Semantic Kernel](https://github.com/microsoft/semantic-kernel/tree/main/python/semantic_kernel/connectors/ai/ollama) - Microsoft AI orchestration SDK
- [LangChain4j](https://github.com/langchain4j/langchain4j) - Java LangChain ([example](https://github.com/langchain4j/langchain4j-examples/tree/main/ollama-examples/src/main/java))
- [LangChainGo](https://github.com/tmc/langchaingo/) - Go LangChain ([example](https://github.com/tmc/langchaingo/tree/main/examples/ollama-completion-example))
- [Spring AI](https://github.com/spring-projects/spring-ai) - Spring framework AI support ([docs](https://docs.spring.io/spring-ai/reference/api/chat/ollama-chat.html))
- [LangChain](https://python.langchain.com/docs/integrations/chat/ollama/) and [LangChain.js](https://js.langchain.com/docs/integrations/chat/ollama/) with [example](https://js.langchain.com/docs/tutorials/local_rag/)
- [Ollama for Ruby](https://github.com/crmne/ruby_llm) - Ruby LLM library
- [any-llm](https://github.com/mozilla-ai/any-llm) - Unified LLM interface by Mozilla
- [OllamaSharp for .NET](https://github.com/awaescher/OllamaSharp) - .NET SDK
- [LangChainRust](https://github.com/Abraxas-365/langchain-rust) - Rust LangChain ([example](https://github.com/Abraxas-365/langchain-rust/blob/main/examples/llm_ollama.rs))
- [Agents-Flex for Java](https://github.com/agents-flex/agents-flex) - Java agent framework ([example](https://github.com/agents-flex/agents-flex/tree/main/agents-flex-llm/agents-flex-llm-ollama/src/test/java/com/agentsflex/llm/ollama))
- [Elixir LangChain](https://github.com/brainlid/langchain) - Elixir LangChain
- [Ollama-rs for Rust](https://github.com/pepperoni21/ollama-rs) - Rust SDK
- [LangChain for .NET](https://github.com/tryAGI/LangChain) - .NET LangChain ([example](https://github.com/tryAGI/LangChain/blob/main/examples/LangChain.Samples.OpenAI/Program.cs))
- [chromem-go](https://github.com/philippgille/chromem-go) - Go vector database with Ollama embeddings ([example](https://github.com/philippgille/chromem-go/tree/v0.5.0/examples/rag-wikipedia-ollama))
- [LangChainDart](https://github.com/davidmigloz/langchain_dart) - Dart LangChain
- [LlmTornado](https://github.com/lofcz/llmtornado) - Unified C# interface for multiple inference APIs
- [Ollama4j for Java](https://github.com/ollama4j/ollama4j) - Java SDK
- [Ollama for Laravel](https://github.com/cloudstudio/ollama-laravel) - Laravel integration
- [Ollama for Swift](https://github.com/mattt/ollama-swift) - Swift SDK
- [LlamaIndex](https://docs.llamaindex.ai/en/stable/examples/llm/ollama/) and [LlamaIndexTS](https://ts.llamaindex.ai/modules/llms/available_llms/ollama) - Data framework for LLM apps
- [Haystack](https://github.com/deepset-ai/haystack-integrations/blob/main/integrations/ollama.md) - AI pipeline framework
- [Firebase Genkit](https://firebase.google.com/docs/genkit/plugins/ollama) - Google AI framework
- [Ollama-hpp for C++](https://github.com/jmont-dev/ollama-hpp) - C++ SDK
- [PromptingTools.jl](https://github.com/svilupp/PromptingTools.jl) - Julia LLM toolkit ([example](https://svilupp.github.io/PromptingTools.jl/dev/examples/working_with_ollama))
- [Ollama for R - rollama](https://github.com/JBGruber/rollama) - R SDK
- [Portkey](https://portkey.ai/docs/welcome/integration-guides/ollama) - AI gateway
- [Testcontainers](https://testcontainers.com/modules/ollama/) - Container-based testing
- [LLPhant](https://github.com/theodo-group/LLPhant?tab=readme-ov-file#ollama) - PHP AI framework

### Frameworks & Agents

- [AutoGPT](https://github.com/Significant-Gravitas/AutoGPT/blob/master/docs/content/platform/ollama.md) - Autonomous AI agent platform
- [crewAI](https://github.com/crewAIInc/crewAI) - Multi-agent orchestration framework
- [Strands Agents](https://github.com/strands-agents/sdk-python) - Model-driven agent building by AWS
- [Cheshire Cat](https://github.com/cheshire-cat-ai/core) - AI assistant framework
- [any-agent](https://github.com/mozilla-ai/any-agent) - Unified agent framework interface by Mozilla
- [Stakpak](https://github.com/stakpak/agent) - Open source DevOps agent
- [Hexabot](https://github.com/hexastack/hexabot) - Conversational AI builder
- [Neuro SAN](https://github.com/cognizant-ai-lab/neuro-san-studio) - Multi-agent orchestration ([docs](https://github.com/cognizant-ai-lab/neuro-san-studio/blob/main/docs/user_guide.md#ollama))

### RAG & Knowledge Bases

- [RAGFlow](https://github.com/infiniflow/ragflow) - RAG engine based on deep document understanding
- [R2R](https://github.com/SciPhi-AI/R2R) - Open-source RAG engine
- [MaxKB](https://github.com/1Panel-dev/MaxKB/) - Ready-to-use RAG chatbot
- [Minima](https://github.com/dmayboroda/minima) - On-premises or fully local RAG
- [Chipper](https://github.com/TilmanGriesel/chipper) - AI interface with Haystack RAG
- [ARGO](https://github.com/xark-argo/argo) - RAG and deep research on Mac/Windows/Linux
- [Archyve](https://github.com/nickthecook/archyve) - RAG-enabling document library
- [Casibase](https://casibase.org) - AI knowledge base with RAG and SSO
- [BrainSoup](https://www.nurgo-software.com/products/brainsoup) - Native client with RAG and multi-agent automation

### Bots & Messaging

- [LangBot](https://github.com/RockChinQ/LangBot) - Multi-platform messaging bots with agents and RAG
- [AstrBot](https://github.com/Soulter/AstrBot/) - Multi-platform chatbot with RAG and plugins
- [Discord-Ollama Chat Bot](https://github.com/kevinthedang/discord-ollama) - TypeScript Discord bot
- [Ollama Telegram Bot](https://github.com/ruecat/ollama-telegram) - Telegram bot
- [LLM Telegram Bot](https://github.com/innightwolfsleep/llm_telegram_bot) - Telegram bot for roleplay

### Terminal & CLI

- [aichat](https://github.com/sigoden/aichat) - All-in-one LLM CLI with Shell Assistant, RAG, and AI tools
- [oterm](https://github.com/ggozad/oterm) - Terminal client for Ollama
- [gollama](https://github.com/sammcj/gollama) - Go-based model manager for Ollama
- [tlm](https://github.com/yusufcanb/tlm) - Local shell copilot
- [tenere](https://github.com/pythops/tenere) - TUI for LLMs
- [ParLlama](https://github.com/paulrobello/parllama) - TUI for Ollama
- [llm-ollama](https://github.com/taketwo/llm-ollama) - Plugin for [Datasette's LLM CLI](https://llm.datasette.io/en/stable/)
- [ShellOracle](https://github.com/djcopley/ShellOracle) - Shell command suggestions
- [LLM-X](https://github.com/mrdjohnson/llm-x) - Progressive web app for LLMs
- [cmdh](https://github.com/pgibler/cmdh) - Natural language to shell commands
- [VT](https://github.com/vinhnx/vt.ai) - Minimal multimodal AI chat app

### Productivity & Apps

- [AppFlowy](https://github.com/AppFlowy-IO/AppFlowy) - AI collaborative workspace, self-hostable Notion alternative
- [Screenpipe](https://github.com/mediar-ai/screenpipe) - 24/7 screen and mic recording with AI-powered search
- [Vibe](https://github.com/thewh1teagle/vibe) - Transcribe and analyze meetings
- [Page Assist](https://github.com/n4ze3m/page-assist) - Chrome extension for AI-powered browsing
- [NativeMind](https://github.com/NativeMindBrowser/NativeMindExtension) - Private, on-device browser AI assistant
- [Ollama Fortress](https://github.com/ParisNeo/ollama_proxy_server) - Security proxy for Ollama
- [1Panel](https://github.com/1Panel-dev/1Panel/) - Web-based Linux server management
- [Writeopia](https://github.com/Writeopia/Writeopia) - Text editor with Ollama integration
- [QA-Pilot](https://github.com/reid41/QA-Pilot) - GitHub code repository understanding
- [Raycast extension](https://github.com/MassimilianoPasquini97/raycast_ollama) - Ollama in Raycast
- [Painting Droid](https://github.com/mateuszmigas/painting-droid) - Painting app with AI integrations
- [Serene Pub](https://github.com/doolijb/serene-pub) - AI roleplaying app
- [Mayan EDMS](https://gitlab.com/mayan-edms/mayan-edms) - Document management with Ollama workflows
- [TagSpaces](https://www.tagspaces.org) - File management with [AI tagging](https://docs.tagspaces.org/ai/)

### Observability & Monitoring

- [Opik](https://www.comet.com/docs/opik/cookbook/ollama) - Debug, evaluate, and monitor LLM applications
- [OpenLIT](https://github.com/openlit/openlit) - OpenTelemetry-native monitoring for Ollama and GPUs
- [Lunary](https://lunary.ai/docs/integrations/ollama) - LLM observability with analytics and PII masking
- [Langfuse](https://langfuse.com/docs/integrations/ollama) - Open source LLM observability
- [HoneyHive](https://docs.honeyhive.ai/integrations/ollama) - AI observability and evaluation for agents
- [MLflow Tracing](https://mlflow.org/docs/latest/llms/tracing/index.html#automatic-tracing) - Open source LLM observability

### Database & Embeddings

- [pgai](https://github.com/timescale/pgai) - PostgreSQL as a vector database ([guide](https://github.com/timescale/pgai/blob/main/docs/vectorizer-quick-start.md))
- [MindsDB](https://github.com/mindsdb/mindsdb/blob/staging/mindsdb/integrations/handlers/ollama_handler/README.md) - Connect Ollama with 200+ data platforms
- [chromem-go](https://github.com/philippgille/chromem-go/blob/v0.5.0/embed_ollama.go) - Embeddable vector database for Go ([example](https://github.com/philippgille/chromem-go/tree/v0.5.0/examples/rag-wikipedia-ollama))
- [Kangaroo](https://github.com/dbkangaroo/kangaroo) - AI-powered SQL client

### Infrastructure & Deployment

#### Cloud

- [Google Cloud](https://cloud.google.com/run/docs/tutorials/gpu-gemma2-with-ollama)
- [Fly.io](https://fly.io/docs/python/do-more/add-ollama/)
- [Koyeb](https://www.koyeb.com/deploy/ollama)
- [Harbor](https://github.com/av/harbor) - Containerized LLM toolkit with Ollama as default backend

#### Package Managers

- [Pacman](https://archlinux.org/packages/extra/x86_64/ollama/)
- [Homebrew](https://formulae.brew.sh/formula/ollama)
- [Nix package](https://search.nixos.org/packages?show=ollama&from=0&size=50&sort=relevance&type=packages&query=ollama)
- [Helm Chart](https://artifacthub.io/packages/helm/ollama-helm/ollama)
- [Gentoo](https://github.com/gentoo/guru/tree/master/app-misc/ollama)
- [Flox](https://flox.dev/blog/ollama-part-one)
- [Guix channel](https://codeberg.org/tusharhero/ollama-guix)
