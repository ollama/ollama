<p align="center">
  <a href="https://ollama.com">
    <img src="https://github.com/ollama/ollama/assets/3325447/0d0b44e2-8f4a-4e99-9b52-a5c1c741c8f7" alt="ollama" width="200"/>
    <img src="https://raw.githubusercontent.com/icedmoca/ollama-vocab-tokenizer/refs/heads/main/llama.png" alt="ollama" width="200"/>
  </a>
</p>

---
> Added lightweight tokenization endpoints that let clients convert text to tokens (and back) using a model’s vocabulary without loading the full model into GPU/VRAM. This enables fast, model aligned utilities (token counting, prompt splitting, input validation) while avoiding inference runner overhead. Targeted a vocab only loading path with a small in process LRU cache; when vocab only isn’t available, the system gracefully falls back to the scheduler backed path.
>
> **`Ollama Fork`** ***+*** **`Tokenizer API`** [https://github.com/ollama/ollama/pull/12030#issuecomment-4106961583](https://github.com/ollama/ollama/pull/12030#issuecomment-4106961583)
>
---
> ### [https://deepwiki.com/icedmoca/ollama-vocab-tokenizer ](https://deepwiki.com/icedmoca/ollama-vocab-tokenizer)
>
---
> [!WARNING]  
> **Experimental API:** *The /api/tokenize and /api/detokenize endpoints are not part of upstream Ollama’s stable API.*  
> **Compatibility risk:** *They may break with future Ollama releases, since internal tokenizer APIs can change without notice.*  
> **Performance trade-offs:** *Cold-starts may still load full model runners; more vocab-only optimization is under active development.*  
> **No stability guarantees:** *This fork is intended for experimentation, benchmarking, and development.*  

---

### [Tokenize / Detokenize: vocab only endpoints with cache + fallback](https://github.com/icedmoca/ollama/commit/48806d3e196a44fd0452eb09354fd9cf1a82b921)


### Goals
- Provide model-aligned tokenization and detokenization over HTTP
- Prefer a minimal, fast, vocab-only path; avoid VRAM/context weights where possible
- Keep public API text-only (no multimodal complexity), with a clean schema
- Avoid persistent model lifetimes per request (no keep_alive)
- Make behavior observable (debug logs, durations), predictable, and easy to wire into upstream llama.cpp vocab-only support

### Public API
- POST `/api/tokenize`
  - Request: `{ "model": "<model>" , "content": "<text>", "options": { … } }`
  - Response: `{ "model": "<model>", "tokens": [<int>...], "total_duration": <ns>, "load_duration": <ns> }`

- POST `/api/detokenize`
  - Request: `{ "model": "<model>" , "tokens": [<int>...], "options": { … } }`
  - Response: `{ "model": "<model>", "content": "<text>", "total_duration": <ns>, "load_duration": <ns> }`

Notes:
- Text-only. No `media_type` in public schema.
- No `keep_alive`. Loading is optimized internally.
- Durations are included for observability.

### Key Implementation Details
- Handlers: `server/routes.go`
  - `TokenizeHandler` and `DetokenizeHandler` parse requests, validate model names, then use `tokenizerloader.Get(ctx, model)` to obtain a tokenizer.
  - Handlers log which path was used (vocab-only vs fallback) and return tokens/content with timing information.
  - Routes are registered alongside other inference endpoints.

- Loader & Cache: `server/tokenizerloader/loader.go`
  - `Tokenizer` interface: `Tokenize(string) ([]int, error)`, `Detokenize([]int) (string, error)`, `Close() error`.
  - `Get(ctx, model)` returns `(Tokenizer, isFallback, error)`.
    - Cache hit → vocab-only path
    - Cache miss → attempts `openVocabOnly(...)`
    - Unavailable → graceful fallback to scheduler-backed tokenizer
  - LRU cache (default capacity 8) with eviction and `Close()` on eviction.
  - Debug logging gated by env var: `OLLAMA_TOKENIZER_DEBUG=1`.
  - Exported sentinel: `ErrVocabOnlyUnavailable`.
  - Test hooks (exported):
    - `ResetForTest()` → clears LRU and restores opener
    - `SetOpenVocabOnlyForTest(func(context.Context, string) (Tokenizer, error))` → dependency injection for tests

- Fallback wiring (scheduler): `server/routes.go`
  - Registers small function hooks to call `scheduleRunner` and execute `r.Tokenize(...)` / `r.Detokenize(...)`.
  - Avoids import cycles by keeping fallback APIs minimal and function-based.

### Vocab-only Path (today vs future)
- Today: `openVocabOnly` defaults to returning `ErrVocabOnlyUnavailable`. This preserves behavior across all models by using the fallback path.
- Future (easy flip): Wire `openVocabOnly` to llama.cpp’s vocab-only loader (via bindings), returning a tiny vocab-only model handle. This should avoid VRAM allocation and make cold times approach warm-cache latencies. The cache will then store those tiny tokenizers.

### Debugging & Observability
- Set both for detailed logs:
  - `OLLAMA_DEBUG=1` (enables slog.Debug)
  - `OLLAMA_TOKENIZER_DEBUG=1` (emits loader-specific logs)
- Expected log lines:
  - Vocab-only cache hit: `tokenizer: vocab-only cache hit`
  - Vocab-only load: `tokenizer: vocab-only load successful`
  - Fallback: `tokenizer: vocab-only unavailable, falling back to scheduler`
  - Handler path: `tokenize: using vocab-only` or `tokenize: using fallback scheduler`

### Tests
- Loader unit tests (`server/tokenizerloader`):
  - LRU eviction without panic
  - Basic concurrency safety (adds/gets under goroutines)
  - Debug flag path executes
  - Vocab-only preferred (mock success), fallback when unavailable (mock sentinel)
  - Exported test hooks simplify dependency injection without import cycles

- Handler tests (`server`):
  - Mount a minimal gin router with just `/api/tokenize` and `/api/detokenize`
  - Test vocab-only success (overridden opener), and fallback path (opener returns `ErrVocabOnlyUnavailable`)
  - Verify tokens/content and HTTP 200 responses

### Benchmarks (local, illustrative)
- Hardware: NVIDIA GeForce RTX 2060 SUPER (8 GiB VRAM); server started from this repo
- Model: `mistral:latest`
- Results observed:
  - Cold: ≈ 0.95s (`time curl … /api/tokenize … > /dev/null`)
  - Warm (5x): ~16–25ms real per call; P50 around ~18–19ms
- Returned examples:
  - Tokenize `"hello"` → `[7080, 29477]`
  - Detokenize `[2050]` → `" fam"`

Notes:
- These warm numbers reflect a resident process and scheduler-backed path today. Once vocab-only is wired, cold times should drop and warm times should remain in the same ballpark or better.

### Developer Workflow (local)
- Build & run from repo root:
  - `go build -o bin/ollama .`
  - `OLLAMA_TOKENIZER_DEBUG=1 OLLAMA_DEBUG=1 ./bin/ollama serve`
- Smoke tests:
  - `curl -s localhost:11434/api/tokenize -H 'Content-Type: application/json' -d '{"model":"mistral:latest","content":"hello"}' | jq`
  - `curl -s localhost:11434/api/detokenize -H 'Content-Type: application/json' -d '{"model":"mistral:latest","tokens":[2050]}' | jq`
- Tests & formatting:
  - `go test ./server/tokenizerloader -race -count=1`
  - `go test ./server -run Tokenize -count=1`
  - `go test ./... -count=1`
  - `go fmt ./...` and `go vet ./...`

### Files & Notable Changes
- Endpoints & routes
  - `server/routes.go`: request/response structs, handlers, route registration, fallback hook registration
- Loader & cache
  - `server/tokenizerloader/loader.go`: `Tokenizer` interface; `Get`; LRU cache; debug logs; exported error; fallback wrapper
  - `server/tokenizerloader/testutil.go`: `ResetForTest`, `SetOpenVocabOnlyForTest`
- Tests
  - `server/tokenizerloader/loader_test.go` (LRU + concurrency)
  - `server/tokenizerloader/loader_vocabbasic_test.go` (vocab-only vs fallback)
  - `server/routes_tokenize_handler_test.go` (gin-mounted handlers, vocab-only + fallback)
- Docs & examples
  - `docs/api.md`: Tokenize / Detokenize section (text-only, no keep_alive, examples, notes)
  - `api/examples/tokenize/bench.md`: reproduction steps + micro-bench
  - `upstream-links/tokenize-python.md`, `upstream-links/tokenize-js.md`: SDK tracking crumbs

### Compatibility & Surface Area
- New endpoints; no breaking changes
- Public schema does not include `keep_alive` or `media_type`
- Internals optimized for future vocab-only; present-day correctness via fallback

### Security & Stability
- Tokenization is read-only over model vocabulary; does not execute inference
- LRU cache is guarded with mutexes; eviction closes tokenizers
- Debug logs gated by environment variables

### Known Limitations / Future Work
- `openVocabOnly` currently returns `ErrVocabOnlyUnavailable`; fallback path is used today
- Wire vocab-only opening with llama.cpp bindings once available; then the cache will hold tiny vocab-only objects, further reducing cold latencies and memory usage
- Consider TTL/refresh policies and cache size tuning based on real workloads

### PR-ready Summary
- Expose `/api/tokenize` and `/api/detokenize` (text-only) with a vocab-only-first design and LRU cache; fall back to scheduler if vocab-only is unavailable
- Remove `keep_alive` and `media_type` from public schema for these endpoints
- Add debug logging behind `OLLAMA_DEBUG` + `OLLAMA_TOKENIZER_DEBUG`
- Provide tests (LRU, concurrency, debug, handler path selection)
- Update docs with examples and micro-bench table; add SDK tracking notes

### Example cURL
```bash
curl -s http://127.0.0.1:11434/api/tokenize \
  -H 'Content-Type: application/json' \
  -d '{"model":"mistral:latest","content":"hello"}' | jq

curl -s http://127.0.0.1:11434/api/detokenize \
  -H 'Content-Type: application/json' \
  -d '{"model":"mistral:latest","tokens":[2050]}' | jq
```

### One-liner for server with logs
```bash
OLLAMA_DEBUG=1 OLLAMA_TOKENIZER_DEBUG=1 ./bin/ollama serve
```

### Bench (quick local reproduction)
```bash
time curl -s http://127.0.0.1:11434/api/tokenize \
  -H 'Content-Type: application/json' \
  -d '{"model":"mistral:latest","content":"hello"}' > /dev/null

for i in {1..5}; do
  time curl -s http://127.0.0.1:11434/api/tokenize \
    -H 'Content-Type: application/json' \
    -d '{"model":"mistral:latest","content":"hello"}' > /dev/null
done
```

### Bottom Line
- Clean, fast, model-aligned tokenization endpoints with a future-proof design: vocab-only target, LRU cache, and safe fallback. It’s production-usable today and ready for direct wiring to llama.cpp vocab-only APIs as they land upstream.


---

---

---



# Ollama Default ReadMe REF:

---

> Start building with open models.

## Download

```shell
curl -fsSL https://ollama.com/install.sh | sh
```

or [download manually](https://ollama.com/download/Ollama.dmg)


```shell
irm https://ollama.com/install.ps1 | iex
```

or [download manually](https://ollama.com/download/OllamaSetup.exe)

`ollama-vocab-r` extends [Ollama](https://github.com/ollama/ollama) with two new HTTP API endpoints:

```shell
curl -fsSL https://ollama.com/install.sh | sh
```

[Manual install instructions](https://docs.ollama.com/linux#manual-install)

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

```json
{
  "model": "mistral:latest",
  "tokens": [7080, 29477],
  "total_duration": 1234567,
  "load_duration": 456789
}
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
