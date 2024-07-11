<div align="center">
 <img alt="ollama" height="200px" src="https://github.com/ollama/ollama/assets/3325447/0d0b44e2-8f4a-4e99-9b52-a5c1c741c8f7">
</div>

# Ollama

[![Discord](https://dcbadge.vercel.app/api/server/ollama?style=flat&compact=true)](https://discord.gg/ollama)

Comece e avance com modelos de linguagem avançados.

### macOS

[Download](https://ollama.com/download/Ollama-darwin.zip)

### Windows preview

[Download](https://ollama.com/download/OllamaSetup.exe)

### Linux

```
curl -fsSL https://ollama.com/install.sh | sh
```

[Instruções de instalação manual](https://github.com/ollama/ollama/blob/main/docs/linux.md)

### Docker

A Imagem oficial [Ollama Docker image](https://hub.docker.com/r/ollama/ollama) `ollama/ollama` está disponível no Docker Hub.

### Bibliotecas

- [ollama-python](https://github.com/ollama/ollama-python)
- [ollama-js](https://github.com/ollama/ollama-js)

## Início rápido

Para executar e conversar [Llama 3](https://ollama.com/library/llama3):

```
ollama run llama3
```

## Biblioteca de modelos

Ollama suporta uma lista de modelos disponíveis em [ollama.com/library](https://ollama.com/library 'ollama model library')

Aqui estão alguns exemplos de modelos que podem ser baixados:

| Modelo              | Parâmetros | Tamanho  | Download                       |
| ------------------ | ---------- | ----- | ------------------------------ |
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

> Nota: Você deve ter pelo menos 8 GB de RAM disponível para executar os modelos de 7B, 16 GB para os modelos de 13B e 32 GB para os modelos de 33B.

## Customize um modelo

### Importe do GGUF

Ollama suporta importar modelos GGUF no arquivo de modelo:

1. Crie um arquivo com o nome de `Modelfile`, com uma `FROM` instrução com o caminho local do arquivo do modelo que você deseja importar.

   ```
   FROM ./vicuna-33b.Q4_0.gguf
   ```

2. Crie o modelo no Ollama

   ```
   ollama create example -f Modelfile
   ```

3. Execute o modelo

   ```
   ollama run example
   ```

### Importe do PyTorch ou do Safetensors

Veja o [guia](docs/import.md) sobre a importação de modelos para mais informações.

### Personalize um prompt.

Os modelos da biblioteca Ollama podem ser personalizados com um prompt. Por exemplo, para personalizar o modelo llama3:

```
ollama pull llama3
```

Crie um `Modelfile`:

```
FROM llama3

# set the temperature to 1 [higher is more creative, lower is more coherent]
PARAMETER temperature 1

# set the system message
SYSTEM """
You are Mario from Super Mario Bros. Answer as Mario, the assistant, only.
"""
```

Proximo, crie e execute o modelo:

```
ollama create mario -f ./Modelfile
ollama run mario
>>> hi
Hello! It's your friend Mario.
```

Para mais exemplos, veja o diretorio de [exemplos](examples). Para mais informações para trabalhar com um arquivo Modelfile, veja a documentação do [Modelfile](docs/modelfile.md).

## Referência de linha de comando

### Crie um modelo

`ollama create` é usado para criar um modelo a partir de um arquivo de modelo.

```
ollama create mymodel -f ./Modelfile
```

### Fazer o pull de um modelo

```
ollama pull llama3
```

> Este comando também pode ser usado para atualizar um modelo local. Apenas a diferença será puxada..

### Remover um modelo

```
ollama rm llama3
```

### Copie um modelo

```
ollama cp llama3 my-model
```

### Entrada multilinha

Para entrada multilinha, você pode envolver o texto com `"""`:

```
>>> """Hello,
... world!
... """
I'm a basic program that prints the famous "Hello, world!" message to the console.
```

### Modelos multimodais

```
>>> What's in this image? /Users/jmorgan/Desktop/smile.png
The image features a yellow smiley face, which is likely the central focus of the picture.
```

### Passe o prompt como um argumento.

```
$ ollama run llama3 "Summarize this file: $(cat README.md)"
 Ollama is a lightweight, extensible framework for building and running language models on the local machine. It provides a simple API for creating, running, and managing models, as well as a library of pre-built models that can be easily used in a variety of applications.
```

### Mostre a informação do modelo

```
ollama show llama3
```

### Liste os modelos no seu computador

```
ollama list
```

### Inicie o Ollama

`ollama serve` é usado quando você deseja iniciar o Ollama sem executar o aplicativo de desktop.

## Building

Veja o [guia de desenvolvimento](https://github.com/ollama/ollama/blob/main/docs/development.md)

### Rodando builds locais

Em seguida, inicie o servidor.:

```
./ollama serve
```

Finalmente, em um shell separado, execute um modelo.:

```
./ollama run llama3
```

## REST API

Ollama possui uma API REST para executar e gerenciar modelos..

### Generate a response

```
curl http://localhost:11434/api/generate -d '{
  "model": "llama3",
  "prompt":"Why is the sky blue?"
}'
```

### Chat with a model

```
curl http://localhost:11434/api/chat -d '{
  "model": "llama3",
  "messages": [
    { "role": "user", "content": "why is the sky blue?" }
  ]
}'
```

See the [API documentation](./docs/api.md) for all endpoints.

## Integrações da comunidade

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
- [LLM-X](https://github.com/mrdjohnson/llm-x) (Aplicativo Web Progressivo)
- [AnythingLLM (Docker + MacOs/Windows/Linux native app)](https://github.com/Mintplex-Labs/anything-llm)
- [Ollama Basic Chat: Uses HyperDiv Reactive UI](https://github.com/rapidarchitect/ollama_basic_chat)
- [Ollama-chats RPG](https://github.com/drazdra/ollama-chats)
- [QA-Pilot](https://github.com/reid41/QA-Pilot) (Chat com repositório de código)
- [ChatOllama](https://github.com/sugarforever/chat-ollama) (Chatbot de código aberto baseado no Ollama com bases de conhecimento)
- [CRAG Ollama Chat](https://github.com/Nagi-ovo/CRAG-Ollama-Chat) (Busca web simples com RAG corretivo)
- [RAGFlow](https://github.com/infiniflow/ragflow) (Motor de Geração com Recuperação Aumentada de código aberto baseado em compreensão profunda de documentos)
- [StreamDeploy](https://github.com/StreamDeploy-DevRel/streamdeploy-llm-app-scaffold) (Estrutura de Aplicação para LLM)
- [chat](https://github.com/swuecho/chat) (Aplicativo web de chat para equipes)
- [Lobe Chat](https://github.com/lobehub/lobe-chat) com [Integrating Doc](https://lobehub.com/docs/self-hosting/examples/ollama)
- [Ollama RAG Chatbot](https://github.com/datvodinh/rag-chatbot.git) (Chat local com múltiplos PDFs usando Ollama e RAG)
- [BrainSoup](https://www.nurgo-software.com/products/brainsoup) (Cliente nativo flexível com automação RAG e multi-agente)
- [macai](https://github.com/Renset/macai) (Cliente para macOS compatível com Ollama, ChatGPT e outros back-ends de API compatíveis.)
- [Olpaka](https://github.com/Otacon/olpaka) (Aplicativo web Flutter amigável para o Ollama)
- [OllamaSpring](https://github.com/CrazyNeil/OllamaSpring) (Cliente Ollama para macOS)
- [LLocal.in](https://github.com/kartikm7/llocal) (Cliente de desktop Electron fácil de usar para o Ollama)

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
- [llm-ollama](https://github.com/taketwo/llm-ollama) para [Datasette's LLM CLI](https://llm.datasette.io/en/stable/).
- [typechat-cli](https://github.com/anaisbetts/typechat-cli)
- [ShellOracle](https://github.com/djcopley/ShellOracle)
- [tlm](https://github.com/yusufcanb/tlm)
- [podman-ollama](https://github.com/ericcurtin/podman-ollama)
- [gollama](https://github.com/sammcj/gollama)

### Banco de Dados

- [MindsDB](https://github.com/mindsdb/mindsdb/blob/staging/mindsdb/integrations/handlers/ollama_handler/README.md) (Conecta modelos do Ollama com quase 200 plataformas de dados e aplicativos.)
- [chromem-go](https://github.com/philippgille/chromem-go/blob/v0.5.0/embed_ollama.go) com [exemplo](https://github.com/philippgille/chromem-go/tree/v0.5.0/examples/rag-wikipedia-ollama)

### Gerenciadores de pacotes

- [Pacman](https://archlinux.org/packages/extra/x86_64/ollama/)
- [Helm Chart](https://artifacthub.io/packages/helm/ollama-helm/ollama)
- [Guix channel](https://codeberg.org/tusharhero/ollama-guix)

### Bibliotecas

- [LangChain](https://python.langchain.com/docs/integrations/llms/ollama) e [LangChain.js](https://js.langchain.com/docs/modules/model_io/models/llms/integrations/ollama) com [exemplo](https://js.langchain.com/docs/use_cases/question_answering/local_retrieval_qa)
- [LangChainGo](https://github.com/tmc/langchaingo/) com [exemplo](https://github.com/tmc/langchaingo/tree/main/examples/ollama-completion-example)
- [LangChain4j](https://github.com/langchain4j/langchain4j) com [exemplo](https://github.com/langchain4j/langchain4j-examples/tree/main/ollama-examples/src/main/java)
- [LangChainRust](https://github.com/Abraxas-365/langchain-rust) com [exemplo](https://github.com/Abraxas-365/langchain-rust/blob/main/examples/llm_ollama.rs)
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
- [PromptingTools.jl](https://github.com/svilupp/PromptingTools.jl) com um [exemplo](https://svilupp.github.io/PromptingTools.jl/dev/examples/working_with_ollama)
- [LlamaScript](https://github.com/Project-Llama/llamascript)

### Mobile

- [Enchanted](https://github.com/AugustDev/enchanted)
- [Maid](https://github.com/Mobile-Artificial-Intelligence/maid)

### Extensões e plugins

- [Raycast extension](https://github.com/MassimilianoPasquini97/raycast_ollama)
- [Discollama](https://github.com/mxyng/discollama) (Bot do Discord dentro do canal do Ollama no Discord)
- [Continue](https://github.com/continuedev/continue)
- [Obsidian Ollama plugin](https://github.com/hinterdupfinger/obsidian-ollama)
- [Logseq Ollama plugin](https://github.com/omagdy7/ollama-logseq)
- [NotesOllama](https://github.com/andersrex/notesollama) (Plugin do Ollama para o Apple Notes)
- [Dagger Chatbot](https://github.com/samalba/dagger-chatbot)
- [Discord AI Bot](https://github.com/mekb-turtle/discord-ai-bot)
- [Ollama Telegram Bot](https://github.com/ruecat/ollama-telegram)
- [Hass Ollama Conversation](https://github.com/ej52/hass-ollama-conversation)
- [Rivet plugin](https://github.com/abrenneke/rivet-plugin-ollama)
- [Obsidian BMO Chatbot plugin](https://github.com/longy2k/obsidian-bmo-chatbot)
- [Cliobot](https://github.com/herval/cliobot) (Bot do Telegram com suporte ao Ollama)
- [Copilot for Obsidian plugin](https://github.com/logancyang/obsidian-copilot)
- [Obsidian Local GPT plugin](https://github.com/pfrankov/obsidian-local-gpt)
- [Open Interpreter](https://docs.openinterpreter.com/language-model-setup/local-models/ollama)
- [Llama Coder](https://github.com/ex3ndr/llama-coder) (Alternativa ao Copilot usando Ollama)
- [Ollama Copilot](https://github.com/bernardo-bruning/ollama-copilot) (Proxy que permite usar Ollama como um copilot, similar ao Github Copilot)
- [twinny](https://github.com/rjmacarthy/twinny) (Alternativa ao Copilot e Copilot Chat usando Ollama)
- [Wingman-AI](https://github.com/RussellCanfield/wingman-ai) (Alternativa ao Copilot: Código e Chat usando Ollama e HuggingFace)
- [Page Assist](https://github.com/n4ze3m/page-assist) (Extensão do Chrome)
- [AI Telegram Bot](https://github.com/tusharhero/aitelegrambot) (Bot do Telegram usando Ollama no backend)
- [AI ST Completion](https://github.com/yaroslavyaroslav/OpenAI-sublime-text) (Plugin do assistente de inteligência artificial para o Sublime Text 4 com suporte ao Ollama)
- [Discord-Ollama Chat Bot](https://github.com/kevinthedang/discord-ollama) (Bot Discord generalizado em TypeScript com documentação de ajustes)
- [Discord AI chat/moderation bot](https://github.com/rapmd73/Companion) Bot de chat/moderação escrito em Python. Usa o Ollama para criar personalidades.
- [Headless Ollama](https://github.com/nischalj10/headless-ollama) (Scripts para instalar automaticamente o cliente e os modelos do Ollama em qualquer sistema operacional para aplicativos que dependem do servidor Ollama)

### Supported backends

- [llama.cpp](https://github.com/ggerganov/llama.cpp) projeto fundado por Georgi Gerganov.
