<div align="center">
  <img alt="ollama" height="200px" src="https://github.com/jmorganca/ollama/assets/3325447/0d0b44e2-8f4a-4e99-9b52-a5c1c741c8f7">
</div>

# Ollama

[![Discord](https://dcbadge.vercel.app/api/server/ollama?style=flat&compact=true)](https://discord.gg/ollama)

大規模な言語モデルをローカルでセットアップし、実行しましょう。

### macOS

[Download](https://ollama.com/download/Ollama-darwin.zip)

### Windows プレビュー

[Download](https://ollama.com/download/OllamaSetup.exe)

### Linux

```
curl -fsSL https://ollama.com/install.sh | sh
```

[手動インストール手順](./docs/ja/linux.md)

### Docker

公式の [Ollama Docker イメージ](https://hub.docker.com/r/ollama/ollama) である `ollama/ollama` は Docker Hub で利用可能です。

### ライブラリー

- [ollama-python](https://github.com/ollama/ollama-python)
- [ollama-js](https://github.com/ollama/ollama-js)

## クイックスタート

[Llama 2](https://ollama.com/library/llama2) を実行してチャットするには：

```
ollama run llama2
```

## モデルライブラリ

Ollama は、[ollama.com/library](https://ollama.com/library 'ollama model library')で利用可能なモデルのリストをサポートしています。

以下は、ダウンロード可能ないくつかのモデルの例です：

| モデル              | パラメーター | サイズ  | ダウンロード                    |
| ------------------ | ----------- | ------- | ------------------------------ |
| Llama 2            | 7B          | 3.8GB   | `ollama run llama2`            |
| Mistral            | 7B          | 4.1GB   | `ollama run mistral`           |
| Dolphin Phi        | 2.7B        | 1.6GB   | `ollama run dolphin-phi`       |
| Phi-2              | 2.7B        | 1.7GB   | `ollama run phi`               |
| Neural Chat        | 7B          | 4.1GB   | `ollama run neural-chat`       |
| Starling           | 7B          | 4.1GB   | `ollama run starling-lm`       |
| Code Llama         | 7B          | 3.8GB   | `ollama run codellama`         |
| Llama 2 Uncensored | 7B          | 3.8GB   | `ollama run llama2-uncensored` |
| Llama 2 13B        | 13B         | 7.3GB   | `ollama run llama2:13b`        |
| Llama 2 70B        | 70B         | 39GB    | `ollama run llama2:70b`        |
| Orca Mini          | 3B          | 1.9GB   | `ollama run orca-mini`         |
| Vicuna             | 7B          | 3.8GB   | `ollama run vicuna`            |
| LLaVA              | 7B          | 4.5GB   | `ollama run llava`             |
| Gemma              | 2B          | 1.4GB   | `ollama run gemma:2b`          |
| Gemma              | 7B          | 4.8GB   | `ollama run gemma:7b`          |

>注意: 7Bモデルを実行するには少なくとも8 GBのRAMが必要であり、13Bモデルを実行するには16 GB、33Bモデルを実行するには32 GBが必要です。

## モデルをカスタマイズする

### GGUF からインポート

Ollama は Modelfile での GGUF モデルのインポートをサポートしています。

1. `Modelfile` という名前のファイルを作成し、インポートしたいモデルのローカルファイルパスを指定する `FROM` 命令を記述します。

   ```
   FROM ./vicuna-33b.Q4_0.gguf
   ```

2. Ollama でモデルを作成します。

   ```
   ollama create example -f Modelfile
   ```

3. モデルを実行します。

   ```
   ollama run example
   ```

### PyTorch または Safetensor からのインポート

詳細については、[ガイド](docs/ja/import.md)を参照してください。

### プロンプトをカスタマイズする

Ollama ライブラリのモデルは、プロンプトでカスタマイズできます。たとえば、`llama2` モデルをカスタマイズするには、次のようにします：

```
ollama pull llama2
```

`Modelfile`を作成してください：

```
FROM llama2

# 温度を1に設定してください（高いほど創造的、低いほど論理的）。
PARAMETER temperature 1

# システムメッセージを設定してください。
SYSTEM """
あなたはスーパーマリオブラザーズのマリオです。マリオ、アシスタントとしてのみお答えください。
"""
```

次に、モデルを作成して実行してください：

```
ollama create mario -f ./Modelfile
ollama run mario
>>> こんにちは！
マリオだよ。
```

さらなる例については、[examples](examples) ディレクトリを参照してください。Modelfileの操作に関する詳細は、[Modelfile](docs/ja/modelfile.md) のドキュメントをご覧ください。

## CLI リファレンス

### モデルを作成する

`ollama create` は、Modelfile からモデルを作成するために使用されます。

```
ollama create mymodel -f ./Modelfile
```

### モデルを引っ張る


```
ollama pull llama2
```

> このコマンドは、ローカルのモデルを更新するためにも使用できます。
差分のみが取得されます。

### モデルを削除する

```
ollama rm llama2
```

### モデルをコピーする

```
ollama cp llama2 my-llama2
```

### 複数行入力

複数行の入力の場合、テキストを `"""` で囲むことができます：

```
>>> """こんにちは、
... 世界！
... """
私は基本的なプログラムで、コンソールに有名な「こんにちは、世界！」のメッセージを表示します。
```

### マルチモーダルモデル

```
>>> この画像には何がありますか？ /Users/jmorgan/Desktop/smile.png
画像には黄色い笑顔の絵文字があり、おそらく画像の中心的な焦点です。
```

### プロンプトを引数として渡します

```
$ ollama run llama2 "このファイルを要約してください：$(cat README_ja.md)"
 Ollama は、ローカルマシン上で言語モデルを構築および実行するための軽量で拡張可能なフレームワークです。モデルの作成、実行、および管理のためのシンプルな API を提供し、さらにさまざまなアプリケーションで簡単に使用できる事前に構築されたモデルのライブラリも提供しています。
```

### コンピュータ上のモデルをリストする

```
ollama list
```

### オラマを開始

`ollama serve` は、デスクトップアプリケーションを実行せずにOllama を起動したい場合に使用します。

## ビルディング

`cmake`と`go`をインストールしてください:

```
brew install cmake go
```

その後、依存関係を生成してください:
```
go generate ./...
```
その後、バイナリをビルドしてください:
```
go build .
```

より詳細な手順は[開発者ガイド](./docs/ja/development.md)に記載されています。


### ローカルビルドの実行

次に、サーバーを起動しますL

```
./ollama serve
```

最後に、別のシェルでモデルを実行します:

```
./ollama run llama2
```

## REST API

Ollama にはモデルの実行と管理のための REST API があります。

### 応答を生成する

```
curl http://localhost:11434/api/generate -d '{
  "model": "llama2",
  "prompt":"空はなぜ青いのでしょうか？"
}'
```

### モデルとチャットする

```
curl http://localhost:11434/api/chat -d '{
  "model": "mistral",
  "messages": [
    { "role": "user", "content": "空はなぜ青いのでしょうか？" }
  ]
}'
```

すべてのエンドポイントについては、[APIドキュメント](./docs/ja/api.md)を参照してください。

## コミュニティの統合

### ウェブとデスクトップ
- [Bionic GPT](https://github.com/bionic-gpt/bionic-gpt)
- [Enchanted (macOS native)](https://github.com/AugustDev/enchanted)
- [HTML UI](https://github.com/rtcfirefly/ollama-ui)
- [Chatbot UI](https://github.com/ivanfioravanti/chatbot-ollama)
- [Typescript UI](https://github.com/ollama-interface/Ollama-Gui?tab=readme-ov-file)
- [Minimalistic React UI for Ollama Models](https://github.com/richawo/minimal-llm-ui)
- [Open WebUI](https://github.com/open-webui/open-webui)
- [Ollamac](https://github.com/kevinhermawan/Ollamac)
- [big-AGI](https://github.com/enricoros/big-AGI/blob/main/docs/config-local-ollama.md)
- [Cheshire Cat assistant framework](https://github.com/cheshire-cat-ai/core)
- [Amica](https://github.com/semperai/amica)
- [chatd](https://github.com/BruceMacD/chatd)
- [Ollama-SwiftUI](https://github.com/kghandour/Ollama-SwiftUI)
- [MindMac](https://mindmac.app)
- [NextJS Web Interface for Ollama](https://github.com/jakobhoeg/nextjs-ollama-llm-ui)
- [Msty](https://msty.app)
- [Chatbox](https://github.com/Bin-Huang/Chatbox)
- [WinForm Ollama Copilot](https://github.com/tgraupmann/WinForm_Ollama_Copilot)
- [NextChat](https://github.com/ChatGPTNextWeb/ChatGPT-Next-Web) with [Get Started Doc](https://docs.nextchat.dev/models/ollama)
- [Odin Runes](https://github.com/leonid20000/OdinRunes)
- [LLM-X: Progressive Web App](https://github.com/mrdjohnson/llm-x)

### ターミナル

- [oterm](https://github.com/ggozad/oterm)
- [Ellama Emacs client](https://github.com/s-kostyaev/ellama)
- [Emacs client](https://github.com/zweifisch/ollama)
- [gen.nvim](https://github.com/David-Kunz/gen.nvim)
- [ollama.nvim](https://github.com/nomnivore/ollama.nvim)
- [ollama-chat.nvim](https://github.com/gerazov/ollama-chat.nvim)
- [ogpt.nvim](https://github.com/huynle/ogpt.nvim)
- [gptel Emacs client](https://github.com/karthink/gptel)
- [Oatmeal](https://github.com/dustinblackman/oatmeal)
- [cmdh](https://github.com/pgibler/cmdh)
- [tenere](https://github.com/pythops/tenere)
- [llm-ollama](https://github.com/taketwo/llm-ollama) for [Datasette's LLM CLI](https://llm.datasette.io/en/stable/).
- [ShellOracle](https://github.com/djcopley/ShellOracle)

### データベース

- [MindsDB](https://github.com/mindsdb/mindsdb/blob/staging/mindsdb/integrations/handlers/ollama_handler/README.md)

### パッケージマネージャー

- [Pacman](https://archlinux.org/packages/extra/x86_64/ollama/)
- [Helm Chart](https://artifacthub.io/packages/helm/ollama-helm/ollama)

### ライブラリー

- [LangChain](https://python.langchain.com/docs/integrations/llms/ollama) and [LangChain.js](https://js.langchain.com/docs/modules/model_io/models/llms/integrations/ollama) with [example](https://js.langchain.com/docs/use_cases/question_answering/local_retrieval_qa)
- [LangChainGo](https://github.com/tmc/langchaingo/) with [example](https://github.com/tmc/langchaingo/tree/main/examples/ollama-completion-example)
- [LangChain4j](https://github.com/langchain4j/langchain4j) with [example](https://github.com/langchain4j/langchain4j-examples/tree/main/ollama-examples/src/main/java)
- [LlamaIndex](https://gpt-index.readthedocs.io/en/stable/examples/llm/ollama.html)
- [LangChain4j](https://github.com/langchain4j/langchain4j/tree/main/langchain4j-ollama)
- [LiteLLM](https://github.com/BerriAI/litellm)
- [OllamaSharp for .NET](https://github.com/awaescher/OllamaSharp)
- [Ollama for Ruby](https://github.com/gbaptista/ollama-ai)
- [Ollama-rs for Rust](https://github.com/pepperoni21/ollama-rs)
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
- [Ollama-ex for Elixir](https://github.com/lebrunel/ollama-ex)
- [Ollama Connector for SAP ABAP](https://github.com/b-tocs/abap_btocs_ollama)

### 携帯

- [Enchanted](https://github.com/AugustDev/enchanted)
- [Maid](https://github.com/Mobile-Artificial-Intelligence/maid)

### 拡張機能とプラグイン

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
- [Llama Coder](https://github.com/ex3ndr/llama-coder) (Copilot alternative using Ollama)
- [Obsidian BMO Chatbot plugin](https://github.com/longy2k/obsidian-bmo-chatbot)
- [Copilot for Obsidian plugin](https://github.com/logancyang/obsidian-copilot)
- [Obsidian Local GPT plugin](https://github.com/pfrankov/obsidian-local-gpt)
- [Open Interpreter](https://docs.openinterpreter.com/language-model-setup/local-models/ollama)
- [twinny](https://github.com/rjmacarthy/twinny) (Copilot and Copilot chat alternative using Ollama)
- [Wingman-AI](https://github.com/RussellCanfield/wingman-ai) (Copilot code and chat alternative using Ollama and HuggingFace)
- [Page Assist](https://github.com/n4ze3m/page-assist) (Chrome Extension)
