# Import a model

このガイドでは、GGUF、PyTorch、またはSafetensorsモデルのインポート手順について説明します。

## インポート（GGUF）

### ステップ1：`Modelfile`を作成します

`Modelfile`を作成して始めましょう。このファイルは、モデルの設計図であり、重み、パラメータ、プロンプトテンプレートなどが指定されています。

```
FROM ./mistral-7b-v0.1.Q4_0.gguf
```

（オプション）多くのチャットモデルは、正しく回答するためにプロンプトテンプレートが必要です。`Modelfile`内の`TEMPLATE`指示でデフォルトのプロンプトテンプレートを指定できます:

```
FROM ./q4_0.bin
TEMPLATE "[INST] {{ .Prompt }} [/INST]"
```

### ステップ2：Ollamaモデルを作成します。

最後に、あなたの `Modelfile` からモデルを作成してください:

```
ollama create example -f Modelfile
```

### ステップ3：モデルを実行します。

次に、`ollama run`でモデルをテストします。

```
ollama run example "あなたのお気に入りの調味料は何ですか？"
```

## インポート（PyTorch＆Safetensors）

### サポートされているモデル

Ollamaは一連のモデルアーキテクチャをサポートしており、今後もサポートが拡充される予定です:

- Llama & Mistral
- Falcon & RW
- BigCode

モデルのアーキテクチャを確認するには、HuggingFaceリポジトリ内の`config.json`ファイルをチェックしてください。`architectures`のエントリーの下に（例：`LlamaForCausalLM`）が表示されるはずです。

### Step 1: HuggingFaceリポジトリをクローンする（オプション）

もしモデルが現在HuggingFaceリポジトリにホストされている場合、まずそのリポジトリをクローンして生のモデルをダウンロードしてください。

```
git lfs install
git clone https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1
cd Mistral-7B-Instruct-v0.1
```

### Step 2: `.bin` ファイルに変換および量子化（オプション、PyTorchおよびSafetensors用）

もしモデルがPyTorchやSafetensors形式の場合、[Dockerイメージ](https://hub.docker.com/r/ollama/quantize)が利用可能で、モデルを変換および量子化するための必要なツールが含まれています。

まず、[Docker](https://www.docker.com/get-started/)をインストールしてください。

次に、モデルを変換および量子化するために、以下を実行してください:

```
docker run --rm -v .:/model ollama/quantize -q q4_0 /model
```
これにより、ディレクトリに2つのファイルが出力されます：

- `f16.bin`: GGUFに変換されたモデル
- `q4_0.bin`: 4ビットの量子化に変換されたモデル（Ollamaはこのファイルを使用してOllamaモデルを作成します）

### Step 3: `Modelfile`の作成

次に、あなたのモデルに対する`Modelfile`を作成してください:

```
FROM ./q4_0.bin
```

（オプション）多くのチャットモデルは、正しく回答するためにはプロンプトのテンプレートが必要です。`Modelfile`内の`TEMPLATE`指示でデフォルトのプロンプトテンプレートを指定できます:

```
FROM ./q4_0.bin
TEMPLATE "[INST] {{ .Prompt }} [/INST]"
```

### Step 4: Ollamaモデルを作成します

最後に、`Modelfile` からモデルを作成します:

```
ollama create example -f Modelfile
```

### Step 5: モデルを実行する

次に、`ollama run` コマンドを使ってモデルをテストします:

```
ollama run example "What is your favourite condiment?"
```

## モデルの公開 (オプション – アーリーアルファ段階)

モデルの公開はアーリーアルファ段階にあります。他の人と共有するためにモデルを公開したい場合は、以下の手順に従ってください：

1. [アカウント](https://ollama.ai/signup)を作成してください。
2. `cat ~/.ollama/id_ed25519.pub` を実行して、Ollamaの公開鍵を表示します。これをクリップボードにコピーします。
3. あなたの公開鍵を[Ollamaのアカウント](https://ollama.ai/settings/keys)に追加します。

次に、モデルをあなたのユーザー名の名前空間にコピーしてください:

```
ollama cp example <your username>/example
```

その後、モデルをプッシュしてください:

```
ollama push <your username>/example
```

公開後、あなたのモデルは `https://ollama.ai/<あなたのユーザー名>/example` で利用可能になります。

## 量子化リファレンス

量子化オプションは以下の通りです（最高から最も低い量子化レベルまで）。注意：Falconなど一部のアーキテクチャはK quantsをサポートしていません。

- `q2_K`
- `q3_K`
- `q3_K_S`
- `q3_K_M`
- `q3_K_L`
- `q4_0` (推奨)
- `q4_1`
- `q4_K`
- `q4_K_S`
- `q4_K_M`
- `q5_0`
- `q5_1`
- `q5_K`
- `q5_K_S`
- `q5_K_M`
- `q6_K`
- `q8_0`
- `f16`

## モデルの手動変換と量子化

### 事前に必要

まず、`llama.cpp` レポジトリを別のディレクトリにマシンにクローンしてください：

```
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
```

次に、Pythonの依存関係をインストールしてください:

```
pip install -r requirements.txt
```

最後に、`quantize` ツールをビルドしてください:

```
make quantize
```

### モデルを変換する

あなたのモデルのアーキテクチャに対応した変換スクリプトを実行してください:

```shell
# LlamaForCausalLM または MistralForCausalLM
python convert.py <path to model directory>

# FalconForCausalLM
python convert-falcon-hf-to-gguf.py <path to model directory>

# GPTBigCodeForCausalLM
python convert-starcoder-hf-to-gguf.py <path to model directory>
```

### モデルを量子化する

```
quantize <path to model dir>/ggml-model-f32.bin <path to model dir>/q4_0.bin q4_0
```
