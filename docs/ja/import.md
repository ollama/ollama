# Import a model

このガイドでは、GGUF、PyTorch、または Safetensors モデルのインポート手順について説明します。

## インポート（GGUF）

### Step 1：`Modelfile` を作成します

`Modelfile` を作成して始めましょう。このファイルは、モデルの設計図であり、重み、パラメータ、プロンプトテンプレートなどが指定されています。

```
FROM ./mistral-7b-v0.1.Q4_0.gguf
```

（オプション）多くのチャットモデルは、正しく回答するためにプロンプトテンプレートが必要です。`Modelfile` 内の `TEMPLATE` 指示でデフォルトのプロンプトテンプレートを指定できます:

```
FROM ./mistral-7b-v0.1.Q4_0.gguf
TEMPLATE "[INST] {{ .Prompt }} [/INST]"
```

### Step 2：Ollama モデルを作成します。

最後に、あなたの `Modelfile` からモデルを作成してください:

```
ollama create example -f Modelfile
```

### Step 3：モデルを実行します。

次に、`ollama run` でモデルをテストします。

```
ollama run example "あなたのお気に入りの調味料は何ですか？"
```

## インポート（PyTorch＆Safetensors）

> PyTorch および Safetensors からのインポートは、GGUF からのインポートよりも時間がかかります。これをより簡単にする改善策は進行中です。

### Step 1：セットアップ

まず、`ollama/ollama` リポジトリをクローンします：

```
git clone git@github.com:ollama/ollama.git ollama
cd ollama
```

次に、`llama.cpp` サブモジュールを取得します：

```shell
git submodule init
git submodule update llm/llama.cpp
```

次に、Python の依存関係をインストールします：

```
python3 -m venv llm/llama.cpp/.venv
source llm/llama.cpp/.venv/bin/activate
pip install -r llm/llama.cpp/requirements.txt
```

その後、 `quantize` ツールをビルドします：

```
make -C llm/llama.cpp quantize
```

### Step 2：HuggingFace リポジトリのクローン（オプション）

もしモデルが現在 HuggingFace リポジトリにホストされている場合、まずそのリポジトリをクローンして生のモデルをダウンロードしてください。

[Git LFS](https://docs.github.com/ja/repositories/working-with-files/managing-large-files/installing-git-large-file-storage) をインストールし、正常にインストールされていることを確認した後、モデルのリポジトリをクローンしてください。

```
git lfs install
git clone https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1 model
```

### Step 3：モデルの変換

> 注：一部のモデルアーキテクチャでは、特定の変換スクリプトを使用する必要があります。たとえば、Qwen モデルの場合は、`convert.py`の代わりに `convert-hf-to-gguf.py` を実行する必要があります。

```
python llm/llama.cpp/convert.py ./model --outtype f16 --outfile converted.bin
```

### Step 4：モデルの量子化

```
llm/llama.cpp/quantize converted.bin quantized.bin q4_0
```

### Step 5: `Modelfile` の作成

次に、あなたのモデルに対する `Modelfile` を作成してください:

```
FROM quantized.bin
TEMPLATE "[INST] {{ .Prompt }} [/INST]"
```

### Step 6: Ollama モデルを作成します

最後に、`Modelfile` からモデルを作成します:

```
ollama create example -f Modelfile
```

### Step 7: モデルを実行する

次に、`ollama run` コマンドを使ってモデルをテストします:

```
ollama run example "What is your favourite condiment?"
```

## モデルの公開 (オプション – アーリーアルファ段階)

モデルの公開はアーリーアルファ段階にあります。他の人と共有するためにモデルを公開したい場合は、以下の手順に従ってください：

1. [アカウント](https://ollama.com/signup)を作成します。
2. Ollama の公開鍵をコピーします：
    - macOS：`cat ~/.ollama/id_ed25519.pub`
    - Windows：`type %USERPROFILE%\.ollama\id_ed25519.pub`
    - Linux：`cat /usr/share/ollama/.ollama/id_ed25519.pub`
3. あなたの公開鍵を [Ollamaアカウント](https://ollama.com/settings/keys) に追加します。

次に、モデルをあなたのユーザー名の名前空間にコピーしてください:

```
ollama cp example <your username>/example
```

その後、モデルをプッシュしてください:

```
ollama push <your username>/example
```

公開後、あなたのモデルは `https://ollama.com/<あなたのユーザー名>/example` で利用可能になります。

## 量子化リファレンス

量子化オプションは以下の通りです（最高から最も低い量子化レベルまで）。注意：Falcon など一部のアーキテクチャは K quants をサポートしていません。

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

