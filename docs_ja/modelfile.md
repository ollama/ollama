# Ollama Model File

> 注意: `Modelfile` の構文は開発中です

モデルファイルは、Ollamaでモデルを作成し共有するための設計図です。

## 目次

- [フォーマット](#フォーマット)
- [例](#例)
- [手順](#手順)
  - [FROM (必須)](#from-必須)
    - [llama2 からビルド](#llama2-からビルド)
    - [バイナリ ファイルからビルド](#バイナリ-ファイルからビルド)
  - [パラメータ](#パラメータ)
    - [有効なパラメータと値](#有効なパラメータと値)
  - [テンプレート](#テンプレート)
    - [テンプレート変数](#テンプレート変数)
  - [システム](#システム)
  - [アダプタ](#アダプタ)
  - [ライセンス](#ライセンス)
  - [メッセージ](#メッセージ)
- [ノート](#ノート)

## フォーマット

`Modelfile`のフォーマット:

```modelfile
# comment
指示 引数
```

| 指示                         | 説明                                                    |
| ----------------------------------- | -------------------------------------------------------------- |
| [`FROM`](#from-必須) (required) | ベースとするモデルを定義します。                                 |
| [`PARAMETER`](#パラメータ)           | Ollamaがモデルを実行する方法のパラメータを設定します。         |
| [`TEMPLATE`](#テンプレート)             | モデルに送信される完全なプロンプトテンプレート。              |
| [`SYSTEM`](#システム)                 | テンプレートに設定されるシステムメッセージを指定します。 |
| [`ADAPTER`](#アダプタ)               | モデルに適用する(Q)LoRAアダプタを定義します。            |
| [`LICENSE`](#ライセンス)               | 法的なライセンスを指定します。                                   |
| [`MESSAGE`](#メッセージ)               | メッセージの履歴を指定します。                                       |

## 例

### Basic `Modelfile`

`Modelfile` でマリオのブループリントを作成する例:

```modelfile
FROM llama2
# 温度を1に設定します [高いほど創造的、低いほど一貫性があります]
PARAMETER temperature 1
# コンテキストウィンドウサイズを4096に設定します。これは、LLMが次のトークンを生成するためのコンテキストとして使用できるトークンの数を制御します。
PARAMETER num_ctx 4096

# チャットアシスタントの挙動を指定するためのカスタムシステムメッセージを設定します。
SYSTEM You are Mario from super mario bros, acting as an assistant.
```

To use this:

1. それをファイル（例: `Modelfile`）として保存してください。
2. `ollama create choose-a-model-name -f <ファイルの場所、例: ./Modelfile>'`
3. `ollama run choose-a-model-name`
4. モデルの使用を開始してください！

より多くの例は [examplesディレクトリ](../examples) にあります。

### `Modelfile`s in [ollama.ai/library][1]

[ollama.ai/library][1] で提供されているモデルのベースとなっている `Modelfile` を見る方法は2つあります。

- オプション1：モデルのタグページから詳細ページを表示：
  1. 特定のモデルのタグページに移動します（例：https://ollama.ai/library/llama2/tags）
  2. タグをクリックします（例：https://ollama.ai/library/llama2:13b）
  3. "Layers" までスクロールします
     - 注意：[`FROM` 指示](#from-必須)が存在しない場合、
       それはモデルがローカルファイルから作成されたことを意味します。
- オプション2：`ollama show`を使用して、次のようにローカルモデルの`Modelfile`を表示します：

  ```bash
  > ollama show --modelfile llama2:13b
  # "ollama show" によって生成された Modelfile
  # 新しい Modelfile をこのものを基に作成するには、FROM行を次のように置き換えます：
  # FROM llama2:13b

  FROM /root/.ollama/models/blobs/sha256:123abc
  TEMPLATE """[INST] {{ if and .First .System }}<<SYS>>{{ .System }}<</SYS>>

  {{ end }}{{ .Prompt }} [/INST] """
  SYSTEM """"""
  PARAMETER stop [INST]
  PARAMETER stop [/INST]
  PARAMETER stop <<SYS>>
  PARAMETER stop <</SYS>>
  ```

## 手順

### FROM (必須)

`FROM` 指示は、モデルを作成する際に使用する基本モデルを定義します。

```modelfile
FROM <model name>:<tag>
```

#### llama2 からビルド

```modelfile
FROM llama2
```

利用可能なベースモデルの一覧:
<https://github.com/jmorganca/ollama#model-library>

#### `バイナリ` ファイルからビルド

```modelfile
FROM ./ollama-model.bin
```

この bin ファイルの場所は、絶対パスまたは `Modelfile` の場所からの相対パスとして指定する必要があります。

### パラメータ

`PARAMETER` 命令は、モデルの実行時に設定できるパラメータを定義します。

```modelfile
PARAMETER <parameter> <parametervalue>
```

### 有効なパラメータと値

| パラメータ      | 説明                                                                                                                                                                                                                                             | 値のタイプ | 使用例        |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- | -------------------- |
| mirostat       | Perplexity を制御するために Mirostat サンプリングを有効にします。（デフォルト：0、0 = 無効、1 = Mirostat、2 = Mirostat 2.0）                                                                                                               | int        | mirostat 0           |
| mirostat_eta   | アルゴリズムが生成されたテキストのフィードバックにどれくらい速く反応するかに影響を与えます。学習率が低いと調整が遅くなり、高い学習率はアルゴリズムをより敏感にします。（デフォルト：0.1）                       | float      | mirostat_eta 0.1     |
| mirostat_tau   | 出力の一貫性と多様性のバランスを制御します。低い値では、より焦点を当て、一貫したテキストが生成されます。（デフォルト：5.0）                                                                                                | float      | mirostat_tau 5.0     |
| num_ctx        | 次のトークンを生成するために使用されるコンテキストウィンドウのサイズを設定します。（デフォルト：2048）                                                                                                                                                                 | int        | num_ctx 4096         |
| num_gqa        | トランスフォーマーレイヤー内のGQA（Grouped Quantized Attention）グループの数。一部のモデルでは必須です。例えば、llama2:70bの場合、これは8です。                                                                                                                                        | int        | num_gqa 1            |
| num_gpu        | GPUに送信するレイヤーの数。macOSでは、Metalサポートを有効にするためにデフォルトで1に設定され、無効にする場合は0になります。                                                                                                                                | int        | num_gpu 50           |
| num_thread     | 計算中に使用するスレッドの数を設定します。デフォルトでは、Ollamaはこれを最適なパフォーマンスのために検出します。お使いのシステムが持つ物理的なCPUコアの数にこの値を設定することが推奨されています（論理的なコアの数ではなく）。 | int        | num_thread 8         |
| repeat_last_n  | モデルが繰り返しを防ぐために遡る範囲を設定します。 (デフォルト: 64、0 = 無効、-1 = num_ctx)                                                                                                                                           | int        | repeat_last_n 64     |
| repeat_penalty | 繰り返しをどれだけ厳しく罰するかを設定します。より高い値（例: 1.5）は、繰り返しをより強く罰しますが、より低い値（例: 0.9）は寛大になります。 (デフォルト: 1.1)                                                                     | float      | repeat_penalty 1.1   |
| temperature    | モデルの温度。温度を上げると、モデルの回答がより創造的になります。 (デフォルト: 0.8)                                                                                                                                     | float      | temperature 0.7      |
| seed           | 生成に使用する乱数シードを設定します。これを特定の数値に設定すると、同じプロンプトに対してモデルが同じテキストを生成します。 (デフォルト: 0)                                                                                      | int        | seed 42              |
| stop           | 停止シーケンスを設定します。このパターンが検出されると、LLMはテキスト生成を停止して返します。複数の停止パターンを設定するには、モデルファイルで複数の別々の `stop` パラメータを指定します。                                      | string     | stop "AI assistant:" |
| tfs_z          | テイルフリーサンプリング (Tail free sampling) は、出力の確率が低いトークンの影響を軽減するために使用されます。より高い値（例：2.0）は、影響をより軽減しますが、1.0の値はこの設定を無効にします。 （デフォルト：1）                                              | float      | tfs_z 1              |
| num_predict    | テキスト生成時に予測するトークンの最大数。 （デフォルト：128、-1 = 無限生成、-2 = コンテキストを埋める）                                                                                                                                 | int        | num_predict 42       |
| top_k          | ナンセンスを生成する確率を低減させます。 より高い値（例：100）はより多様な回答を提供し、より低い値（例：10）はより保守的になります。 （デフォルト：40）                                                                        | int        | top_k 40             |
| top_p          | top-kと連動します。より高い値（例：0.95）はより多様なテキストを生成し、より低い値（例：0.5）はより焦点を絞り込み、保守的なテキストを生成します。 （デフォルト：0.9）                                                                 | float      | top_p 0.9            |

### テンプレート

モデルに渡すフルプロンプトの`TEMPLATE`。システムメッセージとユーザーのプロンプトを（オプションで）含むことができます。これはフルカスタムプロンプトを作成するために使用され、構文はモデルに固有の場合があります。通常、特定のモデルのテンプレートは、そのモデルのreadmeで見つけることができます。


#### テンプレート変数

| 変数          | 説明                                                                                                   |
| ----------------- | ------------------------------------------------------------------------------------------------------------- |
| `{{ .System }}`   | カスタム動作を指定するために使用されるシステムメッセージ。 |
| `{{ .Prompt }}`   | 着信プロンプト。これはモデルファイルには指定されず、入力に基づいて設定されます。                  |
| `{{ .Response }}` | LLMからの応答。指定されていない場合、応答はテンプレートの末尾に追加されます。                 |
| `{{ .First }}`    | セッションの最初の生成に特定のテンプレート情報をレンダリングするために使用されるブール値。          |

```modelfile
TEMPLATE """
{{- if .First }}
### System:
{{ .System }}
{{- end }}

### User:
{{ .Prompt }}

### Response:
"""

SYSTEM """<system message>"""
```

### システム

`SYSTEM`命令は、テンプレートで使用されるシステムメッセージを指定します。必要に応じて。

```modelfile
SYSTEM """<system message>"""
```

### アダプタ

`ADAPTER`命令は、ベースモデルに適用するLoRAアダプタを指定します。この命令の値は、Modelfileからの相対パスまたは絶対パスである必要があり、ファイルはGGMLファイル形式である必要があります。アダプタはベースモデルから調整されている必要があります。それ以外の場合、動作は未定義です。

```modelfile
ADAPTER ./ollama-lora.bin
```

### ライセンス

`LICENSE`命令は、このModelfileで使用されるモデルが共有または配布される際の法的なライセンスを指定するためのものです。

```modelfile
LICENSE """
<license text>
"""
```

### メッセージ

`MESSAGE`命令は、モデルが応答する際に使用するメッセージの履歴を指定するためのものです:

```modelfile
MESSAGE user Is Toronto in Canada?
MESSAGE assistant yes
MESSAGE user Is Sacramento in Canada?
MESSAGE assistant no
MESSAGE user Is Ontario in Canada?
MESSAGE assistant yes
```

## ノート

- the **`Modelfile` is not case sensitive**. In the examples, uppercase instructions are used to make it easier to distinguish it from arguments.
- Instructions can be in any order. In the examples, the `FROM` instruction is first to keep it easily readable.

[1]: https://ollama.ai/library
