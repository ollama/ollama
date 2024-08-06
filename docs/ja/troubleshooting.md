# 問題のトラブルシューティング方法

時々、Ollama が期待通りに機能しないことがあります。何が起こったのかを把握する最良の方法の 1つは、ログを確認することです。**Mac** でログを見つけるには、次のコマンドを実行します：

```shell
cat ~/.ollama/logs/server.log
```

`systemd` を使用している **Linux** システムでは、次のコマンドでログを見つけることができます:

```shell
journalctl -u ollama
```

Ollama を **コンテナ** で実行する場合、ログはコンテナ内の stdout/stderr に送られます:

```shell
docker logs <container-name>
```

（`docker ps` を使用してコンテナ名を見つけてください）

ターミナルで `ollama serve` を手動で実行する場合、ログはそのターミナル上に表示されます。

**Windows** 上で Ollama を実行する場合、いくつかの異なる場所があります。エクスプローラウィンドウでそれらを表示するには、`<cmd>+R` を押して次のコマンドを入力します：
- `explorer %LOCALAPPDATA%\Ollama`：ログを表示します
- `explorer %LOCALAPPDATA%\Programs\Ollama`：バイナリを参照します（インストーラーはこれをユーザーの PATH に追加します）
- `explorer %HOMEPATH%\.ollama`：モデルと設定が保存されている場所を表示します
- `explorer %TEMP%`：一時的な実行ファイルが 1つ以上の `ollama*` ディレクトリに保存されている場所を表示します

問題のトラブルシューティングを支援するために追加のデバッグログを有効にするには、まず **トレイメニューからアプリを終了** し、次に PowerShell ターミナルで次のコマンドを実行します：
```powershell
$env:OLLAMA_DEBUG="1"
& "ollama app.exe"
```

ログの解釈に関するヘルプは [Discord](https://discord.gg/ollama) に参加してください。

## LLM ライブラリ

Ollama には、異なる GPU と CPU ベクトル機能向けにコンパイルされた複数の LLM ライブラリが含まれています。Ollama は、システムの機能に基づいて最適なものを選択しようとします。この自動検出に問題があるか、他の問題（例：GPU のクラッシュ）に遭遇した場合は、特定の LLM ライブラリを強制的に指定することで回避できます。`cpu_avx2` が最も優れており、次に `cpu_avx`、最も互換性があるが最も遅いのが `cpu` です。MacOS の Rosetta エミュレーションは `cpu` ライブラリと動作します。

サーバーログには、次のようなメッセージが表示されます（リリースによって異なります）:

```
Dynamic LLM libraries [rocm_v6 cpu cpu_avx cpu_avx2 cuda_v11 rocm_v5]
```

**実験的 LLM ライブラリのオーバーライド**

OLLAMA_LLM_LIBRARY を利用可能な LLM ライブラリのいずれかに設定すると、自動検出をバイパスできます。たとえば、CUDA カードがあるが AVX2 ベクトルサポートを持つ CPU LLM ライブラリを強制的に使用したい場合は、次のようにします:

```
OLLAMA_LLM_LIBRARY="cpu_avx2" ollama serve
```

あなたの CPU がどの機能を持っているかは、以下の方法で確認できます。

```
cat /proc/cpuinfo| grep flags  | head -1
```

## AMD Radeon GPU サポート

Ollama は AMD ROCm ライブラリを利用しており、すべての AMD GPU をサポートしているわけではありません。一部の場合、類似した LLVM ターゲットを試すようにシステムに強制することができます。たとえば、Radeon RX 5400 は `gfx1034`（別名 10.3.4）ですが、ROCm は現在、このターゲットをサポートしていません。最も近いサポートは `gfx1030` です。環境変数 `HSA_OVERRIDE_GFX_VERSION` を `x.y.z` の構文で使用できます。たとえば、システムを RX 5400 で実行するように強制するには、サーバー用の環境変数として `HSA_OVERRIDE_GFX_VERSION="10.3.0"` を設定します。サポートされていない AMD GPU がある場合は、以下のサポートされているタイプのリストを使用して実験できます。

現時点では、以下の LLVM ターゲットが既知のサポートされている GPU タイプです。この表には、これらのLLVM ターゲットにマップされるいくつかの例の GPU が示されています：

| **LLVMターゲット** | **例のGPU** |
|-----------------|---------------------|
| gfx900 | Radeon RX Vega 56 |
| gfx906 | Radeon Instinct MI50 |
| gfx908 | Radeon Instinct MI100 |
| gfx90a | Radeon Instinct MI210 |
| gfx940 | Radeon Instinct MI300 |
| gfx941 | |
| gfx942 | |
| gfx1030 | Radeon PRO V620 |
| gfx1100 | Radeon PRO W7900 |
| gfx1101 | Radeon PRO W7700 |
| gfx1102 | Radeon RX 7600 |

AMD は、将来のリリースで ROCm v6 を拡張し、さらに多くの GPU をサポートする予定です。

追加のサポートが必要な場合は、[Discord](https://discord.gg/ollama) に連絡するか、[問題](https://github.com/ollama/ollama/issues)を報告してください。

## Linux での古いバージョンのインストール

Linux で問題が発生し、古いバージョンをインストールしたい場合は、インストールスクリプトにインストールするバージョンを指定できます。

```sh
curl -fsSL https://ollama.com/install.sh | OLLAMA_VERSION="0.1.27" sh
```