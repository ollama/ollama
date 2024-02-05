# 問題のトラブルシューティング方法

時々、Ollamaが期待通りに機能しないことがあります。何が起こったかを理解する最良の方法の一つは、ログを確認することです。Mac上でログを見るには、次のコマンドを実行してください:

```shell
cat ~/.ollama/logs/server.log
```

`systemd`を使用しているLinuxシステムでは、次のコマンドでログを見つけることができます:


```shell
journalctl -u ollama
```

Ollamaをコンテナで実行する場合、ログはコンテナ内のstdout/stderrに送られます:

```shell
docker logs <container-name>
```

（`docker ps` を使用してコンテナ名を見つけてください）

ターミナルで`ollama serve`を手動で実行する場合、ログはそのターミナル上に表示されます。

ログの解釈に関するヘルプは[Discord](https://discord.gg/ollama)に参加してください。

## LLM ライブラリ

Ollamaには、異なるGPUとCPUベクトル機能向けにコンパイルされた複数のLLMライブラリが含まれています。Ollamaは、システムの機能に基づいて最適なものを選択しようとします。この自動検出に問題があるか、他の問題（例：GPUのクラッシュ）に遭遇した場合は、特定のLLMライブラリを強制的に指定することで回避できます。`cpu_avx2`が最も優れており、次に`cpu_avx`、最も互換性があるが最も遅いのが`cpu`です。MacOSのRosettaエミュレーションは`cpu`ライブラリと動作します。

サーバーログには、次のようなメッセージが表示されます（リリースによって異なります）:

```
Dynamic LLM libraries [rocm_v6 cpu cpu_avx cpu_avx2 cuda_v11 rocm_v5]
```

**実験的LLMライブラリのオーバーライド**

OLLAMA_LLM_LIBRARYを利用可能なLLMライブラリのいずれかに設定すると、自動検出をバイパスできます。たとえば、CUDAカードがあるがAVX2ベクトルサポートを持つCPU LLMライブラリを強制的に使用したい場合は、次のようにします:

```
OLLAMA_LLM_LIBRARY="cpu_avx2" ollama serve
```

あなたのCPUがどの機能を持っているかは、以下の方法で確認できます。

```
cat /proc/cpuinfo| grep flags  | head -1
```

## 既知の問題

* N/A