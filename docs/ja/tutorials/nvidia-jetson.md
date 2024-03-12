# Running Ollama on NVIDIA Jetson Devices

いくつかの細かい設定で、Ollama は [NVIDIA Jetsonデバイス](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/)でうまく動作します。以下は [JetPack 5.1.2](https://developer.nvidia.com/embedded/jetpack) でのテストが行われました。

NVIDIA Jetson デバイスは、AIアプリケーション向けに特別に設計されたLinuxベースの組み込み型AIコンピュータです。

Jetson にはメモリコントローラに直接接続された統合 GPU があります。このため、`nvidia-smi` コマンドは認識されず、Ollama は「CPUのみ」モードで動作します。これは、jtop などのモニタリングツールを使用して確認できます。

これを解決するために、Jetson の事前インストールされた CUDA ライブラリのパスを単純に `ollama serve` に渡します（tmuxセッション内で）。そして、ターゲットモデルのクローンに `num_gpu` パラメータをハードコードします。

事前に必要:

- curl
- tmux

以下は手順です：

- 標準の Linux コマンドを使用して Ollama をインストールします（404 エラーは無視してください）：`curl https://ollama.com/install.sh | sh`
- Ollama サービスを停止します：`sudo systemctl stop ollama`
- `tmux`セッションで Ollama serve を起動します。これを`ollama_jetson`という tmux セッションとして開始し、CUDA ライブラリのパスを参照します：`tmux has-session -t ollama_jetson 2>/dev/null || tmux new-session -d -s ollama_jetson 'LD_LIBRARY_PATH=/usr/local/cuda/lib64 ollama serve'`
- 使用したいモデル（例：mistral ）を取得します：`ollama pull mistral`
- Jetson で GPU サポートを有効にするための新しい Modelfile を作成します：`touch ModelfileMistralJetson`
- ModelfileMistralJetson ファイルで、以下に示すように FROM モデルと num_gpu PARAMETER を指定します：

```
FROM mistral
PARAMETER num_gpu 999
```

- Modelfile から新しいモデルを作成します：`ollama create mistral-jetson -f ./ModelfileMistralJetson`
- 新しいモデルを実行します：`ollama run mistral-jetson`

jtop などのモニタリングツールを実行すると、Ollama が Jetson の統合 GPU を使用していることが確認できるはずです。

以上で完了です！

