# Running Ollama on NVIDIA Jetson Devices

いくつかの細かい設定で、Ollamaは[NVIDIA Jetsonデバイス](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/)でうまく動作します。以下は[JetPack 5.1.2](https://developer.nvidia.com/embedded/jetpack)でのテストが行われました。

NVIDIA Jetsonデバイスは、AIアプリケーション向けに特別に設計されたLinuxベースの組み込み型AIコンピュータです。

Jetsonにはメモリコントローラに直接接続された統合GPUがあります。このため、`nvidia-smi`コマンドは認識されず、Ollamaは「CPUのみ」モードで動作します。これは、jtopなどのモニタリングツールを使用して確認できます。

これを解決するために、Jetsonの事前インストールされたCUDAライブラリのパスを単純に`ollama serve`に渡します（tmuxセッション内で）。そして、ターゲットモデルのクローンに`num_gpu`パラメータをハードコードします。

事前に必要:

- curl
- tmux

以下は手順です：

- 標準のLinuxコマンドを使用してOllamaをインストールします（404エラーは無視してください）：`curl https://ollama.ai/install.sh | sh`
- Ollamaサービスを停止します：`sudo systemctl stop ollama`
- `tmux`セッションでOllama serveを起動します。これを`ollama_jetson`というtmuxセッションとして開始し、CUDAライブラリのパスを参照します：`tmux has-session -t ollama_jetson 2>/dev/null || tmux new-session -d -s ollama_jetson 'LD_LIBRARY_PATH=/usr/local/cuda/lib64 ollama serve'`
- 使用したいモデル（例：mistral）を取得します：`ollama pull mistral`
- JetsonでGPUサポートを有効にするための新しいModelfileを作成します：`touch ModelfileMistralJetson`
- ModelfileMistralJetsonファイルで、以下に示すようにFROMモデルとnum_gpu PARAMETERを指定します：

```
FROM mistral
PARAMETER num_gpu 999
```

- Modelfileから新しいモデルを作成します：`ollama create mistral-jetson -f ./ModelfileMistralJetson`
- 新しいモデルを実行します：`ollama run mistral-jetson`

jtopなどのモニタリングツールを実行すると、OllamaがJetsonの統合GPUを使用していることが確認できるはずです。

以上で完了です！

