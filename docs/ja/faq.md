# FAQ

## Ollama をアップグレードする方法はどのようになりますか？

macOS と Windows 上の Ollama は自動的にアップデートをダウンロードします。タスクバーまたはメニューバーのアイテムをクリックして、"Restart to update" をクリックすると、アップデートが適用されます。また、最新バージョンを [手動でダウンロード](https://ollama.com/download/) してインストールすることもできます。

Linux では、インストールスクリプトを再実行します：

```
curl -fsSL https://ollama.com/install.sh | sh
```

## ログを表示する方法は？

ログの使用についての詳細については、[トラブルシューティング](./troubleshooting.md)のドキュメントを参照してください。

## コンテキストウィンドウサイズを指定する方法は？

デフォルトでは、Ollama は 2048 トークンのコンテキストウィンドウサイズを使用します。

`ollama run` を使用する場合は、`/set parameter` を使用してこれを変更します：

```
/set parameter num_ctx 4096
```

API を使用する場合は、`num_ctx` パラメータを指定します：

```
curl http://localhost:11434/api/generate -d '{
  "model": "llama2",
  "prompt": "Why is the sky blue?",
  "options": {
    "num_ctx": 4096
  }
}'
```

<div id="how-do-i-configure-ollama-server">
<h2>Ollama サーバーの設定方法は？</h2>
</div>

### Mac での環境変数の設定

Ollama が macOS アプリケーションとして実行される場合、環境変数は `launchctl` を使用して設定する必要があります：

1. 各環境変数に対して、`launchctl setenv` を呼び出します。

    ```bash
    launchctl setenv OLLAMA_HOST "0.0.0.0"
    ```

2. Ollama アプリケーションを再起動します。

### Linux での環境変数の設定

Ollama が systemd サービスとして実行される場合、環境変数は`systemctl` を使用して設定する必要があります：

1. `systemctl edit ollama.service` を呼び出して、systemd サービスを編集します。これによりエディタが開きます。

2. 各環境変数に対して、`[Service]` セクションの下に `Environment` という行を追加します：

    ```ini
    [Service]
    Environment="OLLAMA_HOST=0.0.0.0"
    ```

3. 保存して終了します。

4. `systemd` をリロードし、Ollama を再起動します：

   ```bash
   systemctl daemon-reload
   systemctl restart ollama
   ```
### Windows での環境変数の設定方法

Windows では、Ollama はユーザーとシステムの環境変数を継承します。

1. 最初に、タスクバーで Ollama をクリックして終了します。

2. コントロールパネルからシステムの環境変数を編集します。

3. ユーザーアカウントのために `OLLAMA_HOST`、 `OLLAMA_MODELS` などの変数を編集または新しく作成します。

4. 保存するために OK/Apply をクリックします。

5. 新しいターミナルウィンドウから `ollama` を実行します。

## Ollama をネットワークで公開する方法は？

Ollama はデフォルトで 127.0.0.1 ポート 11434 にバインドされます。バインドアドレスを変更するには、`OLLAMA_HOST` 環境変数を使用してください。

環境変数の設定方法については、[上記](#how-do-i-configure-ollama-server)のセクションを参照してください。

## 追加の web origins が Ollama にアクセスできるようにする方法は？

Ollama はデフォルトで `127.0.0.1` および `0.0.0.0` からのクロスオリジンリクエストを許可します。追加の origins は `OLLAMA_ORIGINS`で構成できます。

環境変数の設定方法については、[上記](#how-do-i-configure-ollama-server)のセクションを参照してください。

## モデルはどこに保存されていますか？

- macOS: `~/.ollama/models`
- Linux: `/usr/share/ollama/.ollama/models`
- Windows: `C:\Users\<username>\.ollama\models`

### それらを異なる場所に設定するにはどうすればよいですか？

別のディレクトリを使用する必要がある場合は、環境変数 `OLLAMA_MODELS` を選択したディレクトリに設定してください。

環境変数の設定方法については、[上記](#how-do-i-configure-ollama-server)のセクションを参照してください。

## Ollama は、プロンプトや回答を ollama.com に送信しますか？

いいえ。Ollama はローカルで実行され、会話データはあなたのマシンから出ません。

## Visual Studio Code で Ollama を使用する方法は？

既に VSCode や他のエディタで Ollama を活用するための多くのプラグインが利用可能です。メインリポジトリの末尾にある[拡張機能とプラグイン](https://github.com/jmorganca/ollama#extensions--plugins)のリストをご覧ください。

## プロキシを使用する方法は？

`HTTP_PROXY` または `HTTPS_PROXY` が構成されている場合、Ollama はプロキシサーバーと互換性があります。これらの変数を使用する場合は、`ollama serve` が値にアクセスできるように設定されていることを確認してください。`HTTPS_PROXY` を使用する場合は、プロキシ証明書がシステム証明書としてインストールされていることを確認してください。環境変数の設定方法については、上記のセクションを参照してください。

### Docker 内でプロキシを使用する方法は？

Ollama Docker コンテナイメージは、コンテナを起動する際に `-e HTTPS_PROXY=https://proxy.example.com` を渡すことでプロキシを使用するように構成できます。

代替として、Docker デーモン自体をプロキシを使用するように構成することもできます。Docker Desktop の[macOS](https://docs.docker.com/desktop/settings/mac/#proxies)、[Windows](https://docs.docker.com/desktop/settings/windows/#proxies)、[Linux](https://docs.docker.com/desktop/settings/linux/#proxies) に関する手順が利用可能であり、またDocker [daemon with systemd](https://docs.docker.com/config/daemon/systemd/#httphttps-proxy) についても指示があります。

HTTPS を使用する場合は、証明書がシステム証明書としてインストールされていることを確認してください。これには、自己署名証明書を使用する場合には新しい Docker イメージが必要となるかもしれません。

```dockerfile
FROM ollama/ollama
COPY my-ca.pem /usr/local/share/ca-certificates/my-ca.crt
RUN update-ca-certificates
```

このイメージをビルドして実行します：

```shell
docker build -t ollama-with-ca .
docker run -d -e HTTPS_PROXY=https://my.proxy.example.com -p 11434:11434 ollama-with-ca
```

## Docker 内で GPU アクセラレーションを使用する方法は？

Ollama Docker コンテナは、Linux または Windows（WSL2を使用する場合）で GPU アクセラレーションを構成することができます。これには [nvidia-container-toolkit](https://github.com/NVIDIA/nvidia-container-toolkit) が必要です。詳細については、[ollama/ollama](https://hub.docker.com/r/ollama/ollama) を参照してください。

GPU アクセラレーションは、macOS の Docker Desktop では GPU のパススルーやエミュレーションの不足のため利用できません。

## Windows 10 の WSL2 でネットワーキングが遅いのはなぜですか？

これは Ollama のインストールやモデルのダウンロードに影響する可能性があります。

`Control Panel > Networking and Internet > View network status and tasks` を開き、左パネルで `Change adapter settings` をクリックします。`vEthernel (WSL)` アダプターを見つけ、右クリックして `Properties` を選択します。
`Configure` をクリックし、`Advanced` タブを開きます。各プロパティを検索し、`Large Send Offload Version 2 (IPv4)` および `Large Send Offload Version 2 (IPv6)` を見つけるまで調べてください。これらのプロパティは *無効* にしてください。

## モデルを事前にロードして応答時間を短縮する方法は？

API を使用している場合、空のリクエストを Ollama サーバーに送信することで、モデルを事前にロードすることができます。これは、`/api/generate` および `/api/chat` API エンドポイントの両方で機能します。

ジェネレートエンドポイントを使用して mistral モデルを事前にロードするには、次のコマンドを使用します：
```shell
curl http://localhost:11434/api/generate -d '{"model": "mistral"}'
```

チャット補完エンドポイントを使用する場合は、次のコマンドを使用します：
```shell
curl http://localhost:11434/api/chat -d '{"model": "mistral"}'
```

## モデルをメモリにロードしたままにする方法、または即時にアンロードする方法は？

デフォルトでは、モデルはメモリに 5分間保持された後にアンロードされます。これにより、LLM に対して多数のリクエストを行う場合に応答時間を短縮できます。ただし、5分が経過する前にメモリを解放したい場合や、モデルを無期限にロードしたい場合があります。モデルがメモリに残る時間を制御するために、`/api/generate` および `/api/chat` API エンドポイントの `keep_alive` パラメーターを使用します。

`keep_alive` パラメーターには、次のような値を設定できます：
* 持続時間文字列（"10m" や "24h" など）
* 秒数での数値（3600 など）
* 負の数値は、モデルをメモリにロードしたままにします（例：-1 または "-1m"）。
* レスポンスの生成後すぐにモデルをアンロードするための '0'

例えば、モデルを事前にロードしてメモリに残す場合は、次のコマンドを使用します：
```shell
curl http://localhost:11434/api/generate -d '{"model": "llama2", "keep_alive": -1}'
```

モデルをアンロードしてメモリを解放するには、次のコマンドを使用します：
```shell
curl http://localhost:11434/api/generate -d '{"model": "llama2", "keep_alive": 0}'
```
