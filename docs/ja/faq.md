# FAQ

## Ollamaをアップグレードする方法はどのようになりますか？

Ollamaをアップグレードするには、インストールプロセスを再実行してください。Macでは、メニューバーにあるOllamaアイコンをクリックし、更新が利用可能な場合は再起動オプションを選択してください。

## ログを表示する方法は？

ログの使用についての詳細については、[トラブルシューティング](./troubleshooting.md)のドキュメントを参照してください。

<div id="how-do-i-configure-ollama-server">
<h2>Ollamaサーバーの設定方法は？</h2>
</div>

Ollamaサーバーは、環境変数を使用して設定できます。

### Macでの環境変数の設定

OllamaがmacOSアプリケーションとして実行される場合、環境変数は`launchctl`を使用して設定する必要があります：

1. 各環境変数に対して、`launchctl setenv`を呼び出します。

    ```bash
    launchctl setenv OLLAMA_HOST "0.0.0.0"
    ```

2. Ollamaアプリケーションを再起動します。

### Linuxでの環境変数の設定

Ollamaがsystemdサービスとして実行される場合、環境変数は`systemctl`を使用して設定する必要があります：

1. `systemctl edit ollama.service`を呼び出して、systemdサービスを編集します。これによりエディタが開きます。

2. 各環境変数に対して、`[Service]`セクションの下に`Environment`という行を追加します：

    ```ini
    [Service]
    Environment="OLLAMA_HOST=0.0.0.0"
    ```

3. 保存して終了します。

4. `systemd`をリロードし、Ollamaを再起動します：

   ```bash
   systemctl daemon-reload
   systemctl restart ollama
   ```

## Ollamaをネットワークで公開する方法は？

Ollamaはデフォルトで127.0.0.1ポート11434にバインドされます。バインドアドレスを変更するには、`OLLAMA_HOST`環境変数を使用してください。

環境変数の設定方法については、[上記](#how-do-i-configure-ollama-server)のセクションを参照してください。

## 追加のウェブ起源がOllamaにアクセスできるようにする方法は？

Ollamaはデフォルトで`127.0.0.1`および`0.0.0.0`からのクロスオリジンリクエストを許可します。追加の起源は`OLLAMA_ORIGINS`で構成できます。

環境変数の設定方法については、[上記](#how-do-i-configure-ollama-server)のセクションを参照してください。

## 追加のウェブ起源がOllamaにアクセスできるようにする方法は？

Ollamaはデフォルトで`127.0.0.1`および`0.0.0.0`からのクロスオリジンリクエストを許可します。追加の起源は`OLLAMA_ORIGINS`で構成できます。

環境変数の設定方法については、[上記](#how-do-i-configure-ollama-server)のセクションを参照してください。

## モデルはどこに保存されていますか？

- macOS: `~/.ollama/models`。
- Linux: `/usr/share/ollama/.ollama/models`。

### それらを異なる場所に設定するにはどうすればよいですか？

別のディレクトリを使用する必要がある場合は、環境変数 `OLLAMA_MODELS` を選択したディレクトリに設定してください。

環境変数の設定方法については、[上記](#how-do-i-configure-ollama-server)のセクションを参照してください。

## Ollamaはプロンプトや回答をOllama.aiに送信して何らかの方法で使用しますか？

いいえ、Ollamaは完全にローカルで実行され、会話データは決してあなたのマシンを離れません。

## Visual Studio CodeでOllamaを使用する方法は？

既にVSCodeや他のエディタでOllamaを活用するための多くのプラグインが利用可能です。メインリポジトリの末尾にある[拡張機能とプラグイン](https://github.com/jmorganca/ollama#extensions--plugins)のリストをご覧ください。

## プロキシを使用する方法は？

`HTTP_PROXY`または`HTTPS_PROXY`が構成されている場合、Ollamaはプロキシサーバーと互換性があります。これらの変数を使用する場合は、`ollama serve`が値にアクセスできるように設定されていることを確認してください。`HTTPS_PROXY`を使用する場合は、プロキシ証明書がシステム証明書としてインストールされていることを確認してください。環境変数の設定方法については、上記のセクションを参照してください。

### Docker内でプロキシを使用する方法は？

Ollama Dockerコンテナイメージは、コンテナを起動する際に `-e HTTPS_PROXY=https://proxy.example.com` を渡すことでプロキシを使用するように構成できます。

代替として、Dockerデーモン自体をプロキシを使用するように構成することもできます。Docker Desktopの[macOS](https://docs.docker.com/desktop/settings/mac/#proxies)、[Windows](https://docs.docker.com/desktop/settings/windows/#proxies)、[Linux](https://docs.docker.com/desktop/settings/linux/#proxies)に関する手順が利用可能であり、またDocker [daemon with systemd](https://docs.docker.com/config/daemon/systemd/#httphttps-proxy)についても指示があります。

HTTPSを使用する場合は、証明書がシステム証明書としてインストールされていることを確認してください。これには、自己署名証明書を使用する場合には新しいDockerイメージが必要となるかもしれません。

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

## Docker内でGPUアクセラレーションを使用する方法は？

Ollama Dockerコンテナは、LinuxまたはWindows（WSL2を使用する場合）でGPUアクセラレーションを構成することができます。これには[nvidia-container-toolkit](https://github.com/NVIDIA/nvidia-container-toolkit)が必要です。詳細については、[ollama/ollama](https://hub.docker.com/r/ollama/ollama)を参照してください。

GPUアクセラレーションは、macOSのDocker DesktopではGPUのパススルーやエミュレーションの不足のため利用できません。

## Windows 10のWSL2でネットワーキングが遅いのはなぜですか？

これはOllamaのインストールやモデルのダウンロードに影響する可能性があります。

`Control Panel > Networking and Internet > View network status and tasks` を開き、左パネルで `Change adapter settings` をクリックします。`vEthernel (WSL)` アダプターを見つけ、右クリックして `Properties` を選択します。
`Configure` をクリックし、`Advanced` タブを開きます。各プロパティを検索し、`Large Send Offload Version 2 (IPv4)` および `Large Send Offload Version 2 (IPv6)` を見つけるまで調べてください。これらのプロパティは *無効* にしてください。
