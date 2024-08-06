# Ollama Windows プレビュー

Ollama Windows プレビューへようこそ。

WSL はもう必要ありません！

Ollama は今やネイティブの Windows アプリケーションとして動作し、NVIDIA および AMD Radeon GPU をサポートしています。
Ollama Windows Preview をインストールした後、Ollama はバックグラウンドで実行され、
`cmd`、`powershell`、またはお気に入りのターミナルアプリケーションで `ollama` コマンドラインが利用可能になります。通常どおり、Ollamaの [API](./api.md) は `http://localhost:11434` で提供されます。

これはプレビューリリースなので、そこかしこにいくつかのバグがあることを予想してください。問題が発生した場合は、[Discord](https://discord.gg/ollama) で連絡するか、[issue](https://github.com/ollama/ollama/issues) を報告してください。
問題を診断するのにログはしばしば役立ちます（以下の [トラブルシューティング](#トラブルシューティング) を参照）。

## システム要件

* Windows 10 以降、Home または Pro
* NVIDIA カードをお持ちの場合は、NVIDIA 452.39 またはそれ以降のドライバ
* Radeon カードをお持ちの場合は、AMD Radeon ドライバ [こちら](https://www.amd.com/en/support) からダウンロード

## API アクセス

こちらは `powershell` からの API アクセスのクイックな例です。
```powershell
(Invoke-WebRequest -method POST -Body '{"model":"llama2", "prompt":"Why is the sky blue?", "stream": false}' -uri http://localhost:11434/api/generate ).Content | ConvertFrom-json
```

## トラブルシューティング

プレビュー中は、常に `OLLAMA_DEBUG` が有効になっています。これにより、アプリケーションのメニューに "view logs" メニューアイテムが追加され、GUI アプリケーションおよびサーバーのログが増えます。

Windows上 の Ollama はいくつかの異なる場所にファイルを保存します。エクスプローラーウィンドウでこれらを表示するには、`<cmd>+R` を押して次のように入力します：
- `%LOCALAPPDATA%\Ollama` には、ログとダウンロードされたアップデートが含まれます
    - *app.log* には、GUI アプリケーションのログが含まれます
    - *server.log* には、サーバーのログが含まれます
    - *upgrade.log* には、アップグレードのログ出力が含まれます
- `%LOCALAPPDATA%\Programs\Ollama` には、バイナリが含まれます（インストーラーはこれをユーザーの PATH に追加します）
- `%HOMEPATH%\.ollama` には、モデルと構成が含まれます
- `%TEMP%` には、1つ以上の `ollama*` ディレクトリに一時的な実行ファイルが含まれます