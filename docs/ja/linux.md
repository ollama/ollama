# Linux上のOllama

## インストール

Ollama をインストールするには、次のワンライナーを実行してください:

>

```bash
curl -fsSL https://ollama.com/install.sh | sh
```
## AMD Radeon GPU サポート

AMD は `amdgpu` ドライバーを公式の Linux カーネルソースにアップストリームで提供していますが、そのバージョンは古く、すべての ROCm 機能をサポートしていない可能性があります。Radeon GPU を最良にサポートするために、最新のドライバーを以下からインストールすることをお勧めします：[https://www.amd.com/en/support/linux-drivers](https://www.amd.com/en/support/linux-drivers)。

## 手動インストール

### `ollama` バイナリをダウンロードしてください

Ollama は自己完結型のバイナリとして配布されています。以下の手順で、ダウンロードして PATH に含まれるディレクトリに保存してください。

```bash
sudo curl -L https://ollama.com/download/ollama-linux-amd64 -o /usr/bin/ollama
sudo chmod +x /usr/bin/ollama
```

### Ollama を起動時サービスに追加する（推奨）

Ollama 用のユーザーを作成してください:

```bash
sudo useradd -r -s /bin/false -m -d /usr/share/ollama ollama
```
`/etc/systemd/system/ollama.service` にサービスファイルを作成してください:

```ini
[Unit]
Description=Ollama Service
After=network-online.target

[Service]
ExecStart=/usr/bin/ollama serve
User=ollama
Group=ollama
Restart=always
RestartSec=3

[Install]
WantedBy=default.target
```

次に、サービスを起動してください:

```bash
sudo systemctl daemon-reload
sudo systemctl enable ollama
```

### CUDA ドライバのインストール（オプション - Nvidia GPU 用）

[CUDA をダウンロードしてインストール](https://developer.nvidia.com/cuda-downloads) してください。

ドライバーがインストールされているか確認するために、以下のコマンドを実行してください。これにより、GPU に関する詳細が表示されるはずです:

```bash
nvidia-smi
```

### ROCm のインストール（オプション - Radeon GPU 用）

[ダウンロードしてインストール](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/tutorial/quick-start.html)

ROCm v6 をインストールしてください。

### Ollama を起動

`systemd` を使用して Ollama を起動します。

```bash
sudo systemctl start ollama
```

## アップデート

再びインストールスクリプトを実行して、Ollama をアップデートします:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

または、Ollama のバイナリをダウンロードすることもできます:

```bash
sudo curl -L https://ollama.com/download/ollama-linux-amd64 -o /usr/bin/ollama
sudo chmod +x /usr/bin/ollama
```

## ログの表示

Ollama が起動サービスとして実行されている際のログを表示するには、次のコマンドを実行してください:

```bash
journalctl -u ollama
```

## アンインストール

Ollama サービスを削除するには:

```bash
sudo systemctl stop ollama
sudo systemctl disable ollama
sudo rm /etc/systemd/system/ollama.service
```

Ollama バイナリを bin ディレクトリから削除してください（`/usr/local/bin`、`/usr/bin`、または `/bin` のいずれか）:

```bash
sudo rm $(which ollama)
```

ダウンロードしたモデルと Ollama サービスのユーザーおよびグループを削除してください：

```bash
sudo rm -r /usr/share/ollama
sudo userdel ollama
sudo groupdel ollama
```
