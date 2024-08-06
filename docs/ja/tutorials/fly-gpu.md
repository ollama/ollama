# Running Ollama on Fly.io GPU Instances

Ollama は、[Fly.io GPUインスタンス](https://fly.io/docs/gpus/gpu-quickstart/)上でほとんどまたはまったく設定なしで実行できます。まだ GPU へのアクセス権がない場合は、[アクセスを申請](https://fly.io/gpu/)してウェイトリストに登録する必要があります。承認されると、開始手順が記載されたメールが届きます。

新しいアプリを作成するには、`fly apps create` を使用します:

```bash
fly apps create
```

次に、次のような新しいフォルダに `fly.toml` ファイルを作成してください。

```toml
app = "sparkling-violet-709"
primary_region = "ord"
vm.size = "a100-40gb" # see https://fly.io/docs/gpus/gpu-quickstart/ for more info

[build]
  image = "ollama/ollama"

[http_service]
  internal_port = 11434
  force_https = false
  auto_stop_machines = true
  auto_start_machines = true
  min_machines_running = 0
  processes = ["app"]

[mounts]
  source = "models"
  destination = "/root/.ollama"
  initial_size = "100gb"
```
次に、アプリケーション用に [新しいプライベートIPv6アドレス](https://fly.io/docs/reference/private-networking/#flycast-private-load-balancing) を作成してください:

```bash
fly ips allocate-v6 --private
```

その後、アプリケーションをデプロイしてください：

```bash
fly deploy
```

そして最後に、新しいFly.ioマシンを使用して対話的にアクセスできます：

```
fly machine run -e OLLAMA_HOST=http://your-app-name.flycast --shell ollama/ollama
```

```bash
$ ollama run openchat:7b-v3.5-fp16
>>> チョコレートチップクッキーの焼き方
 チョコレートチップクッキーを焼くために、以下の手順に従ってください：

1. オーブンを375°F（190°C）に予熱し、天板にはパーチメントペーパーまたはシリコン製のベーキングマットを敷きます。

2. 大きなボウルで、1カップの無塩バター（室温に戻したもの）、3/4カップの砂糖、3/4カップのパッキングされた黒糖を軽く混ぜ合わせ、ふんわりするまで混ぜます。

3. バターの混合物に大きな卵2個を1つずつ加え、各添加後によく混ぜます。1ティースプーンの純粋なバニラエクストラクトを加えて混ぜます。

4. 別のボウルで、2カップのオールパーパス小麦粉、1/2ティースプーンのベーキングソーダ、1/2ティースプーンの塩を混ぜます。ドライな成分を湿った成分に徐々に加え、ちょうど組み合わせるまでかき混ぜます。

5. 生地に2カップのチョコレートチップ（またはチャンク）を折り込みます。

6. 準備したベーキングシートに、丸くて大さじ1杯の生地を約2インチの間隔で並べます。

7. 10〜12分、または端が金褐色になるまで焼きます。中心部は少し柔らかい状態であるべきです。

8. クッキーを焼きたてのうちに、数分間ベーキングシートの上で冷ます後、ワイヤーラックに移して完全に冷やします。

自家製のチョコレートチップクッキーをお楽しみください！
```

これをこのように設定すると、使用が終わると自動的にオフになります。その後、再度アクセスすると、自動的にオンになります。これは、使用していないときに GPU インスタンスの費用を節約する素晴らしい方法です。Ollama インスタンスに対する永続的なウェイクオンユース接続が必要な場合は、[WireGuardを使用したFlyネットワークへの接続を設定](https://fly.io/docs/reference/private-networking/#discovering-apps-through-dns-on-a-wireguard-connection)できます。その後、Ollama インスタンスには `http://your-app-name.flycast` でアクセスできます。

以上で完了です！

