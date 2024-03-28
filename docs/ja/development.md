# 開発

必要なツールをインストールしてください：

- cmake バージョン 3.24 以上
- go バージョン 1.22 以上
- gcc バージョン 11.4.0 以上

```bash
brew install go cmake gcc
```

オプションでデバッグおよび詳細なログを有効にする:

```bash
# ビルド時
export CGO_CFLAGS="-g"

# 実行時
export OLLAMA_DEBUG=1
```

必要なライブラリを取得し、ネイティブ LLM コードをビルドしてください:

```bash
go generate ./...
```

次に、Ollama をビルドしてください:

```bash
go build .
```

これで、`ollama` を実行できます:

```bash
./ollama
```

### Linux

#### Linux CUDA (NVIDIA)

_お使いのオペレーティングシステムディストリビューションには、既に NVIDIA CUDA 用のパッケージが含まれているかもしれません。ディストリビューションパッケージは通常好ましいですが、手順はディストリビューションに依存します。可能であれば、ディストリビューション固有のドキュメントを確認して、依存関係に関する情報を参照してください。_

`cmake` および `golang` をインストールし、[NVIDIA CUDA](https://developer.nvidia.com/cuda-downloads) の開発およびランタイムパッケージもインストールしてください。

通常、ビルドスクリプトは CUDA を自動検出しますが、Linux ディストリビューションやインストールアプローチが異常なパスを使用する場合は、環境変数 `CUDA_LIB_DIR` を共有ライブラリの場所に、`CUDACXX` を nvcc コンパイラの場所に指定することができます。また、`CMAKE_CUDA_ARCHITECTURES` を設定して、対象の CUDA アーキテクチャをカスタマイズできます（例："50;60;70"）。

その後、依存関係を生成してください:

```
go generate ./...
```

その後、バイナリをビルドしてください:

```
go build .
```

#### Linux ROCm (AMD)

_お使いのオペレーティングシステムディストリビューションには、すでに AMD ROCm および CLBlast 用のパッケージが含まれているかもしれません。ディストリビューションパッケージは通常好ましいですが、手順はディストリビューションに依存します。可能であれば、ディストリビューション固有のドキュメントを確認して、依存関係に関する情報を参照してください。_

まず、[CLBlast](https://github.com/CNugteren/CLBlast/blob/master/doc/installation.md)と[ROCm](https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html)の開発パッケージ、および `cmake` と `golang` をインストールしてください。

通常、ビルドスクリプトは ROCm を自動検出しますが、Linux ディストリビューションやインストールアプローチが異常なパスを使用する場合は、環境変数 `ROCM_PATH` を ROCm のインストール場所に（通常は `/opt/rocm`）、`CLBlast_DIR` を CLBlast のインストール場所に（通常は `/usr/lib/cmake/CLBlast`）指定することができます。また、`AMDGPU_TARGETS` を設定して AMD GPU の対象をカスタマイズすることもできます（例：`AMDGPU_TARGETS="gfx1101;gfx1102"`）。


```
go generate ./...
```

その後、バイナリをビルドしてください:

```
go build .
```

ROCm は実行時に GPU にアクセスするために特権が必要です。ほとんどのディストリビューションでは、ユーザーアカウントを `render` グループに追加するか、root として実行することができます。

#### 高度なCPU設定

デフォルトでは、`go generate ./...` を実行すると、一般的な CPU ファミリとベクトル数学の機能に基づいて、いくつかの異なる LLM ライブラリのバリエーションがコンパイルされます。これには、ほとんどの 64 ビット CPU で動作する最も一般的なバージョンも含まれますが、やや遅くなります。実行時に、Ollama は最適なバリエーションを自動検出してロードします。プロセッサにカスタマイズされた CPU ベースのビルドを作成したい場合は、`OLLAMA_CUSTOM_CPU_DEFS` を使用する llama.cpp フラグに設定できます。例えば、Intel i9-9880H 向けに最適化されたバイナリをコンパイルする場合は、次のようにします：

```
OLLAMA_CUSTOM_CPU_DEFS="-DLLAMA_AVX=on -DLLAMA_AVX2=on -DLLAMA_F16C=on -DLLAMA_FMA=on" go generate ./...
go build .
```

#### コンテナ化された Linux ビルド

Docker が利用可能な場合、CUDA および ROCm の依存関係が含まれている `./scripts/build_linux.sh` を使用して Linux 用のバイナリをビルドできます。生成されたバイナリは `./dist` に配置されます。

### Windows

注意：Ollama の Windows ビルドはまだ開発中です。

必要なツールをインストールしてください：

- MSVC ツールチェーン - C/C++ および cmake を最小の要件として - 環境変数を設定した "Developer Shell" からビルドする必要があります
- go のバージョン 1.22 以上
- GCC を搭載した MinGW（いずれかを選択）
  - <https://www.mingw-w64.org/>
  - <https://www.msys2.org/>


```powershell
$env:CGO_ENABLED="1"

go generate ./...

go build .
```

#### Windows CUDA (NVIDIA)

上記で説明した一般的な Windows 開発ツールに加えて、MSVC をインストールした**後**に CUDA をインストールしてください。

- [NVIDIA CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html)


#### Windows ROCm (AMD Radeon)

上記で説明した一般的な Windows 開発ツールに加えて、MSVC をインストールした**後**に AMD の HIP パッケージをインストールしてください。

- [AMD HIP](https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html)