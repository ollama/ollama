# 開発

必要なツールをインストールしてください：

- cmake バージョン3.24以上
- go バージョン1.21以上
- gcc バージョン11.4.0以上

```bash
brew install go cmake gcc
```

オプションでデバッグおよび詳細なログを有効にする:

```bash
# At build time
export CGO_CFLAGS="-g"

# At runtime
export OLLAMA_DEBUG=1
```

必要なライブラリを取得し、ネイティブLLMコードをビルドしてください:

```bash
go generate ./...
```

次に、Ollamaをビルドしてください:

```bash
go build .
```

これで、`ollama`を実行できます:

```bash
./ollama
```

### Linux

#### Linux CUDA (NVIDIA)

*お使いのオペレーティングシステムディストリビューションには、既にNVIDIA CUDA用のパッケージが含まれているかもしれません。ディストリビューションパッケージは通常好ましいですが、手順はディストリビューションに依存します。可能であれば、ディストリビューション固有のドキュメントを確認して、依存関係に関する情報を参照してください。*

`cmake`および`golang`をインストールし、[NVIDIA CUDA](https://developer.nvidia.com/cuda-downloads)の開発およびランタイムパッケージもインストールしてください。

通常、ビルドスクリプトはCUDAを自動検出しますが、Linuxディストリビューションやインストールアプローチが異常なパスを使用する場合は、環境変数`CUDA_LIB_DIR`を共有ライブラリの場所に、`CUDACXX`をnvccコンパイラの場所に指定することができます。また、`CMAKE_CUDA_ARCHITECTURES`を設定して、対象のCUDAアーキテクチャをカスタマイズできます（例："50;60;70"）。

その後、依存関係を生成してください:

```
go generate ./...
```

その後、バイナリをビルドしてください:

```
go build .
```

#### Linux ROCm (AMD)

*お使いのオペレーティングシステムディストリビューションには、すでにAMD ROCmおよびCLBlast用のパッケージが含まれているかもしれません。ディストリビューションパッケージは通常好ましいですが、手順はディストリビューションに依存します。可能であれば、ディストリビューション固有のドキュメントを確認して、依存関係に関する情報を参照してください。*

まず、[CLBlast](https://github.com/CNugteren/CLBlast/blob/master/doc/installation.md)と[ROCm](https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html)の開発パッケージ、および`cmake`と`golang`をインストールしてください。

通常、ビルドスクリプトはROCmを自動検出しますが、Linuxディストリビューションやインストールアプローチが異常なパスを使用する場合は、環境変数`ROCM_PATH`をROCmのインストール場所に（通常は`/opt/rocm`）、`CLBlast_DIR`をCLBlastのインストール場所に（通常は`/usr/lib/cmake/CLBlast`）指定することができます。また、`AMDGPU_TARGETS`を設定してAMD GPUの対象をカスタマイズすることもできます（例：`AMDGPU_TARGETS="gfx1101;gfx1102"`）。


```
go generate ./...
```

その後、バイナリをビルドしてください:

```
go build .
```

ROCmは実行時にGPUにアクセスするために特権が必要です。ほとんどのディストリビューションでは、ユーザーアカウントを`render`グループに追加するか、rootとして実行することができます。

ROCm requires elevated privileges to access the GPU at runtime.  On most distros you can add your user account to the `render` group, or run as root.

#### 高度なCPU設定

デフォルトでは、`go generate ./...`を実行すると、一般的なCPUファミリとベクトル数学の能力に基づいて、いくつかの異なるバリエーションのLLMライブラリがコンパイルされます。これには、ほとんどの64ビットCPUで動作する最低限のバージョンも含まれており、パフォーマンスはやや低いです。実行時に、Ollamaは最適なバリエーションを自動検出してロードします。プロセッサにカスタマイズされたCPUベースのビルドを作成したい場合は、`OLLAMA_CUSTOM_CPU_DEFS`を使用したいllama.cppフラグに設定できます。たとえば、Intel i9-9880H向けに最適化されたバイナリをコンパイルする場合は、次のようにします:

```
OLLAMA_CUSTOM_CPU_DEFS="-DLLAMA_AVX=on -DLLAMA_AVX2=on -DLLAMA_F16C=on -DLLAMA_FMA=on" go generate ./...
go build .
```

#### コンテナ化されたLinuxビルド

Dockerが利用可能な場合、CUDAおよびROCmの依存関係が含まれている`./scripts/build_linux.sh`を使用してLinux用のバイナリをビルドできます。生成されたバイナリは`./dist`に配置されます。

### Windows

注意：OllamaのWindowsビルドはまだ開発中です。

必要なツールをインストールしてください：

- MSVCツールチェーン - C/C++およびcmakeが最小要件です
- go バージョン1.21以上
- GCCを搭載したMinGW（いずれかを選択）
  - <https://www.mingw-w64.org/>
  - <https://www.msys2.org/>


```powershell
$env:CGO_ENABLED="1"

go generate ./...

go build .
```

#### Windows CUDA (NVIDIA)

上記で説明した一般的なWindows開発ツールに加えて、以下をインストールしてください:

- [NVIDIA CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html)
