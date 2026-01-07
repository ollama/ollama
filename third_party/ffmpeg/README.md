# Minimal LGPL FFmpeg Libraries for Ollama

This directory contains the build system for creating minimal, statically-linked FFmpeg libraries used by Ollama's video processing capabilities. This ensures FFmpeg binaries are not required and the final Ollama buid has all the libraries baked into its final build. Approach uses wget to pull the source code and build a LGPL licence compatible build.

## Supported Formats

The current build script `build.sh` has been configured to only support a handful of the most common video formats to avoid creating large library files and bloating the final Ollama binary.

| Container | Codec | Use Case |
|-----------|-------|----------|
| MP4 | H.264 | Most common video format |
| WebM | VP9 | Modern web videos |
| MKV | H.265/H.264 | High-quality videos |
| AVI | H.264 | Legacy format support |

## Building FFmpeg

### Prerequisites

**Linux:**
```bash
sudo apt-get install -y build-essential nasm yasm wget
```

**macOS:**
```bash
brew install nasm wget
```

**Windows:**
```powershell
# Install MSYS2 and MinGW
# Install nasm via MSYS2
```

### Build Command

```bash
# From ollama root directory
./third_party/ffmpeg/build.sh
```

### Skip FFmpeg Build (Local Development)

For faster iteration when not working on video features:

```bash
# Using Makefile
make build SKIP_FFMPEG=1

# Or set environment variable
export SKIP_FFMPEG=1
go build ./...
```

## Build Output

After building, libraries are installed to:
```
third_party/ffmpeg/install/
├── include/
│   ├── libavcodec/
│   ├── libavformat/
│   ├── libavutil/
│   └── libswscale/
└── lib/
    ├── libavcodec.a    (~2.0 MB)
    ├── libavformat.a   (~800 KB)
    ├── libavutil.a     (~400 KB)
    ├── libswscale.a    (~300 KB)
    └── pkgconfig/
```

**Total size:** ~3.5 MB

## FFmpeg Configuration

The build uses aggressive size optimization:

- **Decoder-only:** No encoders included
- **No GPL:** Only LGPL 2.1+ components
- **No network:** File/pipe protocols only
- **No filters:** Minimal processing
- **No hardware acceleration:** Reduces platform dependencies
- **Small binary mode:** `--enable-small` + `--enable-lto`

Full configuration in `build.sh`.

## Integration with Go

FFmpeg libraries are linked via CGO using pkg-config:

```go
// #cgo pkg-config: libavformat libavcodec libavutil libswscale
// #cgo LDFLAGS: -lm -lpthread
import "C"
```

**Build tags:**
- `ffmpeg,cgo` → Use embedded FFmpeg
- No tags → Fallback to system ffmpeg

## License Compliance

This build includes **LGPL 2.1+ components only:**

| Library | License | Purpose |
|---------|---------|---------|
| libavcodec | LGPL 2.1+ | Video decoding |
| libavformat | LGPL 2.1+ | Container parsing |
| libavutil | LGPL 2.1+ | Utility functions |
| libswscale | LGPL 2.1+ | Image scaling/conversion |

See `LICENSE.md` for full details.

## Troubleshooting

### Build fails with "nasm not found"

Install nasm:
```bash
# Linux
sudo apt-get install nasm

# macOS
brew install nasm
```

### Libraries not found during Go build

Set PKG_CONFIG_PATH:
```bash
export PKG_CONFIG_PATH="${PWD}/third_party/ffmpeg/install/lib/pkgconfig"
go build -tags ffmpeg,cgo ./...
```

### Unsupported video format error

This minimal build supports only the most common formats:
Other formats will fall back to system ffmpeg if available.

## CI/CD Integration

GitHub Actions builds FFmpeg automatically. Libraries are cached between builds using the build script checksum as the cache key.

## References

- [FFmpeg Official Site](https://ffmpeg.org/)
- [FFmpeg License](https://ffmpeg.org/legal.html)
- [go-astiav](https://github.com/asticode/go-astiav) - FFmpeg Go bindings
- [LGPL 2.1 License](https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html)
