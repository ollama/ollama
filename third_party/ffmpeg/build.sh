#!/bin/bash
# Builds minimal LGPL FFmpeg (decoder-only) for Ollama
# License: LGPL 2.1+ only (no GPL components)
#
# Usage:
#   ./build.sh           - Build FFmpeg libraries only
#   ./build.sh ollama    - Build FFmpeg then build Ollama with embedded FFmpeg

set -euo pipefail

FFMPEG_VERSION="7.1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
INSTALL_PREFIX="${SCRIPT_DIR}/install"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Detect platform and architecture
PLATFORM=$(uname -s | tr '[:upper:]' '[:lower:]')
ARCH=$(uname -m)

# Normalize architecture names
case "${ARCH}" in
    x86_64) ARCH="amd64" ;;
    aarch64) ARCH="arm64" ;;
    arm64) ARCH="arm64" ;;
esac

echo "Building FFmpeg ${FFMPEG_VERSION} for ${PLATFORM}-${ARCH}..."
echo "Build directory: ${BUILD_DIR}"
echo "Install prefix: ${INSTALL_PREFIX}"

# Create build directory
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# Download FFmpeg source if not already downloaded
if [ ! -d "ffmpeg-${FFMPEG_VERSION}" ]; then
    echo "Downloading FFmpeg ${FFMPEG_VERSION}..."
    if [ ! -f "ffmpeg-${FFMPEG_VERSION}.tar.xz" ]; then
        wget -q "https://ffmpeg.org/releases/ffmpeg-${FFMPEG_VERSION}.tar.xz"
    fi
    echo "Extracting..."
    tar xf "ffmpeg-${FFMPEG_VERSION}.tar.xz"
fi

cd "ffmpeg-${FFMPEG_VERSION}"

# FFmpeg configure flags for minimal LGPL build
CONFIGURE_FLAGS=(
    # License compliance - LGPL only (DO NOT MODIFY)
    --disable-gpl
    --disable-version3
    --disable-nonfree

    # Minimize binary size
    --disable-everything
    --enable-small
    --enable-lto
    --disable-debug
    --disable-programs
    --disable-doc
    --disable-htmlpages
    --disable-manpages
    --disable-podpages
    --disable-txtpages

    # Static linking
    --enable-static
    --disable-shared
    --enable-pic
    --pkg-config-flags="--static"

    # Required core libraries
    --enable-avcodec
    --enable-avformat
    --enable-avutil
    --enable-swscale

    # Video decoders (80/20 rule - top 4 formats)
    --enable-decoder=h264        # MP4, AVI, MKV
    --enable-decoder=hevc        # MKV, MP4 (H.265)
    --enable-decoder=vp9         # WebM

    # Parsers (required for decoders)
    --enable-parser=h264
    --enable-parser=hevc
    --enable-parser=vp9

    # Demuxers (container formats)
    --enable-demuxer=mov         # MP4, MOV, M4V
    --enable-demuxer=matroska    # MKV, WebM
    --enable-demuxer=avi         # AVI

    # Protocols (file access only, no network)
    --enable-protocol=file
    --enable-protocol=pipe

    # Disable everything else
    --disable-network
    --disable-encoders           # Decoder-only
    --disable-muxers
    --disable-filters
    --disable-bsfs
    --disable-devices
    --disable-hwaccels           # No hardware acceleration to minimize size
    --disable-vaapi
    --disable-vdpau
    --disable-videotoolbox
    --disable-audiotoolbox
    --disable-nvenc
    --disable-nvdec
    --disable-cuda
    --disable-cuvid
    --disable-xlib
    --disable-zlib
    --disable-bzlib
    --disable-lzma
    --disable-iconv
    --disable-sdl2
    --disable-securetransport
    --disable-schannel
    --disable-libxml2

    # Install prefix
    --prefix="${INSTALL_PREFIX}"
)

echo "Configuring FFmpeg with minimal LGPL settings..."
./configure "${CONFIGURE_FLAGS[@]}"

echo "Building FFmpeg..."
make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

echo "Installing FFmpeg libraries..."
make install

echo ""
echo "✅ FFmpeg build complete!"
echo "Library sizes:"
du -h "${INSTALL_PREFIX}/lib"/*.a 2>/dev/null || du -h "${INSTALL_PREFIX}/lib"/*.lib 2>/dev/null || true
echo ""
echo "Total library size:"
du -sh "${INSTALL_PREFIX}/lib"
echo ""

# Clean up unnecessary files
echo "Cleaning up unnecessary files..."
rm -rf "${INSTALL_PREFIX}/share"  # Remove example code
rm -rf "${BUILD_DIR}"              # Remove build directory
echo "Removed: share/ directory (examples)"
echo "Removed: build directory"
echo ""

echo "Libraries installed to: ${INSTALL_PREFIX}"
echo "Final size (lib + include):"
du -sh "${INSTALL_PREFIX}"
echo ""
echo "Supported formats:"
echo "  - MP4 (H.264)"
echo "  - WebM (VP9)"
echo "  - MKV (H.265/H.264)"
echo "  - AVI (H.264)"
echo ""
echo "License: LGPL 2.1+ (decoder-only, no GPL components)"

# If "ollama" argument is passed, build Ollama with embedded FFmpeg
if [ "${1:-}" = "ollama" ]; then
    echo ""
    echo "========================================"
    echo "Building Ollama with embedded FFmpeg"
    echo "========================================"
    echo ""

    cd "${PROJECT_ROOT}"

    # Export PKG_CONFIG_PATH for our minimal FFmpeg
    export PKG_CONFIG_PATH="${INSTALL_PREFIX}/lib/pkgconfig"
    export CGO_ENABLED=1

    echo "PKG_CONFIG_PATH: ${PKG_CONFIG_PATH}"
    echo "FFmpeg version: $(pkg-config --modversion libavcodec)"
    echo ""

    go build -tags ffmpeg,cgo -o ./ollama .

    echo ""
    echo "✅ Ollama build complete: ./ollama"
    echo "   - Embedded FFmpeg: YES (statically linked)"
    echo "   - Fallback: System ffmpeg"
fi
