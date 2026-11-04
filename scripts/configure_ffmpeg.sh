#!/bin/sh
# Reproducible decoder-only FFmpeg configuration. Libraries are statically linked
# into separate LGPL tools; Ollama itself does not link against FFmpeg.
# FFmpeg 9.0.2: https://ffmpeg.org/releases/ffmpeg-9.0.2.tar.xz
# The adjacent FFMPEG_SOURCE.tar.xz is the exact corresponding source.
# Rebuild with this script using: sh FFMPEG_NOTICE SOURCE_DIR CC OS ARCH MIN_OS MSVC.
set -eu
source_dir=$1
compiler=$2
target_os=$3
arch=$4
deployment_target=$5
msvc=$6
# ExternalProject extracts the release archive in its binary directory.
cd "$source_dir"
if [ "$target_os" = mingw32 ]; then
    compiler=$(cygpath -u "$compiler")
fi
PATH="$(dirname "$compiler"):$PATH"
export PATH
if [ "$msvc" = 1 ] || [ "$msvc" = TRUE ]; then
    target_os=win32
fi
set -- --cc="$(basename "$compiler")" --arch="$arch" --target-os="$target_os"
if [ "$target_os" = win32 ]; then
    set -- "$@" --toolchain=msvc
elif [ "$target_os" = mingw32 ]; then
    set -- "$@" --extra-ldflags=-static --host-ldflags=-static
    if [ "$arch" = aarch64 ]; then
        # Never run ARM64 configure probes on an x64 Windows build host.
        set -- "$@" --enable-cross-compile --cross-prefix=aarch64-w64-mingw32- --host-cc=x86_64-w64-mingw32-gcc
    else
        set -- "$@" --host-cc="$(basename "$compiler")"
        case "$compiler" in
            *clang*) set -- "$@" --ar=llvm-ar --nm=llvm-nm --ranlib=llvm-ranlib --strip=llvm-strip ;;
        esac
    fi
fi
if [ "$target_os" = darwin ]; then
    flags="-arch $arch"
    [ "$arch" != aarch64 ] || flags="-arch arm64"
    if [ -n "$deployment_target" ]; then
        flags="$flags -mmacosx-version-min=$deployment_target"
    fi
    set -- "$@" --enable-cross-compile --extra-cflags="$flags" --extra-ldflags="$flags"
fi
# MTMD uses these tools for still WebP and Omni video soundtracks as well.
./configure "$@" \
    --disable-autodetect --disable-gpl --disable-version3 --disable-nonfree \
    --disable-network --disable-everything --disable-doc --disable-debug \
    --disable-shared --enable-static --enable-small --disable-x86asm \
    --disable-hwaccels --disable-avdevice --disable-ffplay \
    --enable-ffmpeg --enable-ffprobe \
    --enable-decoder=h264,hevc,vp9,webp,aac,mp3,opus,vorbis,flac,pcm_s16le,pcm_f32le \
    --enable-parser=h264,hevc,vp9,webp,aac,mpegaudio,opus,vorbis,flac \
    --enable-demuxer=mov,matroska,avi,image_webp_pipe --enable-protocol=file,pipe,cache \
    --enable-encoder=rawvideo,pcm_f32le --enable-muxer=rawvideo,pcm_f32le \
    --enable-filter=fps,scale,format,null,trim,transpose,hflip,vflip,aresample,aformat,anull
