#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
build-win.sh - Cross-build ejecutable Windows y opcional installer (sin CUDA)\n\nUso:\n  build-win.sh [opciones] -- <go build extra flags>\n\nOpciones:\n  -v, --version X.Y.Z    Version para PKG_VERSION (default 0.0.0)\n  -a, --arch ARCH        amd64 (default) o arm64 (mapea a aarch64-w64-mingw32)\n  -o, --out DIR          Directorio de salida (default dist)\n  -I, --installer        Generar installer Inno Setup (wine)\n      --iss PATH         Ruta custom a ollama.iss (default app/ollama.iss)\n      --skip-go          No compilar Go (solo libs C si hubiera)\n      --ccache           Habilitar ccache\n      --cpu-libs         Compilar librerías CPU (ggml) con CMake/Ninja\n      --no-app           No compilar app GUI (solo ollama.exe)\n  -h, --help             Esta ayuda\n\nLimitaciones:\n  - No CUDA/ROCm cross: solo CPU.\n  - No link dinámico de runtimes MSVC (usamos mingw).\nEOF
}

ARCH=amd64
OUT=dist
VERSION=0.0.0
DO_INSTALLER=0
CPU_LIBS=0
NO_APP=0
ISS_PATH=app/ollama.iss
SKIP_GO=0
USE_CCACHE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -v|--version) VERSION="$2"; shift 2;;
    -a|--arch) ARCH="$2"; shift 2;;
    -o|--out) OUT="$2"; shift 2;;
    -I|--installer) DO_INSTALLER=1; shift;;
    --iss) ISS_PATH="$2"; shift 2;;
    --skip-go) SKIP_GO=1; shift;;
  --ccache) USE_CCACHE=1; shift;;
  --cpu-libs) CPU_LIBS=1; shift;;
  --no-app) NO_APP=1; shift;;
    -h|--help) usage; exit 0;;
    --) shift; break;;
    *) echo "Opcion desconocida: $1" >&2; usage; exit 1;;
  esac
done

EXTRA_GO_FLAGS=("$@")

case "$ARCH" in
  amd64) GOARCH=amd64; HOST_TRIPLET=x86_64-w64-mingw32;;
  arm64) GOARCH=arm64; HOST_TRIPLET=aarch64-w64-mingw32;;
  *) echo "ARCH no soportada: $ARCH" >&2; exit 1;;
esac

export PKG_VERSION="$VERSION"
export GOOS=windows
export GOARCH
export CGO_ENABLED=1

if [[ $USE_CCACHE -eq 1 ]]; then
  export CC="ccache ${HOST_TRIPLET}-gcc"
  export CXX="ccache ${HOST_TRIPLET}-g++"
else
  export CC="${HOST_TRIPLET}-gcc"
  export CXX="${HOST_TRIPLET}-g++"
fi

mkdir -p "$OUT"

if [[ $CPU_LIBS -eq 1 ]]; then
  echo "==> Compilando librerías CPU (ggml) con CMake"
  BUILD_DIR=build-${HOST_TRIPLET}
  rm -rf "$BUILD_DIR"
  mkdir "$BUILD_DIR"
  pushd "$BUILD_DIR" >/dev/null
  cmake -G Ninja \
    -DCMAKE_SYSTEM_NAME=Windows \
    -DCMAKE_C_COMPILER=${CC} \
    -DCMAKE_CXX_COMPILER=${CXX} \
    -DCMAKE_RC_COMPILER=${HOST_TRIPLET}-windres \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$(pwd)/stage" \
    ..
  cmake --build . --parallel
  cmake --install . --component CPU --strip || true
  mkdir -p "$OUT/lib/ollama"
  if compgen -G "stage/lib/ollama/*" > /dev/null 2>&1; then
    cp -a stage/lib/ollama/* "$OUT/lib/ollama/" || true
  fi
  popd >/dev/null
fi

echo "==> Compilando Ollama (Go) para windows/$GOARCH (VERSION=$VERSION)"
if [[ $SKIP_GO -eq 0 ]]; then
  go build -o "$OUT/ollama.exe" "${EXTRA_GO_FLAGS[@]}" .
fi

if [[ $NO_APP -eq 0 ]]; then
  echo "==> Compilando app GUI (windowsgui)"
  pushd app >/dev/null
  GOOS=windows GOARCH=$GOARCH CGO_ENABLED=0 go build -ldflags="-H windowsgui" -o "../$OUT/windows-${GOARCH}-app.exe" .
  popd >/dev/null
fi

if [[ $DO_INSTALLER -eq 1 ]]; then
  echo "==> Preparando layout para Inno Setup"
  WIN_DIST="$OUT/windows-$GOARCH"
  mkdir -p "$WIN_DIST/lib/ollama"
  cp "$OUT/ollama.exe" "$WIN_DIST/"
  if [[ $NO_APP -eq 0 ]]; then
    cp "$OUT/windows-${GOARCH}-app.exe" "$WIN_DIST/../windows-${GOARCH}-app.exe" 2>/dev/null || true
  fi
  if compgen -G "$OUT/lib/ollama/*" > /dev/null 2>&1; then
    cp -r "$OUT/lib/ollama" "$WIN_DIST/lib/" || true
  fi
  cp app/ollama_welcome.ps1 "$OUT/" 2>/dev/null || true

  echo "==> Ejecutando ISCC (wine)"
  # Usamos la versión extraída en /opt/inno/app
  if ! command -v wine >/dev/null; then
    echo "wine no encontrado" >&2; exit 1
  fi
  if [[ ! -f "$ISS_PATH" ]]; then
    echo "Archivo ISS no encontrado: $ISS_PATH" >&2; exit 1
  fi
  # ISCC necesita path Windows; usamos winepath
  WIN_ISS=$(winepath -w "$ISS_PATH")
  (cd app && wine /opt/inno/app/ISCC.exe /Q "$WIN_ISS")
  echo "Installer generado en dist/ (OllamaSetup.exe)"
fi

echo "==> Hecho"