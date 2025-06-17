#!/bin/bash

set -eu

# Script to duplicate the linux-build workflow from GitHub Actions for MUSA builds
# Usage: ./scripts/build_linux_musa.sh [OS] [ARCH] [TARGET]
# Examples:
#   ./scripts/build_linux_musa.sh linux amd64 archive
#   ./scripts/build_linux_musa.sh linux arm64 archive

# Default values (can be overridden by command line arguments)
OS=${1:-linux}
ARCH=${2:-amd64}
TARGET=${3:-archive}

echo "Building Ollama Linux MUSA Release"
echo "OS: $OS"
echo "ARCH: $ARCH"
echo "TARGET: $TARGET"
echo "Platform: $OS/$ARCH"

# Set up environment variables (similar to setup-environment job)
VERSION=${VERSION:-$(git describe --tags --first-parent --abbrev=7 --long --dirty --always | sed -e "s/^v//g")}
GOFLAGS="'-ldflags=-w -s \"-X=github.com/ollama/ollama/version.Version=$VERSION\" \"-X=github.com/ollama/ollama/server.mode=release\"'"

echo "VERSION: $VERSION"
echo "GOFLAGS: $GOFLAGS"

# Cleanup previous build files
echo "Cleaning up previous build files..."
DIST_DIR="dist/$OS-$ARCH"

# Remove previous build artifacts
rm -f *.tgz
rm -f *.tar.in
echo "Removed previous .tgz and .tar.in files"

# Remove previous dist directory
if [ -d "$DIST_DIR" ]; then
    rm -rf "$DIST_DIR"
    echo "Removed previous dist directory: $DIST_DIR"
fi

# Create fresh dist directory
mkdir -p "$DIST_DIR"
echo "Created fresh dist directory: $DIST_DIR"

echo "Building Docker image with buildx..."

# Build using Docker (equivalent to docker/build-push-action@v6)
echo "Starting Docker build..."
docker buildx build \
    --platform="$OS/$ARCH" \
    --target="$TARGET" \
    --build-arg="GOFLAGS=$GOFLAGS" \
    --build-arg="CGO_CFLAGS=${CGO_CFLAGS:-}" \
    --build-arg="CGO_CXXFLAGS=${CGO_CXXFLAGS:-}" \
    --build-arg="FLAVOR=musa" \
    --output="type=local,dest=$DIST_DIR" \
    .

echo "Build completed. Organizing artifacts..."

# Change to dist directory for artifact organization
cd "$DIST_DIR"

# Organize built components into different archives (equivalent to the shell script in workflow)
if [ -d "bin" ] || [ -d "lib/ollama" ]; then
    for COMPONENT in bin/* lib/ollama/*; do
        if [ -e "$COMPONENT" ]; then
            case "$COMPONENT" in
                bin/ollama)               echo "$COMPONENT" >> "ollama-$OS-$ARCH.tar.in" ;;
                lib/ollama/*.so)          echo "$COMPONENT" >> "ollama-$OS-$ARCH.tar.in" ;;
                lib/ollama/musa_v4)       echo "$COMPONENT" >> "ollama-$OS-$ARCH.tar.in" ;;
                *)                        echo "Skipping $COMPONENT" ;;
            esac
        fi
    done
else
    echo "Warning: No bin/ or lib/ollama/ directories found in build output"
    echo "Available files/directories:"
    ls -la
fi

# Go back to root directory
cd - > /dev/null

# Create compressed archives (equivalent to the tar/pigz command in workflow)
echo "Creating compressed archives..."

for ARCHIVE in "$DIST_DIR"/*.tar.in; do
    if [ -f "$ARCHIVE" ]; then
        ARCHIVE_NAME=$(basename "${ARCHIVE/.tar.in/.tgz}")
        echo "Creating $ARCHIVE_NAME..."

        # Use gzip if pigz is not available
        if command -v pigz > /dev/null; then
            COMPRESS_CMD="pigz -9vc"
        else
            COMPRESS_CMD="gzip -9c"
        fi

        tar c -C "$DIST_DIR" -T "$ARCHIVE" --owner 0 --group 0 | $COMPRESS_CMD > "$ARCHIVE_NAME"
        echo "Created: $ARCHIVE_NAME"
    fi
done

echo "Build artifacts created:"
ls -la *.tgz 2>/dev/null || echo "No .tgz files found"

echo ""
echo "Linux MUSA build completed successfully!"
echo "Artifacts are available in the current directory as .tgz files"
echo "Build output is in: $DIST_DIR"
