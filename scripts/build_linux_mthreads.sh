#!/bin/bash

set -eu

. $(dirname $0)/env.sh

OS="${PLATFORM%%/*}"
ARCH="${PLATFORM#*/}"
if [[ ! "$FLAVORS" =~ ^(musa|vulkan)$ ]]; then
    echo "Error: FLAVORS must be either 'musa' or 'vulkan'"
    exit 1
fi
FLAVOR=${FLAVORS}
if [ "$FLAVOR" == "musa" ]; then
    BACKEND_DIR="musa_v4"
elif [ "$FLAVOR" == "vulkan" ]; then
    BACKEND_DIR="vulkan_v1"
fi

echo "Building Ollama Linux ${FLAVOR^^} Tarball"

# Cleanup previous build files
echo "Cleaning up previous build files..."
DIST_DIR="dist/$OS-$ARCH"

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
    --platform=${PLATFORM} \
    ${OLLAMA_COMMON_BUILD_ARGS} \
    --target=archive \
    --build-arg="FLAVOR=${FLAVOR}" \
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
                bin/ollama)                echo "$COMPONENT" >> "ollama-$OS-$ARCH.tar.in" ;;
                lib/ollama/*.so)           echo "$COMPONENT" >> "ollama-$OS-$ARCH.tar.in" ;;
                lib/ollama/${BACKEND_DIR}) echo "$COMPONENT" >> "ollama-$OS-$ARCH.tar.in" ;;
                *)                         echo "Skipping $COMPONENT" ;;
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
echo "Linux ${FLAVOR^^} build completed successfully!"
echo "Artifacts are available in the current directory as .tgz files"
echo "Build output is in: $DIST_DIR"
