#!/bin/sh

# Note: 
#  While testing, if you double-click on the Ollama.app 
#  some state is left on MacOS and subsequent attempts
#  to build again will fail with:
#
#    hdiutil: create failed - Operation not permitted
#
#  To work around, specify another volume name with:
#
#    VOL_NAME="$(date)" ./scripts/build_darwin.sh
#
VOL_NAME=${VOL_NAME:-"Ollama"}
export VERSION=${VERSION:-$(git describe --tags --first-parent --abbrev=7 --long --dirty --always | sed -e "s/^v//g")}
export GOFLAGS="'-ldflags=-w -s \"-X=github.com/ollama/ollama/version.Version=${VERSION#v}\" \"-X=github.com/ollama/ollama/server.mode=release\"'"
export CGO_CFLAGS="-O3 -mmacosx-version-min=14.0"
export CGO_CXXFLAGS="-O3 -mmacosx-version-min=14.0"
export CGO_LDFLAGS="-mmacosx-version-min=14.0"

set -e

status() { echo >&2 ">>> $@"; }
usage() {
    echo "usage: $(basename $0) [build app [sign]]"
    exit 1
}

mkdir -p dist

# Work around MLX's v3 metallib link leaking the macOS 26 deployment target.
_relink_mlx_metallib() {
    BUILD_DIR="$1"
    KERNEL_DIR="$BUILD_DIR/_deps/mlx-build/mlx/backend/metal/kernels"
    AIR_LIST="$BUILD_DIR/mlx-air-files.txt"
    METALLIB="$KERNEL_DIR/mlx.metallib"

    find "$KERNEL_DIR" -type f -name '*.air' | sort > "$AIR_LIST"
    if [ ! -s "$AIR_LIST" ]; then
        echo "error: could not find MLX AIR files in $KERNEL_DIR" >&2
        exit 1
    fi

    status "Relinking MLX metallib"
    rm -f "$METALLIB"
    xargs xcrun -sdk macosx metallib -o "$METALLIB" < "$AIR_LIST"
}


ARCHS="arm64 amd64"
while getopts "a:h" OPTION; do
    case $OPTION in
        a) ARCHS=$OPTARG ;;
        h) usage ;;
    esac
done

shift $(( $OPTIND - 1 ))

_build_darwin() {
    for ARCH in $ARCHS; do
        status "Building darwin $ARCH"
        INSTALL_PREFIX=dist/darwin-$ARCH/        

        if [ "$ARCH" = "amd64" ]; then
            status "Building darwin $ARCH dynamic backends"
            BUILD_DIR=build/darwin-$ARCH
            cmake -B $BUILD_DIR \
                -DCMAKE_OSX_ARCHITECTURES=x86_64 \
                -DCMAKE_OSX_DEPLOYMENT_TARGET=14.0 \
                -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX \
                -DMLX_ENGINE=ON \
                -DMLX_ENABLE_X64_MAC=ON \
                -DOLLAMA_RUNNER_DIR=./
            cmake --build $BUILD_DIR --target ggml-cpu -j
            cmake --build $BUILD_DIR --target mlx mlxc -j
            cmake --install $BUILD_DIR --component CPU
            cmake --install $BUILD_DIR --component MLX
            cmake --install $BUILD_DIR --component MLX_VENDOR
            # Override CGO flags to point to the amd64 build directory
            MLX_CGO_CFLAGS="-O3 -mmacosx-version-min=14.0"
            MLX_CGO_LDFLAGS="-ldl -lc++ -framework Accelerate -mmacosx-version-min=14.0"
        else
            # CPU backend (ggml-cpu, installed flat to lib/ollama/)
            BUILD_DIR_CPU=build/arm64-cpu
            status "Building arm64 CPU backend"
            cmake -S . -B $BUILD_DIR_CPU \
                -DCMAKE_BUILD_TYPE=Release \
                -DCMAKE_OSX_DEPLOYMENT_TARGET=14.0 \
                -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX
            cmake --build $BUILD_DIR_CPU --target ggml-cpu --parallel
            cmake --install $BUILD_DIR_CPU --component CPU

            # Build MLX twice for arm64
            # Metal 3.x build (backward compatible, macOS 14+)
            BUILD_DIR=build/metal-v3
            status "Building MLX Metal v3 (macOS 14+)"
            cmake -S . -B $BUILD_DIR \
                -DCMAKE_BUILD_TYPE=Release \
                -DMLX_ENGINE=ON \
                -DOLLAMA_RUNNER_DIR=mlx_metal_v3 \
                -DCMAKE_OSX_DEPLOYMENT_TARGET=14.0 \
                -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX
            cmake --build $BUILD_DIR --target mlx mlxc --parallel
            _relink_mlx_metallib $BUILD_DIR
            cmake --install $BUILD_DIR --component MLX
            cmake --install $BUILD_DIR --component MLX_VENDOR

            # Metal 4.x build (NAX-enabled, macOS 26+)
            # Only possible with Xcode 26+ SDK; skip on older toolchains.
            SDK_MAJOR=$(xcrun --show-sdk-version 2>/dev/null | cut -d. -f1)
            if [ "${SDK_MAJOR:-0}" -ge 26 ]; then
                V3_DEPS=$BUILD_DIR/_deps
                BUILD_DIR_V4=build/metal-v4
                status "Building MLX Metal v4 (macOS 26+, NAX)"
                cmake -S . -B $BUILD_DIR_V4 \
                    -DCMAKE_BUILD_TYPE=Release \
                    -DMLX_ENGINE=ON \
                    -DOLLAMA_RUNNER_DIR=mlx_metal_v4 \
                    -DCMAKE_OSX_DEPLOYMENT_TARGET=26.0 \
                    -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX \
                    -DFETCHCONTENT_SOURCE_DIR_MLX=$V3_DEPS/mlx-src \
                    -DFETCHCONTENT_SOURCE_DIR_MLX-C=$V3_DEPS/mlx-c-src \
                    -DFETCHCONTENT_SOURCE_DIR_JSON=$V3_DEPS/json-src \
                    -DFETCHCONTENT_SOURCE_DIR_FMT=$V3_DEPS/fmt-src \
                    -DFETCHCONTENT_SOURCE_DIR_METAL_CPP=$V3_DEPS/metal_cpp-src
                cmake --build $BUILD_DIR_V4 --target mlx mlxc --parallel
                cmake --install $BUILD_DIR_V4 --component MLX
                cmake --install $BUILD_DIR_V4 --component MLX_VENDOR
            else
                status "Skipping MLX Metal v4 (SDK $SDK_MAJOR < 26, need Xcode 26+)"
            fi

            # Use the v3 build for CGO linking (compatible with both)
            MLX_CGO_CFLAGS="-O3 -mmacosx-version-min=14.0"
            MLX_CGO_LDFLAGS="-lc++ -framework Metal -framework Foundation -framework Accelerate -mmacosx-version-min=14.0"
        fi
        GOOS=darwin GOARCH=$ARCH CGO_ENABLED=1 CGO_CFLAGS="$MLX_CGO_CFLAGS" CGO_LDFLAGS="$MLX_CGO_LDFLAGS" go build -o $INSTALL_PREFIX .
        # MLX libraries stay in lib/ollama/ (flat or variant subdirs).
        # The runtime discovery in dynamic.go searches lib/ollama/ relative
        # to the executable, including mlx_* subdirectories.
    done
}

_sign_darwin() {
    status "Creating universal binary..."
    mkdir -p dist/darwin
    lipo -create -output dist/darwin/ollama dist/darwin-*/ollama
    chmod +x dist/darwin/ollama

    if [ -n "$APPLE_IDENTITY" ]; then
        for F in dist/darwin/ollama dist/darwin-*/lib/ollama/* dist/darwin-*/lib/ollama/mlx_metal_v*/*; do
            [ -f "$F" ] && [ ! -L "$F" ] || continue
            codesign -f --timestamp -s "$APPLE_IDENTITY" --identifier ai.ollama.ollama --options=runtime "$F"
        done

        # create a temporary zip for notarization
        TEMP=$(mktemp -u).zip
        ditto -c -k --keepParent dist/darwin/ollama "$TEMP"
        xcrun notarytool submit "$TEMP" --wait --timeout 20m --apple-id $APPLE_ID --password $APPLE_PASSWORD --team-id $APPLE_TEAM_ID
        rm -f "$TEMP"
    fi

    status "Creating universal tarball..."
    tar -cf dist/ollama-darwin.tar --strip-components 2 dist/darwin/ollama
    tar -rf dist/ollama-darwin.tar --strip-components 4 dist/darwin-amd64/lib/
    tar -rf dist/ollama-darwin.tar --strip-components 4 dist/darwin-arm64/lib/
    gzip -9vc <dist/ollama-darwin.tar >dist/ollama-darwin.tgz
}

_build_macapp() {
    if ! command -v npm &> /dev/null; then
        echo "npm is not installed. Please install Node.js and npm first:"
        echo "   Visit: https://nodejs.org/"
        exit 1
    fi

    if ! command -v tsc &> /dev/null; then
        echo "Installing TypeScript compiler..."
        npm install -g typescript
    fi

    echo "Installing required Go tools..."

    cd app/ui/app
    npm install
    npm run build
    cd ../../..

    # Build the Ollama.app bundle
    rm -rf dist/Ollama.app
    cp -a ./app/darwin/Ollama.app dist/Ollama.app

    # update the modified date of the app bundle to now
    touch dist/Ollama.app

    go clean -cache
    GOARCH=amd64 CGO_ENABLED=1 GOOS=darwin go build -o dist/darwin-app-amd64 -ldflags="-s -w -X=github.com/ollama/ollama/app/version.Version=${VERSION}" ./app/cmd/app
    GOARCH=arm64 CGO_ENABLED=1 GOOS=darwin go build -o dist/darwin-app-arm64 -ldflags="-s -w -X=github.com/ollama/ollama/app/version.Version=${VERSION}" ./app/cmd/app
    mkdir -p dist/Ollama.app/Contents/MacOS
    lipo -create -output dist/Ollama.app/Contents/MacOS/Ollama dist/darwin-app-amd64 dist/darwin-app-arm64
    rm -f dist/darwin-app-amd64 dist/darwin-app-arm64

    # Create a mock Squirrel.framework bundle
    mkdir -p dist/Ollama.app/Contents/Frameworks/Squirrel.framework/Versions/A/Resources/
    cp -a dist/Ollama.app/Contents/MacOS/Ollama dist/Ollama.app/Contents/Frameworks/Squirrel.framework/Versions/A/Squirrel
    ln -s ../Squirrel dist/Ollama.app/Contents/Frameworks/Squirrel.framework/Versions/A/Resources/ShipIt
    cp -a ./app/cmd/squirrel/Info.plist dist/Ollama.app/Contents/Frameworks/Squirrel.framework/Versions/A/Resources/Info.plist
    ln -s A dist/Ollama.app/Contents/Frameworks/Squirrel.framework/Versions/Current
    ln -s Versions/Current/Resources dist/Ollama.app/Contents/Frameworks/Squirrel.framework/Resources
    ln -s Versions/Current/Squirrel dist/Ollama.app/Contents/Frameworks/Squirrel.framework/Squirrel

    # Update the version in the Info.plist
    plutil -replace CFBundleShortVersionString -string "$VERSION" dist/Ollama.app/Contents/Info.plist
    plutil -replace CFBundleVersion -string "$VERSION" dist/Ollama.app/Contents/Info.plist

    # Setup the ollama binaries
    mkdir -p dist/Ollama.app/Contents/Resources
    if [ -d dist/darwin-amd64 ]; then
        lipo -create -output dist/Ollama.app/Contents/Resources/ollama dist/darwin-amd64/ollama dist/darwin-arm64/ollama

        # Copy .so files from both architectures (names don't collide: arm64=libggml-cpu.so, amd64=libggml-cpu-*.so)
        cp dist/darwin-arm64/lib/ollama/*.so dist/Ollama.app/Contents/Resources/ 2>/dev/null || true
        cp dist/darwin-amd64/lib/ollama/*.so dist/Ollama.app/Contents/Resources/ 2>/dev/null || true
        # Lipo common dylibs into universal binaries, copy amd64-only ones as-is.
        # Skip MLX dylibs (libmlx*.dylib) — on arm64 these live in variant
        # subdirs (mlx_metal_v3/) and are lipo'd there below. Copying the
        # amd64 flat copy here would produce an x86_64-only dylib in
        # Resources/ that shadows the variant subdirs.
        for F in dist/darwin-amd64/lib/ollama/*.dylib; do
            [ -f "$F" ] && [ ! -L "$F" ] || continue
            BASE=$(basename "$F")
            case "$BASE" in libmlx*) continue ;; esac
            if [ -f "dist/darwin-arm64/lib/ollama/$BASE" ]; then
                lipo -create -output "dist/Ollama.app/Contents/Resources/$BASE" "$F" "dist/darwin-arm64/lib/ollama/$BASE"
            else
                cp "$F" dist/Ollama.app/Contents/Resources/
            fi
        done
        # Recreate ggml-base symlinks
        (cd dist/Ollama.app/Contents/Resources && ln -sf libggml-base.0.0.0.dylib libggml-base.0.dylib && ln -sf libggml-base.0.dylib libggml-base.dylib) 2>/dev/null || true

        # MLX Metal variant subdirs from arm64
        for VARIANT in dist/darwin-arm64/lib/ollama/mlx_metal_v*/; do
            [ -d "$VARIANT" ] || continue
            VNAME=$(basename "$VARIANT")
            DEST=dist/Ollama.app/Contents/Resources/$VNAME
            mkdir -p "$DEST"
            if [ "$VNAME" = "mlx_metal_v3" ]; then
                # v3: lipo amd64 flat + arm64 v3 into universal dylibs
                for LIB in libmlx.dylib libmlxc.dylib; do
                    if [ -f "dist/darwin-amd64/lib/ollama/$LIB" ] && [ -f "$VARIANT$LIB" ]; then
                        lipo -create -output "$DEST/$LIB" "dist/darwin-amd64/lib/ollama/$LIB" "$VARIANT$LIB"
                    elif [ -f "$VARIANT$LIB" ]; then
                        cp "$VARIANT$LIB" "$DEST/"
                    fi
                done
                # Copy remaining files (metallib and auxiliary runtime dylibs)
                # from arm64 v3. libmlx/libmlxc are handled above so v3 can
                # be universal when an x86_64 build is available.
                for F in "$VARIANT"*; do
                    case "$(basename "$F")" in libmlx.dylib|libmlxc.dylib) continue ;; esac
                    [ -f "$F" ] && [ ! -L "$F" ] || continue
                    cp "$F" "$DEST/"
                done
            else
                # v4+: arm64-only, copy all non-symlink files
                for F in "$VARIANT"*; do
                    [ -f "$F" ] && [ ! -L "$F" ] || continue
                    cp "$F" "$DEST/"
                done
            fi
        done
    else
        cp -a dist/darwin/ollama dist/Ollama.app/Contents/Resources/ollama
        # arm64-only build: copy variant subdirs directly
        for VARIANT in dist/darwin-arm64/lib/ollama/mlx_metal_v*/; do
            [ -d "$VARIANT" ] || continue
            VNAME=$(basename "$VARIANT")
            mkdir -p dist/Ollama.app/Contents/Resources/$VNAME
            cp "$VARIANT"* dist/Ollama.app/Contents/Resources/$VNAME/ 2>/dev/null || true
        done
        # CPU backend libs (ggml-base, ggml-cpu) are flat in lib/ollama/
        cp dist/darwin-arm64/lib/ollama/*.so dist/Ollama.app/Contents/Resources/ 2>/dev/null || true
        for F in dist/darwin-arm64/lib/ollama/*.dylib; do
            [ -f "$F" ] && [ ! -L "$F" ] || continue
            cp "$F" dist/Ollama.app/Contents/Resources/
        done
        (cd dist/Ollama.app/Contents/Resources && ln -sf libggml-base.0.0.0.dylib libggml-base.0.dylib && ln -sf libggml-base.0.dylib libggml-base.dylib) 2>/dev/null || true
    fi
    chmod a+x dist/Ollama.app/Contents/Resources/ollama

    # Sign
    if [ -n "$APPLE_IDENTITY" ]; then
        codesign -f --timestamp -s "$APPLE_IDENTITY" --identifier ai.ollama.ollama --options=runtime dist/Ollama.app/Contents/Resources/ollama
        for lib in dist/Ollama.app/Contents/Resources/*.so dist/Ollama.app/Contents/Resources/*.dylib dist/Ollama.app/Contents/Resources/*.metallib dist/Ollama.app/Contents/Resources/mlx_metal_v*/*.dylib dist/Ollama.app/Contents/Resources/mlx_metal_v*/*.metallib dist/Ollama.app/Contents/Resources/mlx_metal_v*/*.so; do
            [ -f "$lib" ] || continue
            codesign -f --timestamp -s "$APPLE_IDENTITY" --identifier ai.ollama.ollama --options=runtime "$lib"
        done
        codesign -f --timestamp -s "$APPLE_IDENTITY" --identifier com.electron.ollama --deep --options=runtime dist/Ollama.app
    fi

    rm -f dist/Ollama-darwin.zip
    ditto -c -k --norsrc --keepParent dist/Ollama.app dist/Ollama-darwin.zip
    (cd dist/Ollama.app/Contents/Resources/; tar -cf - ollama *.so *.dylib *.metallib mlx_metal_v*/ 2>/dev/null) | gzip -9vc > dist/ollama-darwin.tgz

    # Notarize and Staple
    if [ -n "$APPLE_IDENTITY" ]; then
        $(xcrun -f notarytool) submit dist/Ollama-darwin.zip --wait --timeout 20m --apple-id "$APPLE_ID" --password "$APPLE_PASSWORD" --team-id "$APPLE_TEAM_ID"
        rm -f dist/Ollama-darwin.zip
        $(xcrun -f stapler) staple dist/Ollama.app
        ditto -c -k --norsrc --keepParent dist/Ollama.app dist/Ollama-darwin.zip

        rm -f dist/Ollama.dmg

        (cd dist && ../scripts/create-dmg.sh \
            --volname "${VOL_NAME}" \
            --volicon ../app/darwin/Ollama.app/Contents/Resources/icon.icns \
            --background ../app/assets/background.png \
            --window-pos 200 120 \
            --window-size 800 400 \
            --icon-size 128 \
            --icon "Ollama.app" 200 190 \
            --hide-extension "Ollama.app" \
            --app-drop-link 600 190 \
            --text-size 12 \
            "Ollama.dmg" \
            "Ollama.app" \
        ; )
        rm -f dist/rw*.dmg

        codesign -f --timestamp -s "$APPLE_IDENTITY" --identifier ai.ollama.ollama --options=runtime dist/Ollama.dmg
        $(xcrun -f notarytool) submit dist/Ollama.dmg --wait --timeout 20m --apple-id "$APPLE_ID" --password "$APPLE_PASSWORD" --team-id "$APPLE_TEAM_ID"
        $(xcrun -f stapler) staple dist/Ollama.dmg
    else
        echo "WARNING: Code signing disabled, this bundle will not work for upgrade testing"
    fi
}

if [ "$#" -eq 0 ]; then
    _build_darwin
    _sign_darwin
    _build_macapp
    exit 0
fi

for CMD in "$@"; do
    case $CMD in
        build) _build_darwin ;;
        sign) _sign_darwin ;;
        app) _build_macapp ;;
        *) usage ;;
    esac
done
