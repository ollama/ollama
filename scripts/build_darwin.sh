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
export CGO_CFLAGS="-O3 -mmacosx-version-min=14.0"
export CGO_CXXFLAGS="-O3 -mmacosx-version-min=14.0"
export CGO_LDFLAGS="-mmacosx-version-min=14.0"

set -e

status() { echo >&2 ">>> $@"; }
usage() {
    echo "usage: $(basename $0) [build package app sign]"
    exit 1
}

mkdir -p dist

ARCHS="arm64 amd64"
while getopts "a:h" OPTION; do
    case $OPTION in
        a) ARCHS=$OPTARG ;;
        h) usage ;;
    esac
done

shift $(( $OPTIND - 1 ))

_build_darwin() {
    SOURCE_BUILD=build/darwin-sources
    status "Preparing shared native sources"
    cmake -S . -B "$SOURCE_BUILD" -DOLLAMA_MLX_BACKENDS=metal_v3 -DOLLAMA_LLAMA_BACKENDS=
    cmake --build "$SOURCE_BUILD" --target ollama-llama-cpp-source --target ollama-mlx-sources
    LLAMA_CPP_SHARED_SRC="$(pwd)/$SOURCE_BUILD/_deps/llama_cpp-src"
    MLX_SHARED_SRC="$(pwd)/$SOURCE_BUILD/_deps/mlx-src"
    MLX_C_SHARED_SRC="$(pwd)/$SOURCE_BUILD/_deps/mlx-c-src"

    for ARCH in $ARCHS; do
        status "Building darwin $ARCH"
        INSTALL_PREFIX=dist/darwin-$ARCH/
        BUILD_DIR=build/darwin-$ARCH

        if [ "$ARCH" = "amd64" ]; then
            CMAKE_ARCH=x86_64
            MLX_BACKENDS=metal_v3
            MLX_EXTRA_ARGS="-DMLX_ENABLE_X64_MAC=ON"
            MLX_CGO_CFLAGS="-O3 -mmacosx-version-min=14.0"
            MLX_CGO_LDFLAGS="-ldl -lc++ -framework Accelerate -mmacosx-version-min=14.0"
        else
            CMAKE_ARCH=arm64
            MLX_BACKENDS="metal_v3;metal_v4"
            MLX_EXTRA_ARGS=
            MLX_CGO_CFLAGS="-O3 -mmacosx-version-min=14.0"
            MLX_CGO_LDFLAGS="-lc++ -framework Metal -framework Foundation -framework Accelerate -mmacosx-version-min=14.0"
        fi

        cmake -S . -B "$BUILD_DIR" \
            -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_OSX_ARCHITECTURES=$CMAKE_ARCH \
            -DCMAKE_OSX_DEPLOYMENT_TARGET=14.0 \
            -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX \
            -DOLLAMA_PAYLOAD_INSTALL_PREFIX=$INSTALL_PREFIX \
            -DOLLAMA_GO_OUTPUT=$INSTALL_PREFIX/ollama \
            -DOLLAMA_VERSION="$VERSION" \
            -DOLLAMA_MLX_BACKENDS="$MLX_BACKENDS" \
            -DOLLAMA_LLAMA_BACKENDS= \
            -DFETCHCONTENT_SOURCE_DIR_LLAMA_CPP=$LLAMA_CPP_SHARED_SRC \
            -DFETCHCONTENT_SOURCE_DIR_MLX=$MLX_SHARED_SRC \
            -DFETCHCONTENT_SOURCE_DIR_MLX-C=$MLX_C_SHARED_SRC \
            $MLX_EXTRA_ARGS

        GOOS=darwin GOARCH=$ARCH CGO_ENABLED=1 CGO_CFLAGS="$MLX_CGO_CFLAGS" CGO_LDFLAGS="$MLX_CGO_LDFLAGS" \
            cmake --build "$BUILD_DIR" --target ollama-local --target ollama-mlx-backends --parallel
    done
}

_merge_darwin_payload() {
    status "Preparing universal Darwin runtime payload"
    rm -rf dist/darwin/lib
    mkdir -p dist/darwin/lib/ollama

    for ROOT in dist/darwin-amd64/lib/ollama dist/darwin-arm64/lib/ollama; do
        [ -d "$ROOT" ] || continue
        for F in "$ROOT"/*; do
            [ -e "$F" ] || continue
            BASE=$(basename "$F")
            case "$BASE" in
                llama-server|llama-quantize|mlx_*) continue ;;
            esac
            [ -e "dist/darwin/lib/ollama/$BASE" ] || cp -P "$F" dist/darwin/lib/ollama/
        done
    done

    for VARIANT in dist/darwin-arm64/lib/ollama/mlx_metal_v*/; do
        [ -d "$VARIANT" ] || continue
        VNAME=$(basename "$VARIANT")
        DEST=dist/darwin/lib/ollama/$VNAME
        AMD_VARIANT=dist/darwin-amd64/lib/ollama/$VNAME
        [ -d "$AMD_VARIANT" ] || AMD_VARIANT=dist/darwin-amd64/lib/ollama
        mkdir -p "$DEST"

        for LIB in libmlx.dylib libmlxc.dylib; do
            if [ -f "$AMD_VARIANT/$LIB" ] && [ -f "$VARIANT$LIB" ]; then
                lipo -create -output "$DEST/$LIB" "$AMD_VARIANT/$LIB" "$VARIANT$LIB"
            elif [ -f "$VARIANT$LIB" ]; then
                cp "$VARIANT$LIB" "$DEST/"
            elif [ -f "$AMD_VARIANT/$LIB" ]; then
                cp "$AMD_VARIANT/$LIB" "$DEST/"
            fi
        done

        for F in "$VARIANT"*; do
            [ -f "$F" ] && [ ! -L "$F" ] || continue
            case "$(basename "$F")" in
                libmlx.dylib|libmlxc.dylib) continue ;;
            esac
            cp "$F" "$DEST/"
        done
    done
}

_prepare_darwin_runtime() {
    status "Creating universal binary..."
    mkdir -p dist/darwin
    lipo -create -output dist/darwin/ollama dist/darwin-amd64/ollama dist/darwin-arm64/ollama
    chmod +x dist/darwin/ollama
    lipo dist/darwin/ollama -verify_arch x86_64 arm64

    lipo -create -output dist/darwin/llama-server dist/darwin-amd64/lib/ollama/llama-server dist/darwin-arm64/lib/ollama/llama-server
    chmod +x dist/darwin/llama-server
    lipo dist/darwin/llama-server -verify_arch x86_64 arm64

    lipo -create -output dist/darwin/llama-quantize dist/darwin-amd64/lib/ollama/llama-quantize dist/darwin-arm64/lib/ollama/llama-quantize
    chmod +x dist/darwin/llama-quantize
    lipo dist/darwin/llama-quantize -verify_arch x86_64 arm64

    _merge_darwin_payload
}

_create_darwin_runtime_tarball() {
    status "Creating universal tarball..."
    rm -f dist/ollama-darwin.tar dist/ollama-darwin.tgz
    tar -cf dist/ollama-darwin.tar --strip-components 2 dist/darwin/ollama dist/darwin/llama-server dist/darwin/llama-quantize
    tar -rf dist/ollama-darwin.tar --strip-components 4 dist/darwin/lib/ollama
    gzip -9vc <dist/ollama-darwin.tar >dist/ollama-darwin.tgz
}

_package_darwin_runtime() {
    _prepare_darwin_runtime
    _create_darwin_runtime_tarball
}

_sign_darwin() {
    _prepare_darwin_runtime
    if [ -n "$APPLE_IDENTITY" ]; then
        for F in dist/darwin/ollama dist/darwin/llama-server dist/darwin/llama-quantize dist/darwin/lib/ollama/* dist/darwin/lib/ollama/mlx_metal_v*/*; do
            [ -f "$F" ] && [ ! -L "$F" ] || continue
            codesign -f --timestamp -s "$APPLE_IDENTITY" --identifier ai.ollama.ollama --options=runtime "$F"
        done

        # create a temporary zip for notarization
        TEMP=$(mktemp -u).zip
        ditto -c -k --keepParent dist/darwin/ollama "$TEMP"
        xcrun notarytool submit "$TEMP" --wait --timeout 20m --apple-id $APPLE_ID --password $APPLE_PASSWORD --team-id $APPLE_TEAM_ID
        rm -f "$TEMP"
    fi

    _create_darwin_runtime_tarball
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
    [ -d dist/darwin/lib/ollama ] || _merge_darwin_payload
    cp -a dist/darwin/ollama dist/Ollama.app/Contents/Resources/ollama
    cp dist/darwin/llama-server dist/Ollama.app/Contents/Resources/
    cp dist/darwin/llama-quantize dist/Ollama.app/Contents/Resources/
    if [ -d dist/darwin/lib/ollama ]; then
        cp -a dist/darwin/lib/ollama/. dist/Ollama.app/Contents/Resources/
    fi
    chmod a+x dist/Ollama.app/Contents/Resources/ollama

    # Sign
    if [ -n "$APPLE_IDENTITY" ]; then
        codesign -f --timestamp -s "$APPLE_IDENTITY" --identifier ai.ollama.ollama --options=runtime dist/Ollama.app/Contents/Resources/ollama
        codesign -f --timestamp -s "$APPLE_IDENTITY" --identifier ai.ollama.ollama --options=runtime dist/Ollama.app/Contents/Resources/llama-server
        codesign -f --timestamp -s "$APPLE_IDENTITY" --identifier ai.ollama.ollama --options=runtime dist/Ollama.app/Contents/Resources/llama-quantize
        for lib in dist/Ollama.app/Contents/Resources/*.so dist/Ollama.app/Contents/Resources/*.dylib dist/Ollama.app/Contents/Resources/*.metallib dist/Ollama.app/Contents/Resources/mlx_metal_v*/*.dylib dist/Ollama.app/Contents/Resources/mlx_metal_v*/*.metallib dist/Ollama.app/Contents/Resources/mlx_metal_v*/*.so; do
            [ -f "$lib" ] || continue
            codesign -f --timestamp -s "$APPLE_IDENTITY" --identifier ai.ollama.ollama --options=runtime "$lib"
        done
        codesign -f --timestamp -s "$APPLE_IDENTITY" --identifier com.electron.ollama --deep --options=runtime dist/Ollama.app
    fi

    rm -f dist/Ollama-darwin.zip
    ditto -c -k --norsrc --keepParent dist/Ollama.app dist/Ollama-darwin.zip
    (cd dist/Ollama.app/Contents/Resources/; tar -cf - ollama llama-server llama-quantize *.so *.dylib *.metallib mlx_metal_v*/ 2>/dev/null) | gzip -9vc > dist/ollama-darwin.tgz

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
        package) _package_darwin_runtime ;;
        sign) _sign_darwin ;;
        app) _build_macapp ;;
        *) usage ;;
    esac
done
