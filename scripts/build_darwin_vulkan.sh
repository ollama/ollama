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
export CGO_CFLAGS="-mmacosx-version-min=14.0"
export CGO_CXXFLAGS="-mmacosx-version-min=14.0"
export CGO_LDFLAGS="-mmacosx-version-min=14.0"

set -e

PLISTBUDDY=${PLISTBUDDY:-/usr/libexec/PlistBuddy}

status() { echo >&2 ">>> $@"; }
usage() {
    echo "usage: $(basename $0) [build app [sign]]"
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
    for ARCH in $ARCHS; do
        status "Building darwin $ARCH"
        INSTALL_PREFIX=dist/darwin-$ARCH/
        GOOS=darwin GOARCH=$ARCH CGO_ENABLED=1 go build -o $INSTALL_PREFIX .
        if [[ $ARCH == "amd64" ]]; then
            OSXARCHBUILD="x86_64"
        else 
            OSXARCHBUILD="arm64"
        fi
        cmake --preset MoltenVK \
                -DCMAKE_OSX_ARCHITECTURES=$OSXARCHBUILD \
                -DCMAKE_INSTALL_PREFIX=$PWD/dist/darwin-$ARCH
        status "Building darwin $ARCH vulkan backend"
        cmake -B build/darwin-$ARCH \
            -DCMAKE_OSX_ARCHITECTURES=$OSXARCHBUILD \
            -DCMAKE_OSX_DEPLOYMENT_TARGET=11.3 \
            -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX
        cmake --build build/darwin-$ARCH --target ggml-cpu -j
        cmake --install build/darwin-$ARCH --component CPU
        cmake --build --preset MoltenVK
        cmake --install build --component Vulkan
    done
}

_sign_darwin() {
    status "Creating binary..."
    mkdir -p dist/darwin
    lipo -create -output dist/darwin/ollama dist/darwin-*/ollama
    chmod +x dist/darwin/ollama

    if [ -n "$APPLE_IDENTITY" ]; then
        for F in dist/darwin/ollama dist/darwin-amd64/lib/ollama/*; do
            codesign -f --timestamp -s "$APPLE_IDENTITY" --identifier ai.ollama.ollama --options=runtime $F
        done

        # create a temporary zip for notarization
        TEMP=$(mktemp -u).zip
        ditto -c -k --keepParent dist/darwin/ollama "$TEMP"
        xcrun notarytool submit "$TEMP" --wait --timeout 10m --apple-id $APPLE_ID --password $APPLE_PASSWORD --team-id $APPLE_TEAM_ID
        rm -f "$TEMP"
    fi

    status "Creating tarball..."
    tar -cf dist/ollama-darwin.tar --strip-components 2 dist/darwin/ollama
    tar -rf dist/ollama-darwin.tar --strip-components 4 dist/darwin-amd64/lib/
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
    local resources_path="dist/Ollama.app/Contents/Resources"
    if [ -d dist/darwin-amd64 ]; then
        lipo -create -output dist/Ollama.app/Contents/Resources/ollama dist/darwin-amd64/ollama dist/darwin-arm64/ollama
        cp -a dist/darwin-amd64/lib/ollama/. "$resources_path/"
    else
        cp -a dist/darwin/ollama dist/Ollama.app/Contents/Resources/ollama
        if [ -d dist/darwin/lib/ollama ]; then
            cp -a dist/darwin/lib/ollama/. "$resources_path/"
        else
            cp dist/darwin/*.so dist/darwin/*.dylib "$resources_path/"
        fi
    fi
    if [ -d "$resources_path/vulkan" ]; then
        cp -a "$resources_path/vulkan/." "$resources_path/"
        rm -rf "$resources_path/vulkan"
    fi
    chmod a+x dist/Ollama.app/Contents/Resources/ollama
    APP_BUNDLES="dist/Ollama.app"

    # Sign
    if [ -n "$APPLE_IDENTITY" ]; then
        for APP in $APP_BUNDLES; do
            codesign -f --timestamp -s "$APPLE_IDENTITY" --identifier ai.ollama.ollama --options=runtime "$APP/Contents/Resources/ollama"
            for lib in "$APP"/Contents/Resources/*.so "$APP"/Contents/Resources/*.dylib ; do
                [ -e "$lib" ] || continue
                codesign -f --timestamp -s "$APPLE_IDENTITY" --identifier ai.ollama.ollama --options=runtime ${lib}
            done
            codesign -f --timestamp -s "$APPLE_IDENTITY" --identifier com.electron.ollama --deep --options=runtime "$APP"
        done
    fi

    for APP in $APP_BUNDLES; do
        APP_NAME=$(basename "$APP" .app)
        ZIP_PATH="dist/${APP_NAME}-darwin.zip"
        rm -f "$ZIP_PATH"
        ditto -c -k --keepParent "$APP" "$ZIP_PATH"
    done

    (cd dist/Ollama.app/Contents/Resources/; tar -cf - ollama *.so *.dylib) | gzip -9vc > dist/ollama-darwin.tgz

    # Notarize and Staple
    if [ -n "$APPLE_IDENTITY" ]; then
        rm -rf dist/OllamaDisk
        mkdir -p dist/OllamaDisk
        for APP in $APP_BUNDLES; do
            cp -a "$APP" dist/OllamaDisk/
        done

        for APP in $APP_BUNDLES; do
            APP_NAME=$(basename "$APP" .app)
            ZIP_PATH="dist/${APP_NAME}-darwin.zip"
            $(xcrun -f notarytool) submit "$ZIP_PATH" --wait --timeout 10m --apple-id "$APPLE_ID" --password "$APPLE_PASSWORD" --team-id "$APPLE_TEAM_ID"
            rm -f "$ZIP_PATH"
            $(xcrun -f stapler) staple "$APP"
            ditto -c -k --keepParent "$APP" "$ZIP_PATH"
        done

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
            "OllamaDisk" \
        ; )
        rm -f dist/rw*.dmg

        codesign -f --timestamp -s "$APPLE_IDENTITY" --identifier ai.ollama.ollama --options=runtime dist/Ollama.dmg
        $(xcrun -f notarytool) submit dist/Ollama.dmg --wait --timeout 10m --apple-id "$APPLE_ID" --password "$APPLE_PASSWORD" --team-id "$APPLE_TEAM_ID"
        $(xcrun -f stapler) staple dist/Ollama.dmg
        rm -rf dist/OllamaDisk
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
