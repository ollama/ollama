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
    echo "usage: $(basename "$0") [-a archs] [build package app sign self-test]"
    exit 1
}

# NOTARY_PROFILE selects a notarytool keychain profile (the release Mac uses
# AC_PASSWORD). Without it, notarization uses APPLE_ID, APPLE_PASSWORD, and
# APPLE_TEAM_ID, which is what the GitHub workflow sets.
# OLLAMA_SKIP_NOTARIZE=1 still signs when APPLE_IDENTITY is set, and skips
# notarytool and stapler. scripts/release-fork.sh --build-only uses that.
_notarize() {
    if [ -n "${OLLAMA_SKIP_NOTARIZE:-}" ]; then
        status "skipping notarization of $1"
        return 0
    fi
    if [ -n "${NOTARY_PROFILE:-}" ]; then
        $(xcrun -f notarytool) submit "$1" --wait --timeout 20m --keychain-profile "$NOTARY_PROFILE"
    else
        $(xcrun -f notarytool) submit "$1" --wait --timeout 20m --apple-id "$APPLE_ID" --password "$APPLE_PASSWORD" --team-id "$APPLE_TEAM_ID"
    fi
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
    BUILD_CPUS=$(getconf _NPROCESSORS_ONLN)
    BUILD_JOBS=${OLLAMA_BUILD_PARALLEL:-$BUILD_CPUS}
    BUILD_LOAD=${OLLAMA_BUILD_LOAD:-$BUILD_CPUS}
    status "Build parallelism: $BUILD_JOBS, load limit: $BUILD_LOAD"

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
            cmake --build "$BUILD_DIR" --target ollama-local --target ollama-mlx-backends --parallel "$BUILD_JOBS" -- -l "$BUILD_LOAD"
        _assert_no_build_rpath "$INSTALL_PREFIX" "$BUILD_DIR"
    done
}

# CMake's Apple install invokes install_name_tool without checking its exit
# status. If a build-tree LC_RPATH survives, the binary only runs on this
# checkout. Fail the arch build instead of lipo-ing that into the app.
_assert_no_build_rpath() {
    root=$1
    build_dir=$2
    tool=${OTOOL:-otool}
    if [ ! -d "$root" ]; then
        echo "ERROR: $root was not installed" >&2
        return 1
    fi
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "ERROR: $tool is required to check installed rpaths under $root" >&2
        return 1
    fi
    list=$(mktemp)
    rpath_bad=0
    find "$root" -type f >"$list"
    while IFS= read -r f; do
        [ -n "$f" ] || continue
        info=$("$tool" -l "$f" 2>/dev/null) || continue
        printf '%s\n' "$info" | awk -v path="$f" -v build="$build_dir" '
            $1 == "cmd" && $2 == "LC_RPATH" { inr = 1; next }
            inr && $1 == "path" {
                if (index($2, build) > 0 || index($2, "/build/darwin-") > 0) {
                    printf "ERROR: %s still has build-tree LC_RPATH %s\n", path, $2 > "/dev/stderr"
                    printf "Remove %s and rebuild so the binary relinks with @loader_path.\n", build > "/dev/stderr"
                    bad = 1
                }
                inr = 0
            }
            END { exit bad ? 1 : 0 }
        ' || rpath_bad=1
        [ "$rpath_bad" -eq 0 ] || break
    done <"$list"
    rm -f "$list"
    [ "$rpath_bad" -eq 0 ] || return 1
}

# Xcode 27 lipo (dyld other-tools) accepts one architecture per -verify_arch.
# `lipo file -verify_arch x86_64 arm64` treats arm64 as a second input file and
# exits with "-verify_arch requires exactly one input file". The universal
# file from -create is fine; the check was the release failure.
_lipo_universal() {
    out=$1
    shift
    for input in "$@"; do
        if [ ! -f "$input" ]; then
            echo "ERROR: universal build is missing $input" >&2
            return 1
        fi
    done
    mkdir -p "$(dirname "$out")"
    lipo -create -output "$out" "$@"
    chmod +x "$out"
    lipo "$out" -verify_arch x86_64
    lipo "$out" -verify_arch arm64
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

        for LIB in libmlx.dylib libmlxc.dylib libollama_xgrammar.dylib; do
            if [ -f "$AMD_VARIANT/$LIB" ] && [ -f "$VARIANT$LIB" ]; then
                _lipo_universal "$DEST/$LIB" "$AMD_VARIANT/$LIB" "$VARIANT$LIB"
            elif [ -f "$VARIANT$LIB" ]; then
                cp "$VARIANT$LIB" "$DEST/"
            elif [ -f "$AMD_VARIANT/$LIB" ]; then
                cp "$AMD_VARIANT/$LIB" "$DEST/"
            fi
        done

        for F in "$VARIANT"*; do
            [ -f "$F" ] && [ ! -L "$F" ] || continue
            case "$(basename "$F")" in
                libmlx.dylib|libmlxc.dylib|libollama_xgrammar.dylib) continue ;;
            esac
            cp "$F" "$DEST/"
        done
    done
}

_prepare_darwin_runtime() {
    status "Creating universal binary..."
    _lipo_universal dist/darwin/ollama \
        dist/darwin-amd64/ollama dist/darwin-arm64/ollama
    _lipo_universal dist/darwin/llama-server \
        dist/darwin-amd64/lib/ollama/llama-server \
        dist/darwin-arm64/lib/ollama/llama-server
    _lipo_universal dist/darwin/llama-quantize \
        dist/darwin-amd64/lib/ollama/llama-quantize \
        dist/darwin-arm64/lib/ollama/llama-quantize
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
            case "$F" in *_LICENSE|*_NOTICE) continue ;; esac
            codesign -f --timestamp -s "$APPLE_IDENTITY" --identifier ai.ollama.ollama --options=runtime "$F"
        done

        if [ -z "${OLLAMA_SKIP_NOTARIZE:-}" ]; then
            # create a temporary zip for notarization; _notarize uses
            # NOTARY_PROFILE when set, else APPLE_ID/APPLE_PASSWORD/APPLE_TEAM_ID
            TEMP=$(mktemp -u).zip
            ditto -c -k --keepParent dist/darwin/ollama "$TEMP"
            _notarize "$TEMP"
            rm -f "$TEMP"
        fi
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
    _lipo_universal dist/Ollama.app/Contents/MacOS/Ollama dist/darwin-app-amd64 dist/darwin-app-arm64
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
    (cd dist/Ollama.app/Contents/Resources/; tar -cf - ollama llama-server llama-quantize *.so *.dylib *.metallib *_LICENSE *_NOTICE mlx_metal_v*/ 2>/dev/null) | gzip -9vc > dist/ollama-darwin.tgz

    # Notarize and Staple. --build-only signs when it can and skips this.
    if [ -n "$APPLE_IDENTITY" ] && [ -z "${OLLAMA_SKIP_NOTARIZE:-}" ]; then
        _notarize dist/Ollama-darwin.zip
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
        _notarize dist/Ollama.dmg
        $(xcrun -f stapler) staple dist/Ollama.dmg
    elif [ -n "$APPLE_IDENTITY" ]; then
        echo "WARNING: Notarization skipped. dist/Ollama.app is signed but not stapled."
    else
        echo "WARNING: Code signing disabled, this bundle will not work for upgrade testing"
    fi
}

# Linux-runnable stand-in for Xcode 27 lipo and for the rpath check.
_self_test() {
    root=$(CDPATH='' cd -- "$(dirname "$0")/.." && pwd)
    if ! grep -q 'CMAKE_BUILD_WITH_INSTALL_RPATH ON' "$root/llama/server/CMakeLists.txt"; then
        echo "self-test: llama/server is missing CMAKE_BUILD_WITH_INSTALL_RPATH" >&2
        return 1
    fi

    tmp=$(mktemp -d)
    log="$tmp/lipo.log"
    cat >"$tmp/lipo" <<'EOF'
#!/bin/sh
printf '%s\n' "$*" >>"$LIPO_LOG"
out=
cmd=
verify=
prev=
inputs=
for arg in "$@"; do
    case "$prev" in
        -output|-o)
            out=$arg
            prev=
            continue
            ;;
        -verify_arch)
            verify=$arg
            prev=
            continue
            ;;
    esac
    case "$arg" in
        -create|-archs|-info)
            cmd=$arg
            ;;
        -output|-o|-verify_arch)
            prev=$arg
            ;;
        -*)
            echo "lipo: unknown option $arg" >&2
            exit 1
            ;;
        *)
            inputs="$inputs $arg"
            ;;
    esac
done
# shellcheck disable=SC2086
set -- $inputs
if [ -n "$verify" ]; then
    if [ "$#" -ne 1 ]; then
        echo "lipo: -verify_arch requires exactly one input file" >&2
        echo "usage: lipo <input_file> <command> [<options> ...]" >&2
        exit 1
    fi
    exit 0
fi
if [ "$cmd" = "-create" ]; then
    [ -n "$out" ] || exit 1
    [ "$#" -ge 1 ] || exit 1
    for f in "$@"; do
        [ -f "$f" ] || {
            echo "lipo: missing $f" >&2
            exit 1
        }
    done
    : >"$out"
    exit 0
fi
echo "lipo: unhandled" >&2
exit 1
EOF
    chmod +x "$tmp/lipo"
    cat >"$tmp/otool" <<'EOF'
#!/bin/sh
file=
prev=
for arg in "$@"; do
    case "$prev" in
        -l) file=$arg; prev= ;;
    esac
    case "$arg" in
        -l) prev=-l ;;
    esac
done
case "$file" in
    *bad-rpath*)
        printf '%s\n' "cmd LC_RPATH" "path /tmp/build/darwin-amd64/llama-server-local/bin (offset 12)"
        ;;
    *)
        printf '%s\n' "cmd LC_RPATH" "path @loader_path (offset 12)"
        ;;
esac
exit 0
EOF
    chmod +x "$tmp/otool"

    old_path=$PATH
    old_otool=${OTOOL:-}
    PATH="$tmp:$PATH"
    LIPO_LOG=$log
    export LIPO_LOG
    OTOOL="$tmp/otool"
    export OTOOL

    work=$tmp/work
    mkdir -p "$work/in"
    : >"$work/in/amd64"
    : >"$work/in/arm64"
    _lipo_universal "$work/out/ollama" "$work/in/amd64" "$work/in/arm64"
    if grep -q -- '-verify_arch x86_64 arm64' "$log"; then
        echo "self-test: verify_arch was passed two architectures in one lipo call" >&2
        return 1
    fi
    verifies=$(grep -c -- '-verify_arch ' "$log" || true)
    if [ "$verifies" -ne 2 ]; then
        echo "self-test: expected 2 verify_arch calls, got $verifies" >&2
        cat "$log" >&2
        return 1
    fi
    miss=$tmp/missing.err
    if _lipo_universal "$work/out/missing" "$work/in/no-such-arch" "$work/in/arm64" 2>"$miss"; then
        echo "self-test: missing arch input was accepted" >&2
        return 1
    fi
    if ! grep -q "missing $work/in/no-such-arch" "$miss"; then
        echo "self-test: missing-input error did not name the file" >&2
        cat "$miss" >&2
        return 1
    fi

    mkdir -p "$work/good/lib"
    : >"$work/good/lib/libok.dylib"
    _assert_no_build_rpath "$work/good" "build/darwin-amd64"
    mkdir -p "$work/bad/lib"
    : >"$work/bad/lib/bad-rpath.dylib"
    bad=$tmp/bad-rpath.err
    if _assert_no_build_rpath "$work/bad" "build/darwin-amd64" 2>"$bad"; then
        echo "self-test: build-tree LC_RPATH was accepted" >&2
        return 1
    fi
    if ! grep -q "build-tree LC_RPATH" "$bad"; then
        echo "self-test: rpath error did not name the load command" >&2
        cat "$bad" >&2
        return 1
    fi

    PATH=$old_path
    if [ -n "$old_otool" ]; then
        OTOOL=$old_otool
    else
        unset OTOOL
    fi
    unset LIPO_LOG
    rm -rf "$tmp"
    echo "build_darwin self-test ok"
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
        self-test) _self_test ;;
        *) usage ;;
    esac
done
