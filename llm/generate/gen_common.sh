# common logic across linux and darwin

init_vars() {
    case "${GOARCH}" in
    "amd64")
        ARCH="x86_64"
        ;;
    "arm64")
        ARCH="arm64"
        ;;
    *)
        echo "GOARCH must be set"
        echo "this script is meant to be run from within go generate"
        exit 1
        ;;
    esac

    LLAMACPP_DIR=../llama.cpp
    CMAKE_DEFS="-DCMAKE_SKIP_RPATH=on"
    CMAKE_TARGETS="--target ollama_llama_server"
    if echo "${CGO_CFLAGS}" | grep -- '-g' >/dev/null; then
        CMAKE_DEFS="-DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_VERBOSE_MAKEFILE=on -DLLAMA_GPROF=on -DLLAMA_SERVER_VERBOSE=on ${CMAKE_DEFS}"
    else
        # TODO - add additional optimization flags...
        CMAKE_DEFS="-DCMAKE_BUILD_TYPE=Release -DLLAMA_SERVER_VERBOSE=off ${CMAKE_DEFS}"
    fi
    case $(uname -s) in
    "Darwin")
        LIB_EXT="dylib"
        WHOLE_ARCHIVE="-Wl,-force_load"
        NO_WHOLE_ARCHIVE=""
        GCC_ARCH="-arch ${ARCH}"
        DIST_BASE=../../dist/darwin-${GOARCH}/
        PAYLOAD_BASE=../../build/darwin/${GOARCH}
        ;;
    "Linux")
        LIB_EXT="so"
        WHOLE_ARCHIVE="-Wl,--whole-archive"
        NO_WHOLE_ARCHIVE="-Wl,--no-whole-archive"

        # Cross compiling not supported on linux - Use docker
        GCC_ARCH=""
        DIST_BASE=../../dist/linux-${GOARCH}/
        PAYLOAD_BASE=../../build/linux/${GOARCH}
        ;;
    *)
        ;;
    esac
    if [ -z "${CMAKE_CUDA_ARCHITECTURES}" ] ; then
        CMAKE_CUDA_ARCHITECTURES="50;52;61;70;75;80"
    fi
    GZIP=$(command -v pigz 2>/dev/null || echo "gzip")
    RUNNER_BASE="${DIST_BASE}/lib/ollama/runners"
}

git_module_setup() {
    if [ -n "${OLLAMA_SKIP_PATCHING}" ]; then
        echo "Skipping submodule initialization"
        return
    fi
    # Make sure the tree is clean after the directory moves
    if [ -d "${LLAMACPP_DIR}/gguf" ]; then
        echo "Cleaning up old submodule"
        rm -rf ${LLAMACPP_DIR}
    fi
    git submodule init
    git submodule update --force ${LLAMACPP_DIR}

}

apply_patches() {
    # apply temporary patches until fix is upstream
    for patch in ../patches/*.patch; do
        git -c 'user.name=nobody' -c 'user.email=<>' -C ${LLAMACPP_DIR} am ${patch}
    done
}

build() {
    cmake -S ${LLAMACPP_DIR} -B ${BUILD_DIR} ${CMAKE_DEFS}
    cmake --build ${BUILD_DIR} ${CMAKE_TARGETS} -j8
    # remove unnecessary build artifacts
    rm -f ${BUILD_DIR}/bin/ggml-common.h ${BUILD_DIR}/bin/ggml-metal.metal
}

dist() {
    [ -z "${RUNNER}" ] && exit 1
    mkdir -p ${RUNNER_BASE}/${RUNNER}/
    for f in ${BUILD_DIR}/bin/* ; do
        cp ${f} ${RUNNER_BASE}/${RUNNER}/
    done
    # check for lib directory
    if [ -d ${BUILD_DIR}/lib ]; then
        for f in ${BUILD_DIR}/lib/* ; do
            cp ${f} ${RUNNER_BASE}/${RUNNER}/
        done
    fi
}

# Compress from the build $BUILD_DIR into the $PAYLOAD_BASE/$RUNNER dir
compress() {
    [ -z "${RUNNER}" ] && exit 1
    echo "Compressing payloads with ${GZIP} to reduce overall binary size..."
    rm -rf "${PAYLOAD_BASE}/${RUNNER}/"
    mkdir -p "${PAYLOAD_BASE}/${RUNNER}/"
    for f in ${BUILD_DIR}/bin/* ; do
        ${GZIP} -c --best ${f} > "${PAYLOAD_BASE}/${RUNNER}/$(basename ${f}).gz" &
        compress_pids+=" $!"
    done
    # check for lib directory
    if [ -d ${BUILD_DIR}/lib ]; then
        for f in ${BUILD_DIR}/lib/* ; do
            ${GZIP} -c --best ${f} > "${PAYLOAD_BASE}/${RUNNER}/$(basename ${f}).gz" &
            compress_pids+=" $!"
        done
    fi
    echo
}

wait_for_compress() {
    for pid in ${compress_pids}; do
        wait $pid
    done
    echo "Finished compression"
}

install() {
    echo "Installing libraries to bin dir ${BUILD_DIR}/bin/"
    for lib in $(find ${BUILD_DIR} -name \*.${LIB_EXT} | grep -v "${BUILD_DIR}/bin/" ); do
        rm -f "${BUILD_DIR}/bin/$(basename ${lib})"
        cp -af "${lib}" "${BUILD_DIR}/bin/"
    done
}

# Keep the local tree clean after we're done with the build
cleanup() {
    git submodule update --force ${LLAMACPP_DIR}
}
