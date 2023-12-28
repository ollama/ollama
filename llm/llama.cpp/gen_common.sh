# common logic accross linux and darwin

init_vars() {
    LLAMACPP_DIR=gguf
    PATCHES="0001-Expose-callable-API-for-server.patch"
    CMAKE_DEFS="-DLLAMA_ACCELERATE=on"
    # TODO - LLAMA_K_QUANTS is stale and needs to be mapped to newer cmake settings
    CMAKE_TARGETS="--target ggml --target ggml_static --target llama --target build_info --target common --target ext_server --target llava_static"
    if echo "${CGO_CFLAGS}" | grep -- '-g' >/dev/null; then
        CMAKE_DEFS="-DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_VERBOSE_MAKEFILE=on -DLLAMA_GPROF=on -DLLAMA_SERVER_VERBOSE=on ${CMAKE_DEFS}"
    else
        # TODO - add additional optimization flags...
        CMAKE_DEFS="-DCMAKE_BUILD_TYPE=Release -DLLAMA_SERVER_VERBOSE=off ${CMAKE_DEFS}"
    fi
}

git_module_setup() {
    if [ -n "${OLLAMA_SKIP_PATCHING}" ]; then
        echo "Skipping submodule initialization"
        return
    fi
    git submodule init
    git submodule update --force gguf

}

apply_patches() {
    if [ -n "${OLLAMA_SKIP_PATCHING}" ]; then
        echo "Skipping submodule patching"
        return
    fi
    # Workaround git apply not handling creation well for iteration
    rm -f gguf/examples/server/server.h
    for patch in ${PATCHES}; do
        git -C gguf apply ../patches/${patch}
    done
}

build() {
    cmake -S ${LLAMACPP_DIR} -B ${BUILD_DIR} ${CMAKE_DEFS}
    cmake --build ${BUILD_DIR} ${CMAKE_TARGETS} -j8
}
