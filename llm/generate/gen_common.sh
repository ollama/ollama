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
        ARCH=$(uname -m | sed -e "s/aarch64/arm64/g")
    esac

    LLAMACPP_DIR=../llama.cpp
    NEURAL_SPEED_DIR=../neural_speed
    CMAKE_DEFS=""
    CMAKE_TARGETS="--target ext_server"
    NS_CMAKE_TARGETS="--target ns_ext_server"
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
        ;;
    "Linux")
        LIB_EXT="so"
        WHOLE_ARCHIVE="-Wl,--whole-archive"
        NO_WHOLE_ARCHIVE="-Wl,--no-whole-archive"

        # Cross compiling not supported on linux - Use docker
        GCC_ARCH=""
        ;;
    *)
        ;;
    esac
    if [ -z "${CMAKE_CUDA_ARCHITECTURES}" ] ; then 
        CMAKE_CUDA_ARCHITECTURES="50;52;61;70;75;80"
    fi
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
    git submodule sync
    git submodule update --recursive --remote --force ${LLAMACPP_DIR} ${NEURAL_SPEED_DIR}

}

apply_patches() {
    # Wire up our CMakefile
    if ! grep ollama ${LLAMACPP_DIR}/CMakeLists.txt; then
        echo 'add_subdirectory(../ext_server ext_server) # ollama' >>${LLAMACPP_DIR}/CMakeLists.txt
    fi
    if ! grep ollama ${NEURAL_SPEED_DIR}/CMakeLists.txt; then
        echo 'add_subdirectory(../ns_ext_server ns_ext_server) # ollama' >>${NEURAL_SPEED_DIR}/CMakeLists.txt
    fi

    for submodule in ${LLAMACPP_DIR} ${NEURAL_SPEED_DIR}; do
        dir=$(basename $submodule)
        if [ -n "$(ls -A ../patches/${dir}/*.diff)" ]; then
            # apply temporary patches until fix is upstream
            (cd ../${dir}; git clean -f -d)
            for patch in ../patches/${dir}/*.diff; do
                (cd ${submodule} && git apply ${patch})
            done
        fi
    done
}

build() {
    if [ -z "$1" -o "$1" = "llama.cpp" ]; then 
        cmake -S ${LLAMACPP_DIR} -B ${BUILD_DIR} ${CMAKE_DEFS}
        cmake --build ${BUILD_DIR} ${CMAKE_TARGETS} -j8
        mkdir -p ${BUILD_DIR}/lib/
        ls ${BUILD_DIR}
        g++ -fPIC -g -shared -o ${BUILD_DIR}/lib/libext_server.${LIB_EXT} \
            ${GCC_ARCH} \
            ${WHOLE_ARCHIVE} ${BUILD_DIR}/ext_server/libext_server.a ${NO_WHOLE_ARCHIVE} \
            ${BUILD_DIR}/common/libcommon.a \
            ${BUILD_DIR}/libllama.a \
            -Wl,-rpath,\$ORIGIN \
            -lpthread -ldl -lm \
            ${EXTRA_LIBS}
    fi

    if [ "$1" = "neural_speed" ]; then 
        cmake -S ${NEURAL_SPEED_DIR} -B ${BUILD_DIR} ${CMAKE_DEFS}  
        cmake --build ${BUILD_DIR} ${NS_CMAKE_TARGETS} -j8
        mkdir -p ${BUILD_DIR}/lib/
        ls ${BUILD_DIR}
        g++ -fPIC -g -shared -o ${BUILD_DIR}/lib/lib_ext_server.${LIB_EXT} \
            ${GCC_ARCH} \
            ${WHOLE_ARCHIVE} ${BUILD_DIR}/lib/libns_ext_server.a ${NO_WHOLE_ARCHIVE} \
            -Wl,-rpath,\$ORIGIN \
            ${EXTRA_LIBS}
    fi
}

compress_libs() {
    echo "Compressing payloads to reduce overall binary size..."
    pids=""
    rm -rf ${BUILD_DIR}/lib/*.${LIB_EXT}*.gz
    for lib in ${BUILD_DIR}/lib/*.${LIB_EXT}* ; do
        gzip -n --best -f ${lib} &
        pids+=" $!"
    done
    echo 
    for pid in ${pids}; do
        wait $pid
    done
    echo "Finished compression"
}

# Keep the local tree clean after we're done with the build
cleanup() {
    (cd ${LLAMACPP_DIR}/ && git checkout -- .)
    (cd ${NEURAL_SPEED_DIR}/ && git clean -f -d)
}
