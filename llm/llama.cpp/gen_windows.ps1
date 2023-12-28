#!powershell

$ErrorActionPreference = "Stop"

function init_vars {
    $script:patches = @("0001-Expose-callable-API-for-server.patch")
    $script:cmakeDefs = @("-DBUILD_SHARED_LIBS=on", "-DLLAMA_NATIVE=off", "-DLLAMA_F16C=off", "-DLLAMA_FMA=off", "-DLLAMA_AVX512=off", "-DLLAMA_AVX2=off", "-DLLAMA_AVX=on", "-DLLAMA_K_QUANTS=on", "-DLLAMA_ACCELERATE=on", "-A","x64")

    if ($env:CGO_CFLAGS -contains "-g") {
        $script:cmakeDefs += @("-DCMAKE_VERBOSE_MAKEFILE=on", "-DLLAMA_SERVER_VERBOSE=on")
        $script:config = "RelWithDebInfo"
    } else {
        $script:cmakeDefs += @("-DLLAMA_SERVER_VERBOSE=off")
        $script:config = "Release"
    }
}

function git_module_setup {
    # TODO add flags to skip the init/patch logic to make it easier to mod llama.cpp code in-repo
    & git submodule init
    if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
    & git submodule update --force gguf
    if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
}

function apply_patches {
    rm -erroraction ignore -path "gguf/examples/server/server.h"
    foreach ($patch in $script:patches) {
        write-host "Applying patch $patch"
        & git -C gguf apply ../patches/$patch
        if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
    }
}

function build {
    write-host "generating config with: cmake -S gguf -B $script:buildDir $script:cmakeDefs"
    & cmake --version
    & cmake -S gguf -B $script:buildDir $script:cmakeDefs
    if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
    write-host "building with: cmake --build $script:buildDir --config $script:config"
    & cmake --build $script:buildDir --config $script:config
    if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
}

function install {
    rm -erroraction ignore -recurse -force -path $script:installDir
    & cmake --install $script:buildDir --prefix $script:installDir --config $script:config
    if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}

}

init_vars
git_module_setup
apply_patches

# first build CPU based
$script:buildDir="gguf/build/wincpu"
$script:installDir="gguf/build/wincpu/dist"

build
# install

md gguf/build/lib -ea 0
md gguf/build/wincpu/dist/lib -ea 0
mv gguf/build/wincpu/bin/$script:config/ext_server_shared.dll gguf/build/wincpu/dist/lib/cpu_server.dll


# Nope, this barfs on lots of symbol problems
#mv gguf/build/wincpu/examples/server/$script:config/ext_server_shared.dll gguf/build/wincpu/dist/lib/cpu_server.lib
# Nope: this needs lots of include paths to pull in things like msvcprt.lib and other deps
# & cl.exe `
#     gguf/build/wincpu/examples/server/$script:config/ext_server.lib `
#     gguf/build/wincpu/common/$script:config/common.lib `
#     gguf/build/wincpu/$script:config/llama.lib `
#     gguf/build/wincpu/$script:config/ggml_static.lib `
#     /link /DLL /DEF:cpu_server.def /NOENTRY /MACHINE:X64  /OUT:gguf/build/wincpu/dist/lib/cpu_server.dll
# if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}

# Then build cuda as a dynamically loaded library
init_vars
$script:buildDir="gguf/build/wincuda"
$script:installDir="gguf/build/wincuda/dist"
$script:cmakeDefs += @("-DLLAMA_CUBLAS=ON", "-DBUILD_SHARED_LIBS=on")
build
install
cp gguf/build/wincuda/dist/bin/ext_server_shared.dll gguf/build/lib/cuda_server.dll

# TODO - more to do here to create a usable dll


# TODO - implement ROCm support on windows
md gguf/build/winrocm/lib -ea 0
echo $null >> gguf/build/winrocm/lib/.generated
