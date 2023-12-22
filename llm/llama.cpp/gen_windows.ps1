#!powershell

$ErrorActionPreference = "Stop"

function init_vars {
    $script:patches = @("0001-Expose-callable-API-for-server.patch")
    $script:cmakeDefs = @("-DBUILD_SHARED_LIBS=on", "-DLLAMA_NATIVE=off", "-DLLAMA_F16C=off", "-DLLAMA_FMA=off", "-DLLAMA_AVX512=off", "-DLLAMA_AVX2=off", "-DLLAMA_AVX=on", "-DLLAMA_K_QUANTS=on", "-DLLAMA_ACCELERATE=on", "-A","x64")
    $script:cmakeTargets = @("ggml", "ggml_static", "llama", "build_info", "common", "ext_server_shared", "llava_static")
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
    # Wire up our CMakefile
    if (!(Select-String -Path "gguf/examples/server/CMakeLists.txt" -Pattern 'ollama.txt')) {
        Add-Content -Path "gguf/examples/server/CMakeLists.txt" -Value 'include (../../../ollama.txt)'
    }
    # Avoid duplicate main symbols when we link into the cgo binary
    $content = Get-Content -Path "./gguf/examples/server/server.cpp"
    $content = $content -replace 'int main\(', 'int __main('
    Set-Content -Path "./gguf/examples/server/server.cpp" -Value $content
}

function build {
    write-host "generating config with: cmake -S gguf -B $script:buildDir $script:cmakeDefs"
    & cmake --version
    & cmake -S gguf -B $script:buildDir $script:cmakeDefs
    if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
    write-host "building with: cmake --build $script:buildDir --config $script:config ($script:cmakeTargets | ForEach-Object { "--target", $_ })"
    & cmake --build $script:buildDir --config $script:config ($script:cmakeTargets | ForEach-Object { "--target", $_ })
    if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
}

function cleanup {
    Set-Location "gguf/examples/server"
    git checkout CMakeLists.txt server.cpp
}

init_vars
git_module_setup
apply_patches

# first build CPU based
$script:buildDir="gguf/build/wincpu"

build
# install

md gguf/build/lib -ea 0
md gguf/build/wincpu/dist/lib -ea 0
cp -force gguf/build/wincpu/bin/$script:config/ext_server_shared.dll gguf/build/lib/ext_server_shared.dll
cp -force gguf/build/wincpu/bin/$script:config/llama.dll gguf/build/lib/llama.dll

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
$script:cmakeDefs += @("-DLLAMA_CUBLAS=ON", "-DBUILD_SHARED_LIBS=on")
build
# install
cp -force gguf/build/wincuda/bin/$script:config/ext_server_shared.dll gguf/build/lib/cuda_server.dll

# TODO - more to do here to create a usable dll


# TODO - implement ROCm support on windows
md gguf/build/winrocm/lib -ea 0
echo $null >> gguf/build/winrocm/lib/.generated

cleanup

write-host "go generate completed"