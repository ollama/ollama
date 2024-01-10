#!powershell

$ErrorActionPreference = "Stop"

function init_vars {
    $script:llamacppDir = "../llama.cpp"
    $script:cmakeDefs = @("-DBUILD_SHARED_LIBS=on", "-DLLAMA_NATIVE=off",  "-A","x64")
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
    & git submodule update --force "${script:llamacppDir}"
    if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
}

function apply_patches {
    # Wire up our CMakefile
    if (!(Select-String -Path "${script:llamacppDir}/examples/server/CMakeLists.txt" -Pattern 'ollama')) {
        Add-Content -Path "${script:llamacppDir}/examples/server/CMakeLists.txt" -Value 'include (../../../ext_server/CMakeLists.txt) # ollama'
    }
    # Avoid duplicate main symbols when we link into the cgo binary
    $content = Get-Content -Path "${script:llamacppDir}/examples/server/server.cpp"
    $content = $content -replace 'int main\(', 'int __main('
    Set-Content -Path "${script:llamacppDir}/examples/server/server.cpp" -Value $content
}

function build {
    write-host "generating config with: cmake -S ${script:llamacppDir} -B $script:buildDir $script:cmakeDefs"
    & cmake --version
    & cmake -S "${script:llamacppDir}" -B $script:buildDir $script:cmakeDefs
    if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
    write-host "building with: cmake --build $script:buildDir --config $script:config ($script:cmakeTargets | ForEach-Object { "--target", $_ })"
    & cmake --build $script:buildDir --config $script:config ($script:cmakeTargets | ForEach-Object { "--target", $_ })
    if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
}

function install {
    rm -ea 0 -recurse -force -path "${script:buildDir}/lib"
    md "${script:buildDir}/lib" -ea 0 > $null
    cp "${script:buildDir}/bin/${script:config}/ext_server_shared.dll" "${script:buildDir}/lib"
    cp "${script:buildDir}/bin/${script:config}/llama.dll" "${script:buildDir}/lib"

    # Display the dll dependencies in the build log
    dumpbin /dependents "${script:buildDir}/bin/${script:config}/ext_server_shared.dll" | select-string ".dll"
}

function cleanup {
    Set-Location "${script:llamacppDir}/examples/server"
    git checkout CMakeLists.txt server.cpp
}

init_vars
git_module_setup
apply_patches

# -DLLAMA_AVX -- 2011 Intel Sandy Bridge & AMD Bulldozer
# -DLLAMA_F16C -- 2012 Intel Ivy Bridge & AMD 2011 Bulldozer (No significant improvement over just AVX)
# -DLLAMA_AVX2 -- 2013 Intel Haswell & 2015 AMD Excavator / 2017 AMD Zen
# -DLLAMA_FMA (FMA3) -- 2013 Intel Haswell & 2012 AMD Piledriver

$script:commonCpuDefs = @("-DCMAKE_POSITION_INDEPENDENT_CODE=on", "-DLLAMA_NATIVE=off")

$script:cmakeDefs = $script:commonCpuDefs + @("-DLLAMA_AVX=off", "-DLLAMA_AVX2=off", "-DLLAMA_AVX512=off", "-DLLAMA_FMA=off", "-DLLAMA_F16C=off") + $script:cmakeDefs
$script:buildDir="${script:llamacppDir}/build/windows/cpu"
write-host "Building LCD CPU"
build
install

$script:cmakeDefs = $script:commonCpuDefs + @("-DLLAMA_AVX=on", "-DLLAMA_AVX2=off", "-DLLAMA_AVX512=off", "-DLLAMA_FMA=off", "-DLLAMA_F16C=off") + $script:cmakeDefs
$script:buildDir="${script:llamacppDir}/build/windows/cpu_avx"
write-host "Building AVX CPU"
build
install

$script:cmakeDefs = $script:commonCpuDefs + @("-DLLAMA_AVX=on", "-DLLAMA_AVX2=on", "-DLLAMA_AVX512=off", "-DLLAMA_FMA=on", "-DLLAMA_F16C=on") + $script:cmakeDefs
$script:buildDir="${script:llamacppDir}/build/windows/cpu_avx2"
write-host "Building AVX2 CPU"
build
install

# Then build cuda as a dynamically loaded library
# TODO figure out how to detect cuda version
init_vars
$script:buildDir="${script:llamacppDir}/build/windows/cuda"
$script:cmakeDefs += @("-DLLAMA_CUBLAS=ON", "-DLLAMA_AVX=on")
build
install

# TODO - actually implement ROCm support on windows
$script:buildDir="${script:llamacppDir}/build/windows/rocm"

rm -ea 0 -recurse -force -path "${script:buildDir}/lib"
md "${script:buildDir}/lib" -ea 0 > $null
echo $null >> "${script:buildDir}/lib/.generated"

cleanup
write-host "`ngo generate completed"