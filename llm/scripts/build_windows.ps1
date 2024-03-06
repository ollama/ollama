#!powershell

$ErrorActionPreference = "Stop"

function init_vars {
    $script:SRC_DIR = $(resolve-path "..\..\")
    $script:llamacppDir = "llama.cpp"
    $script:cmakeDefs = @("-DBUILD_SHARED_LIBS=on", "-DLLAMA_NATIVE=off", "-DLLAMA_SERVER_VERBOSE=off",  "-A", "x64")
    $script:cmakeTargets = @("server")
    $script:ARCH = "amd64" # arm not yet supported.
    if ($env:CGO_CFLAGS -contains "-g") {
        $script:cmakeDefs += @("-DCMAKE_VERBOSE_MAKEFILE=on")
        $script:config = "RelWithDebInfo"
    } else {
        $script:config = "Release"
    }
    # Try to find the CUDA dir
    if ($env:CUDA_LIB_DIR -eq $null) {
        $d=(get-command -ea 'silentlycontinue' nvcc).path
        if ($d -ne $null) {
            $script:CUDA_LIB_DIR=($d| split-path -parent)
            $script:CUDA_INCLUDE_DIR=($script:CUDA_LIB_DIR|split-path -parent)+"\include"
        }
    } else {
        $script:CUDA_LIB_DIR=$env:CUDA_LIB_DIR
    }
    $script:GZIP=(get-command -ea 'silentlycontinue' gzip).path
    $script:DUMPBIN=(get-command -ea 'silentlycontinue' dumpbin).path
    if ($null -eq $env:CMAKE_CUDA_ARCHITECTURES) {
        $script:CMAKE_CUDA_ARCHITECTURES="50;52;61;70;75;80"
    } else {
        $script:CMAKE_CUDA_ARCHITECTURES=$env:CMAKE_CUDA_ARCHITECTURES
    }
    # Note: 10 Windows Kit signtool crashes with GCP's plugin
    ${script:SignTool}="C:\Program Files (x86)\Windows Kits\8.1\bin\x64\signtool.exe"
    if ("${env:KEY_CONTAINER}") {
        ${script:OLLAMA_CERT}=$(resolve-path "${script:SRC_DIR}\ollama_inc.crt")
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
    # Apply temporary patches until fix is upstream
    $patches = Get-ChildItem "patches/*.diff"
    foreach ($patch in $patches) {
        # Extract file paths from the patch file
        $filePaths = Get-Content $patch.FullName | Where-Object { $_ -match '^\+\+\+ ' } | ForEach-Object {
            $parts = $_ -split ' '
            ($parts[1] -split '/', 2)[1]
        }

        # Checkout each file
        foreach ($file in $filePaths) {
            git -C "${script:llamacppDir}" checkout $file
        }
    }

    # Apply each patch
    foreach ($patch in $patches) {
        git -C "${script:llamacppDir}" apply $patch.FullName
    }
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

function sign {
    if ("${env:KEY_CONTAINER}") {
        write-host "Signing ${script:buildDir}/bin/${script:config}/*.exe"
        foreach ($file in (get-childitem "${script:buildDir}/bin/${script:config}/*.exe")){
            & "${script:SignTool}" sign /v /fd sha256 /t http://timestamp.digicert.com /f "${script:OLLAMA_CERT}" `
                /csp "Google Cloud KMS Provider" /kc "${env:KEY_CONTAINER}" $file
            if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
        }
    }
}

function compress {
    if ($script:GZIP -eq $null) {
        write-host "gzip not installed, not compressing files"
        return
    }
    write-host "Compressing binaries..."
    $binaries = dir "${script:buildDir}/bin/${script:config}/*.exe"
    foreach ($file in $binaries) {
        & "$script:GZIP" --best -f $file
    }

    write-host "Compressing dlls..."
    $binaries = dir "${script:buildDir}/bin/${script:config}/*.dll"
    foreach ($file in $dlls) {
        & "$script:GZIP" --best -f $file
    }
}

function cleanup {
    $patches = Get-ChildItem "patches/*.diff"
    foreach ($patch in $patches) {
        # Extract file paths from the patch file
        $filePaths = Get-Content $patch.FullName | Where-Object { $_ -match '^\+\+\+ ' } | ForEach-Object {
            $parts = $_ -split ' '
            ($parts[1] -split '/', 2)[1]
        }

        # Checkout each file
        foreach ($file in $filePaths) {            
            git -C "${script:llamacppDir}" checkout $file
        }
    }
}

init_vars
git_module_setup
apply_patches

# -DLLAMA_AVX -- 2011 Intel Sandy Bridge & AMD Bulldozer
# -DLLAMA_AVX2 -- 2013 Intel Haswell & 2015 AMD Excavator / 2017 AMD Zen
# -DLLAMA_FMA (FMA3) -- 2013 Intel Haswell & 2012 AMD Piledriver

$script:commonCpuDefs = @("-DCMAKE_POSITION_INDEPENDENT_CODE=on")

init_vars
$script:cmakeDefs = $script:commonCpuDefs + @("-DLLAMA_AVX=off", "-DLLAMA_AVX2=off", "-DLLAMA_AVX512=off", "-DLLAMA_FMA=off") + $script:cmakeDefs
$script:buildDir="build/windows/${script:ARCH}/cpu"
write-host "Building LCD CPU"
build
sign
compress

init_vars
$script:cmakeDefs = $script:commonCpuDefs + @("-DLLAMA_AVX=on", "-DLLAMA_AVX2=off", "-DLLAMA_AVX512=off", "-DLLAMA_FMA=off") + $script:cmakeDefs
$script:buildDir="build/windows/${script:ARCH}/cpu_avx"
write-host "Building AVX CPU"
build
sign
compress

init_vars
$script:cmakeDefs = $script:commonCpuDefs + @("-DLLAMA_AVX=on", "-DLLAMA_AVX2=on", "-DLLAMA_AVX512=off", "-DLLAMA_FMA=on") + $script:cmakeDefs
$script:buildDir="build/windows/${script:ARCH}/cpu_avx2"
write-host "Building AVX2 CPU"
build
sign
compress

if ($null -ne $script:CUDA_LIB_DIR) {
    # Then build cuda as a dynamically loaded library
    $nvcc = "$script:CUDA_LIB_DIR\nvcc.exe"
    $script:CUDA_VERSION=(get-item ($nvcc | split-path | split-path)).Basename
    if ($null -ne $script:CUDA_VERSION) {
        $script:CUDA_VARIANT="_"+$script:CUDA_VERSION
    }
    init_vars
    $script:buildDir="build/windows/${script:ARCH}/cuda$script:CUDA_VARIANT"
    $script:cmakeDefs += @("-DLLAMA_CUBLAS=ON", "-DLLAMA_AVX=on", "-DLLAMA_AVX2=off", "-DCUDAToolkit_INCLUDE_DIR=$script:CUDA_INCLUDE_DIR", "-DCMAKE_CUDA_ARCHITECTURES=${script:CMAKE_CUDA_ARCHITECTURES}")
    build
    sign
    compress
}
# TODO - actually implement ROCm support on windows
$script:buildDir="build/windows/${script:ARCH}/rocm"

rm -ea 0 -recurse -force -path "${script:buildDir}/lib"
md "${script:buildDir}/lib" -ea 0 > $null
echo $null >> "${script:buildDir}/lib/.generated"

write-host "Copying llama.dll"
Copy-Item -Path "build/windows/${script:ARCH}/cpu/bin/${script:config}/llama.dll" -Destination ..

cleanup
write-host "`ngo generate completed"
