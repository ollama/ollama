#!powershell

$ErrorActionPreference = "Stop"

function amdGPUs {
    if ($env:AMDGPU_TARGETS) {
        return $env:AMDGPU_TARGETS
    }
    # Current supported rocblas list from ROCm v6.1.2 on windows
    # https://rocm.docs.amd.com/projects/install-on-windows/en/latest/reference/system-requirements.html#windows-supported-gpus
    $GPU_LIST = @(
        "gfx1030"
        "gfx1100"
        "gfx1101"
        "gfx1102"
    )
    $GPU_LIST -join ';'
}


function init_vars {
    if (!$script:SRC_DIR) {
        $script:SRC_DIR = $(resolve-path "..\..\")
    }
    if (!$script:llamacppDir) {
        $script:llamacppDir = "../llama.cpp"
    }
    if (!$script:cmakeTargets) {
        $script:cmakeTargets = @("ollama_llama_server")
    }
    $script:cmakeDefs = @(
        "-DBUILD_SHARED_LIBS=on",
        "-DGGML_NATIVE=off",
        "-DGGML_OPENMP=off"
        )
    $script:commonCpuDefs = @("-DCMAKE_POSITION_INDEPENDENT_CODE=on")
    $script:ARCH = $Env:PROCESSOR_ARCHITECTURE.ToLower()
    $script:DIST_BASE = "${script:SRC_DIR}\dist\windows-${script:ARCH}\lib\ollama\runners"
    md "$script:DIST_BASE" -ea 0 > $null
    if ($env:CGO_CFLAGS -contains "-g") {
        $script:cmakeDefs += @("-DCMAKE_VERBOSE_MAKEFILE=on", "-DLLAMA_SERVER_VERBOSE=on", "-DCMAKE_BUILD_TYPE=RelWithDebInfo")
        $script:config = "RelWithDebInfo"
    } else {
        $script:cmakeDefs += @("-DLLAMA_SERVER_VERBOSE=off", "-DCMAKE_BUILD_TYPE=Release")
        $script:config = "Release"
    }
    if ($null -ne $env:CMAKE_SYSTEM_VERSION) {
        $script:cmakeDefs += @("-DCMAKE_SYSTEM_VERSION=${env:CMAKE_SYSTEM_VERSION}")
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
    $script:DUMPBIN=(get-command -ea 'silentlycontinue' dumpbin).path
    if ($null -eq $env:CMAKE_CUDA_ARCHITECTURES) {
        $script:CMAKE_CUDA_ARCHITECTURES="50;52;61;70;75;80"
    } else {
        $script:CMAKE_CUDA_ARCHITECTURES=$env:CMAKE_CUDA_ARCHITECTURES
    }
    # Note: Windows Kits 10 signtool crashes with GCP's plugin
    if ($null -eq $env:SIGN_TOOL) {
        ${script:SignTool}="C:\Program Files (x86)\Windows Kits\8.1\bin\x64\signtool.exe"
    } else {
        ${script:SignTool}=${env:SIGN_TOOL}
    }
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
    # Wire up our CMakefile
    if (!(Select-String -Path "${script:llamacppDir}/CMakeLists.txt" -Pattern 'ollama')) {
        Add-Content -Path "${script:llamacppDir}/CMakeLists.txt" -Value 'add_subdirectory(../ext_server ext_server) # ollama'
    }

    # Apply temporary patches until fix is upstream
    $patches = Get-ChildItem "../patches/*.diff"
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
    if ($cmakeDefs -contains "-G") {
        $extra=@("-j8")
    } else {
        $extra= @("--", "/maxCpuCount:8")
    }
    write-host "building with: cmake --build $script:buildDir --config $script:config $($script:cmakeTargets | ForEach-Object { `"--target`", $_ }) $extra"
    & cmake --build $script:buildDir --config $script:config ($script:cmakeTargets | ForEach-Object { "--target", $_ }) $extra
    if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
    # Rearrange output to be consistent between different generators
    if ($null -ne ${script:config} -And (test-path -path "${script:buildDir}/bin/${script:config}" ) ) {
        mv -force "${script:buildDir}/bin/${script:config}/*" "${script:buildDir}/bin/"
        remove-item "${script:buildDir}/bin/${script:config}"
    }
}

function sign {
    if ("${env:KEY_CONTAINER}") {
        write-host "Signing ${script:buildDir}/bin/*.exe  ${script:buildDir}/bin/*.dll"
        foreach ($file in @(get-childitem "${script:buildDir}/bin/*.exe") + @(get-childitem "${script:buildDir}/bin/*.dll")){
            & "${script:SignTool}" sign /v /fd sha256 /t http://timestamp.digicert.com /f "${script:OLLAMA_CERT}" `
                /csp "Google Cloud KMS Provider" /kc "${env:KEY_CONTAINER}" $file
            if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
        }
    }
}

function install {
    write-host "Installing binaries to dist dir ${script:distDir}"
    mkdir ${script:distDir} -ErrorAction SilentlyContinue
    $binaries = dir "${script:buildDir}/bin/*.exe"
    foreach ($file in $binaries) {
        copy-item -Path $file -Destination ${script:distDir} -Force
    }

    write-host "Installing dlls to dist dir ${script:distDir}"
    $dlls = dir "${script:buildDir}/bin/*.dll"
    foreach ($file in $dlls) {
        copy-item -Path $file -Destination ${script:distDir} -Force
    }
}

function cleanup {
    $patches = Get-ChildItem "../patches/*.diff"
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
        git -C "${script:llamacppDir}" checkout CMakeLists.txt
    }
}