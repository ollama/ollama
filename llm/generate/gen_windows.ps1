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
    write-host "Checking for cmake..."
    get-command cmake
    write-host "Checking for ninja..."
    $d=(get-command -ea 'silentlycontinue' ninja).path
    if ($null -eq $d) {
        $MSVC_INSTALL=(Get-CimInstance MSFT_VSInstance -Namespace root/cimv2/vs)[0].InstallLocation
        $matches=(gci -path $MSVC_INSTALL -r -fi ninja.exe)
        if ($matches.count -eq 0) {
            throw "Unable to locate ninja"
        }
        $ninjaDir=($matches[0].FullName | split-path -parent)
        $env:PATH="$env:PATH;$ninjaDir"
    }
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
    # Apply temporary patches until fix is upstream
    foreach ($patch in $(Get-ChildItem "../patches/*.patch")) {
        git -c 'user.name=nobody' -c 'user.email=<>' -C "${script:llamacppDir}" am $patch.FullName
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


# -DGGML_AVX -- 2011 Intel Sandy Bridge & AMD Bulldozer
# -DGGML_AVX2 -- 2013 Intel Haswell & 2015 AMD Excavator / 2017 AMD Zen
# -DGGML_FMA (FMA3) -- 2013 Intel Haswell & 2012 AMD Piledriver


function build_cpu_x64 {
    if ((-not "${env:OLLAMA_SKIP_CPU_GENERATE}" ) -and ((-not "${env:OLLAMA_CPU_TARGET}") -or ("${env:OLLAMA_CPU_TARGET}" -eq "cpu"))) {
        init_vars
        $script:cmakeDefs = $script:commonCpuDefs + @("-A", "x64", "-DGGML_AVX=off", "-DGGML_AVX2=off", "-DGGML_AVX512=off", "-DGGML_FMA=off", "-DGGML_F16C=off") + $script:cmakeDefs
        $script:buildDir="../build/windows/${script:ARCH}/cpu"
        $script:distDir="$script:DIST_BASE\cpu"
        write-host "Building LCD CPU"
        build
        sign
        install
    } else {
        write-host "Skipping CPU generation step as requested"
    }
}

function build_cpu_arm64 {
    if ((-not "${env:OLLAMA_SKIP_CPU_GENERATE}" ) -and ((-not "${env:OLLAMA_CPU_TARGET}") -or ("${env:OLLAMA_CPU_TARGET}" -eq "cpu"))) {
        init_vars
        write-host "Checking for clang..."
        get-command clang
        $env:CFLAGS="-march=armv8.7-a -fvectorize -ffp-model=fast -fno-finite-math-only"
        $env:CXXFLAGS="$env:CFLAGS"
        $env:LDFLAGS="-static-libstdc++"
        $script:cmakeDefs = $script:commonCpuDefs + @(
            "-DCMAKE_VERBOSE_MAKEFILE=on",
            "-DCMAKE_C_COMPILER=clang.exe",
            "-DCMAKE_CXX_COMPILER=clang++.exe",
            "-DMSVC_RUNTIME_LIBRARY=MultiThreaded"
        ) + $script:cmakeDefs
        $script:buildDir="../build/windows/${script:ARCH}/cpu"
        $script:distDir="$script:DIST_BASE\cpu"
        write-host "Building LCD CPU"
        build
        sign
        install
    } else {
        write-host "Skipping CPU generation step as requested"
    }
}


function build_cpu_avx() {
    if ((-not "${env:OLLAMA_SKIP_CPU_GENERATE}" ) -and ((-not "${env:OLLAMA_CPU_TARGET}") -or ("${env:OLLAMA_CPU_TARGET}" -eq "cpu_avx"))) {
        init_vars
        $script:cmakeDefs = $script:commonCpuDefs + @("-A", "x64", "-DGGML_AVX=on", "-DGGML_AVX2=off", "-DGGML_AVX512=off", "-DGGML_FMA=off", "-DGGML_F16C=off") + $script:cmakeDefs
        $script:buildDir="../build/windows/${script:ARCH}/cpu_avx"
        $script:distDir="$script:DIST_BASE\cpu_avx"
        write-host "Building AVX CPU"
        build
        sign
        install
    } else {
        write-host "Skipping CPU AVX generation step as requested"
    }
}

function build_cpu_avx2() {
    if ((-not "${env:OLLAMA_SKIP_CPU_GENERATE}" ) -and ((-not "${env:OLLAMA_CPU_TARGET}") -or ("${env:OLLAMA_CPU_TARGET}" -eq "cpu_avx2"))) {
        init_vars
        $script:cmakeDefs = $script:commonCpuDefs + @("-A", "x64", "-DGGML_AVX=on", "-DGGML_AVX2=on", "-DGGML_AVX512=off", "-DGGML_FMA=on", "-DGGML_F16C=on") + $script:cmakeDefs
        $script:buildDir="../build/windows/${script:ARCH}/cpu_avx2"
        $script:distDir="$script:DIST_BASE\cpu_avx2"
        write-host "Building AVX2 CPU"
        build
        sign
        install
    } else {
        write-host "Skipping CPU AVX2 generation step as requested"
    }
}

function build_cuda() {
    if ((-not "${env:OLLAMA_SKIP_CUDA_GENERATE}") -and ("${script:CUDA_LIB_DIR}")) {
        # Then build cuda as a dynamically loaded library
        $nvcc = "$script:CUDA_LIB_DIR\nvcc.exe"
        $script:CUDA_VERSION=((get-item ($nvcc | split-path | split-path)).Basename -Split "\.")[0]
        if ($null -ne $script:CUDA_VERSION) {
            $script:CUDA_VARIANT="_"+$script:CUDA_VERSION
        }
        init_vars
        $script:buildDir="../build/windows/${script:ARCH}/cuda$script:CUDA_VARIANT"
        $script:distDir="$script:DIST_BASE\cuda$script:CUDA_VARIANT"
        $script:cmakeDefs += @(
            "-A", "x64",
            "-DGGML_CUDA=ON",
            "-DGGML_AVX=on",
            "-DGGML_AVX2=off",
            "-DCMAKE_CUDA_FLAGS=-t6",
            "-DCMAKE_CUDA_ARCHITECTURES=${script:CMAKE_CUDA_ARCHITECTURES}",
            "-DCMAKE_CUDA_COMPILER_TOOLKIT_ROOT=$env:CUDA_PATH"
            )
        if ($null -ne $env:OLLAMA_CUSTOM_CUDA_DEFS) {
            write-host "OLLAMA_CUSTOM_CUDA_DEFS=`"${env:OLLAMA_CUSTOM_CUDA_DEFS}`""
            $script:cmakeDefs +=@("${env:OLLAMA_CUSTOM_CUDA_DEFS}")
            write-host "building custom CUDA GPU"
        }
        build
        sign
        install

        md "${script:SRC_DIR}\dist\windows-${script:ARCH}\lib\ollama\" -ea 0 > $null
        write-host "copying CUDA dependencies to ${script:SRC_DIR}\dist\windows-${script:ARCH}\lib\ollama\"
        cp "${script:CUDA_LIB_DIR}\cudart64_*.dll" "${script:SRC_DIR}\dist\windows-${script:ARCH}\lib\ollama\"
        cp "${script:CUDA_LIB_DIR}\cublas64_*.dll" "${script:SRC_DIR}\dist\windows-${script:ARCH}\lib\ollama\"
        cp "${script:CUDA_LIB_DIR}\cublasLt64_*.dll" "${script:SRC_DIR}\dist\windows-${script:ARCH}\lib\ollama\"
    } else {
        write-host "Skipping CUDA generation step"
    }
}

function build_oneapi() {
  if ((-not "${env:OLLAMA_SKIP_ONEAPI_GENERATE}") -and ("${env:ONEAPI_ROOT}"))  {
    # Get oneAPI version
    $script:ONEAPI_VERSION = icpx --version
    $script:ONEAPI_VERSION = [regex]::Match($script:ONEAPI_VERSION, '(?<=oneAPI DPC\+\+/C\+\+ Compiler )(?<version>\d+\.\d+\.\d+)').Value
    if ($null -ne $script:ONEAPI_VERSION) {
      $script:ONEAPI_VARIANT = "_v" + $script:ONEAPI_VERSION
    }
    init_vars
    $script:buildDir = "../build/windows/${script:ARCH}/oneapi$script:ONEAPI_VARIANT"
    $script:distDir ="$script:DIST_BASE\oneapi$script:ONEAPI_VARIANT"
    $script:cmakeDefs += @(
      "-G", "MinGW Makefiles",
      "-DGGML_SYCL=ON",
      "-DCMAKE_C_COMPILER=icx",
      "-DCMAKE_CXX_COMPILER=icx",
      "-DCMAKE_BUILD_TYPE=Release"
    )

    Write-Host "Building oneAPI"
    build
    # Ninja doesn't prefix with config name
    if ($null -ne $script:DUMPBIN) {
      & "$script:DUMPBIN" /dependents "${script:buildDir}/bin/ollama_llama_server.exe" | Select-String ".dll"
    }
    sign
    install

    md "${script:SRC_DIR}\dist\windows-${script:ARCH}\lib\ollama\" -ea 0 > $null
    cp "${env:ONEAPI_ROOT}\compiler\latest\bin\libirngmd.dll" "${script:SRC_DIR}\dist\windows-${script:ARCH}\lib\ollama\"
    cp "${env:ONEAPI_ROOT}\compiler\latest\bin\libmmd.dll" "${script:SRC_DIR}\dist\windows-${script:ARCH}\lib\ollama\"
    cp "${env:ONEAPI_ROOT}\compiler\latest\bin\pi_level_zero.dll" "${script:SRC_DIR}\dist\windows-${script:ARCH}\lib\ollama\"
    cp "${env:ONEAPI_ROOT}\compiler\latest\bin\pi_unified_runtime.dll" "${script:SRC_DIR}\dist\windows-${script:ARCH}\lib\ollama\"
    cp "${env:ONEAPI_ROOT}\compiler\latest\bin\pi_win_proxy_loader.dll" "${script:SRC_DIR}\dist\windows-${script:ARCH}\lib\ollama\"
    cp "${env:ONEAPI_ROOT}\compiler\latest\bin\svml_dispmd.dll" "${script:SRC_DIR}\dist\windows-${script:ARCH}\lib\ollama\"
    cp "${env:ONEAPI_ROOT}\compiler\latest\bin\sycl7.dll" "${script:SRC_DIR}\dist\windows-${script:ARCH}\lib\ollama\"
    cp "${env:ONEAPI_ROOT}\mkl\latest\bin\mkl_core.2.dll" "${script:SRC_DIR}\dist\windows-${script:ARCH}\lib\ollama\"
    cp "${env:ONEAPI_ROOT}\mkl\latest\bin\mkl_sycl_blas.4.dll" "${script:SRC_DIR}\dist\windows-${script:ARCH}\lib\ollama\"
    cp "${env:ONEAPI_ROOT}\mkl\latest\bin\mkl_tbb_thread.2.dll" "${script:SRC_DIR}\dist\windows-${script:ARCH}\lib\ollama\"
  } else {
    Write-Host "Skipping oneAPI generation step"
  }
}

function build_rocm() {
    if ((-not "${env:OLLAMA_SKIP_ROCM_GENERATE}") -and ("${env:HIP_PATH}")) {
        $script:ROCM_VERSION=(get-item $env:HIP_PATH).Basename
        if ($null -ne $script:ROCM_VERSION) {
            $script:ROCM_VARIANT="_v"+$script:ROCM_VERSION
        }

        init_vars
        $script:buildDir="../build/windows/${script:ARCH}/rocm$script:ROCM_VARIANT"
        $script:distDir="$script:DIST_BASE\rocm$script:ROCM_VARIANT"
        $script:cmakeDefs += @(
            "-G", "Ninja",
            "-DCMAKE_C_COMPILER=clang.exe",
            "-DCMAKE_CXX_COMPILER=clang++.exe",
            "-DGGML_HIPBLAS=on",
            "-DHIP_PLATFORM=amd",
            "-DGGML_AVX=on",
            "-DGGML_AVX2=off",
            "-DCMAKE_POSITION_INDEPENDENT_CODE=on",
            "-DAMDGPU_TARGETS=$(amdGPUs)",
            "-DGPU_TARGETS=$(amdGPUs)"
            )

        # Make sure the ROCm binary dir is first in the path
        $env:PATH="$env:HIP_PATH\bin;$env:PATH"

        # We have to clobber the LIB var from the developer shell for clang to work properly
        $env:LIB=""
        if ($null -ne $env:OLLAMA_CUSTOM_ROCM_DEFS) {
            write-host "OLLAMA_CUSTOM_ROCM_DEFS=`"${env:OLLAMA_CUSTOM_ROCM_DEFS}`""
            $script:cmakeDefs += @("${env:OLLAMA_CUSTOM_ROCM_DEFS}")
            write-host "building custom ROCM GPU"
        }
        write-host "Building ROCm"
        build
        # Ninja doesn't prefix with config name
        ${script:config}=""
        if ($null -ne $script:DUMPBIN) {
            & "$script:DUMPBIN" /dependents "${script:buildDir}/bin/ollama_llama_server.exe" | select-string ".dll"
        }
        sign
        install

        md "${script:SRC_DIR}\dist\windows-${script:ARCH}\lib\ollama\rocblas\library\" -ea 0 > $null
        cp "${env:HIP_PATH}\bin\hipblas.dll" "${script:SRC_DIR}\dist\windows-${script:ARCH}\lib\ollama\"
        cp "${env:HIP_PATH}\bin\rocblas.dll" "${script:SRC_DIR}\dist\windows-${script:ARCH}\lib\ollama\"
        # amdhip64.dll dependency comes from the driver and must be installed on the host to use AMD GPUs
        cp "${env:HIP_PATH}\bin\rocblas\library\*" "${script:SRC_DIR}\dist\windows-${script:ARCH}\lib\ollama\rocblas\library\"
    } else {
        write-host "Skipping ROCm generation step"
    }
}

init_vars
if ($($args.count) -eq 0) {
    git_module_setup
    apply_patches
    if ($script:ARCH -eq "arm64") {
        build_cpu_arm64
    } else { # amd64
        build_cpu_x64
        build_cpu_avx
        build_cpu_avx2
        build_cuda
        build_oneapi
        build_rocm
    }

    cleanup
    write-host "`ngo generate completed.  LLM runners: $(get-childitem -path $script:DIST_BASE)"
} else {
    for ( $i = 0; $i -lt $args.count; $i++ ) {
        write-host "performing $($args[$i])"
        & $($args[$i])
    }
}