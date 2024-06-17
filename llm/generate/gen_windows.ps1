#!powershell

function amdGPUs {
    if ($env:AMDGPU_TARGETS) {
        return $env:AMDGPU_TARGETS
    }
    # TODO - load from some common data file for linux + windows build consistency
    $GPU_LIST = @(
        "gfx900"
        "gfx906:xnack-"
        "gfx908:xnack-"
        "gfx90a:xnack+"
        "gfx90a:xnack-"
        "gfx940"
        "gfx941"
        "gfx942"
        "gfx1010"
        "gfx1012"
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
        "-DLLAMA_NATIVE=off",
        "-DLLAMA_OPENMP=off"
        )
    $script:commonCpuDefs = @("-DCMAKE_POSITION_INDEPENDENT_CODE=on")
    $script:ARCH = $Env:PROCESSOR_ARCHITECTURE.ToLower()
    $script:DIST_BASE = "${script:SRC_DIR}\dist\windows-${script:ARCH}\ollama_runners"
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
    if ($LASTEXITCODE -ne 0) { throw($LASTEXITCODE)}
    & git submodule update --force "${script:llamacppDir}"
    if ($LASTEXITCODE -ne 0) { throw($LASTEXITCODE)}
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
    if ($LASTEXITCODE -ne 0) { throw($LASTEXITCODE)}
    if ($cmakeDefs -contains "-G") {
        $extra=@("-j8")
    } else {
        $extra= @("--", "/p:CL_MPcount=8")
    }
    write-host "building with: cmake --build $script:buildDir --config $script:config $($script:cmakeTargets | ForEach-Object { `"--target`", $_ }) $extra"
    & cmake --build $script:buildDir --config $script:config ($script:cmakeTargets | ForEach-Object { "--target", $_ }) $extra
    if ($LASTEXITCODE -ne 0) { write-host "cmake build exit status $LASTEXITCODE"; throw($LASTEXITCODE)}
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
            if ($LASTEXITCODE -ne 0) { throw($LASTEXITCODE)}
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


# -DLLAMA_AVX -- 2011 Intel Sandy Bridge & AMD Bulldozer
# -DLLAMA_AVX2 -- 2013 Intel Haswell & 2015 AMD Excavator / 2017 AMD Zen
# -DLLAMA_FMA (FMA3) -- 2013 Intel Haswell & 2012 AMD Piledriver


function build_static() {
    if ((-not "${env:OLLAMA_SKIP_STATIC_GENERATE}") -and ((-not "${env:OLLAMA_CPU_TARGET}") -or ("${env:OLLAMA_CPU_TARGET}" -eq "static"))) {
        # GCC build for direct linking into the Go binary
        init_vars
        # cmake will silently fallback to msvc compilers if mingw isn't in the path, so detect and fail fast
        # as we need this to be compiled by gcc for golang to be able to link with itx
        write-host "Checking for MinGW..."
        # error action ensures we exit on failure
        get-command gcc
        get-command mingw32-make
        $oldTargets = $script:cmakeTargets
        $script:cmakeTargets = @("llama", "ggml")
        $script:cmakeDefs = @(
            "-G", "MinGW Makefiles"
            "-DCMAKE_C_COMPILER=gcc.exe",
            "-DCMAKE_CXX_COMPILER=g++.exe",
            "-DBUILD_SHARED_LIBS=off",
            "-DLLAMA_NATIVE=off",
            "-DLLAMA_AVX=off",
            "-DLLAMA_AVX2=off",
            "-DLLAMA_AVX512=off",
            "-DLLAMA_F16C=off",
            "-DLLAMA_FMA=off",
            "-DLLAMA_OPENMP=off")
        $script:buildDir="../build/windows/${script:ARCH}_static"
        write-host "Building static library"
        build
        $script:cmakeTargets = $oldTargets
    } else {
        write-host "Skipping CPU generation step as requested"
    }
}

function build_cpu() {
    if ($script:ARCH -eq "arm64") {
        $gen_arch = "ARM64"
    } else { # amd64
        $gen_arch = "x64"
    }

    if ((-not "${env:OLLAMA_SKIP_CPU_GENERATE}" ) -and ((-not "${env:OLLAMA_CPU_TARGET}") -or ("${env:OLLAMA_CPU_TARGET}" -eq "cpu"))) {
        # remaining llama.cpp builds use MSVC 
        init_vars
        $script:cmakeDefs = $script:commonCpuDefs + @("-A", $gen_arch, "-DLLAMA_AVX=off", "-DLLAMA_AVX2=off", "-DLLAMA_AVX512=off", "-DLLAMA_FMA=off", "-DLLAMA_F16C=off") + $script:cmakeDefs
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
        $script:cmakeDefs = $script:commonCpuDefs + @("-A", "x64", "-DLLAMA_AVX=on", "-DLLAMA_AVX2=off", "-DLLAMA_AVX512=off", "-DLLAMA_FMA=off", "-DLLAMA_F16C=off") + $script:cmakeDefs
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
        $script:cmakeDefs = $script:commonCpuDefs + @("-A", "x64", "-DLLAMA_AVX=on", "-DLLAMA_AVX2=on", "-DLLAMA_AVX512=off", "-DLLAMA_FMA=on", "-DLLAMA_F16C=on") + $script:cmakeDefs
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
        $script:CUDA_VERSION=(get-item ($nvcc | split-path | split-path)).Basename
        if ($null -ne $script:CUDA_VERSION) {
            $script:CUDA_VARIANT="_"+$script:CUDA_VERSION
        }
        init_vars
        $script:buildDir="../build/windows/${script:ARCH}/cuda$script:CUDA_VARIANT"
        $script:distDir="$script:DIST_BASE\cuda$script:CUDA_VARIANT"
        $script:cmakeDefs += @(
            "-A", "x64",
            "-DLLAMA_CUDA=ON",
            "-DLLAMA_AVX=on",
            "-DLLAMA_AVX2=off",
            "-DCUDAToolkit_INCLUDE_DIR=$script:CUDA_INCLUDE_DIR",
            "-DCMAKE_CUDA_FLAGS=-t8"
            "-DCMAKE_CUDA_ARCHITECTURES=${script:CMAKE_CUDA_ARCHITECTURES}"
            )
        if ($null -ne $env:OLLAMA_CUSTOM_CUDA_DEFS) {
            write-host "OLLAMA_CUSTOM_CUDA_DEFS=`"${env:OLLAMA_CUSTOM_CUDA_DEFS}`""
            $script:cmakeDefs +=@("${env:OLLAMA_CUSTOM_CUDA_DEFS}")
            write-host "building custom CUDA GPU"
        }
        build
        sign
        install

        write-host "copying CUDA dependencies to ${script:SRC_DIR}\dist\windows-${script:ARCH}\"
        cp "${script:CUDA_LIB_DIR}\cudart64_*.dll" "${script:SRC_DIR}\dist\windows-${script:ARCH}\"
        cp "${script:CUDA_LIB_DIR}\cublas64_*.dll" "${script:SRC_DIR}\dist\windows-${script:ARCH}\"
        cp "${script:CUDA_LIB_DIR}\cublasLt64_*.dll" "${script:SRC_DIR}\dist\windows-${script:ARCH}\"
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
      "-DLLAMA_SYCL=ON",
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

    cp "${env:ONEAPI_ROOT}\compiler\latest\bin\libirngmd.dll" "${script:distDir}"
    cp "${env:ONEAPI_ROOT}\compiler\latest\bin\libmmd.dll" "${script:distDir}"
    cp "${env:ONEAPI_ROOT}\compiler\latest\bin\pi_level_zero.dll" "${script:distDir}"
    cp "${env:ONEAPI_ROOT}\compiler\latest\bin\pi_unified_runtime.dll" "${script:distDir}"
    cp "${env:ONEAPI_ROOT}\compiler\latest\bin\pi_win_proxy_loader.dll" "${script:distDir}"
    cp "${env:ONEAPI_ROOT}\compiler\latest\bin\svml_dispmd.dll" "${script:distDir}"
    cp "${env:ONEAPI_ROOT}\compiler\latest\bin\sycl7.dll" "${script:distDir}"
    cp "${env:ONEAPI_ROOT}\mkl\latest\bin\mkl_core.2.dll" "${script:distDir}"
    cp "${env:ONEAPI_ROOT}\mkl\latest\bin\mkl_sycl_blas.4.dll" "${script:distDir}"
    cp "${env:ONEAPI_ROOT}\mkl\latest\bin\mkl_tbb_thread.2.dll" "${script:distDir}"
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
            "-DLLAMA_HIPBLAS=on",
            "-DHIP_PLATFORM=amd",
            "-DLLAMA_AVX=on",
            "-DLLAMA_AVX2=off",
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

        # Assumes v5.7, may need adjustments for v6
        rm -ea 0 -recurse -force -path "${script:SRC_DIR}\dist\windows-${script:ARCH}\rocm\"
        md "${script:SRC_DIR}\dist\windows-${script:ARCH}\rocm\rocblas\library\" -ea 0 > $null
        cp "${env:HIP_PATH}\bin\hipblas.dll" "${script:SRC_DIR}\dist\windows-${script:ARCH}\rocm\"
        cp "${env:HIP_PATH}\bin\rocblas.dll" "${script:SRC_DIR}\dist\windows-${script:ARCH}\rocm\"
        # amdhip64.dll dependency comes from the driver and must be installed on the host to use AMD GPUs
        cp "${env:HIP_PATH}\bin\rocblas\library\*" "${script:SRC_DIR}\dist\windows-${script:ARCH}\rocm\rocblas\library\"
    } else {
        write-host "Skipping ROCm generation step"
    }
}

init_vars
if ($($args.count) -eq 0) {
    git_module_setup
    apply_patches

    $tasks = @("build_static", "build_cpu")
    $jobs = @()
    if ($script:ARCH -ne "arm64") {
        $tasks += $("build_cpu_avx", "build_cpu_avx2", "build_cuda", "build_oneapi", "build_rocm")
    }
    foreach ($t in $tasks) {
        $jobs += @(Start-ThreadJob -ThrottleLimit 12 -FilePath .\gen_windows.ps1 -ArgumentList $t -Name $t)
    }
    get-job
    foreach ($job in $jobs) {
        write-host "----" $job.Name output follows
        receive-job -wait -job $job
        write-host "----" $job.Name $job.State
        write-host ""
        if ($job.State -contains 'Failed') {
            cleanup
            write-host "Terminating remaining jobs (this takes a while, you can ^C)"
            # TODO find some way to kill the spawned cmake processes faster
            remove-job -force -job $jobs
            exit(-1)
        }
        get-job
    }

    cleanup
    write-host "`ngo generate completed.  LLM runners: $(get-childitem -path $script:DIST_BASE)"
} else {
    for ( $i = 0; $i -lt $args.count; $i++ ) {
        write-host "performing $($args[$i])"
        & $($args[$i])
    } 
}