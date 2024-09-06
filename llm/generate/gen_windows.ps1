#!powershell

function get_script_path($script_name){
    $ScriptPath = Join-Path -Path $PSScriptRoot -ChildPath $script_name
    return $scriptPath
}

. $(get_script_path("gen_windows_utils.ps1"))

# -DGGML_AVX -- 2011 Intel Sandy Bridge & AMD Bulldozer
# -DGGML_AVX2 -- 2013 Intel Haswell & 2015 AMD Excavator / 2017 AMD Zen
# -DGGML_FMA (FMA3) -- 2013 Intel Haswell & 2012 AMD Piledriver


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
            "-DGGML_NATIVE=off",
            "-DGGML_AVX=off",
            "-DGGML_AVX2=off",
            "-DGGML_AVX512=off",
            "-DGGML_F16C=off",
            "-DGGML_FMA=off",
            "-DGGML_OPENMP=off")
        $script:buildDir="../build/windows/${script:ARCH}_static"
        write-host "Building static library"
        build
        $script:cmakeTargets = $oldTargets
    } else {
        write-host "Skipping CPU generation step as requested"
    }
}

function build_cpu($gen_arch) {
    if ((-not "${env:OLLAMA_SKIP_CPU_GENERATE}" ) -and ((-not "${env:OLLAMA_CPU_TARGET}") -or ("${env:OLLAMA_CPU_TARGET}" -eq "cpu"))) {
        # remaining llama.cpp builds use MSVC 
        init_vars
        $script:cmakeDefs = $script:commonCpuDefs + @("-A", $gen_arch, "-DGGML_AVX=off", "-DGGML_AVX2=off", "-DGGML_AVX512=off", "-DGGML_FMA=off", "-DGGML_F16C=off") + $script:cmakeDefs
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
    if (Test-Path -Path "C:\Program Files (x86)\Intel\oneAPI\setvars.bat") {
        cmd.exe "/C" '"C:\Program Files (x86)\Intel\oneAPI\setvars.bat" && powershell -Command "& '$(get_script_path("gen_oneapi.ps1"))'"'
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
            "-DGGML_CUDA_NO_PEER_COPY=on",
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
    build_static
    if ($script:ARCH -eq "arm64") {
        build_cpu("ARM64")
    } else { # amd64
        build_cpu("x64")
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