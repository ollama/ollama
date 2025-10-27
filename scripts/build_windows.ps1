#!powershell
#
# powershell -ExecutionPolicy Bypass -File .\scripts\build_windows.ps1
#
# gcloud auth application-default login

$ErrorActionPreference = "Stop"

function checkEnv() {
    if ($null -ne $env:ARCH ) {
        $script:ARCH = $env:ARCH
    } else {
        $arch=([System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture)
        if ($null -ne $arch) {
            $script:ARCH = ($arch.ToString().ToLower()).Replace("x64", "amd64")
        } else {
            write-host "WARNING: old powershell detected, assuming amd64 architecture - set `$env:ARCH to override"
            $script:ARCH="amd64"
        }
    }
    $script:TARGET_ARCH=$script:ARCH
    Write-host "Building for ${script:TARGET_ARCH}"
    write-host "Locating required tools and paths"
    $script:SRC_DIR=$PWD
    if ($null -eq $env:VCToolsRedistDir) {
        $MSVC_INSTALL=(Get-CimInstance MSFT_VSInstance -Namespace root/cimv2/vs)[0].InstallLocation
        $env:VCToolsRedistDir=(get-item "${MSVC_INSTALL}\VC\Redist\MSVC\*")[0]
    }
    # Locate CUDA versions
    $cudaList=(get-item "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*\bin\" -ea 'silentlycontinue')
    if ($cudaList.length -eq 0) {
        $d=(get-command -ea 'silentlycontinue' nvcc).path
        if ($null -ne $d) {
            $script:CUDA_DIRS=@($d| split-path -parent)
        }
    } else {
        $script:CUDA_DIRS=$cudaList
    }
    
    $inoSetup=(get-item "C:\Program Files*\Inno Setup*\")
    if ($inoSetup.length -gt 0) {
        $script:INNO_SETUP_DIR=$inoSetup[0]
    }

    $script:DIST_DIR="${script:SRC_DIR}\dist\windows-${script:TARGET_ARCH}"
    $env:CGO_ENABLED="1"
    Write-Output "Checking version"
    if (!$env:VERSION) {
        $data=(git describe --tags --first-parent --abbrev=7 --long --dirty --always)
        $pattern="v(.+)"
        if ($data -match $pattern) {
            $script:VERSION=$matches[1]
        }
    } else {
        $script:VERSION=$env:VERSION
    }
    $pattern = "(\d+[.]\d+[.]\d+).*"
    if ($script:VERSION -match $pattern) {
        $script:PKG_VERSION=$matches[1]
    } else {
        $script:PKG_VERSION="0.0.0"
    }
    write-host "Building Ollama $script:VERSION with package version $script:PKG_VERSION"

    # Note: Windows Kits 10 signtool crashes with GCP's plugin
    if ($null -eq $env:SIGN_TOOL) {
        ${script:SignTool}="C:\Program Files (x86)\Windows Kits\8.1\bin\x64\signtool.exe"
    } else {
        ${script:SignTool}=${env:SIGN_TOOL}
    }
    if ("${env:KEY_CONTAINER}") {
        ${script:OLLAMA_CERT}=$(resolve-path "${script:SRC_DIR}\ollama_inc.crt")
        Write-host "Code signing enabled"
    } else {
        write-host "Code signing disabled - please set KEY_CONTAINERS to sign and copy ollama_inc.crt to the top of the source tree"
    }
    $script:JOBS=((Get-CimInstance Win32_ComputerSystem).NumberOfLogicalProcessors)
}


function buildCPU() {
    mkdir -Force -path "${script:DIST_DIR}\"
    if ($script:ARCH -ne "arm64") {
        Remove-Item -ea 0 -recurse -force -path "${script:SRC_DIR}\dist\windows-${script:ARCH}"
        New-Item "${script:SRC_DIR}\dist\windows-${script:ARCH}\lib\ollama\" -ItemType Directory -ea 0

        & cmake -B build\cpu --preset CPU --install-prefix $script:DIST_DIR
        if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
        & cmake --build build\cpu --target ggml-cpu --config Release --parallel $script:JOBS
        if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
        & cmake --install build\cpu --component CPU --strip
        if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
    }
}

function buildCUDA11() {
    # CUDA v11 claims to be compatible with MSVC 2022, but the latest updates are no longer compatible
    # 19.40 is the last compiler version that works, but recent udpates are 19.43
    # So this pins to MSVC 2019 for best compatibility
    mkdir -Force -path "${script:DIST_DIR}\"
    if ($script:ARCH -ne "arm64") {
        $hashEnv = @{}
        Get-ChildItem env: | foreach { $hashEnv[$_.Name] = $_.Value }
        if ("$script:CUDA_DIRS".Contains("v11")) {
            $hashEnv.Keys | foreach { if ($_.Contains("CUDA_PATH_V11")) { $x=$hashEnv[$_]; if (test-path -literalpath "$x\bin\nvcc.exe" ) { $cuda=$x}  }}
            write-host "Building CUDA v11 backend libraries $cuda"
            $env:CUDAToolkit_ROOT=$cuda
            & cmake -B build\cuda_v11 --preset "CUDA 11" -T cuda="$cuda" -DCMAKE_CUDA_COMPILER="$cuda\bin\nvcc.exe" -G "Visual Studio 16 2019" --install-prefix $script:DIST_DIR -DOLLAMA_RUNNER_DIR="cuda_v11"
            if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
            & cmake --build build\cuda_v11 --target ggml-cuda --config Release --parallel $script:JOBS
            if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
            & cmake --install build\cuda_v11 --component "CUDA" --strip
            if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
        }
    }
}

function buildCUDA12() {
    mkdir -Force -path "${script:DIST_DIR}\"
    if ($script:ARCH -ne "arm64") {
        $hashEnv = @{}
        Get-ChildItem env: | foreach { $hashEnv[$_.Name] = $_.Value }
        if ("$script:CUDA_DIRS".Contains("v12.8")) {
            $hashEnv.Keys | foreach { if ($_.Contains("CUDA_PATH_V12_8")) { $x=$hashEnv[$_]; if (test-path -literalpath "$x\bin\nvcc.exe" ) { $cuda=$x}  }}
            write-host "Building CUDA v12 backend libraries $cuda"
            $env:CUDAToolkit_ROOT=$cuda
            & cmake -B build\cuda_v12 --preset "CUDA 12" -T cuda="$cuda" --install-prefix $script:DIST_DIR -DOLLAMA_RUNNER_DIR="cuda_v12"
            if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
            & cmake --build build\cuda_v12 --target ggml-cuda --config Release --parallel $script:JOBS
            if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
            & cmake --install build\cuda_v12 --component "CUDA" --strip
            if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
        }
    }
}

function buildCUDA13() {
    mkdir -Force -path "${script:DIST_DIR}\"
    if ($script:ARCH -ne "arm64") {
        $hashEnv = @{}
        Get-ChildItem env: | foreach { $hashEnv[$_.Name] = $_.Value }
        if ("$script:CUDA_DIRS".Contains("v13")) {
            $hashEnv.Keys | foreach { if ($_.Contains("CUDA_PATH_V13")) { $x=$hashEnv[$_]; if (test-path -literalpath "$x\bin\nvcc.exe" ) { $cuda=$x}  }}
            $env:CUDAToolkit_ROOT=$cuda
            write-host "Building CUDA v13 backend libraries $cuda"
            & cmake -B build\cuda_v13 --preset "CUDA 13" -T cuda="$cuda" --install-prefix $script:DIST_DIR -DOLLAMA_RUNNER_DIR="cuda_v13"
            if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
            & cmake --build build\cuda_v13 --target ggml-cuda --config Release --parallel $script:JOBS
            if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
            & cmake --install build\cuda_v13 --component "CUDA" --strip
            if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
        }
    }
}

function buildROCm() {
    mkdir -Force -path "${script:DIST_DIR}\"
    if ($script:ARCH -ne "arm64") {
        if ($env:HIP_PATH) {
            write-host "Building ROCm backend libraries"
            if (-Not (get-command -ErrorAction silent ninja)) {
                $NINJA_DIR=(gci -path (Get-CimInstance MSFT_VSInstance -Namespace root/cimv2/vs)[0].InstallLocation -r -fi ninja.exe).Directory.FullName
                $env:PATH="$NINJA_DIR;$env:PATH"
            }
            $env:HIPCXX="${env:HIP_PATH}\bin\clang++.exe"
            $env:HIP_PLATFORM="amd"
            $env:CMAKE_PREFIX_PATH="${env:HIP_PATH}"
            & cmake --fresh -B build\rocm --preset "ROCm 6" -G Ninja -DOLLAMA_RUNNER_DIR="rocm" `
                -DCMAKE_C_COMPILER=clang `
                -DCMAKE_CXX_COMPILER=clang++ `
                -DCMAKE_C_FLAGS="-parallel-jobs=4 -Wno-ignored-attributes -Wno-deprecated-pragma" `
                -DCMAKE_CXX_FLAGS="-parallel-jobs=4 -Wno-ignored-attributes -Wno-deprecated-pragma" `
                --install-prefix $script:DIST_DIR -DOLLAMA_RUNNER_DIR="rocm"
            if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
            $env:HIPCXX=""
            $env:HIP_PLATFORM=""
            $env:CMAKE_PREFIX_PATH=""
            & cmake --build build\rocm --target ggml-hip --config Release --parallel $script:JOBS
            if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
            & cmake --install build\rocm --component "HIP" --strip
            if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
            Remove-Item -Path $script:DIST_DIR\lib\ollama\rocm\rocblas\library\*gfx906* -ErrorAction SilentlyContinue
        }
    }
}

function buildVulkan(){
    if ($env:VULKAN_SDK) {
        write-host "Building Vulkan backend libraries"
        & cmake --fresh --preset Vulkan --install-prefix $script:DIST_DIR -DOLLAMA_RUNNER_DIR="vulkan"
        if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
        & cmake --build --preset Vulkan  --config Release --parallel $script:JOBS
        if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
        & cmake --install build --component Vulkan --strip
        if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
    }
}

function buildOllama() {
    mkdir -Force -path "${script:DIST_DIR}\"
    write-host "Building ollama CLI"
    & go build -trimpath -ldflags "-s -w -X=github.com/ollama/ollama/version.Version=$script:VERSION -X=github.com/ollama/ollama/server.mode=release" .
    if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
    cp .\ollama.exe "${script:DIST_DIR}\"
}

function buildApp() {
    write-host "Building Ollama App"
    cd "${script:SRC_DIR}\app"
    & windres -l 0 -o ollama.syso ollama.rc
    & go build -trimpath -ldflags "-s -w -H windowsgui -X=github.com/ollama/ollama/version.Version=$script:VERSION -X=github.com/ollama/ollama/server.mode=release" -o "${script:SRC_DIR}\dist\windows-${script:TARGET_ARCH}-app.exe" .
    if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
}

function gatherDependencies() {
    if ($null -eq $env:VCToolsRedistDir) {
        write-error "Unable to locate VC Install location - please use a Developer shell"
        exit 1
    }
    write-host "Gathering runtime dependencies from $env:VCToolsRedistDir"
    cd "${script:SRC_DIR}"
    md "${script:DIST_DIR}\lib\ollama" -ea 0 > $null

    # TODO - this varies based on host build system and MSVC version - drive from dumpbin output
    # currently works for Win11 + MSVC 2019 + Cuda V11
    if ($script:TARGET_ARCH -eq "amd64") {
        $depArch="x64"
    } else {
        $depArch=$script:TARGET_ARCH
    }
    if ($depArch -eq "x64") {
        write-host "cp ${env:VCToolsRedistDir}\${depArch}\Microsoft.VC*.CRT\msvcp140*.dll ${script:DIST_DIR}\lib\ollama\"
        cp "${env:VCToolsRedistDir}\${depArch}\Microsoft.VC*.CRT\msvcp140*.dll" "${script:DIST_DIR}\lib\ollama\"
        write-host "cp ${env:VCToolsRedistDir}\${depArch}\Microsoft.VC*.CRT\vcruntime140.dll ${script:DIST_DIR}\lib\ollama\"
        cp "${env:VCToolsRedistDir}\${depArch}\Microsoft.VC*.CRT\vcruntime140.dll" "${script:DIST_DIR}\lib\ollama\"
        write-host "cp ${env:VCToolsRedistDir}\${depArch}\Microsoft.VC*.CRT\vcruntime140_1.dll ${script:DIST_DIR}\lib\ollama\"
        cp "${env:VCToolsRedistDir}\${depArch}\Microsoft.VC*.CRT\vcruntime140_1.dll" "${script:DIST_DIR}\lib\ollama\"
        $llvmCrtDir="$env:VCToolsRedistDir\..\..\..\Tools\Llvm\${depArch}\bin"
        foreach ($part in $("runtime", "stdio", "filesystem", "math", "convert", "heap", "string", "time", "locale", "environment")) {
            write-host "cp ${llvmCrtDir}\api-ms-win-crt-${part}*.dll ${script:DIST_DIR}\lib\ollama\"
            cp "${llvmCrtDir}\api-ms-win-crt-${part}*.dll" "${script:DIST_DIR}\lib\ollama\"
        }
    } else {
        # Carying the dll's doesn't seem to work, so use the redist installer
        copy-item -path "${env:VCToolsRedistDir}\vc_redist.arm64.exe" -destination "${script:DIST_DIR}" -verbose
    }

    cp "${script:SRC_DIR}\app\ollama_welcome.ps1" "${script:SRC_DIR}\dist\"
}

function sign() {
    if ("${env:KEY_CONTAINER}") {
        write-host "Signing Ollama executables, scripts and libraries"
        & "${script:SignTool}" sign /v /fd sha256 /t http://timestamp.digicert.com /f "${script:OLLAMA_CERT}" `
            /csp "Google Cloud KMS Provider" /kc ${env:KEY_CONTAINER} `
            $(get-childitem -path "${script:SRC_DIR}\dist" -r -include @('ollama_welcome.ps1')) `
            $(get-childitem -path "${script:SRC_DIR}\dist\windows-*" -r -include @('*.exe', '*.dll'))
        if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
    } else {
        write-host "Signing not enabled"
    }
}

function buildInstaller() {
    if ($null -eq ${script:INNO_SETUP_DIR}) {
        write-host "Inno Setup not present, skipping installer build"
        return
    }
    write-host "Building Ollama Installer"
    cd "${script:SRC_DIR}\app"
    $env:PKG_VERSION=$script:PKG_VERSION
    if ("${env:KEY_CONTAINER}") {
        & "${script:INNO_SETUP_DIR}\ISCC.exe" /DARCH=$script:TARGET_ARCH /SMySignTool="${script:SignTool} sign /fd sha256 /t http://timestamp.digicert.com /f ${script:OLLAMA_CERT} /csp `$qGoogle Cloud KMS Provider`$q /kc ${env:KEY_CONTAINER} `$f" .\ollama.iss
    } else {
        & "${script:INNO_SETUP_DIR}\ISCC.exe" /DARCH=$script:TARGET_ARCH .\ollama.iss
    }
    if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
}

function distZip() {
    foreach ($arch in @('amd64', 'arm64')) {
        if (Test-Path -Path "${script:SRC_DIR}\dist\windows-${arch}") {
            # CPU bundle
            $name = "cpu"
            write-host "Creating ${arch} ${name} bundle"
            $gpuDirs = @('rocm', 'cuda_v12', 'cuda_v13')
            Remove-Item -ea 0 -r "${script:SRC_DIR}\dist\windows-${arch}-${name}"
            Copy-Item -Recurse -path "${script:SRC_DIR}\dist\windows-${arch}" -destination "${script:SRC_DIR}\dist\windows-${arch}-${name}"
            foreach ($dir in $gpuDirs) {
                Remove-Item -ea 0 -r "${script:SRC_DIR}\dist\windows-${arch}-${name}\lib\ollama\${dir}"
            }
            Compress-Archive -CompressionLevel Optimal -Path "${script:SRC_DIR}\dist\windows-${arch}-${name}\*" -DestinationPath "${script:SRC_DIR}\dist\ollama-windows-${arch}-${name}.zip" -Force
            Remove-Item -ea 0 -r "${script:SRC_DIR}\dist\windows-${arch}-${name}"

            # GPU bundles if present
            foreach ($dir in $gpuDirs) {
                $name = $dir -replace '_', '-'
                if (Test-Path -Path "${script:SRC_DIR}\dist\windows-${arch}\lib\ollama\${dir}") {
                    write-host "Creating ${arch} ${name} bundle"
                    Remove-Item -ea 0 -r "${script:SRC_DIR}\dist\windows-${arch}-${name}"
                    mkdir -Force -path "${script:SRC_DIR}\dist\windows-${arch}-${name}\lib\ollama"
                    Copy-Item -Recurse -path "${script:SRC_DIR}\dist\windows-${arch}\lib\ollama\${dir}" -destination "${script:SRC_DIR}\dist\windows-${arch}-${name}\lib\ollama"
                    Compress-Archive -CompressionLevel Optimal -Path "${script:SRC_DIR}\dist\windows-${arch}-${name}\*" -DestinationPath "${script:SRC_DIR}\dist\ollama-windows-${arch}-${name}.zip" -Force
                    Remove-Item -ea 0 -r "${script:SRC_DIR}\dist\windows-${arch}-${name}"
                }
            }
        }
    }
}

checkEnv
try {
    if ($($args.count) -eq 0) {
        buildCPU
        buildCUDA12
        buildCUDA13
        buildROCm
        buildVulkan
        buildOllama
        buildApp
        gatherDependencies
        sign
        buildInstaller
        distZip
    } else {
        for ( $i = 0; $i -lt $args.count; $i++ ) {
            write-host "performing $($args[$i])"
            & $($args[$i])
        } 
    }
} catch {
    write-host "Build Failed"
    write-host $_
} finally {
    set-location $script:SRC_DIR
    $env:PKG_VERSION=""
}