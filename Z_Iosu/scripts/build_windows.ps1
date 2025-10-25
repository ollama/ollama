#!powershell
#
# powershell -ExecutionPolicy Bypass -File .\scripts\build_windows.ps1
#
# gcloud auth application-default login

$ErrorActionPreference = "Stop"

# Auto-cargar entorno Vulkan si existe configuración local
try {
    $vulkanEnvPath = Join-Path (Split-Path $PSScriptRoot -Parent) "config\vulkan-env.ps1"
    if (Test-Path $vulkanEnvPath) {
        . $vulkanEnvPath
        Write-Host "[INFO] Vulkan env cargado: $vulkanEnvPath" -ForegroundColor Gray
    }
} catch {
    Write-Host "[WARN] No se pudo cargar vulkan-env.ps1: $($_.Exception.Message)" -ForegroundColor Yellow
}

function initVS2022Env() {
    # Initialize Visual Studio 2022 Developer Environment using vcvarsall.bat
    $vcvarsPath = "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvarsall.bat"
    if (Test-Path $vcvarsPath) {
        Write-Host "Initializing Visual Studio 2022 environment..."
        # Execute vcvarsall.bat and capture environment variables
        $tempFile = [System.IO.Path]::GetTempFileName()
        cmd /c "`"$vcvarsPath`" amd64 && set > `"$tempFile`""
        Get-Content $tempFile | ForEach-Object {
            if ($_ -match '^([^=]+)=(.*)$') {
                [System.Environment]::SetEnvironmentVariable($matches[1], $matches[2])
            }
        }
        Remove-Item $tempFile
        Write-Host "Visual Studio 2022 environment initialized successfully"
    } else {
        Write-Host "Warning: VS 2022 vcvarsall.bat not found at $vcvarsPath"
        $MSVC_INSTALL=(Get-CimInstance MSFT_VSInstance -Namespace root/cimv2/vs | Where-Object {$_.InstallLocation -like "*2022*"})[0].InstallLocation
        if ($MSVC_INSTALL) {
            $vcvarsAlt = "$MSVC_INSTALL\VC\Auxiliary\Build\vcvarsall.bat"
            if (Test-Path $vcvarsAlt) {
                $tempFile = [System.IO.Path]::GetTempFileName()
                cmd /c "`"$vcvarsAlt`" amd64 && set > `"$tempFile`""
                Get-Content $tempFile | ForEach-Object {
                    if ($_ -match '^([^=]+)=(.*)$') {
                        [System.Environment]::SetEnvironmentVariable($matches[1], $matches[2])
                    }
                }
                Remove-Item $tempFile
                Write-Host "Visual Studio 2022 environment initialized from $MSVC_INSTALL"
            }
        }
    }
}

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

        # Enable ccache for CMake
        $env:CMAKE_C_COMPILER_LAUNCHER="ccache"
        $env:CMAKE_CXX_COMPILER_LAUNCHER="ccache"
        
        & cmake --fresh --preset CPU --install-prefix $script:DIST_DIR
        if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
        & cmake --build --preset CPU  --config Release --parallel $script:JOBS
        if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
        & cmake --install build --component CPU --strip
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
            & cmake --fresh --preset "CUDA 11" -T cuda="$cuda" -DCMAKE_CUDA_COMPILER="$cuda\bin\nvcc.exe" -G "Visual Studio 16 2019" --install-prefix $script:DIST_DIR -DOLLAMA_RUNNER_DIR="cuda_v11"
            if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
            & cmake --build --preset "CUDA 11"  --config Release --parallel $script:JOBS
            if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
            & cmake --install build --component "CUDA" --strip
            if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
        }
    }
}

function buildCUDA12() {
    mkdir -Force -path "${script:DIST_DIR}\"
    if ($script:ARCH -ne "arm64") {
        $hashEnv = @{}
        Get-ChildItem env: | foreach { $hashEnv[$_.Name] = $_.Value }
        if ("$script:CUDA_DIRS".Contains("v13")) {
            $hashEnv.Keys | foreach { if ($_.Contains("CUDA_PATH_V13")) { $x=$hashEnv[$_]; if (test-path -literalpath "$x\bin\nvcc.exe" ) { $cuda=$x}  }}
            write-host "Building CUDA v13 backend libraries $cuda"
            $env:CUDAToolkit_ROOT=$cuda
            # Enable ccache for CMake
            $env:CMAKE_C_COMPILER_LAUNCHER="ccache"
            $env:CMAKE_CXX_COMPILER_LAUNCHER="ccache"
            $env:CMAKE_CUDA_COMPILER_LAUNCHER="ccache"
            & cmake --fresh --preset "CUDA 13" -G "Visual Studio 17 2022" -T cuda="$cuda" --install-prefix $script:DIST_DIR -DOLLAMA_RUNNER_DIR="cuda_v13"
            if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
            & cmake --build --preset "CUDA 12"  --config Release --parallel $script:JOBS
            if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
            & cmake --install build --component "CUDA" --strip
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
            # Enable ccache for CMake
            $env:CMAKE_C_COMPILER_LAUNCHER="ccache"
            $env:CMAKE_CXX_COMPILER_LAUNCHER="ccache"
            $env:CMAKE_CUDA_COMPILER_LAUNCHER="ccache"
            & cmake --fresh --preset "CUDA 13" -G "Visual Studio 17 2022" -T cuda="$cuda" --install-prefix $script:DIST_DIR -DOLLAMA_RUNNER_DIR="cuda_v13"
            if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
            & cmake --build --preset "CUDA 13"  --config Release --parallel $script:JOBS
            if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
            & cmake --install build --component "CUDA" --strip
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
            & cmake --fresh --preset "ROCm 6" -G Ninja -DOLLAMA_RUNNER_DIR="rocm" `
                -DCMAKE_C_COMPILER=clang `
                -DCMAKE_CXX_COMPILER=clang++ `
                -DCMAKE_C_FLAGS="-parallel-jobs=4 -Wno-ignored-attributes -Wno-deprecated-pragma" `
                -DCMAKE_CXX_FLAGS="-parallel-jobs=4 -Wno-ignored-attributes -Wno-deprecated-pragma" `
                --install-prefix $script:DIST_DIR
            if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
            $env:HIPCXX=""
            $env:HIP_PLATFORM=""
            $env:CMAKE_PREFIX_PATH=""
            & cmake --build --preset "ROCm 6" --config Release --parallel $script:JOBS
            if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
            & cmake --install build --component "HIP" --strip
            if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
            Remove-Item -Path $script:DIST_DIR\lib\ollama\rocm\rocblas\library\*gfx906* -ErrorAction SilentlyContinue
        }
    }
}

function buildVulkan() {
    mkdir -Force -path "${script:DIST_DIR}\"
    if ($script:ARCH -ne "arm64") {
        if (-not $env:VULKAN_SDK) {
            $defaultVulkanSdk = "C:\VulkanSDK\1.4.328.1"
            if (Test-Path $defaultVulkanSdk) {
                $env:VULKAN_SDK = $defaultVulkanSdk
            } else {
                $latestSdk = Get-ChildItem -Path "C:\VulkanSDK\*" -Directory -ErrorAction SilentlyContinue |
                    Sort-Object Name -Descending |
                    Select-Object -First 1
                if ($latestSdk) {
                    $env:VULKAN_SDK = $latestSdk.FullName
                }
            }
        }

        if ($env:VULKAN_SDK) {
            write-host "Building Vulkan backend libraries using SDK: $env:VULKAN_SDK" -ForegroundColor Cyan

            $vulkanBin = Join-Path $env:VULKAN_SDK "Bin"
            if (Test-Path $vulkanBin) {
                $env:PATH = "$vulkanBin;$env:PATH"
            }
            $env:CMAKE_PREFIX_PATH = $env:VULKAN_SDK

            & cmake --fresh --preset Vulkan --install-prefix $script:DIST_DIR -DOLLAMA_RUNNER_DIR="vulkan"
            if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}

            & cmake --build --preset Vulkan --config Release --parallel $script:JOBS
            if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}

            & cmake --install build --component Vulkan --strip
            if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}

            write-host "✅ Vulkan backend compiled successfully" -ForegroundColor Green
        } else {
            write-host "⚠️  Vulkan SDK not found - skipping Vulkan backend" -ForegroundColor Yellow
            write-host "   Install Vulkan SDK with: powershell -ExecutionPolicy Bypass -File Z_Iosu\scripts\install-vulkan-sdk.ps1" -ForegroundColor Cyan
        }
    }
}

function buildOllama() {
    mkdir -Force -path "${script:DIST_DIR}\"
    write-host "Building ollama CLI with llvm-mingw" -ForegroundColor Cyan
    
    # Save current environment
    $savedPATH = $env:PATH
    $savedCGO = $env:CGO_ENABLED
    $savedCC = $env:CC
    $savedCXX = $env:CXX
    
    try {
        # Use the EXACT same configuration that works manually
        $env:CGO_ENABLED = "1"
        $llvmPath = (Resolve-Path "C:\llvm-mingw-ucrt\llvm-mingw-*").Path
        $env:PATH = "$llvmPath\bin;$env:PATH"
        $env:CC = "$llvmPath\bin\gcc.exe"
        $env:CXX = "$llvmPath\bin\g++.exe"
        
        write-host "Using llvm-mingw UCRT from: $llvmPath" -ForegroundColor Green
        write-host "Starting Go build..." -ForegroundColor Yellow
        
        # Build directly to the distribution directory to avoid cluttering root
        New-Item -ItemType Directory -Force -Path "${script:DIST_DIR}" | Out-Null
        $outputPath = "${script:DIST_DIR}\ollama.exe"
        
        # Execute build command with output directly to distribution directory
        & go build -v -a -trimpath -ldflags "-s -w -X=github.com/ollama/ollama/version.Version=$script:VERSION -X=github.com/ollama/ollama/server.mode=release" -o "$outputPath" .
        
        # Check if ollama.exe was created in the correct location
        if (Test-Path "$outputPath") {
            $size = [math]::Round((Get-Item "$outputPath").Length/1MB, 2)
            write-host "✅ ollama.exe built successfully ($size MB) at $outputPath" -ForegroundColor Green
        } else {
            write-error "Build failed: ollama.exe not found at $outputPath"
            exit 1
        }
    } finally {
        # Restore environment
        $env:PATH = $savedPATH
        $env:CGO_ENABLED = $savedCGO
        $env:CC = $savedCC
        $env:CXX = $savedCXX
    }
}

function buildApp() {
    write-host "Building Ollama App with MSVC (NOT llvm-mingw to avoid context menu bug)"
    cd "${script:SRC_DIR}\app"
    
    # Clean previous build artifacts
    Remove-Item "ollama.syso" -ErrorAction SilentlyContinue
    Remove-Item "${script:SRC_DIR}\dist\windows-${script:TARGET_ARCH}-app.exe" -ErrorAction SilentlyContinue
    
    # Generate Windows resources
    & windres -l 0 -o ollama.syso ollama.rc
    if ($LASTEXITCODE -ne 0) { 
        write-error "windres failed"
        exit($LASTEXITCODE)
    }
    
    # CRITICAL: Clear llvm-mingw environment to use MSVC for Win32 API compatibility
    # llvm-mingw produces broken context menus in system tray apps
    $env:CGO_ENABLED="1"
    $env:CC=""
    $env:CXX=""
    Remove-Item env:\CGO_CFLAGS -ErrorAction SilentlyContinue
    Remove-Item env:\CGO_CXXFLAGS -ErrorAction SilentlyContinue
    
    write-host "Compiling GUI app with MSVC (ensures working context menus)"
    & go build -trimpath -ldflags "-s -w -H windowsgui -X=github.com/ollama/ollama/version.Version=$script:VERSION -X=github.com/ollama/ollama/server.mode=release" -o "${script:SRC_DIR}\dist\windows-${script:TARGET_ARCH}-app.exe" .
    if ($LASTEXITCODE -ne 0) { 
        write-error "App build failed"
        exit($LASTEXITCODE)
    }
    
    $appSize = [math]::Round((Get-Item "${script:SRC_DIR}\dist\windows-${script:TARGET_ARCH}-app.exe").Length/1MB, 2)
    write-host "App built successfully with MSVC (size: ${appSize} MB)" -ForegroundColor Green
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

    # Copiar bibliotecas Vulkan si están disponibles
    $vulkanDll = "build\lib\ggml-vulkan.dll"
    if (Test-Path $vulkanDll) {
        write-host "Copying Vulkan backend: $vulkanDll → ${script:DIST_DIR}\lib\ollama\"
        cp $vulkanDll "${script:DIST_DIR}\lib\ollama\"
        write-host "✅ ggml-vulkan.dll copied" -ForegroundColor Green
    }

    if ($env:VULKAN_SDK) {
        $runtimeDirs = @(
            (Join-Path $env:VULKAN_SDK "RuntimeInstaller"),
            (Join-Path $env:VULKAN_SDK "Redist"),
            (Join-Path $env:VULKAN_SDK "Bin"))
        $runtimeInstaller = $null
        foreach ($dir in $runtimeDirs) {
            if (-not (Test-Path $dir)) { continue }
            $candidate = Get-ChildItem -Path $dir -Filter "Vulkan*.exe" -ErrorAction SilentlyContinue |
                Sort-Object LastWriteTime -Descending |
                Select-Object -First 1
            if ($candidate) {
                $runtimeInstaller = $candidate
                break
            }
        }
        if ($runtimeInstaller) {
            $destInstaller = Join-Path "${script:SRC_DIR}\dist" "vulkan_runtime_installer.exe"
            Copy-Item -Path $runtimeInstaller.FullName -Destination $destInstaller -Force
            write-host "Including Vulkan runtime installer: $($runtimeInstaller.FullName)" -ForegroundColor Cyan
        } else {
            write-host "Vulkan runtime installer not found under $env:VULKAN_SDK" -ForegroundColor Yellow
        }
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
    if (Test-Path -Path "${script:SRC_DIR}\dist\windows-amd64") {
        if (Test-Path -Path "${script:SRC_DIR}\dist\windows-amd64\lib\ollama\rocm") {
            write-host "Generating stand-alone distribution zip file ${script:SRC_DIR}\dist\ollama-windows-amd64-rocm.zip"
            # Temporarily adjust paths so we can retain the same directory structure
            Remove-Item -ea 0 -r "${script:SRC_DIR}\dist\windows-amd64-rocm"
            mkdir -Force -path "${script:SRC_DIR}\dist\windows-amd64-rocm\lib\ollama"
            Write-Output "Extract this ROCm zip file to the same location where you extracted ollama-windows-amd64.zip" > "${script:SRC_DIR}\dist\windows-amd64-rocm\README.txt"
            Move-Item -path "${script:SRC_DIR}\dist\windows-amd64\lib\ollama\rocm" -destination "${script:SRC_DIR}\dist\windows-amd64-rocm\lib\ollama"
            Compress-Archive -CompressionLevel Optimal -Path "${script:SRC_DIR}\dist\windows-amd64-rocm\*" -DestinationPath "${script:SRC_DIR}\dist\ollama-windows-amd64-rocm.zip" -Force
        }

        write-host "Generating stand-alone distribution zip file ${script:SRC_DIR}\dist\ollama-windows-amd64.zip"
        Compress-Archive -CompressionLevel Optimal -Path "${script:SRC_DIR}\dist\windows-amd64\*" -DestinationPath "${script:SRC_DIR}\dist\ollama-windows-amd64.zip" -Force
        if (Test-Path -Path "${script:SRC_DIR}\dist\windows-amd64-rocm") {
            Move-Item -destination "${script:SRC_DIR}\dist\windows-amd64\lib\ollama\rocm" -path "${script:SRC_DIR}\dist\windows-amd64-rocm\lib\ollama\rocm"
        }
    }

    if (Test-Path -Path "${script:SRC_DIR}\dist\windows-arm64") {
        write-host "Generating stand-alone distribution zip file ${script:SRC_DIR}\dist\ollama-windows-arm64.zip"
        Compress-Archive -CompressionLevel Optimal -Path "${script:SRC_DIR}\dist\windows-arm64\*" -DestinationPath "${script:SRC_DIR}\dist\ollama-windows-arm64.zip" -Force
    }
}

initVS2022Env
checkEnv
try {
    if ($($args.count) -eq 0) {
        buildCPU
        buildCUDA12
        buildCUDA13
        buildVulkan
        buildROCm
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
