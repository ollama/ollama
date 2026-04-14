#!powershell
#
# powershell -ExecutionPolicy Bypass -File .\scripts\build_windows.ps1
#
# gcloud auth application-default login

# Use "Continue" so that stderr output from native commands (e.g. CGo warnings)
# is not promoted to a terminating exception by the try/catch block.
# All native commands already check $LASTEXITCODE explicitly.
$ErrorActionPreference = "Continue"

mkdir -Force -path .\dist | Out-Null

function checkEnv {
    if ($null -ne $env:ARCH ) {
        $script:ARCH = $env:ARCH
    } else {
        $arch=([System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture)
        if ($null -ne $arch) {
            $script:ARCH = ($arch.ToString().ToLower()).Replace("x64", "amd64")
        } else {
            Write-Output "WARNING: old powershell detected, assuming amd64 architecture - set `$env:ARCH to override"
            $script:ARCH="amd64"
        }
    }
    $script:TARGET_ARCH=$script:ARCH
    Write-host "Building for ${script:TARGET_ARCH}"
    Write-Output "Locating required tools and paths"
    $script:SRC_DIR=$PWD

    # Locate CUDA versions
    $cudaList=(get-item "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*\bin\" -ea 'silentlycontinue')
    if ($cudaList.length -eq 0) {
        $d=(get-command -ea 'silentlycontinue' nvcc).path
        if ($null -ne $d) {
            $script:CUDA_DIRS=@($d| split-path -parent)
        }
    } else {
        # Favor newer patch versions if available
        $script:CUDA_DIRS=($cudaList | sort-object -Descending)
    }
    if ($script:CUDA_DIRS.length -gt 0) {
        Write-Output "Available CUDA Versions: $script:CUDA_DIRS"
    } else {
        Write-Output "No CUDA versions detected"
    }

    # Locate ROCm v6
    $rocmDir=(get-item "C:\Program Files\AMD\ROCm\6.*" -ea 'silentlycontinue' | sort-object -Descending | select-object -First 1)
    if ($null -ne $rocmDir) {
        $script:HIP_PATH=$rocmDir.FullName
    } elseif ($null -ne $env:HIP_PATH -and $env:HIP_PATH -match '[/\\]6\.') {
        $script:HIP_PATH=$env:HIP_PATH
    }
    
    $inoSetup=(get-item "C:\Program Files*\Inno Setup*\")
    if ($inoSetup.length -gt 0) {
        $script:INNO_SETUP_DIR=$inoSetup[0]
    }

    $script:DIST_DIR="${script:SRC_DIR}\dist\windows-${script:TARGET_ARCH}"
    $env:CGO_ENABLED="1"
    if (-not $env:CGO_CFLAGS) {
        $env:CGO_CFLAGS = "-O3"
    }
    if (-not $env:CGO_CXXFLAGS) {
        $env:CGO_CXXFLAGS = "-O3"
    }
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
    Write-Output "Building Ollama $script:VERSION with package version $script:PKG_VERSION"

    # Note: Windows Kits 10 signtool crashes with GCP's plugin
    if ($null -eq $env:SIGN_TOOL) {
        ${script:SignTool}="C:\Program Files (x86)\Windows Kits\8.1\bin\x64\signtool.exe"
    } else {
        ${script:SignTool}=${env:SIGN_TOOL}
    }
    if ("${env:KEY_CONTAINER}") {
        if (Test-Path "${script:SRC_DIR}\ollama_inc.crt") {
            ${script:OLLAMA_CERT}=$(resolve-path "${script:SRC_DIR}\ollama_inc.crt")
            Write-host "Code signing enabled"
        } else {
            Write-Output "WARNING: KEY_CONTAINER is set but ollama_inc.crt not found at ${script:SRC_DIR}\ollama_inc.crt - code signing disabled"
        }
    } else {
        Write-Output "Code signing disabled - please set KEY_CONTAINERS to sign and copy ollama_inc.crt to the top of the source tree"
    }
    if ($env:OLLAMA_BUILD_PARALLEL) {
        $script:JOBS=[int]$env:OLLAMA_BUILD_PARALLEL
    } else {
        # Use physical core count rather than logical processors (hyperthreads)
        # to avoid saturating the system during builds
        try {
            $cores = (Get-CimInstance Win32_Processor | Measure-Object -Property NumberOfCores -Sum).Sum
        } catch {
            $cores = 0
        }
        if ($cores -gt 0) {
            $script:JOBS = $cores
        } else {
            $script:JOBS = [Environment]::ProcessorCount
        }
    }
    Write-Output "Build parallelism: $script:JOBS (set OLLAMA_BUILD_PARALLEL to override)"
}


function cpu {
    mkdir -Force -path "${script:DIST_DIR}\" | Out-Null
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

function cuda11 {
    # CUDA v11 claims to be compatible with MSVC 2022, but the latest updates are no longer compatible
    # 19.40 is the last compiler version that works, but recent udpates are 19.43
    # So this pins to MSVC 2019 for best compatibility
    mkdir -Force -path "${script:DIST_DIR}\" | Out-Null
    $cudaMajorVer="11"
    if ($script:ARCH -ne "arm64") {
        if ("$script:CUDA_DIRS".Contains("v$cudaMajorVer")) {
            foreach ($d in $Script:CUDA_DIRS){ 
                if ($d.FullName.Contains("v$cudaMajorVer")) {
                    if (test-path -literalpath (join-path -path $d -childpath "nvcc.exe" ) ) {
                        $cuda=($d.FullName|split-path -parent)
                        break
                    }
                }
            }
            Write-Output "Building CUDA v$cudaMajorVer backend libraries $cuda"
            $env:CUDAToolkit_ROOT=$cuda
            & cmake -B build\cuda_v$cudaMajorVer --preset "CUDA $cudaMajorVer" -T cuda="$cuda" -DCMAKE_CUDA_COMPILER="$cuda\bin\nvcc.exe" -G "Visual Studio 16 2019" --install-prefix "$script:DIST_DIR"
            if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
            & cmake --build build\cuda_v$cudaMajorVer --target ggml-cuda --config Release --parallel $script:JOBS
            if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
            & cmake --install build\cuda_v$cudaMajorVer --component "CUDA" --strip
            if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
        } else {
            Write-Output "CUDA v$cudaMajorVer not detected, skipping"
        }
    } else {
        Write-Output "not arch we wanted"
    }
    Write-Output "done"
}

function cudaCommon {
    param (
        [string]$cudaMajorVer
    )
    mkdir -Force -path "${script:DIST_DIR}\" | Out-Null
    if ($script:ARCH -ne "arm64") {
        if ("$script:CUDA_DIRS".Contains("v$cudaMajorVer")) {
            foreach ($d in $Script:CUDA_DIRS){ 
                if ($d.FullName.Contains("v$cudaMajorVer")) {
                    if (test-path -literalpath (join-path -path $d -childpath "nvcc.exe" ) ) {
                        $cuda=($d.FullName|split-path -parent)
                        break
                    }
                }
            }
            Write-Output "Building CUDA v$cudaMajorVer backend libraries $cuda"
            $env:CUDAToolkit_ROOT=$cuda
            & cmake -B build\cuda_v$cudaMajorVer --preset "CUDA $cudaMajorVer" -T cuda="$cuda" --install-prefix "$script:DIST_DIR"
            if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
            & cmake --build build\cuda_v$cudaMajorVer --target ggml-cuda --config Release --parallel $script:JOBS
            if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
            & cmake --install build\cuda_v$cudaMajorVer --component "CUDA" --strip
            if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
        } else {
            Write-Output "CUDA v$cudaMajorVer not detected, skipping"
        }
    }
}

function cuda12 {
    cudaCommon("12")
}

function cuda13 {
    cudaCommon("13")
}

function rocm6 {
    mkdir -Force -path "${script:DIST_DIR}\" | Out-Null
    if ($script:ARCH -ne "arm64") {
        if ($script:HIP_PATH) {
            Write-Output "Building ROCm backend libraries $script:HIP_PATH"
            if (-Not (get-command -ErrorAction silent ninja)) {
                $NINJA_DIR=(gci -path (Get-CimInstance MSFT_VSInstance -Namespace root/cimv2/vs)[0].InstallLocation -r -fi ninja.exe).Directory.FullName
                $env:PATH="$NINJA_DIR;$env:PATH"
            }
            $env:HIPCXX="${script:HIP_PATH}\bin\clang++.exe"
            $env:HIP_PLATFORM="amd"
            $env:CMAKE_PREFIX_PATH="${script:HIP_PATH}"
            # Set CC/CXX via environment instead of -D flags to avoid triggering
            # spurious compiler-change reconfigures that reset CMAKE_INSTALL_PREFIX
            $env:CC="${script:HIP_PATH}\bin\clang.exe"
            $env:CXX="${script:HIP_PATH}\bin\clang++.exe"
            & cmake -B build\rocm --preset "ROCm 6" -G Ninja `
                -DCMAKE_C_FLAGS="-parallel-jobs=4 -Wno-ignored-attributes -Wno-deprecated-pragma" `
                -DCMAKE_CXX_FLAGS="-parallel-jobs=4 -Wno-ignored-attributes -Wno-deprecated-pragma" `
                --install-prefix $script:DIST_DIR
            if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
            $env:HIPCXX=""
            $env:HIP_PLATFORM=""
            $env:CMAKE_PREFIX_PATH=""
            $env:CC=""
            $env:CXX=""
            & cmake --build build\rocm --target ggml-hip --config Release --parallel $script:JOBS
            if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
            & cmake --install build\rocm --component "HIP" --strip
            if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
            Remove-Item -Path $script:DIST_DIR\lib\ollama\rocm\rocblas\library\*gfx906* -ErrorAction SilentlyContinue
        } else {
            Write-Output "ROCm not detected, skipping"
        }
    }
}

function vulkan {
    if ($env:VULKAN_SDK) {
        Write-Output "Building Vulkan backend libraries"
        & cmake -B build\vulkan --preset Vulkan --install-prefix $script:DIST_DIR
        if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
        & cmake --build build\vulkan --target ggml-vulkan --config Release --parallel $script:JOBS
        if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
        & cmake --install build\vulkan  --component Vulkan --strip
        if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
    } else {
        Write-Output "Vulkan not detected, skipping"
    }
}

function mlxCuda13 {
    mkdir -Force -path "${script:DIST_DIR}\" | Out-Null
    $cudaMajorVer="13"
    if ($script:ARCH -ne "arm64") {
        if ("$script:CUDA_DIRS".Contains("v$cudaMajorVer")) {
            foreach ($d in $Script:CUDA_DIRS){
                if ($d.FullName.Contains("v$cudaMajorVer")) {
                    if (test-path -literalpath (join-path -path $d -childpath "nvcc.exe" ) ) {
                        $cuda=($d.FullName|split-path -parent)
                        break
                    }
                }
            }

            # Check for cuDNN - required for MLX CUDA backend
            # Supports two layouts:
            # 1. CI/zip extract: CUDNN\include\cudnn.h, lib\x64\, bin\x64\
            # 2. Official installer: CUDNN\v*\include\{cuda-ver}\cudnn.h, lib\{cuda-ver}\x64\, bin\{cuda-ver}\
            if ($env:CUDNN_INCLUDE_PATH -and $env:CUDNN_LIBRARY_PATH) {
                Write-Output "Using cuDNN from environment: $env:CUDNN_INCLUDE_PATH"
            } elseif (Test-Path "C:\Program Files\NVIDIA\CUDNN\include\cudnn.h") {
                # CI/zip layout (flat)
                $cudnnRoot = "C:\Program Files\NVIDIA\CUDNN"
                $env:CUDNN_ROOT_DIR = $cudnnRoot
                $env:CUDNN_INCLUDE_PATH = "$cudnnRoot\include"
                $env:CUDNN_LIBRARY_PATH = "$cudnnRoot\lib\x64"
                Write-Output "Found cuDNN at $cudnnRoot (flat layout)"
            } else {
                # Official installer layout (versioned)
                $cudnnRoot = $null
                $resolved = Resolve-Path -Path "C:\Program Files\NVIDIA\CUDNN\v*" -ErrorAction SilentlyContinue | Sort-Object -Descending | Select-Object -First 1
                if ($resolved -and (Test-Path "$($resolved.Path)\include\$cudaMajorVer.0\cudnn.h")) {
                    $cudnnRoot = $resolved.Path
                    $env:CUDNN_ROOT_DIR = $cudnnRoot
                    $env:CUDNN_INCLUDE_PATH = "$cudnnRoot\include\$cudaMajorVer.0"
                    $env:CUDNN_LIBRARY_PATH = "$cudnnRoot\lib\$cudaMajorVer.0\x64"
                    Write-Output "Found cuDNN at $cudnnRoot (official installer, CUDA $cudaMajorVer.0)"
                } else {
                    Write-Output "cuDNN not found - set CUDNN_INCLUDE_PATH and CUDNN_LIBRARY_PATH environment variables"
                    Write-Output "Skipping MLX build"
                    return
                }
            }

            Write-Output "Building MLX CUDA v$cudaMajorVer backend libraries $cuda"
            $env:CUDAToolkit_ROOT=$cuda
            & cmake -B build\mlx_cuda_v$cudaMajorVer --preset "MLX CUDA $cudaMajorVer" -T cuda="$cuda" --install-prefix "$script:DIST_DIR"
            if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
            & cmake --build build\mlx_cuda_v$cudaMajorVer --target mlx --target mlxc --config Release --parallel $script:JOBS -- /nodeReuse:false
            if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
            & cmake --install build\mlx_cuda_v$cudaMajorVer --component "MLX" --strip
            if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
        } else {
            Write-Output "CUDA v$cudaMajorVer not detected, skipping MLX build"
        }
    }
}

function ollama {
    mkdir -Force -path "${script:DIST_DIR}\" | Out-Null
    Write-Output "Building ollama CLI"
    & go build -trimpath -ldflags "-s -w -X=github.com/ollama/ollama/version.Version=$script:VERSION -X=github.com/ollama/ollama/server.mode=release" .
    if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
    cp .\ollama.exe "${script:DIST_DIR}\"
}

function app {
    Write-Output "Building Ollama App $script:VERSION with package version $script:PKG_VERSION"

    if (!(Get-Command npm -ErrorAction SilentlyContinue)) {
        Write-Output "npm is not installed. Please install Node.js and npm first:"
        Write-Output "   Visit: https://nodejs.org/"
        exit 1
    }

    if (!(Get-Command tsc -ErrorAction SilentlyContinue)) {
        Write-Output "Installing TypeScript compiler..."
        npm install -g typescript
    }
    if (!(Get-Command tscriptify -ErrorAction SilentlyContinue)) {
        Write-Output "Installing tscriptify..."
        go install github.com/tkrajina/typescriptify-golang-structs/tscriptify@latest
    }
    if (!(Get-Command tscriptify -ErrorAction SilentlyContinue)) {
        $env:PATH="$env:PATH;$(go env GOPATH)\bin"
    }

    Push-Location app/ui/app
    npm install
    if ($LASTEXITCODE -ne 0) { 
        Write-Output "ERROR: npm install failed with exit code $LASTEXITCODE"
        exit $LASTEXITCODE
    }

    Write-Output "Building React application..."
    npm run build
    if ($LASTEXITCODE -ne 0) { 
        Write-Output "ERROR: npm run build failed with exit code $LASTEXITCODE"
        exit $LASTEXITCODE
    }

    # Check if dist directory exists and has content
    if (!(Test-Path "dist")) {
        Write-Output "ERROR: dist directory was not created by npm run build"
        exit 1
    }

    $distFiles = Get-ChildItem "dist" -Recurse
    if ($distFiles.Count -eq 0) {
        Write-Output "ERROR: dist directory is empty after npm run build"
        exit 1
    }

    Pop-Location

    Write-Output "Running go generate"
    & go generate ./...
    if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
	& go build -trimpath -ldflags "-s -w -H windowsgui -X=github.com/ollama/ollama/app/version.Version=$script:VERSION" -o .\dist\windows-ollama-app-${script:ARCH}.exe ./app/cmd/app/
    if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
}

function deps {
    Write-Output "Download MSVC Redistributables"
    mkdir -Force -path "${script:SRC_DIR}\dist\\windows-arm64" | Out-Null
    mkdir -Force -path "${script:SRC_DIR}\dist\\windows-amd64" | Out-Null
    invoke-webrequest -Uri "https://aka.ms/vs/17/release/vc_redist.arm64.exe" -OutFile  "${script:SRC_DIR}\dist\windows-arm64\vc_redist.arm64.exe" -ErrorAction Stop
    invoke-webrequest -Uri "https://aka.ms/vs/17/release/vc_redist.x64.exe" -OutFile  "${script:SRC_DIR}\dist\windows-amd64\vc_redist.x64.exe" -ErrorAction Stop
    Write-Output "Done."
}

function sign {
    # Copy install.ps1 to dist for release packaging
    Write-Output "Copying install.ps1 to dist"
    Copy-Item -Path "${script:SRC_DIR}\scripts\install.ps1" -Destination "${script:SRC_DIR}\dist\install.ps1" -ErrorAction Stop

    if ("${env:KEY_CONTAINER}") {
        Write-Output "Signing Ollama executables, scripts and libraries"
        & "${script:SignTool}" sign /v /fd sha256 /t http://timestamp.digicert.com /f "${script:OLLAMA_CERT}" `
            /csp "Google Cloud KMS Provider" /kc ${env:KEY_CONTAINER} `
            $(get-childitem -path "${script:SRC_DIR}\dist\windows-*" -r -include @('*.exe', '*.dll'))
        if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}

        Write-Output "Signing install.ps1"
        & "${script:SignTool}" sign /v /fd sha256 /t http://timestamp.digicert.com /f "${script:OLLAMA_CERT}" `
            /csp "Google Cloud KMS Provider" /kc ${env:KEY_CONTAINER} `
            "${script:SRC_DIR}\dist\install.ps1"
        if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
    } else {
        Write-Output "Signing not enabled"
    }
}

function installer {
    if ($null -eq ${script:INNO_SETUP_DIR}) {
        Write-Output "ERROR: missing Inno Setup installation directory - install from https://jrsoftware.org/isdl.php"
        exit 1
    }
    Write-Output "Building Ollama Installer"
    cd "${script:SRC_DIR}\app"
    $env:PKG_VERSION=$script:PKG_VERSION
    if ("${env:KEY_CONTAINER}") {
        & "${script:INNO_SETUP_DIR}\ISCC.exe" /DARCH=$script:TARGET_ARCH /SMySignTool="${script:SignTool} sign /fd sha256 /t http://timestamp.digicert.com /f ${script:OLLAMA_CERT} /csp `$qGoogle Cloud KMS Provider`$q /kc ${env:KEY_CONTAINER} `$f" .\ollama.iss
    } else {
        & "${script:INNO_SETUP_DIR}\ISCC.exe" /DARCH=$script:TARGET_ARCH .\ollama.iss
    }
    if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
}

function newZipJob($sourceDir, $destZip) {
    $use7z = [bool](Get-Command 7z -ErrorAction SilentlyContinue)
    Start-Job -ScriptBlock {
        param($src, $dst, $use7z)
        if ($use7z) {
            & 7z a -tzip -mx=9 -mmt=on $dst "${src}\*"
            if ($LASTEXITCODE -ne 0) { throw "7z failed with exit code $LASTEXITCODE" }
        } else {
            Compress-Archive -CompressionLevel Optimal -Path "${src}\*" -DestinationPath $dst -Force
        }
    } -ArgumentList $sourceDir, $destZip, $use7z
}

function stageComponents($mainDir, $stagingDir, $pattern, $readmePrefix) {
    $components = Get-ChildItem -Path "${mainDir}\lib\ollama" -Directory -Filter $pattern -ErrorAction SilentlyContinue
    if ($components) {
        Remove-Item -ea 0 -r $stagingDir
        mkdir -Force -path "${stagingDir}\lib\ollama" | Out-Null
        Write-Output "Extract this ${readmePrefix} zip file to the same location where you extracted ollama-windows-amd64.zip" > "${stagingDir}\README_${readmePrefix}.txt"
        foreach ($dir in $components) {
            Write-Output "  Staging $($dir.Name)"
            Move-Item -path $dir.FullName -destination "${stagingDir}\lib\ollama\$($dir.Name)"
        }
        return $true
    }
    return $false
}

function restoreComponents($mainDir, $stagingDir) {
    if (Test-Path -Path "${stagingDir}\lib\ollama") {
        foreach ($dir in (Get-ChildItem -Path "${stagingDir}\lib\ollama" -Directory)) {
            Move-Item -path $dir.FullName -destination "${mainDir}\lib\ollama\$($dir.Name)"
        }
    }
    Remove-Item -ea 0 -r $stagingDir
}

function zip {
    $jobs = @()
    $distDir = "${script:SRC_DIR}\dist"
    $amd64Dir = "${distDir}\windows-amd64"

    # Remove any stale zip files before starting
    Remove-Item -ea 0 "${distDir}\ollama-windows-*.zip"

    try {
        if (Test-Path -Path $amd64Dir) {
            # Stage ROCm into its own directory for independent compression
            if (stageComponents $amd64Dir "${distDir}\windows-amd64-rocm" "rocm*" "ROCm") {
                Write-Output "Generating ${distDir}\ollama-windows-amd64-rocm.zip"
                $jobs += newZipJob "${distDir}\windows-amd64-rocm" "${distDir}\ollama-windows-amd64-rocm.zip"
            }

            # Stage MLX into its own directory for independent compression
            if (stageComponents $amd64Dir "${distDir}\windows-amd64-mlx" "mlx_*" "MLX") {
                Write-Output "Generating ${distDir}\ollama-windows-amd64-mlx.zip"
                $jobs += newZipJob "${distDir}\windows-amd64-mlx" "${distDir}\ollama-windows-amd64-mlx.zip"
            }

            # Compress the main amd64 zip (without rocm/mlx)
            Write-Output "Generating ${distDir}\ollama-windows-amd64.zip"
            $jobs += newZipJob $amd64Dir "${distDir}\ollama-windows-amd64.zip"
        }

        if (Test-Path -Path "${distDir}\windows-arm64") {
            Write-Output "Generating ${distDir}\ollama-windows-arm64.zip"
            $jobs += newZipJob "${distDir}\windows-arm64" "${distDir}\ollama-windows-arm64.zip"
        }

        if ($jobs.Count -gt 0) {
            Write-Output "Waiting for $($jobs.Count) parallel zip jobs..."
            $jobs | Wait-Job | Out-Null
            $failed = $false
            foreach ($job in $jobs) {
                if ($job.State -eq 'Failed') {
                    Write-Error "Zip job failed: $($job.ChildJobs[0].JobStateInfo.Reason)"
                    $failed = $true
                }
                Receive-Job $job
                Remove-Job $job
            }
            if ($failed) { throw "One or more zip jobs failed" }
        }
    } finally {
        # Always restore staged components back into the main tree
        restoreComponents $amd64Dir "${distDir}\windows-amd64-rocm"
        restoreComponents $amd64Dir "${distDir}\windows-amd64-mlx"
    }
}

function clean {
    Remove-Item -ea 0 -r "${script:SRC_DIR}\dist\"
    Remove-Item -ea 0 -r "${script:SRC_DIR}\build\"
}

checkEnv
try {
    if ($($args.count) -eq 0) {
        cpu
        cuda12
        cuda13
        rocm6
        vulkan
        mlxCuda13
        ollama
        app
        deps
        sign
        installer
        zip
    } else {
        for ( $i = 0; $i -lt $args.count; $i++ ) {
            Write-Output "running build step $($args[$i])"
            & $($args[$i])
        } 
    }
} catch {
    Write-Error "Build Failed: $($_.Exception.Message)"
    Write-Error "$($_.ScriptStackTrace)"
} finally {
    set-location $script:SRC_DIR
    $env:PKG_VERSION=""
}