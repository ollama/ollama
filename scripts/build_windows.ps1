#!powershell
#
# powershell -ExecutionPolicy Bypass -File .\scripts\build_windows.ps1
#
# gcloud auth application-default login

$ErrorActionPreference = "Stop"

mkdir -Force -path .\dist | Out-Null

function checkEnv {
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
        write-host "Available CUDA Versions: $script:CUDA_DIRS"
    } else {
        write-host "No CUDA versions detected"
    }

    # Locate ROCm version
    if ($null -ne $env:HIP_PATH) {
        $script:HIP_PATH=$env:HIP_PATH
    } else {
        $script:HIP_PATH=(get-item "C:\Program Files\AMD\ROCm\*\bin\" -ea 'silentlycontinue' | sort-object -Descending)
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
    $script:JOBS=([Environment]::ProcessorCount)
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
            write-host "Building CUDA v$cudaMajorVer backend libraries $cuda"
            $env:CUDAToolkit_ROOT=$cuda
            & cmake -B build\cuda_v$cudaMajorVer --preset "CUDA $cudaMajorVer" -T cuda="$cuda" -DCMAKE_CUDA_COMPILER="$cuda\bin\nvcc.exe" -G "Visual Studio 16 2019" --install-prefix "$script:DIST_DIR"
            if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
            & cmake --build build\cuda_v$cudaMajorVer --target ggml-cuda --config Release --parallel $script:JOBS
            if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
            & cmake --install build\cuda_v$cudaMajorVer --component "CUDA" --strip
            if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
        } else {
            write-host "CUDA v$cudaMajorVer not detected, skipping"
        }
    } else {
        write-host "not arch we wanted"
    }
    write-host "done"
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
            write-host "Building CUDA v$cudaMajorVer backend libraries $cuda"
            $env:CUDAToolkit_ROOT=$cuda
            & cmake -B build\cuda_v$cudaMajorVer --preset "CUDA $cudaMajorVer" -T cuda="$cuda" --install-prefix "$script:DIST_DIR"
            if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
            & cmake --build build\cuda_v$cudaMajorVer --target ggml-cuda --config Release --parallel $script:JOBS
            if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
            & cmake --install build\cuda_v$cudaMajorVer --component "CUDA" --strip
            if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
        } else {
            write-host "CUDA v$cudaMajorVer not detected, skipping"
        }
    }
}

function cuda12 {
    cudaCommon("12")
}

function cuda13 {
    cudaCommon("13")
}

function rocm {
    mkdir -Force -path "${script:DIST_DIR}\" | Out-Null
    if ($script:ARCH -ne "arm64") {
        if ($script:HIP_PATH) {
            write-host "Building ROCm backend libraries $script:HIP_PATH"
            if (-Not (get-command -ErrorAction silent ninja)) {
                $NINJA_DIR=(gci -path (Get-CimInstance MSFT_VSInstance -Namespace root/cimv2/vs)[0].InstallLocation -r -fi ninja.exe).Directory.FullName
                $env:PATH="$NINJA_DIR;$env:PATH"
            }
            $env:HIPCXX="${script:HIP_PATH}\bin\clang++.exe"
            $env:HIP_PLATFORM="amd"
            $env:CMAKE_PREFIX_PATH="${script:HIP_PATH}"
            & cmake -B build\rocm --preset "ROCm 6" -G Ninja `
                -DCMAKE_C_COMPILER=clang `
                -DCMAKE_CXX_COMPILER=clang++ `
                -DCMAKE_C_FLAGS="-parallel-jobs=4 -Wno-ignored-attributes -Wno-deprecated-pragma" `
                -DCMAKE_CXX_FLAGS="-parallel-jobs=4 -Wno-ignored-attributes -Wno-deprecated-pragma" `
                --install-prefix $script:DIST_DIR
            if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
            $env:HIPCXX=""
            $env:HIP_PLATFORM=""
            $env:CMAKE_PREFIX_PATH=""
            & cmake --build build\rocm --target ggml-hip --config Release --parallel $script:JOBS
            if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
            & cmake --install build\rocm --component "HIP" --strip
            if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
            Remove-Item -Path $script:DIST_DIR\lib\ollama\rocm\rocblas\library\*gfx906* -ErrorAction SilentlyContinue
        } else {
            write-host "ROCm not detected, skipping"
        }
    }
}

function vulkan {
    if ($env:VULKAN_SDK) {
        write-host "Building Vulkan backend libraries"
        & cmake -B build\vulkan --preset Vulkan --install-prefix $script:DIST_DIR
        if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
        & cmake --build build\vulkan --target ggml-vulkan --config Release --parallel $script:JOBS
        if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
        & cmake --install build\vulkan  --component Vulkan --strip
        if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
    } else {
        write-host "Vulkan not detected, skipping"
    }
}

function ollama {
    mkdir -Force -path "${script:DIST_DIR}\" | Out-Null
    write-host "Building ollama CLI"
    & go build -trimpath -ldflags "-s -w -X=github.com/ollama/ollama/version.Version=$script:VERSION -X=github.com/ollama/ollama/server.mode=release" .
    if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
    cp .\ollama.exe "${script:DIST_DIR}\"
}

function app {
    write-host "Building Ollama App $script:VERSION with package version $script:PKG_VERSION"

    if (!(Get-Command npm -ErrorAction SilentlyContinue)) {
        write-host "npm is not installed. Please install Node.js and npm first:"
        write-host "   Visit: https://nodejs.org/"
        exit 1
    }

    if (!(Get-Command tsc -ErrorAction SilentlyContinue)) {
        write-host "Installing TypeScript compiler..."
        npm install -g typescript
    }
    if (!(Get-Command tscriptify -ErrorAction SilentlyContinue)) {
        write-host "Installing tscriptify..."
        go install github.com/tkrajina/typescriptify-golang-structs/tscriptify@latest
    }
    if (!(Get-Command tscriptify -ErrorAction SilentlyContinue)) {
        $env:PATH="$env:PATH;$(go env GOPATH)\bin"
    }

    Push-Location app/ui/app
    npm install
    if ($LASTEXITCODE -ne 0) { 
        write-host "ERROR: npm install failed with exit code $LASTEXITCODE"
        exit $LASTEXITCODE
    }

    write-host "Building React application..."
    npm run build
    if ($LASTEXITCODE -ne 0) { 
        write-host "ERROR: npm run build failed with exit code $LASTEXITCODE"
        exit $LASTEXITCODE
    }

    # Check if dist directory exists and has content
    if (!(Test-Path "dist")) {
        write-host "ERROR: dist directory was not created by npm run build"
        exit 1
    }

    $distFiles = Get-ChildItem "dist" -Recurse
    if ($distFiles.Count -eq 0) {
        write-host "ERROR: dist directory is empty after npm run build"
        exit 1
    }

    Pop-Location

    write-host "Running go generate"
    & go generate ./...
    if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
	& go build -trimpath -ldflags "-s -w -H windowsgui -X=github.com/ollama/ollama/app/version.Version=$script:VERSION" -o .\dist\windows-ollama-app-${script:ARCH}.exe ./app/cmd/app/
    if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
}

function deps {
    write-host "Download MSVC Redistributables"
    mkdir -Force -path "${script:SRC_DIR}\dist\\windows-arm64" | Out-Null
    mkdir -Force -path "${script:SRC_DIR}\dist\\windows-amd64" | Out-Null
    invoke-webrequest -Uri "https://aka.ms/vs/17/release/vc_redist.arm64.exe" -OutFile  "${script:SRC_DIR}\dist\windows-arm64\vc_redist.arm64.exe"
    invoke-webrequest -Uri "https://aka.ms/vs/17/release/vc_redist.x64.exe" -OutFile  "${script:SRC_DIR}\dist\windows-amd64\vc_redist.x64.exe"
    write-host "Done."
}

function sign {
    if ("${env:KEY_CONTAINER}") {
        write-host "Signing Ollama executables, scripts and libraries"
        & "${script:SignTool}" sign /v /fd sha256 /t http://timestamp.digicert.com /f "${script:OLLAMA_CERT}" `
            /csp "Google Cloud KMS Provider" /kc ${env:KEY_CONTAINER} `
            $(get-childitem -path "${script:SRC_DIR}\dist\windows-*" -r -include @('*.exe', '*.dll'))
        if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
    } else {
        write-host "Signing not enabled"
    }
}

function installer {
    if ($null -eq ${script:INNO_SETUP_DIR}) {
        write-host "ERROR: missing Inno Setup installation directory - install from https://jrsoftware.org/isdl.php"
        exit 1
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

function zip {
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
        rocm
        vulkan
        ollama
        app
        deps
        sign
        installer
        zip
    } else {
        for ( $i = 0; $i -lt $args.count; $i++ ) {
            write-host "running build step $($args[$i])"
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