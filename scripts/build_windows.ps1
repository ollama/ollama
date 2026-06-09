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

function findVisualStudioInstall {
    if ($env:VSINSTALLDIR -and (Test-Path $env:VSINSTALLDIR)) {
        return $env:VSINSTALLDIR
    }

    $programFilesX86 = [Environment]::GetEnvironmentVariable("ProgramFiles(x86)")
    if ($programFilesX86) {
        $vswhere = Join-Path $programFilesX86 "Microsoft Visual Studio\Installer\vswhere.exe"
        if (Test-Path $vswhere) {
            $install = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>$null | Select-Object -First 1
            if ($install) {
                return $install
            }
        }
    }

    $instance = Get-CimInstance MSFT_VSInstance -Namespace root/cimv2/vs -ErrorAction SilentlyContinue | Sort-Object -Property Version -Descending | Select-Object -First 1
    if ($instance) {
        return $instance.InstallLocation
    }

    return $null
}

function findDumpbin {
    $dumpbin = Get-Command -Name "dumpbin.exe" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($dumpbin) {
        return $dumpbin.Path
    }

    $vsInstall = findVisualStudioInstall
    if ($vsInstall) {
        $candidate = Get-ChildItem -Path (Join-Path $vsInstall "VC\Tools\MSVC\*\bin\Hostx64\x64\dumpbin.exe") -ErrorAction SilentlyContinue |
            Sort-Object -Property FullName -Descending |
            Select-Object -First 1
        if ($candidate) {
            return $candidate.FullName
        }
    }

    return $null
}

function normalizePathForCompare {
    param([string]$Path)

    if (-not $Path) {
        return ""
    }

    return ([IO.Path]::GetFullPath($Path).TrimEnd('\')).Replace('/', '\').ToLowerInvariant()
}

function newCompilerPair($name, $cc, $cxx) {
    if ((Test-Path $cc) -and (Test-Path $cxx)) {
        return [pscustomobject]@{
            Name = $name
            CC = (Resolve-Path $cc).Path
            CXX = (Resolve-Path $cxx).Path
        }
    }
    return $null
}

function findWindowsCPUCompiler {
    $llvmMingwBins = @()
    if ($env:ProgramFiles) {
        $llvmMingwBins += Resolve-Path "$env:ProgramFiles\llvm-mingw-*-x86_64*\bin" -ErrorAction SilentlyContinue
    }
    if ($env:LOCALAPPDATA) {
        $llvmMingwBins += Resolve-Path "$env:LOCALAPPDATA\Microsoft\WinGet\Packages\MartinStorsjo.LLVM-MinGW*\llvm-mingw-*-x86_64*\bin" -ErrorAction SilentlyContinue
    }
    foreach ($bin in ($llvmMingwBins | Sort-Object -Property Path -Descending)) {
        $compiler = newCompilerPair "llvm-mingw" (Join-Path $bin.Path "x86_64-w64-mingw32-gcc.exe") (Join-Path $bin.Path "x86_64-w64-mingw32-g++.exe")
        if ($compiler) { return $compiler }
        $compiler = newCompilerPair "llvm-mingw" (Join-Path $bin.Path "gcc.exe") (Join-Path $bin.Path "g++.exe")
        if ($compiler) { return $compiler }
    }

    $compiler = newCompilerPair "MSYS2 clang64" "C:\msys64\clang64\bin\clang.exe" "C:\msys64\clang64\bin\clang++.exe"
    if ($compiler) { return $compiler }

    $compiler = newCompilerPair "MSYS2 UCRT64 GCC" "C:\msys64\ucrt64\bin\gcc.exe" "C:\msys64\ucrt64\bin\g++.exe"
    if ($compiler) { return $compiler }

    return $null
}

function ensureMsvcForNinja {
    if ($env:CMAKE_GENERATOR -notlike "Ninja*") {
        return
    }

    if (-not (Get-Command -Name "cl.exe" -ErrorAction SilentlyContinue)) {
        $vsInstall = findVisualStudioInstall
        if ($vsInstall) {
            $devShell = Join-Path $vsInstall "Common7\Tools\Microsoft.VisualStudio.DevShell.dll"
            if (Test-Path $devShell) {
                Import-Module $devShell
                Enter-VsDevShell -VsInstallPath $vsInstall -SkipAutomaticLocation -DevCmdArguments "-arch=x64 -no_logo"
            }
        }
    }

    if (-not (Get-Command -Name "cl.exe" -ErrorAction SilentlyContinue)) {
        Write-Error "Ninja builds require MSVC cl.exe. Install Visual Studio C++ tools or run from a VS Developer shell."
        exit(1)
    }
    Write-Output "MSVC cl.exe available for Ninja builds"
}

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

    $arm64CCPath = (Get-Command -Name "aarch64-w64-mingw32-gcc.exe" -ErrorAction SilentlyContinue | Select-Object -First 1).Path
    $arm64CXXPath = (Get-Command -Name "aarch64-w64-mingw32-g++.exe" -ErrorAction SilentlyContinue | Select-Object -First 1).Path
    if (-not $arm64CCPath -or -not $arm64CXXPath) {
        $arm64Toolchain = Resolve-Path "C:\Program Files\llvm-mingw-*-x86_64*\bin" -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($arm64Toolchain) {
            $arm64CCPath = (Get-Item (Join-Path $arm64Toolchain.Path "aarch64-w64-mingw32-gcc.exe") -ErrorAction SilentlyContinue).FullName
            $arm64CXXPath = (Get-Item (Join-Path $arm64Toolchain.Path "aarch64-w64-mingw32-g++.exe") -ErrorAction SilentlyContinue).FullName
        }
    }
    # TODO: support other Windows ARM64 cross-compile toolchain layouts as needed.
    if ($arm64CCPath -and $arm64CXXPath -and $arm64CCPath -notlike "*\clangarm64\*" -and $arm64CXXPath -notlike "*\clangarm64\*") {
        $script:WINDOWS_ARM64_CROSS_COMPILE = $true
        $script:WINDOWS_ARM64_CC = $arm64CCPath
        $script:WINDOWS_ARM64_CXX = $arm64CXXPath
    } else {
        $script:WINDOWS_ARM64_CROSS_COMPILE = $false
    }

    # Locate ROCm installations
    $rocm7Dir=(get-item "C:\Program Files\AMD\ROCm\7.*" -ea 'silentlycontinue' | sort-object { [version]$_.Name } -Descending | select-object -First 1)
    if ($null -ne $rocm7Dir) {
        $script:HIP_PATH_V7=$rocm7Dir.FullName
    } elseif ($null -ne $env:HIP_PATH -and $env:HIP_PATH -match '[/\\]7\.') {
        $script:HIP_PATH_V7=$env:HIP_PATH
    }
    $rocm6Dir=(get-item "C:\Program Files\AMD\ROCm\6.*" -ea 'silentlycontinue' | sort-object { [version]$_.Name } -Descending | select-object -First 1)
    if ($null -ne $rocm6Dir) {
        $script:HIP_PATH_V6=$rocm6Dir.FullName
    } elseif ($null -ne $env:HIP_PATH -and $env:HIP_PATH -match '[/\\]6\.') {
        $script:HIP_PATH_V6=$env:HIP_PATH
    }
    # Default to v7
    $script:HIP_PATH=$script:HIP_PATH_V7
    if (-not $script:HIP_PATH) {
        $script:HIP_PATH=$script:HIP_PATH_V6
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
    if (!$env:CMAKE_GENERATOR) {
        $ninja = Get-Command -Name "ninja.exe" -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($ninja) {
            $env:CMAKE_GENERATOR = "Ninja"
            Write-Output "Using CMake generator: Ninja"
        } else {
            Write-Output "Ninja not detected; using CMake default generator"
        }
    } else {
        Write-Output "Using CMake generator: $env:CMAKE_GENERATOR"
    }
    ensureMsvcForNinja
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

        $oldCC = $env:CC
        $oldCXX = $env:CXX
        $setCPUCompiler = $false
        try {
            if (-not $env:CC -and -not $env:CXX) {
                $cpuCompiler = findWindowsCPUCompiler
                if ($cpuCompiler) {
                    $env:CC = $cpuCompiler.CC
                    $env:CXX = $cpuCompiler.CXX
                    $setCPUCompiler = $true
                    Write-Output "Using $($cpuCompiler.Name) for Windows CPU backend variants"
                } else {
                    Write-Output "WARNING: llvm-mingw/MSYS2 compiler not found; CPU variants unsupported by MSVC will be skipped"
                }
            } elseif (-not $env:CC -or -not $env:CXX) {
                Write-Output "WARNING: set both CC and CXX for a custom Windows CPU compiler"
            }

            $cpuBuildDir = "${script:SRC_DIR}\build\llama-server-cpu"
            $cache = Join-Path $cpuBuildDir "CMakeCache.txt"
            if ($env:CXX -and (Test-Path $cache)) {
                $cachedCXX = Select-String -Path $cache -Pattern '^CMAKE_CXX_COMPILER:FILEPATH=(.*)$' | Select-Object -First 1
                if ($cachedCXX -and (normalizePathForCompare $cachedCXX.Matches[0].Groups[1].Value) -ne (normalizePathForCompare $env:CXX)) {
                    Write-Output "Reconfiguring Windows CPU build after compiler change"
                    Remove-Item -ea 0 -recurse -force -path $cpuBuildDir
                }
            }

            # Build llama-server from upstream source (CPU + base)
            & cmake -S llama\server --preset cpu_windows --install-prefix $script:DIST_DIR
            if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
            $oldPath = $env:PATH
            if ($env:CXX -and [System.IO.Path]::IsPathRooted($env:CXX)) {
                # llama-ui-embed runs even with UI disabled; this ensures DLL dependencies are found.
                $cpuCompilerDir = Split-Path -Parent $env:CXX
                $env:PATH = "$cpuCompilerDir;$env:PATH"
            }
            & cmake --build build\llama-server-cpu --config Release --parallel $script:JOBS
            $env:PATH = $oldPath
            if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
            & cmake --install build\llama-server-cpu --component llama-server --strip
            if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
        } finally {
            if ($oldPath) {
                $env:PATH = $oldPath
            }
            if ($setCPUCompiler) {
                if ($null -eq $oldCC) {
                    Remove-Item Env:CC -ErrorAction SilentlyContinue
                } else {
                    $env:CC = $oldCC
                }
                if ($null -eq $oldCXX) {
                    Remove-Item Env:CXX -ErrorAction SilentlyContinue
                } else {
                    $env:CXX = $oldCXX
                }
            }
        }
    }
}

function cpuArm64 {
    if (-not $script:WINDOWS_ARM64_CROSS_COMPILE) {
        Write-Output "WARNING: skipping cpuArm64; Windows ARM64 cross-compiling is disabled due to missing tools"
        return
    }

    $arm64DistDir = "${script:SRC_DIR}\dist\windows-arm64"
    mkdir -Force -path "${arm64DistDir}\lib\ollama\" | Out-Null
    Remove-Item -ea 0 -recurse -force -path "${arm64DistDir}\lib\ollama"
    New-Item "${arm64DistDir}\lib\ollama\" -ItemType Directory -ea 0 | Out-Null

    # Cross-compile the Windows ARM64 CPU llama-server payload from an x64 host
    # with llvm-mingw. GPU backends are not built for Windows ARM64.
    $oldCC = $env:CC
    $oldCXX = $env:CXX
    $oldGenerator = $env:CMAKE_GENERATOR
    $oldGeneratorPlatform = $env:CMAKE_GENERATOR_PLATFORM
    $oldGeneratorToolset = $env:CMAKE_GENERATOR_TOOLSET
    $env:CC = $null
    $env:CXX = $null
    $env:CMAKE_GENERATOR_PLATFORM = $null
    $env:CMAKE_GENERATOR_TOOLSET = $null
    & cmake -S llama\server --preset cpu_arm64 --install-prefix $arm64DistDir
    if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
    & cmake --build build\llama-server-cpu_arm64 --config Release --parallel $script:JOBS
    if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
    & cmake --install build\llama-server-cpu_arm64 --component llama-server --strip
    if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
    $env:CC = $oldCC
    $env:CXX = $oldCXX
    $env:CMAKE_GENERATOR = $oldGenerator
    $env:CMAKE_GENERATOR_PLATFORM = $oldGeneratorPlatform
    $env:CMAKE_GENERATOR_TOOLSET = $oldGeneratorToolset
}

function cuda11 {
    Write-Output "CUDA v11 is no longer supported"
}

function cudaCMakeArgs {
    param (
        [string]$cuda
    )

    $env:CUDACXX = "$cuda\bin\nvcc.exe"
    if ($env:CMAKE_GENERATOR -like "Ninja*") {
        return @()
    }
    return @("-T", "cuda=$cuda", "-DCMAKE_CUDA_COMPILER=$cuda\bin\nvcc.exe")
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
            # Build llama-server CUDA backend from upstream source
            Write-Output "Building llama-server CUDA v$cudaMajorVer backend"
            $env:CUDAToolkit_ROOT=$cuda
            $preset = "llama_cuda_v$($cudaMajorVer)_windows"
            $cudaToolsetArgs = cudaCMakeArgs $cuda
            $configureArgs = @("-S", "llama\server", "--preset", $preset) + $cudaToolsetArgs + @("--install-prefix", "$script:DIST_DIR")
            & cmake @configureArgs
            if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
            & cmake --build "build\llama-server-cuda_v$cudaMajorVer" --config Release --parallel $script:JOBS
            if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
            & cmake --install "build\llama-server-cuda_v$cudaMajorVer" --component llama-server --strip
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
    # KNOWN ISSUE: ROCm v6 on Windows is currently broken with upstream llama.cpp b8591+.
    # The vendors/hip.h guard (#if HIP_VERSION >= 60200000) assumes __hip_fp8_e4m3 exists,
    # but Windows ROCm 6.2 only has the _fnuz variant (__hip_fp8_e4m3_fnuz).
    # This causes a compile error in ggml-cuda/vendors/hip.h:240.
    # Use rocm7 instead, or wait for an upstream fix.
    mkdir -Force -path "${script:DIST_DIR}\" | Out-Null
    if ($script:ARCH -ne "arm64") {
        if ($script:HIP_PATH_V6) {
            Write-Output "WARNING: ROCm v6 build is currently broken (FP8 type mismatch). Skipping."
            Write-Output "Use rocm7 instead."
        } else {
            Write-Output "ROCm v6 not detected, skipping"
        }
    }
}

function rocm7 {
    mkdir -Force -path "${script:DIST_DIR}\" | Out-Null
    if ($script:ARCH -ne "arm64") {
        if ($script:HIP_PATH_V7) {
            Write-Output "Building llama-server ROCm v7 backend $script:HIP_PATH_V7"
            $rocmVersion = Split-Path -Leaf $script:HIP_PATH_V7
            if ($rocmVersion -notmatch '^(\d+)\.(\d+)') {
                Write-Output "Unable to determine ROCm version from $script:HIP_PATH_V7"
                exit(1)
            }
            $rocmBackend = "rocm_v$($Matches[1])_$($Matches[2])"
            $rocmPreset = "${rocmBackend}_windows"
            if (-Not (get-command -ErrorAction silent ninja)) {
                $NINJA_DIR=(gci -path (Get-CimInstance MSFT_VSInstance -Namespace root/cimv2/vs)[0].InstallLocation -r -fi ninja.exe).Directory.FullName
                $env:PATH="$NINJA_DIR;$env:PATH"
            }
            $oldHIPCXX = $env:HIPCXX
            $oldHIP_PLATFORM = $env:HIP_PLATFORM
            $oldCMAKE_PREFIX_PATH = $env:CMAKE_PREFIX_PATH
            $oldCC = $env:CC
            $oldCXX = $env:CXX
            $env:HIPCXX="${script:HIP_PATH_V7}\bin\clang++.exe"
            $env:HIP_PLATFORM="amd"
            $env:CMAKE_PREFIX_PATH="${script:HIP_PATH_V7}"
            $env:CC="${script:HIP_PATH_V7}\bin\clang.exe"
            $env:CXX="${script:HIP_PATH_V7}\bin\clang++.exe"
            & cmake -S llama\server --preset $rocmPreset -G Ninja `
                --install-prefix $script:DIST_DIR
            if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
            $env:HIPCXX=$oldHIPCXX
            $env:HIP_PLATFORM=$oldHIP_PLATFORM
            $env:CMAKE_PREFIX_PATH=$oldCMAKE_PREFIX_PATH
            $env:CC=$oldCC
            $env:CXX=$oldCXX
            & cmake --build "build\llama-server-$rocmBackend" --config Release --parallel $script:JOBS
            if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
            & cmake --install "build\llama-server-$rocmBackend" --component llama-server --strip
            if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
        } else {
            Write-Output "ROCm v7 not detected, skipping"
        }
    }
}

function vulkan {
    if ($env:VULKAN_SDK) {
        Write-Output "Building llama-server Vulkan backend"
        # Use short build path to avoid Windows MAX_PATH issues — the Vulkan
        # shader generator uses ExternalProject_Add which creates deep nesting
        & cmake -S llama\server --preset vulkan -B build\ls-vk --install-prefix $script:DIST_DIR
        if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
        & cmake --build build\ls-vk --config Release --parallel $script:JOBS
        if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
        & cmake --install build\ls-vk --component llama-server --strip
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
            $cudaFlags = @()
            if ($env:OLLAMA_CMAKE_CUDA_FLAGS) {
                $cudaFlags += "-DCMAKE_CUDA_FLAGS=$env:OLLAMA_CMAKE_CUDA_FLAGS"
            }
            $cudaToolsetArgs = cudaCMakeArgs $cuda
            $configureArgs = @("-S", ".", "-B", "build\mlx_cuda_v$cudaMajorVer", "-DOLLAMA_MLX_BACKENDS=cuda_v$cudaMajorVer") + $cudaToolsetArgs + $cudaFlags + @("-DOLLAMA_PAYLOAD_INSTALL_PREFIX=$script:DIST_DIR", "--install-prefix", "$script:DIST_DIR")
            & cmake @configureArgs
            if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
            $buildArgs = @("--build", "build\mlx_cuda_v$cudaMajorVer", "--target", "ollama-mlx-cuda_v$cudaMajorVer", "--config", "Release", "--parallel", "$script:JOBS")
            if ($env:CMAKE_GENERATOR -notlike "Ninja*") {
                $buildArgs += @("--", "/nodeReuse:false")
            }
            & cmake @buildArgs
            if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
        } else {
            Write-Output "CUDA v$cudaMajorVer not detected, skipping MLX build"
        }
    }
}

function withWindowsArm64GoEnv {
    param (
        [scriptblock]$body
    )
    $oldGOOS = $env:GOOS
    $oldGOARCH = $env:GOARCH
    $oldCGO_ENABLED = $env:CGO_ENABLED
    $oldCC = $env:CC
    $oldCXX = $env:CXX
    $oldPath = $env:PATH
    $compilerDir = Split-Path -Parent $script:WINDOWS_ARM64_CC
    $compiler = Split-Path -Leaf $script:WINDOWS_ARM64_CC
    $compilerXX = Split-Path -Leaf $script:WINDOWS_ARM64_CXX
    try {
        $env:GOOS = "windows"
        $env:GOARCH = "arm64"
        $env:CGO_ENABLED = "1"
        $env:PATH = "$compilerDir;$oldPath"
        $env:CC = $compiler
        $env:CXX = $compilerXX
        & $body
    } finally {
        $env:GOOS = $oldGOOS
        $env:GOARCH = $oldGOARCH
        $env:CGO_ENABLED = $oldCGO_ENABLED
        $env:CC = $oldCC
        $env:CXX = $oldCXX
        $env:PATH = $oldPath
    }
}

function buildOllamaCLI {
    param (
        [string]$distDir
    )
    mkdir -Force -path "${distDir}\" | Out-Null
    & go build -trimpath -ldflags "-s -w -X=github.com/ollama/ollama/version.Version=$script:VERSION -X=github.com/ollama/ollama/server.mode=release" -o "${distDir}\ollama.exe" .
    if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
}

function ollama {
    Write-Output "Building ollama CLI"
    buildOllamaCLI $script:DIST_DIR
}

function ollamaArm64 {
    if (-not $script:WINDOWS_ARM64_CROSS_COMPILE) {
        Write-Output "WARNING: skipping ollamaArm64; Windows ARM64 cross-compiling is disabled due to missing tools"
        return
    }

    Write-Output "Building ollama CLI for arm64"
    withWindowsArm64GoEnv {
        buildOllamaCLI "${script:SRC_DIR}\dist\windows-arm64"
    }
}

function prepareApp {
    if ($script:APP_PREPARED) {
        return
    }

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
    $script:APP_PREPARED = $true
}

function buildApp {
    param (
        [string]$arch
    )
	& go build -trimpath -ldflags "-s -w -H windowsgui -X=github.com/ollama/ollama/app/version.Version=$script:VERSION" -o .\dist\windows-ollama-app-${arch}.exe ./app/cmd/app/
    if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
}

function app {
    prepareApp
    buildApp $script:ARCH
}

function appArm64 {
    if (-not $script:WINDOWS_ARM64_CROSS_COMPILE) {
        Write-Output "WARNING: skipping appArm64; Windows ARM64 cross-compiling is disabled due to missing tools"
        return
    }

    prepareApp
    Write-Output "Building Ollama App for arm64"
    withWindowsArm64GoEnv {
        buildApp "arm64"
    }
}

function deps {
    # MSVC CRT DLLs (vcruntime140.dll, msvcp140.dll, etc.) are now bundled
    # directly alongside the executables by CMake's RUNTIME_DEPENDENCIES
    # mechanism during install. No need to download vc_redist.exe.
    Write-Output "deps: no external dependencies to download (CRT DLLs bundled by CMake install)"
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
            & 7z a -tzip -mx=7 -mmt=on $dst "${src}\*"
            if ($LASTEXITCODE -ne 0) { throw "7z failed with exit code $LASTEXITCODE" }
        } else {
            Compress-Archive -CompressionLevel Optimal -Path "${src}\*" -DestinationPath $dst -Force
        }
    } -ArgumentList $sourceDir, $destZip, $use7z
}

function newDependencyAuditJob($payloadDir, $label, $reportPath, $dependencyDirs = @()) {
    $dumpbin = findDumpbin
    if (-not $dumpbin) {
        throw "Unable to locate dumpbin.exe for dependency audit"
    }

    $dependencyDirText = [string]::Join([IO.Path]::PathSeparator, @($dependencyDirs))
    Start-Job -ScriptBlock {
        param($root, $name, $report, $dumpbinPath, $dependencyRootText)

        $ErrorActionPreference = "Stop"

        $systemDlls = @(
            "advapi32.dll", "bcrypt.dll", "cfgmgr32.dll", "combase.dll",
            "comctl32.dll", "comdlg32.dll", "crypt32.dll", "d3d12.dll",
            "dbghelp.dll", "dxcore.dll", "dxgi.dll", "gdi32.dll",
            "gdi32full.dll", "imm32.dll", "iphlpapi.dll", "kernel32.dll",
            "mpr.dll", "msasn1.dll", "msvcrt.dll", "ncrypt.dll",
            "normaliz.dll", "ntdll.dll", "ole32.dll", "oleaut32.dll",
            "powrprof.dll", "propsys.dll", "rpcrt4.dll", "sechost.dll",
            "secur32.dll", "setupapi.dll", "shell32.dll", "shlwapi.dll",
            "ucrtbase.dll", "user32.dll", "userenv.dll", "version.dll",
            "winhttp.dll", "winmm.dll", "ws2_32.dll"
        )
        $driverDlls = @(
            "nvcuda.dll", "nvml.dll"
        )

        $dependencyRoots = @()
        if ($dependencyRootText) {
            $dependencyRoots = $dependencyRootText -split [regex]::Escape([IO.Path]::PathSeparator)
        }

        $availableRoots = @($root) + $dependencyRoots | Where-Object { $_ -and (Test-Path -Path $_) } | Select-Object -Unique
        $binaries = Get-ChildItem -Path $root -Recurse -File -Include *.dll,*.exe | Sort-Object FullName
        $available = @{}
        foreach ($availableRoot in $availableRoots) {
            foreach ($binary in (Get-ChildItem -Path $availableRoot -Recurse -File -Include *.dll,*.exe)) {
                $available[$binary.Name.ToLowerInvariant()] = $true
            }
        }

        $reportLines = [System.Collections.Generic.List[string]]::new()
        $reportLines.Add("Dependency roots:")
        foreach ($availableRoot in $availableRoots) {
            $reportLines.Add("  $availableRoot")
        }
        $reportLines.Add("")
        $missing = [System.Collections.Generic.List[string]]::new()

        foreach ($binary in $binaries) {
            $reportLines.Add("[$($binary.FullName)]")
            $output = & $dumpbinPath /nologo /dependents $binary.FullName 2>&1
            if ($LASTEXITCODE -ne 0) {
                throw "dumpbin failed for $($binary.FullName) with exit code $LASTEXITCODE"
            }

            foreach ($line in $output) {
                if ($line -match '^\s+([A-Za-z0-9._-]+\.dll)\s*$') {
                    $dep = $matches[1]
                    $depLower = $dep.ToLowerInvariant()
                    $reportLines.Add("  $dep")
                    if ($available.ContainsKey($depLower)) {
                        continue
                    }
                    if ($systemDlls -contains $depLower) {
                        continue
                    }
                    if ($driverDlls -contains $depLower) {
                        continue
                    }
                    if ($depLower -like 'api-ms-win-*.dll' -or $depLower -like 'ext-ms-*.dll') {
                        continue
                    }
                    $missing.Add("$($binary.FullName) -> $dep")
                }
            }
        }

        $reportDir = Split-Path -Parent $report
        if ($reportDir) {
            New-Item -ItemType Directory -Force -Path $reportDir | Out-Null
        }
        Set-Content -Path $report -Value $reportLines

        if ($missing.Count -gt 0) {
            $summary = [System.String]::Join([Environment]::NewLine, $missing)
            throw "Dependency audit failed for $name`n$summary"
        }
    } -ArgumentList $payloadDir, $label, $reportPath, $dumpbin, $dependencyDirText
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
            # Stage ROCm into its own directory for independent compression.
            if (stageComponents $amd64Dir "${distDir}\windows-amd64-rocm" "rocm_v*" "ROCm") {
                Write-Output "Generating ${distDir}\ollama-windows-amd64-rocm.zip"
                $jobs += newZipJob "${distDir}\windows-amd64-rocm" "${distDir}\ollama-windows-amd64-rocm.zip"
                $jobs += newDependencyAuditJob "${distDir}\windows-amd64-rocm" "windows-amd64-rocm" "${distDir}\dependency-audit-windows-amd64-rocm.txt" $amd64Dir
            }

            # Stage MLX into its own directory for independent compression
            if (stageComponents $amd64Dir "${distDir}\windows-amd64-mlx" "mlx_*" "MLX") {
                Write-Output "Generating ${distDir}\ollama-windows-amd64-mlx.zip"
                $jobs += newZipJob "${distDir}\windows-amd64-mlx" "${distDir}\ollama-windows-amd64-mlx.zip"
                $jobs += newDependencyAuditJob "${distDir}\windows-amd64-mlx" "windows-amd64-mlx" "${distDir}\dependency-audit-windows-amd64-mlx.txt" $amd64Dir
            }

            # Compress the main amd64 zip (without rocm/mlx)
            Write-Output "Generating ${distDir}\ollama-windows-amd64.zip"
            $jobs += newZipJob $amd64Dir "${distDir}\ollama-windows-amd64.zip"
            $jobs += newDependencyAuditJob $amd64Dir "windows-amd64" "${distDir}\dependency-audit-windows-amd64.txt"
        }

        $arm64Dir = "${distDir}\windows-arm64"
        if (Test-Path -Path $arm64Dir) {
            if ((Test-Path -Path "${arm64Dir}\ollama.exe") -and (Test-Path -Path "${arm64Dir}\lib\ollama\llama-server.exe")) {
                Write-Output "Generating ${distDir}\ollama-windows-arm64.zip"
                $jobs += newZipJob $arm64Dir "${distDir}\ollama-windows-arm64.zip"
                $jobs += newDependencyAuditJob $arm64Dir "windows-arm64" "${distDir}\dependency-audit-windows-arm64.txt"
            } else {
                Write-Output "Skipping ${distDir}\ollama-windows-arm64.zip; missing ARM64 ollama.exe or llama-server.exe"
            }
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
        rocm7
        vulkan
        mlxCuda13
        ollama
        app
        cpuArm64
        ollamaArm64
        appArm64
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
