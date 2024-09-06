#!powershell

function get_script_path($script_name){
    $ScriptPath = Join-Path -Path $PSScriptRoot -ChildPath $script_name
    return $scriptPath
}

. $(get_script_path("gen_windows_utils.ps1"))

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
      "-DCMAKE_C_COMPILER=cl",
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

    rm -ea 0 -recurse -force -path "${script:SRC_DIR}\dist\windows-${script:ARCH}\oneapi\"
    md "${script:SRC_DIR}\dist\windows-${script:ARCH}\oneapi\" -ea 0 > $null
    cp "${env:ONEAPI_ROOT}\compiler\latest\bin\libirngmd.dll" "${script:SRC_DIR}\dist\windows-${script:ARCH}\oneapi\"
    cp "${env:ONEAPI_ROOT}\compiler\latest\bin\libmmd.dll" "${script:SRC_DIR}\dist\windows-${script:ARCH}\oneapi\"
    cp "${env:ONEAPI_ROOT}\compiler\latest\bin\pi_level_zero.dll" "${script:SRC_DIR}\dist\windows-${script:ARCH}\oneapi\"
    cp "${env:ONEAPI_ROOT}\compiler\latest\bin\pi_unified_runtime.dll" "${script:SRC_DIR}\dist\windows-${script:ARCH}\oneapi\"
    cp "${env:ONEAPI_ROOT}\compiler\latest\bin\pi_win_proxy_loader.dll" "${script:SRC_DIR}\dist\windows-${script:ARCH}\oneapi\"
    cp "${env:ONEAPI_ROOT}\compiler\latest\bin\svml_dispmd.dll" "${script:SRC_DIR}\dist\windows-${script:ARCH}\oneapi\"
    cp "${env:ONEAPI_ROOT}\compiler\latest\bin\sycl7.dll" "${script:SRC_DIR}\dist\windows-${script:ARCH}\oneapi\"
    cp "${env:ONEAPI_ROOT}\mkl\latest\bin\mkl_core.2.dll" "${script:SRC_DIR}\dist\windows-${script:ARCH}\oneapi\"
    cp "${env:ONEAPI_ROOT}\mkl\latest\bin\mkl_sycl_blas.4.dll" "${script:SRC_DIR}\dist\windows-${script:ARCH}\oneapi\"
    cp "${env:ONEAPI_ROOT}\mkl\latest\bin\mkl_tbb_thread.2.dll" "${script:SRC_DIR}\dist\windows-${script:ARCH}\oneapi\"
  } else {
    Write-Host "Skipping oneAPI generation step"
  }
}

build_oneapi