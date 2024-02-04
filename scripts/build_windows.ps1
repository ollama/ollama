#!powershell
#
# powershell -ExecutionPolicy Bypass -File .\scripts\build_windows.ps1

$ErrorActionPreference = "Stop"

function checkEnv() {
    write-host "Locating required tools and paths"
    $script:SRC_DIR=$PWD
    if (!$env:VCToolsRedistDir) {
        $MSVC_INSTALL=(Get-CimInstance MSFT_VSInstance -Namespace root/cimv2/vs)[0].InstallLocation
        $env:VCToolsRedistDir=(get-item "${MSVC_INSTALL}\VC\Redist\MSVC\*")[0]
    }
    $script:NVIDIA_DIR=(get-item "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*\bin\")[0]
    $script:INNO_SETUP_DIR=(get-item "C:\Program Files*\Inno Setup*\")[0]

    $script:DEPS_DIR="${script:SRC_DIR}\dist\windeps"
    $env:CGO_ENABLED="1"
}


function buildOllama() {
    write-host "Building ollama CLI"
    & go generate ./...
    if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
    & go build .
    if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
}

function buildApp() {
    write-host "Building Ollama App"
    cd "${script:SRC_DIR}\app"
    & go build -ldflags="-H windowsgui" .
    if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
}

function gatherDependencies() {
    write-host "Gathering runtime dependencies"
    cd "${script:SRC_DIR}"
    rm -ea 0 -recurse -force -path "${script:DEPS_DIR}"
    md "${script:DEPS_DIR}" -ea 0 > $null

    cp "${env:VCToolsRedistDir}\x64\Microsoft.VC*.CRT\msvcp140.dll" "${script:DEPS_DIR}\"
    cp "${env:VCToolsRedistDir}\x64\Microsoft.VC*.CRT\vcruntime140.dll" "${script:DEPS_DIR}\"
    cp "${env:VCToolsRedistDir}\x64\Microsoft.VC*.CRT\vcruntime140.dll" "${script:DEPS_DIR}\"

    cp "${script:NVIDIA_DIR}\cudart64_*.dll" "${script:DEPS_DIR}\"
    cp "${script:NVIDIA_DIR}\cublas64_*.dll" "${script:DEPS_DIR}\"
}

function buildInstaller() {
    write-host "Building Ollama Installer"
    cd "${script:SRC_DIR}\app"
    & "${script:INNO_SETUP_DIR}\ISCC.exe" .\ollama.iss
    if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
}

try {
    checkEnv
    buildOllama
    buildApp
    gatherDependencies
    buildInstaller
} catch {
    write-host "Build Failed"
    write-host $_
} finally {
    set-location $script:SRC_DIR
}