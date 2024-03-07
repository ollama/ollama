#!powershell
#
# powershell -ExecutionPolicy Bypass -File .\scripts\build_windows.ps1
#
# gcloud auth application-default login

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
    echo "Checking version"
    if (!$env:VERSION) {
        $data=(git describe --tags --first-parent --abbrev=7 --long --dirty --always)
        $pattern="v(.+)"
        if ($data -match $pattern) {
            $script:VERSION=$matches[1]
        }
    } else {
        $script:VERSION=$env:VERSION
    }
    $pattern = "(\d+[.]\d+[.]\d+)-(\d+)-"
    if ($script:VERSION -match $pattern) {
        $script:PKG_VERSION=$matches[1] + "." + $matches[2]
    } else {
        $script:PKG_VERSION=$script:VERSION
    }
    write-host "Building Ollama $script:VERSION with package version $script:PKG_VERSION"

    # Check for signing key
    if ("${env:KEY_CONTAINER}") {
        ${script:OLLAMA_CERT}=$(resolve-path "${script:SRC_DIR}\ollama_inc.crt")
        Write-host "Code signing enabled"
        # Note: 10 Windows Kit signtool crashes with GCP's plugin
        ${script:SignTool}="C:\Program Files (x86)\Windows Kits\8.1\bin\x64\signtool.exe"
    } else {
        write-host "Code signing disabled - please set KEY_CONTAINERS to sign and copy ollama_inc.crt to the top of the source tree"
    }

}


function buildOllama() {
    write-host "Building ollama CLI"
    if ($null -eq ${env:OLLAMA_SKIP_GENERATE}) {
        & go generate ./...
        if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}    
    } else {
        write-host "Skipping generate step with OLLAMA_SKIP_GENERATE set"
    }
    & go build -ldflags "-s -w -X=github.com/jmorganca/ollama/version.Version=$script:VERSION -X=github.com/jmorganca/ollama/server.mode=release" .
    if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
    if ("${env:KEY_CONTAINER}") {
        & "${script:SignTool}" sign /v /fd sha256 /t http://timestamp.digicert.com /f "${script:OLLAMA_CERT}" `
            /csp "Google Cloud KMS Provider" /kc ${env:KEY_CONTAINER} ollama.exe
        if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
    }
    New-Item -ItemType Directory -Path .\dist -Force
    cp .\ollama.exe .\dist\ollama-windows-amd64.exe
}

function buildApp() {
    write-host "Building Ollama App"
    cd "${script:SRC_DIR}\app"
    & windres -l 0 -o ollama.syso ollama.rc
    & go build -ldflags "-s -w -H windowsgui -X=github.com/jmorganca/ollama/version.Version=$script:VERSION -X=github.com/jmorganca/ollama/server.mode=release" .
    if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
    if ("${env:KEY_CONTAINER}") {
        & "${script:SignTool}" sign /v /fd sha256 /t http://timestamp.digicert.com /f "${script:OLLAMA_CERT}" `
            /csp "Google Cloud KMS Provider" /kc ${env:KEY_CONTAINER} app.exe
        if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
    }
}

function gatherDependencies() {
    write-host "Gathering runtime dependencies"
    cd "${script:SRC_DIR}"
    rm -ea 0 -recurse -force -path "${script:DEPS_DIR}"
    md "${script:DEPS_DIR}" -ea 0 > $null

    # TODO - this varies based on host build system and MSVC version - drive from dumpbin output
    # currently works for Win11 + MSVC 2019 + Cuda V11
    cp "${env:VCToolsRedistDir}\x64\Microsoft.VC*.CRT\msvcp140.dll" "${script:DEPS_DIR}\"
    cp "${env:VCToolsRedistDir}\x64\Microsoft.VC*.CRT\vcruntime140.dll" "${script:DEPS_DIR}\"
    cp "${env:VCToolsRedistDir}\x64\Microsoft.VC*.CRT\vcruntime140_1.dll" "${script:DEPS_DIR}\"

    cp "${script:NVIDIA_DIR}\cudart64_*.dll" "${script:DEPS_DIR}\"
    cp "${script:NVIDIA_DIR}\cublas64_*.dll" "${script:DEPS_DIR}\"
    cp "${script:NVIDIA_DIR}\cublasLt64_*.dll" "${script:DEPS_DIR}\"

    cp "${script:SRC_DIR}\app\ollama_welcome.ps1" "${script:SRC_DIR}\dist\"
    if ("${env:KEY_CONTAINER}") {
        write-host "about to sign"
        foreach ($file in (get-childitem "${script:DEPS_DIR}/cu*.dll") + @("${script:SRC_DIR}\dist\ollama_welcome.ps1")){
            write-host "signing $file"
            & "${script:SignTool}" sign /v /fd sha256 /t http://timestamp.digicert.com /f "${script:OLLAMA_CERT}" `
                /csp "Google Cloud KMS Provider" /kc ${env:KEY_CONTAINER} $file
            if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
        }
    }

}

function buildInstaller() {
    write-host "Building Ollama Installer"
    cd "${script:SRC_DIR}\app"
    $env:PKG_VERSION=$script:PKG_VERSION
    if ("${env:KEY_CONTAINER}") {
        & "${script:INNO_SETUP_DIR}\ISCC.exe" /SMySignTool="${script:SignTool} sign /fd sha256 /t http://timestamp.digicert.com /f ${script:OLLAMA_CERT} /csp `$qGoogle Cloud KMS Provider`$q /kc ${env:KEY_CONTAINER} `$f" .\ollama.iss
    } else {
        & "${script:INNO_SETUP_DIR}\ISCC.exe" .\ollama.iss
    }
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
    $env:PKG_VERSION=""
}
