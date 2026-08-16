param(
    [switch]$Install
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$sdkVersion = "1.8.260804001"
$foundationVersion = "1.8.260803002"
$runtimeDir = "dist\app-runtime"
$foundationZip = Join-Path $env:RUNNER_TEMP "Microsoft.WindowsAppSDK.Foundation.zip"
$foundationDir = Join-Path $env:RUNNER_TEMP "Microsoft.WindowsAppSDK.Foundation"

New-Item -ItemType Directory -Force -Path "$runtimeDir\x64", "$runtimeDir\arm64", "$runtimeDir\licenses" | Out-Null
$downloads = @(
    @("https://aka.ms/windowsappsdk/1.8/$sdkVersion/windowsappruntimeinstall-x64.exe", "$runtimeDir\x64\WindowsAppRuntimeInstall.exe", "D20630D3078EA1514FB2D1FBE827254F04AF49D86502ACE8ECFDA626A8631192"),
    @("https://aka.ms/windowsappsdk/1.8/$sdkVersion/windowsappruntimeinstall-arm64.exe", "$runtimeDir\arm64\WindowsAppRuntimeInstall.exe", "1A92653A08AEEF382340DBC953249342A2A8516B97CF48AC3938F77DCF495B1B"),
    @("https://api.nuget.org/v3-flatcontainer/microsoft.windowsappsdk.foundation/$foundationVersion/microsoft.windowsappsdk.foundation.$foundationVersion.nupkg", $foundationZip, "B9232041AFD605B606C6F78F442D92EAD0076453F1F2A3260D2B7F8089BCAB0E")
)
foreach ($download in $downloads) {
    Invoke-WebRequest -Uri $download[0] -OutFile $download[1]
    if ((Get-FileHash $download[1] -Algorithm SHA256).Hash -ne $download[2]) {
        throw "SHA256 mismatch for $($download[0])"
    }
}

# Ollama installs per-user. Reject a runtime redistributable that asks Windows
# to elevate rather than inheriting the non-elevated installer process.
foreach ($arch in @("x64", "arm64")) {
    $installer = "$runtimeDir\$arch\WindowsAppRuntimeInstall.exe"
    $imageText = [Text.Encoding]::ASCII.GetString([IO.File]::ReadAllBytes($installer))
    if ($imageText -notmatch "requestedExecutionLevel level='asInvoker'") {
        throw "$installer does not declare requestedExecutionLevel=asInvoker"
    }
}

Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $foundationDir
Expand-Archive -Path $foundationZip -DestinationPath $foundationDir
Copy-Item "$foundationDir\runtimes\win-x64\native\Microsoft.WindowsAppRuntime.Bootstrap.dll" "$runtimeDir\x64"
Copy-Item "$foundationDir\runtimes\win-arm64\native\Microsoft.WindowsAppRuntime.Bootstrap.dll" "$runtimeDir\arm64"
Copy-Item "$foundationDir\license.txt" "$runtimeDir\licenses\LICENSE.txt"

if ($Install -and -not (Get-AppxPackage -Name Microsoft.WindowsAppRuntime.1.8 | Where-Object Architecture -eq X64)) {
    & "$runtimeDir\x64\WindowsAppRuntimeInstall.exe" --quiet
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}
if ($Install -and -not (Get-AppxPackage -Name Microsoft.WindowsAppRuntime.1.8 | Where-Object Architecture -eq X64)) {
    throw "Windows App Runtime 1.8 x64 was not installed"
}
