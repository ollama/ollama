Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
public class DllTest {
    [DllImport("kernel32.dll", SetLastError=true)]
    public static extern IntPtr LoadLibrary(string lpFileName);
    [DllImport("kernel32.dll")]
    public static extern bool FreeLibrary(IntPtr hModule);
}
"@

$dllPath = "$env:LOCALAPPDATA\Programs\Ollama\lib\ollama\directml\ggml-directml.dll"
Write-Host "Loading: $dllPath"
$h = [DllTest]::LoadLibrary($dllPath)
if ($h -eq [IntPtr]::Zero) {
    $err = [System.Runtime.InteropServices.Marshal]::GetLastWin32Error()
    Write-Host "LoadLibrary FAILED, error code: $err (0x$($err.ToString('X8')))"
    $ex = New-Object System.ComponentModel.Win32Exception($err)
    Write-Host "Error message: $($ex.Message)"
} else {
    Write-Host "DLL loaded successfully!"
    [DllTest]::FreeLibrary($h) | Out-Null
}
