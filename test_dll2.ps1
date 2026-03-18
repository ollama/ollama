Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
public class DllTest2 {
    [DllImport("kernel32.dll", SetLastError=true)]
    public static extern IntPtr LoadLibrary(string lpFileName);
    [DllImport("kernel32.dll", SetLastError=true)]
    public static extern bool SetDllDirectory(string lpPathName);
    [DllImport("kernel32.dll")]
    public static extern bool FreeLibrary(IntPtr hModule);
}
"@

$dllDir = "$env:LOCALAPPDATA\Programs\Ollama\lib\ollama\directml"
Write-Host "Setting DLL directory: $dllDir"
[DllTest2]::SetDllDirectory($dllDir) | Out-Null

$h = [DllTest2]::LoadLibrary("$dllDir\ggml-directml.dll")
if ($h -eq [IntPtr]::Zero) {
    $err = [System.Runtime.InteropServices.Marshal]::GetLastWin32Error()
    $ex = New-Object System.ComponentModel.Win32Exception($err)
    Write-Host "LoadLibrary FAILED: $($ex.Message) (error $err)"
} else {
    Write-Host "DLL loaded successfully!"
    [DllTest2]::FreeLibrary($h) | Out-Null
}
