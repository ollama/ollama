# Chained uninstall for Ollama MSI packages.
# Called asynchronously when the core MSI is uninstalled (via ARP or msiexec /x).
# Waits for the core msiexec to release the MSI mutex, then removes all
# remaining backend and deps MSIs using the Windows Installer COM API.

Start-Sleep 2
$installer = New-Object -ComObject WindowsInstaller.Installer
# Backends first, then deps (reverse of install order)
$upgradeCodes = @(
  @{ UC="3F8A2D1E-5B6C-4E7F-A9D0-1C2B3E4F5A6D" }  # cuda_v12
  @{ UC="9C7E3A1B-2D4F-4E5A-B6C8-0D1E2F3A4B5C" }  # cuda_v13
  @{ UC="4B2E8F1A-6C3D-4A5E-9F7B-0D1C2E3A4B5D" }  # rocm
  @{ UC="6D4A2E8F-1B3C-4F5E-A7D9-0C1B2E3F4A5D" }  # vulkan
  @{ UC="A1E3B5C7-2D4F-6A8E-9B0C-1D2E3F4A5B6C" }  # cuda_v12_deps
  @{ UC="8F2A4E6C-1B3D-5C7E-A9F0-2D1E3B4A5C6D" }  # cuda_v13_deps
  @{ UC="E5C7A9B1-3D2F-4E6A-8F0C-1B2D3E4A5F6C" }  # rocm_deps
  @{ UC="2A4C6E8F-0B1D-3E5A-7C9F-1D2B3A4E5C6F" }  # vulkan_deps
)
foreach ($pkg in $upgradeCodes) {
  try {
    $related = $installer.RelatedProducts("{$($pkg.UC)}")
    foreach ($pc in $related) {
      Start-Process msiexec -ArgumentList '/x',$pc,'/quiet','/norestart' -Wait -EA 0
    }
  } catch { }
}
