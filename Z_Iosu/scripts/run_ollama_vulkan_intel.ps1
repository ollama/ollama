# Fuerza a Ollama a usar Vulkan en la iGPU Intel y deshabilita CUDA (NVIDIA)
# Requisitos: haber construido con soporte Vulkan y tener ggml-vulkan.dll en dist\windows-amd64\lib\ollama

param(
    # Índice de la GPU Vulkan a usar según vulkaninfo (por defecto 0 = Intel en tu máquina)
    [int]$VkDeviceIndex = 0,
    # Seleccionar automáticamente la GPU cuyo vendor sea NVIDIA (ignora VkDeviceIndex si se encuentra)
    [switch]$UseNvidia,
    # O bien especifica un patrón de proveedor para auto-selección (por ejemplo: "AMD|Radeon" o "NVIDIA|GeForce|RTX")
    [string]$VendorPattern,
    # Puerto del servidor Ollama
    [int]$Port = 11434,
    # Ruta del ejecutable de Ollama (si no se provee, se detecta el instalado)
    [string]$OllamaExe
)

$ErrorActionPreference = 'Stop'

# Detectar la ruta del Ollama instalado (Program Files / LocalAppData) si no se especifica
function Find-InstalledOllamaExe {
    # Por requerimiento: usar la instalación en LocalAppData por defecto
    if ($env:LOCALAPPDATA) {
        $fixed = Join-Path $env:LOCALAPPDATA 'Programs\Ollama\ollama.exe'
        if (Test-Path $fixed) { return $fixed }
    }
    # Opcionalmente, si no está ahí, probar PATH
    try {
        $cmd = Get-Command ollama -ErrorAction SilentlyContinue
        if ($cmd -and $cmd.Path -and (Test-Path $cmd.Path)) { return $cmd.Path }
    } catch { }
    # Como último recurso, intentar el binario de desarrollo
    $dev = Join-Path (Split-Path -Parent $PSCommandPath) '..\..\dist\windows-amd64\ollama.exe'
    return (Resolve-Path -LiteralPath $dev -ErrorAction SilentlyContinue)
}

# Mostrar GPUs disponibles via Vulkan para ayudar a elegir el índice
function Show-VulkanDevicesSummary {
    try {
        $vkInfo = Join-Path $env:VULKAN_SDK 'Bin\\vulkaninfoSDK.exe'
        if (Test-Path $vkInfo) {
            Write-Host 'Dispositivos Vulkan detectados:' -ForegroundColor Cyan
            & $vkInfo --summary | Select-String -Pattern '^GPU\d+:' -Context 0,8 | ForEach-Object { $_.ToString() }
        } else {
            Write-Warning "No se encontró vulkaninfoSDK.exe en VULKAN_SDK. Continuo igualmente."
        }
    } catch { Write-Warning $_ }
}

function Get-VulkanDevices {
    # Devuelve una lista de objetos con Index, Name y Block (trozo de texto del summary)
    $result = @()
    try {
        $vkInfo = Join-Path $env:VULKAN_SDK 'Bin\vulkaninfoSDK.exe'
        if (-not (Test-Path $vkInfo)) { return $result }
        $summary = & $vkInfo --summary 2>$null
        if (-not $summary) { return $result }
        $lines = $summary -split "`r?`n"
        for ($i = 0; $i -lt $lines.Length; $i++) {
            $m = [regex]::Match($lines[$i], '^GPU(\d+):\s*(.*)$')
            if ($m.Success) {
                $idx = [int]$m.Groups[1].Value
                $name = $m.Groups[2].Value
                $end = $i + 1
                while ($end -lt $lines.Length -and -not ($lines[$end] -match '^GPU\d+:')) { $end++ }
                $block = ($lines[$i..($end-1)] -join "`n")
                if (-not $name -or $name.Trim().Length -eq 0) {
                    # Intenta extraer de campos típicos (soporta ':' o '=')
                    $nm = [regex]::Match($block, '(?im)^\s*(deviceName|DeviceName|Name)\s*[:=]\s*(.+)$')
                    if ($nm.Success) { $name = $nm.Groups[2].Value.Trim() }
                }
                $obj = [pscustomobject]@{ Index = $idx; Name = $name; Block = $block }
                $result += $obj
            }
        }
    } catch { }
    return $result
}

function Get-VulkanDeviceIndexByVendor {
    param(
        [Parameter(Mandatory=$true)][string]$VendorPattern
    )
    try {
        $vkInfo = Join-Path $env:VULKAN_SDK 'Bin\\vulkaninfoSDK.exe'
        if (-not (Test-Path $vkInfo)) { return $null }
        $summary = & $vkInfo --summary 2>$null
        if (-not $summary) { return $null }
        $lines = $summary -split "`r?`n"
        for ($i = 0; $i -lt $lines.Length; $i++) {
            $m = [regex]::Match($lines[$i], '^GPU(\d+):')
            if ($m.Success) {
                $idx = [int]$m.Groups[1].Value
                $end = [Math]::Min($i + 15, $lines.Length - 1)
                $block = ($lines[$i..$end] -join "`n")
                if ($block -match $VendorPattern) {
                    return $idx
                }
            }
        }
        return $null
    } catch { return $null }
}

Show-VulkanDevicesSummary

# Auto-selección por vendor si se solicita
if ($UseNvidia) {
    if (-not $VendorPattern) { $VendorPattern = 'NVIDIA|GeForce|RTX' }
}
if ($VendorPattern) {
    $autoIdx = Get-VulkanDeviceIndexByVendor -VendorPattern $VendorPattern
    if ($null -ne $autoIdx) {
        Write-Host "Auto-seleccionado VkDeviceIndex=$autoIdx por vendor '$VendorPattern'" -ForegroundColor Cyan
        $VkDeviceIndex = $autoIdx
    } else {
        Write-Warning "No se pudo localizar una GPU Vulkan que coincida con '$VendorPattern'. Se mantiene VkDeviceIndex=$VkDeviceIndex."
    }
}

# Mostrar cuál es el dispositivo seleccionado por índice
$vkDevices = Get-VulkanDevices
$selected = $vkDevices | Where-Object { $_.Index -eq $VkDeviceIndex } | Select-Object -First 1
if ($selected) {
    Write-Host ("Usando Vulkan GPU{0}: {1}" -f $selected.Index, $selected.Name) -ForegroundColor Green
} else {
    Write-Host ("Usando Vulkan GPU{0} (nombre no disponible - ejecuta vulkaninfoSDK.exe --summary para más detalles)" -f $VkDeviceIndex) -ForegroundColor Green
}

# 1) Ocultar NVIDIA / evitar CUDA
$env:CUDA_VISIBLE_DEVICES = "-1"  # equivalente a ocultar todas las GPUs NVIDIA

# 2) Limitar Vulkan al índice deseado (Intel = 0 según tu salida de vulkaninfo)
$env:GGML_VK_VISIBLE_DEVICES = "$VkDeviceIndex"

# 3) Forzar familia de backend a Vulkan y aumentar verbosidad de logs (opcional)
$env:OLLAMA_LLM_LIBRARY = "vulkan"
$env:GGML_LOG_LEVEL = "INFO"

# 4) Ejecutable de Ollama (preferir instalado)
if (-not $OllamaExe -or -not (Test-Path $OllamaExe)) {
    $detected = Find-InstalledOllamaExe
    if ($detected) { $OllamaExe = $detected }
}
if (-not $OllamaExe -or -not (Test-Path $OllamaExe)) {
    throw "No se encontró el ejecutable de Ollama instalado ni el de desarrollo. Especifica -OllamaExe explicitamente."
}
Write-Host "Ejecutable de Ollama: $OllamaExe" -ForegroundColor Cyan

# 5) Verificación de la DLL de Vulkan junto al ejecutable seleccionado
$ollamaRoot = Split-Path -Parent $OllamaExe
$libDir  = Join-Path $ollamaRoot 'lib\\ollama'
$vkDll = Join-Path $libDir 'ggml-vulkan.dll'
if (-not (Test-Path $vkDll)) {
    Write-Warning "No se encontró '$vkDll' en la instalación. Intentando copiarlo desde el build local (dist)..."
    try {
        $repoRoot = Resolve-Path (Join-Path (Split-Path -Parent $PSCommandPath) '..\..') -ErrorAction SilentlyContinue
        if ($repoRoot) {
            $devLib = Join-Path $repoRoot 'dist\windows-amd64\lib\ollama'
            $srcVk = Join-Path $devLib 'ggml-vulkan.dll'
            if (Test-Path $srcVk) {
                Copy-Item -Path $srcVk -Destination $libDir -Force
                Write-Host "Copiado ggml-vulkan.dll -> $libDir" -ForegroundColor Yellow
            } else {
                Write-Warning "No se encontró ggml-vulkan.dll en $devLib. Vulkan no estará disponible."
            }
        }
    } catch { Write-Warning $_ }
}

Write-Host "Lanzando Ollama con Vulkan (GGML_VK_VISIBLE_DEVICES=$($env:GGML_VK_VISIBLE_DEVICES)) y CUDA oculto (CUDA_VISIBLE_DEVICES=-1)" -ForegroundColor Green

# 6) Si ya hay un servidor escuchando en el puerto, no arrancar otra instancia
function Test-PortOpen {
    param([int]$p)
    try {
        return (Test-NetConnection -ComputerName '127.0.0.1' -Port $p -WarningAction SilentlyContinue -InformationLevel Quiet)
    } catch { return $false }
}

if (Test-PortOpen -p $Port) {
    Write-Host "Ya hay un Ollama escuchando en 127.0.0.1:$Port. No se lanzará otra instancia." -ForegroundColor Yellow
} else {
    # Arrancar servidor (usar OLLAMA_HOST en lugar de flags de CLI)
    $env:OLLAMA_HOST = "127.0.0.1:$Port"
    & $OllamaExe serve
}
