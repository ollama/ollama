# Smart Ollama Launcher with Intel iGPU Vulkan Protection
# Automatically detects problematic models and applies appropriate workarounds

param(
    [string]$Model = "",
    [switch]$ForceVulkan = $false,
    [switch]$ForceCPU = $false,
    [switch]$Help = $false
)

function Show-Help {
    Write-Host @"
Smart Ollama Launcher with Intel iGPU Vulkan Protection
======================================================

USAGE:
  start_ollama_intel_smart.ps1 [-Model <model_name>] [-ForceVulkan] [-ForceCPU] [-Help]

PARAMETERS:
  -Model <model_name>  : Specify model name to run immediately
  -ForceVulkan        : Force Vulkan even for problematic models (risky)
  -ForceCPU           : Force CPU execution regardless of model
  -Help               : Show this help message

EXAMPLES:
  .\start_ollama_intel_smart.ps1
  .\start_ollama_intel_smart.ps1 -Model "gpt-oss:20b"
  .\start_ollama_intel_smart.ps1 -Model "llama3:8b" -ForceVulkan

AUTOMATIC BEHAVIOR:
  - Detects problematic models (gpt-oss, etc.)
  - Applies CPU fallback for crash-prone models
  - Uses Vulkan with safety flags for compatible models
  - Shows recommendations and warnings

"@
}

if ($Help) {
    Show-Help
    exit 0
}

# Known problematic models that cause crashes on Intel iGPU Vulkan
$ProblematicModels = @(
    "gpt-oss",
    "gptoss",
    "gpt4",
    "gpt4all" # Add more as discovered
)

# Function to check if a model is problematic
function Test-ProblematicModel {
    param([string]$ModelName)
    
    $lowerModel = $ModelName.ToLower()
    foreach ($problematic in $ProblematicModels) {
        if ($lowerModel -like "*$problematic*") {
            return $true
        }
    }
    return $false
}

# Function to get recommended configuration
function Get-ModelConfig {
    param([string]$ModelName)
    
    $isProblematic = Test-ProblematicModel -ModelName $ModelName
    
    if ($ForceCPU) {
        return @{
            Library = "cpu"
            Reason = "User forced CPU execution"
            Safe = $true
        }
    }
    
    if ($isProblematic -and -not $ForceVulkan) {
        return @{
            Library = "cpu"
            Reason = "Model known to cause crashes on Intel iGPU Vulkan"
            Safe = $true
        }
    }
    
    if ($ForceVulkan -or -not $isProblematic) {
        return @{
            Library = "vulkan"
            Reason = if ($isProblematic) { "User forced Vulkan (risky)" } else { "Model compatible with Vulkan" }
            Safe = -not $isProblematic
        }
    }
    
    # Default safe configuration
    return @{
        Library = "vulkan"
        Reason = "Default Vulkan with safety measures"
        Safe = $true
    }
}

# Clear any existing environment variables
Write-Host "🧹 Cleaning environment variables..." -ForegroundColor Yellow
$EnvVarsToClean = @(
    "OLLAMA_LLM_LIBRARY",
    "OLLAMA_INTEL_GPU", 
    "GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM",
    "OLLAMA_VK_IGPU_MEMORY_LIMIT_MB",
    "OLLAMA_VK_DISABLE_IGPU_CLAMP"
)

foreach ($var in $EnvVarsToClean) {
    [Environment]::SetEnvironmentVariable($var, $null, "Process")
}

# Determine configuration
if ($Model) {
    $config = Get-ModelConfig -ModelName $Model
    $modelInfo = "for model '$Model'"
} else {
    # When no model specified, use safe Vulkan config
    $config = @{
        Library = "vulkan"
        Reason = "Default safe configuration"
        Safe = $true
    }
    $modelInfo = "default configuration"
}

Write-Host "🤖 Smart Intel iGPU Configuration $modelInfo" -ForegroundColor Cyan
Write-Host "   Library: $($config.Library.ToUpper())" -ForegroundColor $(if ($config.Library -eq "vulkan") { "Green" } else { "Yellow" })
Write-Host "   Reason:  $($config.Reason)" -ForegroundColor Gray
Write-Host "   Safe:    $(if ($config.Safe) { "✅ Yes" } else { "⚠️  Risky" })" -ForegroundColor $(if ($config.Safe) { "Green" } else { "Red" })

# Apply configuration
if ($config.Library -eq "cpu") {
    Write-Host "📋 Applying CPU configuration..." -ForegroundColor Yellow
    [Environment]::SetEnvironmentVariable("OLLAMA_LLM_LIBRARY", "cpu", "Process")
    Write-Host "   ✅ OLLAMA_LLM_LIBRARY=cpu" -ForegroundColor Green
    
    if (Test-ProblematicModel -ModelName $Model) {
        Write-Host "   ⚠️  This model is known to cause system crashes on Intel iGPU Vulkan" -ForegroundColor Red
        Write-Host "   💡 CPU execution will prevent crashes but may be slower" -ForegroundColor Yellow
    }
} else {
    Write-Host "📋 Applying Vulkan configuration..." -ForegroundColor Yellow
    [Environment]::SetEnvironmentVariable("OLLAMA_LLM_LIBRARY", "vulkan", "Process")
    [Environment]::SetEnvironmentVariable("OLLAMA_INTEL_GPU", "1", "Process")
    [Environment]::SetEnvironmentVariable("GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM", "1", "Process")
    
    Write-Host "   ✅ OLLAMA_LLM_LIBRARY=vulkan" -ForegroundColor Green
    Write-Host "   ✅ OLLAMA_INTEL_GPU=1" -ForegroundColor Green  
    Write-Host "   ✅ GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1 (crash prevention)" -ForegroundColor Green
    
    if (Test-ProblematicModel -ModelName $Model) {
        Write-Host "   ⚠️  WARNING: This model is known to cause crashes!" -ForegroundColor Red
        Write-Host "   💡 Consider using -ForceCPU instead" -ForegroundColor Yellow
        if (-not $ForceVulkan) {
            Write-Host "   🛑 Use -ForceVulkan to override this warning" -ForegroundColor Red
        }
    }
}

Write-Host ""

# Show current environment
Write-Host "🌍 Environment summary:" -ForegroundColor Cyan
$ollama_lib = [Environment]::GetEnvironmentVariable('OLLAMA_LLM_LIBRARY', 'Process')
$ollama_gpu = [Environment]::GetEnvironmentVariable('OLLAMA_INTEL_GPU', 'Process')
$ggml_vk_disable = [Environment]::GetEnvironmentVariable('GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM', 'Process')

Write-Host "   OLLAMA_LLM_LIBRARY               = $ollama_lib" -ForegroundColor White
Write-Host "   OLLAMA_INTEL_GPU                 = $ollama_gpu" -ForegroundColor White
Write-Host "   GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM = $ggml_vk_disable" -ForegroundColor White

Write-Host ""

# Start Ollama
Write-Host "🚀 Starting Ollama serve..." -ForegroundColor Green
try {
    if ($Model) {
        Write-Host "   Will run model: $Model" -ForegroundColor Gray
        Start-Process -FilePath "ollama" -ArgumentList "serve" -NoNewWindow -PassThru
        Start-Sleep -Seconds 3
        Write-Host "   Running model..." -ForegroundColor Yellow
        ollama run $Model
    } else {
        ollama serve
    }
} catch {
    Write-Host "❌ Error starting Ollama: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "   Check if Ollama is properly installed" -ForegroundColor Yellow
    exit 1
}