param(
    [string]$OllamaExecutable = 'C:\IA\tools\ollama\dist\windows-amd64\ollama.exe',
    [string[]]$OllamaArguments = @()
)

$env:OLLAMA_LLM_LIBRARY = "vulkan"
$env:OLLAMA_INTEL_GPU = "1"
# $env:OLLAMA_VK_IGPU_MEMORY_LIMIT_MB = "32768"
$env:GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM = "1"

$varsToUnset = @(
    'OLLAMA_GPU_OVERHEAD',
    # 'OLLAMA_FLASH_ATTENTION',
    # 'OLLAMA_NUM_PARALLEL',
    # 'OLLAMA_SCHED_SPREAD',
    # 'OLLAMA_REMOTES',
    # 'OLLAMA_ORIGINS',
    'ROCR_VISIBLE_DEVICES',
    'ROCM_VISIBLE_DEVICES',
    'HIP_VISIBLE_DEVICES'
)

foreach ($name in $varsToUnset) {
    Remove-Item Env:$name -ErrorAction SilentlyContinue
}

Write-Host "Launching Ollama with Vulkan backend and Intel iGPU enabled..."
Write-Host "  OLLAMA_LLM_LIBRARY=$($env:OLLAMA_LLM_LIBRARY)"
Write-Host "  OLLAMA_INTEL_GPU=$($env:OLLAMA_INTEL_GPU)"
Write-Host "  OLLAMA_VK_IGPU_MEMORY_LIMIT_MB=$($env:OLLAMA_VK_IGPU_MEMORY_LIMIT_MB)"
Write-Host "  GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=$($env:GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM)"

if (!(Test-Path -Path $OllamaExecutable -PathType Leaf)) {
    Write-Error "Ollama executable not found at '$OllamaExecutable'."
    exit 1
}

& $OllamaExecutable serve @OllamaArguments


# # powershell -ExecutionPolicy Bypass -File Z_Iosu\scripts\start_ollama_vulkan_intel.ps1 *> C:\IA\tools\ollama\logs\ollama.log
# PS C:\Users\iosuc> ollama ps
# NAME            ID              SIZE     PROCESSOR          CONTEXT    UNTIL
# gpt-oss:120b    a951a23b46a1    66 GB    28%/72% CPU/GPU    8192       4 minutes from now
# OLLAMA_GPU_OVERHEAD=536870912
# OLLAMA_FLASH_ATTENTION=1
# OLLAMA_NUM_PARALLEL=1


#https://github.com/ggml-org/llama.cpp/issues/10528 For anybody running into the bugcheck 0x10e, please try setting the env var GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM to workaround it. https://github.com/ggml-org/llama.cpp/issues/10528#issuecomment-3165629609 
