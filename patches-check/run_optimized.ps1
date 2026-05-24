# OLLaMA ROCm 7.x Maximum Aggression Launcher (Windows)
# Apply before running ollama serve or ollama run

$env:OLLAMA_MAX_LOADED_MODELS = "1"
$env:OLLAMA_NUM_PARALLEL = "1"
$env:OLLAMA_NOPRUNE = "1"

# ROCm/HIP Runtime Tuning
$env:HSA_ENABLE_SDMA = "1"
$env:HSA_SDMA_PERF_MODE = "1"
$env:HSA_USE_HOST_ALLOC = "0"
$env:HIP_FORCE_DEV_KERNARG = "1"
$env:AMD_DIRECT_DISPATCH = "1"
$env:AMD_SHADER_CACHE = "1"
$env:AMD_SHADER_CACHE_PATH = "$env:TEMP\amd_shader_cache"
$env:GPU_MAX_HW_QUEUES = "8"
$env:HIP_VISIBLE_DEVICES = "0"

# Memory & Allocation Aggression
$env:HSA_XNACK = "0"
$env:HIP_HOST_COHERENT = "0"
$env:HSA_USERPTR_FOR_PAGED_MEM = "0"
$env:GPU_SINGLE_ALLOC_PERCENT = "80"
$env:GPU_MAX_ALLOC_PERCENT = "95"
$env:GPU_ENABLE_ALLOC_CACHE = "1"

# GGML/LLaMA.cpp Backend Tuning
$env:GGML_HIP_GRAPHS = "1"
$env:GGML_HIP_UMA = "0"
$env:GGML_CUDA_FORCE_CUBLAS = "0"
$env:GGML_CUDA_NO_VMM = "0"

# Threading
$env:OMP_PROC_BIND = "close"
$env:OMP_PLACES = "cores"

# Scheduler Aggression
$env:OLLAMA_SCHED_SPREAD = "1"
$env:OLLAMA_KEEP_ALIVE = "-1"

# TurboQuant
$env:LLAMA_CACHE_TYPE_K = "tbq3"
$env:LLAMA_CACHE_TYPE_V = "tbq3"
$env:LLAMA_ATTENTION_SHARPEN = "1"

# Create shader cache dir
New-Item -ItemType Directory -Force -Path "$env:TEMP\amd_shader_cache" | Out-Null

# Run
ollama $args