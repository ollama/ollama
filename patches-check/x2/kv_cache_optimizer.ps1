# kv_cache_optimizer.ps1 — Auto-configure Ollama for max context + speed on RX 9070 XT
# Place in repo root, run before benchmarking

param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("llama3","qwen2.5","gemma4","devstral","mistral")]
    [string]$ModelFamily,

    [int]$DesiredContext = 8192,
    [ValidateSet("F16","Q8_0","Q4_0")]
    [string]$KVType = "F16",
    [switch]$ForceFullOffload
)

$ErrorActionPreference = "Stop"

# ── GPU Specs ────────────────────────────────────────────────────────────────
$VRAM_TOTAL = 16.0
$VRAM_RESERVED = 2.3
$VRAM_AVAILABLE = $VRAM_TOTAL - $VRAM_RESERVED

# ── Model Database ───────────────────────────────────────────────────────────
$models = @{
    "llama3"   = @{ Params=8;  Layers=32; QHeads=32; KVHeads=8;  HeadDim=128; GQA=4;  QuantSizes=@{"Q4_0"=4.5; "Q5_K_M"=5.2; "Q8_0"=8.2} }
    "qwen2.5"  = @{ Params=7;  Layers=28; QHeads=28; KVHeads=4;  HeadDim=128; GQA=7;  QuantSizes=@{"Q4_0"=4.0; "Q5_K_M"=4.6; "Q8_0"=7.2} }
    "gemma4"   = @{ Params=9;  Layers=42; QHeads=16; KVHeads=16; HeadDim=256; GQA=1;  QuantSizes=@{"Q4_0"=5.2; "Q5_K_M"=6.0; "Q8_0"=9.5} }
    "devstral" = @{ Params=12.2;Layers=41; QHeads=32; KVHeads=8;  HeadDim=128; GQA=4;  QuantSizes=@{"Q4_0"=7.0; "Q5_K_M"=8.1; "Q8_0"=12.5}}
    "mistral"  = @{ Params=7;  Layers=32; QHeads=32; KVHeads=8;  HeadDim=128; GQA=4;  QuantSizes=@{"Q4_0"=4.0; "Q5_K_M"=4.6; "Q8_0"=7.2} }
}

$cfg = $models[$ModelFamily]
$bytesPerElem = switch ($KVType) { "F16" {2} "Q8_0" {1} "Q4_0" {0.5} }

function Get-KVCacheSize($layers, $kvHeads, $headDim, $seqLen, $bytes) {
    $perLayer = 2 * $kvHeads * $seqLen * $headDim * $bytes
    return ($layers * $perLayer) / [math]::Pow(1024, 3)
}

Write-Host "`n=== KV Cache Optimizer for RX 9070 XT ===" -ForegroundColor Cyan
Write-Host "Model: $ModelFamily ($($cfg.Params)B, $($cfg.Layers) layers, GQA=$($cfg.GQA))" -ForegroundColor White
Write-Host "Desired context: $DesiredContext tokens" -ForegroundColor White
Write-Host "KV type: $KVType ($bytesPerElem bytes/element)" -ForegroundColor White
Write-Host "Available VRAM: $([math]::Round($VRAM_AVAILABLE,1)) GiB" -ForegroundColor White

$kvSize = Get-KVCacheSize $cfg.Layers $cfg.KVHeads $cfg.HeadDim $DesiredContext $bytesPerElem

$bestQuant = $null
$bestModelSize = $null
foreach ($q in @("Q4_0","Q5_K_M","Q8_0")) {
    $modelSize = $cfg.QuantSizes[$q]
    $total = $modelSize + $kvSize + 0.5
    if ($total -le $VRAM_AVAILABLE) {
        $bestQuant = $q
        $bestModelSize = $modelSize
        break
    }
}

if (-not $bestQuant) {
    Write-Host "`n❌ IMPOSSIBLE: Even Q4_0 + $KVType KV doesn't fit!" -ForegroundColor Red
    $maxCtx = [math]::Floor((($VRAM_AVAILABLE - $cfg.QuantSizes["Q4_0"] - 0.5) * [math]::Pow(1024,3)) / 
                ($cfg.Layers * 2 * $cfg.KVHeads * $cfg.HeadDim * $bytesPerElem))
    Write-Host "`n💡 Maximum context with Q4_0 + $KVType`: $maxCtx tokens" -ForegroundColor Yellow
    exit 1
}

$totalVram = $bestModelSize + $kvSize + 0.5

Write-Host "`n✅ SOLUTION FOUND:" -ForegroundColor Green
Write-Host "   Quantization: $bestQuant (model ~$bestModelSize GiB)" -ForegroundColor Green
Write-Host "   KV Cache: $([math]::Round($kvSize,2)) GiB" -ForegroundColor Green
Write-Host "   Total: $([math]::Round($totalVram,2)) GiB / $VRAM_AVAILABLE GiB" -ForegroundColor Green
Write-Host "   Headroom: $([math]::Round($VRAM_AVAILABLE - $totalVram,2)) GiB" -ForegroundColor Green

Write-Host "`n=== Set these environment variables ===" -ForegroundColor Cyan
Write-Host "`$env:OLLAMA_NUM_GPU = `"$($cfg.Layers)`"" -ForegroundColor Yellow
Write-Host "`$env:OLLAMA_CTX_SIZE = `"$DesiredContext`"" -ForegroundColor Yellow
Write-Host "`$env:OLLAMA_KV_CACHE_TYPE = `"$($KVType.ToLower())`"" -ForegroundColor Yellow
Write-Host "`$env:OLLAMA_FLASH_ATTENTION = `"1`"" -ForegroundColor Yellow
Write-Host "`$env:HSA_OVERRIDE_GFX_VERSION = `"12.0.1`"" -ForegroundColor Yellow

if ($ForceFullOffload) {
    Write-Host "`n⚠️  FORCE FULL OFFLOAD enabled" -ForegroundColor Red
}

$baseDecode = 76
$basePrefill = switch ($ModelFamily) {
    "qwen2.5"  { 1800 }
    "llama3"   { 1500 }
    "mistral"  { 1400 }
    "gemma4"   { 1100 }
    "devstral" { 1300 }
    default    { 1200 }
}
$quantFactor = switch ($bestQuant) { "Q4_0" {1.15} "Q5_K_M" {1.05} "Q8_0" {1.0} }
$ctxFactor = [math]::Max(0.6, 1.0 - ($DesiredContext / 131072) * 0.4)
$estPrefill = [math]::Round($basePrefill * $quantFactor)
$estDecode = [math]::Round($baseDecode * $ctxFactor)

Write-Host "`n=== Expected Performance ===" -ForegroundColor Cyan
Write-Host "   Prefill: ~$estPrefill tok/s" -ForegroundColor White
Write-Host "   Decode:  ~$estDecode tok/s" -ForegroundColor White

if ($cfg.GQA -eq 1) {
    Write-Host "`n⚠️  WARNING: Dense attention (no GQA). KV cache is $($cfg.QHeads)x larger." -ForegroundColor Yellow
}
if ($DesiredContext -gt 32768 -and $KVType -eq "F16") {
    Write-Host "`n⚠️  WARNING: Context >32K with F16 KV may OOM. Use Q8_0 or Q4_0." -ForegroundColor Yellow
}

Write-Host "`n=== Ready to benchmark! ===" -ForegroundColor Green
Write-Host "   ollama run ${ModelFamily}:$($bestQuant.ToLower()) `"Your prompt`" --verbose" -ForegroundColor White
