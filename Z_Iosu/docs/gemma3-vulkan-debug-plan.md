# Gemma3 Vulkan Bug Investigation Plan

## Current Status
- Branch: `12-07-b5`
- Issue: `oscar_while/gemma-3-4b-tools:Q4_K_M` fails with `GGML_ASSERT(ggml_can_repeat(b, a))` error when using Vulkan backend
- Temporary workaround: Commented out assertion in `ml/backend/ggml/ggml/src/ggml.c` line 1921
- Current state: Model works with workaround on all backends (CPU, CUDA, Vulkan)

## What We Know
1. Model works on v0.12.6 **without Vulkan** (CPU backend)
2. Model works on 12-07-b5 **with workaround** on all backends
3. Model `qwen3_vl_4b_instruct` also had issues (patch_bias assertion in clip.cpp - already fixed)

## What We DON'T Know Yet
- Does v0.12.6 official fail with Vulkan backend?
- Did we introduce the bug in our branch 12-07-b5, or was it already present in v0.12.6?

## Investigation Steps (NOT COMPLETED YET)

### Step 1: Test 12-07-b5 WITHOUT Workaround
```powershell
# Navigate to repo
cd C:\IA\tools\ollama

# Make sure we're on 12-07-b5
git checkout 12-07-b5

# Remove the workaround in ml/backend/ggml/ggml/src/ggml.c
# Restore line 1921 to: GGML_ASSERT(ggml_can_repeat(b, a));
# Remove the debug fprintf statements (lines 1916-1923)

# Recompile
powershell -ExecutionPolicy Bypass -File Z_Iosu\scripts\build_windows.ps1 buildCPU buildVulkan buildOllama buildApp

# Test with Vulkan - SHOULD FAIL
cd dist\windows-amd64
.\ollama.exe serve
# In another terminal:
.\ollama.exe run oscar_while/gemma-3-4b-tools:Q4_K_M
# Expected: Crash with ggml_can_repeat error
```

### Step 2: Test v0.12.6 Official WITH Vulkan
```powershell
# Checkout v0.12.6 (stash changes first)
git stash
git checkout v0.12.6

# Compile v0.12.6 with Vulkan
# Note: v0.12.6 doesn't have Z_Iosu build scripts, need alternative method
# Try: go generate ./... then manual build
# OR: Cherry-pick build scripts from 12-07-b5

# Test with Vulkan
cd dist\windows-amd64
.\ollama.exe serve
# In another terminal:
.\ollama.exe run oscar_while/gemma-3-4b-tools:Q4_K_M

# Expected outcomes:
# - If it FAILS: Bug existed in v0.12.6, our workaround is valid
# - If it WORKS: We introduced the bug in 12-07-b5, need to find which commit
```

### Step 3A: If v0.12.6 Works (We Introduced Bug)
```powershell
# Git bisect to find problematic commit
git checkout 12-07-b5
git bisect start
git bisect bad 12-07-b5
git bisect good v0.12.6

# For each bisect step:
# 1. Compile with Vulkan
# 2. Test model
# 3. Mark as good/bad
# 4. Continue until found

# Once found, analyze the commit and fix properly
```

### Step 3B: If v0.12.6 Fails (Bug Already Existed)
```powershell
# Our workaround is valid
# Proceed with proper documentation and commit

# Optional: Check if upstream has fixed this in newer versions
git fetch upstream
git log v0.12.6..upstream/main -- ml/backend/ggml/ggml/src/ggml.c

# Consider reporting to upstream if not fixed
```

## Files Modified
1. `ml/backend/ggml/ggml/src/ggml.c` - Line 1921 (ggml_can_repeat assertion)
2. `llama/llama.cpp/tools/mtmd/clip.cpp` - Line 667 (patch_bias assertion) - Already committed

## Important Notes
- All testing MUST be done with Vulkan backend specifically
- Don't test with CPU/CUDA as they work fine
- The bug is specific to Vulkan + Gemma3 multimodal
- Model `oscar_while/gemma-3-4b-tools:Q4_K_M` is the test case
- Model `qwen3_vl_4b_instruct` works after clip.cpp fix

## Commits Made
- `4696eaab0` - "Fix gemma3 multimodal compatibility (temporary workaround)"
- `1da3d018d` - Cherry-picked llama.cpp bump with multimodal fixes

## Next Session TODO
1. Remove workaround from ggml.c
2. Test 12-07-b5 with Vulkan (should fail)
3. Compile and test v0.12.6 with Vulkan
4. Based on result, either bisect or document workaround properly
5. Write proper commit message in English explaining findings
