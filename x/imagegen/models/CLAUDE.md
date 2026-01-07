# Model Implementation Guide

See `README.md` for memory management (critical for Go + MLX).

## Phase 1: Import & Forward Pass

- Read Python reference implementation (PyTorch/Transformers)
- Create Go struct mirroring layer hierarchy
- Implement weight loading from safetensors (see `safetensors.go`)
- Port forward pass layer-by-layer, bottom-up
- For tokenizers: check if BPE (`bpe.go`) or custom needed

**Key files to reference:** `llama` (dense LLM), `gpt_oss` (MoE LLM), `zimage` (image generation), `qwen_image_edit` (image editing)

### Vision Models: Image Preprocessing

When implementing vision models (image-to-text, image editing, etc.), image preprocessing must match Python exactly. Common pitfalls:

1. **Resolution constraints**: Many vision models use `min_pixels` and `max_pixels` to constrain image size, not a fixed target area. Check the Python processor's `smart_resize` logic.

2. **Patch alignment**: Images must be resized to multiples of `factor = patch_size * spatial_merge_size` (e.g., 14 \* 2 = 28 for Qwen2.5-VL).

3. **Normalization**: Vision encoders use ImageNet stats (mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]), not simple [-1, 1] scaling.

4. **Temporal dimension**: Video/image models may expect a temporal dimension (e.g., `[B, T, C, H, W]`). For single images, duplicate frames if `temporal_patch_size > 1`.

**Verification**: Always compare Go preprocessed image shape and statistics against Python to catch sizing mismatches early.

### Tokenizer & Chat Templates

Most instruction-tuned models require:

1. **BOS token**: Added at start of input (token ID 2 for most models)
2. **Chat template**: Wraps user prompt in model-specific format

**Common chat templates:**

| Model   | Format                                                                                                                                     |
| ------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| Llama 3 | `<\|begin_of_text\|><\|start_header_id\|>user<\|end_header_id\|>\n{prompt}<\|eot_id\|><\|start_header_id\|>assistant<\|end_header_id\|>\n` |
| Gemma 3 | `<bos><start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n`                                                                  |
| Qwen    | `<\|im_start\|>user\n{prompt}<\|im_end\|>\n<\|im_start\|>assistant\n`                                                                      |

**Checking tokenization:**

```bash
source .venv/bin/activate && python3 -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('./weights/model-name')
tokens = tok.encode('Hello', add_special_tokens=True)
print('Tokens:', tokens)
print('Decoded:', [tok.decode([t]) for t in tokens])
"
```

### Text Model Checklist

Before moving to vision components, ensure the text model is fully working:

1. **Sliding window cache**: Some models (Gemma 3, GPT-OSS) use sliding window attention on certain layers. Use `cache.NewRotatingKVCache(windowSize)` for those layers, not `cache.NewKVCache()`. Check config for `sliding_window` and `sliding_window_pattern`.

2. **Unicode/UTF-8 decoding**: If output shows garbled characters like `Â` before spaces, the tokenizer's byte-level encoding isn't being decoded properly. Check `Decode()` handles UTF-8 byte sequences correctly.

3. **EOS tokens from vocabulary**: Don't hardcode EOS token IDs. The tokenizer should extract them from `added_tokens` in `tokenizer.json`. Multiple EOS tokens are common (e.g., Gemma has both `<eos>` and `<end_of_turn>`).

4. **Chat template**: Instruction-tuned models need chat formatting. Test with and without to ensure the model responds coherently.

5. **Compare with reference**: Always test against `mlx_lm.generate` with same prompt and `--temp 0` to verify outputs match.

## Phase 2: Correctness Testing

Run the model and look at the output. Make sure it outputs something coherent.

To compare correctness, add hooks to the python model and compare the output with debug statements in Go.

## Phase 3: Memory Verification

After loading, verify peak memory is close to final model size:

```bash
# Run and check peak vs active memory
/tmp/engine -model ./weights/MyModel -steps 1 2>&1 | grep -E "(peak|GB)"
```

**Expected:** Peak should be ~1.1x final size (small overhead is OK). If peak is 2-3x final size, you have a memory problem.

### Checking Weight Dtypes

```bash
# Check dtype of weights in safetensors files
python3 -c "
from safetensors import safe_open
f = safe_open('model.safetensors', 'pt')
for k in list(f.keys())[:5]:
    print(k, f.get_tensor(k).dtype)
"
```

### f32 Weights Need Special Handling

If weights are f32 but model runs in bf16, use `GetTensorBF16()` instead of `GetTensor()`:

- `GetTensor()` uses MLX's native loader (loads all tensors from file at once)
- `GetTensorBF16()` loads one tensor at a time, converts to bf16, frees f32 immediately

This prevents peak memory from being 2x model size during loading.

## Phase 4: Performance

### Evaluation Strategy

- Call `mlx.Eval()` once per token/step, not inside loops
- Use `mlx.AsyncEval()` to pipeline: build next step's graph while current executes
- Never call `mlx.Eval()` inside attention or MLP - batch it at the end

### Fast Operations (Already Built-in)

These Go functions use MLX's fast fused kernels internally:

- `mlx.RMSNorm(x, weight, eps)` → uses `mlx_fast_rms_norm`
- `mlx.RoPE(x, dims, traditional, base, scale, offset)` → uses `mlx_fast_rope`
- `mlx.ScaledDotProductAttention(q, k, v, scale, causalMask)` → uses `mlx_fast_scaled_dot_product_attention`

### Type Promotion Gotchas

- `mlx.Mul(bf16Array, mlx.Full(shape, 2.0, mlx.Float32))` → upcasts everything to f32
- Use `mlx.MulScalar(bf16Array, 2.0)` to preserve dtype (if available), or ensure scalar arrays match input dtype

### Profiling

- Use `mactop` to check GPU utilization - should be ~100%
- If low, bottleneck is likely Go code (tokenization, data prep), not MLX
- Use `pprof` for CPU profiling to find Go-side overhead (CGO calls, tokenization, etc.)
- Use Metal debugger for kernel-level profiling (see docs/performance.md)
- Profile with `time.Since()` around major blocks
- Compare tok/s against reference (llama.cpp, MLX-LM)

## Phase 5: Polish

- Remove debug prints
- Add proper error handling
- Document config.json fields used

## Tips

- MLX is lazy; call `Eval()` only when you need values
- Check `model.safetensors.index.json` for weight→file mapping

## Common Gotchas

### MLX Transpose requires Contiguous

`mlx.Transpose()` returns a view with modified strides - calling `Data()` returns the original memory layout. Always follow with `mlx.Contiguous()` if you need correct data ordering:

```go
// Wrong - Data() returns original layout
x = mlx.Transpose(x, 0, 2, 3, 4, 1)
data := x.Data()  // Bug: data is in wrong order

// Correct
x = mlx.Contiguous(mlx.Transpose(x, 0, 2, 3, 4, 1))
data := x.Data()  // Data is in transposed order
```

### Missing Biases in Weight Loading

Python layers often have optional biases. Check the safetensors files for bias tensors:

```bash
python3 -c "from safetensors import safe_open; f=safe_open('model.safetensors','pt'); print([k for k in f.keys() if 'bias' in k])"
```

### Don't Spam ClearCache() or Eval()

- `mlx.ClearCache()` clears the GPU cache but doesn't free arrays - it has minimal effect on memory. Don't call it repeatedly.
- `mlx.Eval()` forces synchronous evaluation and frees non-kept arrays. Call it once per step/token, not inside loops.

### Lazy Eval and Free() - The Critical Pattern

MLX arrays are lazy - operations build a graph, actual computation happens at `Eval()`. This has a critical implication for `Free()`:

```go
// WRONG: Lazy array references freed input
func BadForward(x *mlx.Array) *mlx.Array {
    return mlx.Add(compute(x), x)  // Returns lazy array referencing x
}

func Caller() {
    result := BadForward(input)
    input.Free()       // Frees input, but result still references it!
    mlx.Eval(result)   // CRASH: "expected a non-empty mlx_array"
}

// CORRECT: Eval before caller can free inputs
func GoodForward(x *mlx.Array) *mlx.Array {
    out := mlx.Add(compute(x), x)
    mlx.Eval(out)  // Materialize before returning
    return out
}
```

**Rule**: If your function returns an array that references its input (residual connections, skip connections), you MUST `Eval()` before returning - otherwise the caller may free the input while the result still needs it.

**Debugging**: Errors like "expected a non-empty mlx_array" at Eval time often mean a tensor was freed while still referenced by a lazy graph. Add logging BEFORE the Free() calls to find which one, not inside the lazy operations.

### Data() and DataInt32() Trigger Eval

Calling `.Data()` or `.DataInt32()` on an array does an implicit `Eval()`, which frees any un-eval'd arrays:

```go
// WRONG: tokenArray gets freed when we eval image
tokenArray := mlx.NewArrayInt32(tokens, shape)
image := processImage(path)  // This evals image internally
mlx.Eval(image)              // This frees tokenArray!

tokenData := tokenArray.DataInt32()  // CRASH: tokenArray was freed

// CORRECT: Eval arrays you need to keep before other evals
tokenArray := mlx.NewArrayInt32(tokens, shape)
mlx.Eval(tokenArray)  // Materialize it first
image := processImage(path)

tokenData := tokenArray.DataInt32()  // Works fine
```

**Rule**: Before calling any function that might do an `Eval()` internally, make sure to `Eval()` any arrays you'll need later. When passing arrays to model forward functions, eval them first if they were just created.

### Diffusers Pipeline vs Scheduler Defaults

Diffusers pipelines often pass custom parameters that override scheduler defaults. When writing tests, match what the **pipeline** does, not the raw scheduler:

```python
# Scheduler default (when no sigmas passed):
#   sigmas from 1.0 to 1/1000 = 0.001

# But pipeline passes custom sigmas:
sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
scheduler.set_timesteps(sigmas=sigmas, ...)  # 1.0 to 1/30 for 30 steps
```

Always check the pipeline source to see what parameters it passes to components.

### Diffusion Models: Timestep Scaling

Diffusion transformers use sinusoidal timestep embeddings with internal scaling. **Critical**: Check what the pipeline actually passes to the transformer, not just what the scheduler stores.

**Common pattern in diffusers (tricky!):**

- `scheduler.sigmas` = values in [0, 1] range (e.g., 1.0, 0.608, 0.02)
- `scheduler.timesteps` = sigmas × 1000 (e.g., 1000, 608, 20)
- **BUT** the pipeline often divides by 1000 before passing to transformer: `timestep=t / 1000`
- Transformer's `Timesteps` class has `scale=1000`, multiplying input by 1000
- Net effect: transformer receives sigma (0.608), scales to 608

**Verification - check the actual pipeline source:**

```bash
grep -A2 "timestep=" .venv/.../pipeline_*.py
# Look for: timestep=timestep / 1000  ← pipeline normalizes!
```

**Go approach (skip the multiply/divide dance):**

```go
// Store sigmas directly as timesteps - equivalent to Python's
// scheduler.timesteps / 1000 that the pipeline passes to transformer
s.Timesteps[i] = sigmas[i]  // 0.608
// Transformer does: 0.608 * 1000 = 608 ✓
```

**Symptoms of wrong timestep scaling:**

- Noise predictions have wrong magnitude (off by orders of magnitude)
- Output images are completely noisy/corrupted or have extreme contrast
- Latents diverge from Python after first denoising step

**Key lesson:** Don't assume scheduler.timesteps is what the transformer receives - always check the pipeline's forward pass for any normalization.
