# Issue #16111: kimi-k2.6 Model Alias Investigation

## Summary
The model endpoint `kimi-k2.6` is incorrectly mapped to Kimi K2.5 instead of Kimi K2.6. Both `kimi-k2.6` and `kimi-k2.5` endpoints return the same underlying model (K2.5).

## Root Cause
The model-to-endpoint mapping for cloud models does **not** live in the `ollama/ollama` open-source repository. 

The ollama server proxies cloud model requests to `ollama.com`'s backend API, which handles the mapping from model names (e.g., `kimi-k2.6`) to actual inference endpoints. This mapping is in the **closed-source ollama.com cloud backend**.

### How Cloud Models Flow
1. User requests model `kimi-k2.6:cloud`
2. `server/cloud_proxy.go` - `cloudPassthroughMiddleware` detects `:cloud` suffix
3. The `:cloud` suffix is stripped → model name becomes `kimi-k2.6`  
4. Request is proxied to `https://ollama.com/api/generate` with model `kimi-k2.6`
5. **ollama.com backend** maps `kimi-k2.6` to an inference endpoint ← BUG IS HERE
6. Currently both `kimi-k2.6` and `kimi-k2.5` resolve to the same K2.5 endpoint

### Files in ollama/ollama repo related to kimi-k2.6
- `server/model_recommendations.go:369` - Default model recommendation `kimi-k2.6:cloud`
- `cmd/launch/models.go:26` - Client-side recommended model `kimi-k2.6:cloud`
- `cmd/launch/models.go:67` - Cloud model limit `kimi-k2.6`
- `server/cloud_proxy.go` - Cloud proxy middleware (strips `:cloud` suffix)
- `internal/modelref/modelref.go` - Model reference parsing

## Fix Required
The fix must be applied on the **ollama.com cloud backend** to correctly route `kimi-k2.6` to the Kimi K2.6 inference endpoint instead of the K2.5 endpoint it currently resolves to.

## Testing
Once the backend mapping is corrected:
1. `curl -d '{"model":"kimi-k2.6:cloud","prompt":"State your exact model name and version.","stream":false}' https://ollama.com/api/generate` should return K2.6
2. `curl -d '{"model":"kimi-k2.5:cloud","prompt":"State your exact model name and version.","stream":false}' https://ollama.com/api/generate` should return K2.5
