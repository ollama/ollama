package llm

import "log/slog"

// llamaServerNoKVShiftArchs lists GGUF architectures whose llama.cpp memory
// reports get_can_shift() == false. llama-server then disables context
// shifting itself and warns ("KV cache shifting is not supported for this
// context"), so Ollama should not ask for it. Keep this in sync with the
// vendored llama.cpp:
//   - llama_kv_cache_dsv4::get_can_shift (DeepSeek V4 compressed KV cache)
//   - llama_kv_cache::get_can_shift (LLM_ARCH_STEP35, and any model with
//     n_pos_per_embd() > 1, i.e. LLAMA_ROPE_TYPE_MROPE / IMROPE in
//     llama_model_rope_type)
//
// glm5-next (GLM 5.3 Flash) is not listed: its loads never logged the warning.
var llamaServerNoKVShiftArchs = map[string]string{
	"deepseek4":  "dsv4 compressed kv cache",
	"deepseek41": "dsv4 compressed kv cache",
	"step35":     "per-layer rope dims",
	"qwen2vl":    "m-rope",
	"paddleocr":  "m-rope",
	"dflash":     "m-rope",
	"qwen3vl":    "im-rope",
	"qwen3vlmoe": "im-rope",
	"qwen35":     "im-rope",
	"qwen35moe":  "im-rope",
	"qwen4exp":   "im-rope",
	"clef":       "im-rope",
	"qwen3tts":   "im-rope",
}

// llamaServerSupportsContextShift reports whether llama-server can honor
// --context-shift for a model. llama-server also turns it off whenever an
// mmproj is loaded ("ctx_shift is not supported by multimodal").
func llamaServerSupportsContextShift(arch string, hasProjector bool) (bool, string) {
	if reason, ok := llamaServerNoKVShiftArchs[arch]; ok {
		return false, reason
	}
	if hasProjector {
		return false, "multimodal projector"
	}
	return true, ""
}

// serverContextShift reports whether to pass --context-shift. Ollama's own
// prompt truncation still follows config.ContextShift; this only avoids
// requesting a llama-server feature it would disable anyway.
func (launch llamaServerLaunchConfig) serverContextShift() bool {
	if !launch.config.ContextShift {
		return false
	}
	ok, reason := llamaServerSupportsContextShift(launch.modelArch, len(launch.projectors) > 0)
	if !ok {
		slog.Debug("not passing --context-shift to llama-server", "arch", launch.modelArch, "reason", reason)
	}
	return ok
}
