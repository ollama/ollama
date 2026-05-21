package llm

import (
	"log/slog"
	"strings"

	"github.com/ollama/ollama/api"
)

// applyFastOptimizations applies user-specified fast optimization options from API
// Returns (kvCacheType, flashAttentionOverride, flashAttentionUserSet)
func applyFastOptimizations(opts api.Options, envKVCacheType string, envFlashAttention bool, envFlashAttentionUserSet bool) (string, bool, bool) {
	kvct := strings.ToLower(opts.KVCacheType)
	if kvct == "" {
		kvct = strings.ToLower(envKVCacheType)
	}

	fa := envFlashAttention
	faUserSet := envFlashAttentionUserSet

	// Flash attention: API override takes precedence
	if opts.FlashAttention != "" {
		faUserSet = true
		switch strings.ToLower(opts.FlashAttention) {
		case "enabled":
			fa = true
			slog.Info("flash attention enabled via API option")
		case "disabled":
			fa = false
			slog.Info("flash attention disabled via API option")
		case "auto":
			// Use default logic
		default:
			slog.Warn("invalid flash_attention value, use 'enabled', 'disabled', or 'auto'", "value", opts.FlashAttention)
		}
	}

	return kvct, fa, faUserSet
}
