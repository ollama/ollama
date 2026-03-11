// Package fitcheck provides hardware profiling and model compatibility scoring.
package fitcheck

// ModelRequirement describes the hardware needed to run one model variant.
//
// VRAM rule of thumb used for estimates:
//
//	Q4_K_M ~= params_B × 600 MB + 512 MB overhead
//	Q8_0   ~= params_B × 1100 MB + 512 MB overhead
//	F16    ~= params_B × 2000 MB + 512 MB overhead
//
// Tags: code, vision, embed, small (≤4B), reasoning, multilingual
type ModelRequirement struct {
	Name       string   `json:"name"`
	Family     string   `json:"family"`
	Quant      string   `json:"quant"`
	DiskSizeMB uint64   `json:"disk_size_mb,omitempty"`
	VRAMMinMB  uint64   `json:"vram_min_mb,omitempty"`
	RAMMinMB   uint64   `json:"ram_min_mb,omitempty"`
	Tags       []string `json:"tags,omitempty"`
}

// Requirements is the built-in model catalogue sourced from ollama.com/library.
var Requirements = []ModelRequirement{

	// ── Llama 3.2 ────────────────────────────────────────────────────────────
	{Name: "llama3.2:1b", Family: "llama3.2", Quant: "Q4_K_M", DiskSizeMB: 1300, VRAMMinMB: 1536, RAMMinMB: 2048, Tags: []string{"small"}},
	{Name: "llama3.2:3b", Family: "llama3.2", Quant: "Q4_K_M", DiskSizeMB: 2000, VRAMMinMB: 2560, RAMMinMB: 4096, Tags: []string{"small"}},

	// ── Llama 3.1 ────────────────────────────────────────────────────────────
	{Name: "llama3.1:8b", Family: "llama3.1", Quant: "Q4_K_M", DiskSizeMB: 4900, VRAMMinMB: 5632, RAMMinMB: 8192},
	{Name: "llama3.1:8b-instruct-q8_0", Family: "llama3.1", Quant: "Q8_0", DiskSizeMB: 8600, VRAMMinMB: 9728, RAMMinMB: 12288},
	{Name: "llama3.1:70b", Family: "llama3.1", Quant: "Q4_K_M", DiskSizeMB: 43008, VRAMMinMB: 45056, RAMMinMB: 64000},
	{Name: "llama3.1:405b", Family: "llama3.1", Quant: "Q4_K_M", DiskSizeMB: 248832, VRAMMinMB: 262144, RAMMinMB: 512000},

	// ── Llama 3.3 ────────────────────────────────────────────────────────────
	{Name: "llama3.3:70b", Family: "llama3.3", Quant: "Q4_K_M", DiskSizeMB: 44032, VRAMMinMB: 46080, RAMMinMB: 64000},

	// ── Llama 3 ──────────────────────────────────────────────────────────────
	{Name: "llama3:8b", Family: "llama3", Quant: "Q4_K_M", DiskSizeMB: 4812, VRAMMinMB: 5632, RAMMinMB: 8192},
	{Name: "llama3:70b", Family: "llama3", Quant: "Q4_K_M", DiskSizeMB: 40960, VRAMMinMB: 43008, RAMMinMB: 64000},

	// ── Llama 2 ──────────────────────────────────────────────────────────────
	{Name: "llama2:7b", Family: "llama2", Quant: "Q4_K_M", DiskSizeMB: 3891, VRAMMinMB: 5120, RAMMinMB: 8192},
	{Name: "llama2:13b", Family: "llama2", Quant: "Q4_K_M", DiskSizeMB: 7578, VRAMMinMB: 9216, RAMMinMB: 16384},
	{Name: "llama2:70b", Family: "llama2", Quant: "Q4_K_M", DiskSizeMB: 39936, VRAMMinMB: 43008, RAMMinMB: 64000},

	// ── Llama 3.2 Vision ─────────────────────────────────────────────────────
	{Name: "llama3.2-vision:11b", Family: "llama3.2-vision", Quant: "Q4_K_M", DiskSizeMB: 7987, VRAMMinMB: 8704, RAMMinMB: 16384, Tags: []string{"vision"}},
	{Name: "llama3.2-vision:90b", Family: "llama3.2-vision", Quant: "Q4_K_M", DiskSizeMB: 56320, VRAMMinMB: 59392, RAMMinMB: 96000, Tags: []string{"vision"}},

	// ── Mistral ───────────────────────────────────────────────────────────────
	{Name: "mistral:7b", Family: "mistral", Quant: "Q4_K_M", DiskSizeMB: 4506, VRAMMinMB: 5632, RAMMinMB: 8192},
	{Name: "mistral:7b-instruct-q8_0", Family: "mistral", Quant: "Q8_0", DiskSizeMB: 7700, VRAMMinMB: 9216, RAMMinMB: 12288},

	// ── Mistral Nemo ──────────────────────────────────────────────────────────
	{Name: "mistral-nemo:12b", Family: "mistral-nemo", Quant: "Q4_K_M", DiskSizeMB: 7270, VRAMMinMB: 8192, RAMMinMB: 16384, Tags: []string{"multilingual"}},

	// ── Mistral Small ─────────────────────────────────────────────────────────
	{Name: "mistral-small:22b", Family: "mistral-small", Quant: "Q4_K_M", DiskSizeMB: 13312, VRAMMinMB: 14336, RAMMinMB: 24576},
	{Name: "mistral-small:24b", Family: "mistral-small", Quant: "Q4_K_M", DiskSizeMB: 14336, VRAMMinMB: 16384, RAMMinMB: 24576},

	// ── Mixtral ───────────────────────────────────────────────────────────────
	{Name: "mixtral:8x7b", Family: "mixtral", Quant: "Q4_K_M", DiskSizeMB: 26624, VRAMMinMB: 28672, RAMMinMB: 48000, Tags: []string{"multilingual"}},
	{Name: "mixtral:8x22b", Family: "mixtral", Quant: "Q4_K_M", DiskSizeMB: 81920, VRAMMinMB: 86016, RAMMinMB: 128000, Tags: []string{"multilingual"}},

	// ── Phi 3 ─────────────────────────────────────────────────────────────────
	{Name: "phi3:3.8b", Family: "phi3", Quant: "Q4_K_M", DiskSizeMB: 2253, VRAMMinMB: 3072, RAMMinMB: 4096, Tags: []string{"small"}},
	{Name: "phi3:14b", Family: "phi3", Quant: "Q4_K_M", DiskSizeMB: 8089, VRAMMinMB: 9216, RAMMinMB: 16384},

	// ── Phi 3.5 ───────────────────────────────────────────────────────────────
	{Name: "phi3.5:3.8b", Family: "phi3.5", Quant: "Q4_K_M", DiskSizeMB: 2253, VRAMMinMB: 3072, RAMMinMB: 4096, Tags: []string{"small"}},

	// ── Phi 4 ─────────────────────────────────────────────────────────────────
	{Name: "phi4:14b", Family: "phi4", Quant: "Q4_K_M", DiskSizeMB: 9318, VRAMMinMB: 10240, RAMMinMB: 16384},
	{Name: "phi4:14b-q8_0", Family: "phi4", Quant: "Q8_0", DiskSizeMB: 15360, VRAMMinMB: 16896, RAMMinMB: 24576},

	// ── Phi 4 Mini ────────────────────────────────────────────────────────────
	{Name: "phi4-mini:3.8b", Family: "phi4-mini", Quant: "Q4_K_M", DiskSizeMB: 2560, VRAMMinMB: 3072, RAMMinMB: 4096, Tags: []string{"small"}},

	// ── Gemma ─────────────────────────────────────────────────────────────────
	{Name: "gemma:2b", Family: "gemma", Quant: "Q4_K_M", DiskSizeMB: 1741, VRAMMinMB: 2560, RAMMinMB: 4096, Tags: []string{"small"}},
	{Name: "gemma:7b", Family: "gemma", Quant: "Q4_K_M", DiskSizeMB: 5120, VRAMMinMB: 6144, RAMMinMB: 8192},

	// ── Gemma 2 ───────────────────────────────────────────────────────────────
	{Name: "gemma2:2b", Family: "gemma2", Quant: "Q4_K_M", DiskSizeMB: 1638, VRAMMinMB: 2560, RAMMinMB: 4096, Tags: []string{"small"}},
	{Name: "gemma2:9b", Family: "gemma2", Quant: "Q4_K_M", DiskSizeMB: 5530, VRAMMinMB: 6656, RAMMinMB: 10240},
	{Name: "gemma2:27b", Family: "gemma2", Quant: "Q4_K_M", DiskSizeMB: 16384, VRAMMinMB: 18432, RAMMinMB: 32768},

	// ── Gemma 3 ───────────────────────────────────────────────────────────────
	{Name: "gemma3:1b", Family: "gemma3", Quant: "Q4_K_M", DiskSizeMB: 815, VRAMMinMB: 1536, RAMMinMB: 2048, Tags: []string{"small", "vision"}},
	{Name: "gemma3:4b", Family: "gemma3", Quant: "Q4_K_M", DiskSizeMB: 3379, VRAMMinMB: 4096, RAMMinMB: 6144, Tags: []string{"small", "vision"}},
	{Name: "gemma3:12b", Family: "gemma3", Quant: "Q4_K_M", DiskSizeMB: 8294, VRAMMinMB: 9728, RAMMinMB: 16384, Tags: []string{"vision"}},
	{Name: "gemma3:27b", Family: "gemma3", Quant: "Q4_K_M", DiskSizeMB: 17408, VRAMMinMB: 19456, RAMMinMB: 32768, Tags: []string{"vision"}},

	// ── Qwen 2 ────────────────────────────────────────────────────────────────
	{Name: "qwen2:0.5b", Family: "qwen2", Quant: "Q4_K_M", DiskSizeMB: 352, VRAMMinMB: 1024, RAMMinMB: 2048, Tags: []string{"small", "multilingual"}},
	{Name: "qwen2:1.5b", Family: "qwen2", Quant: "Q4_K_M", DiskSizeMB: 935, VRAMMinMB: 1536, RAMMinMB: 2048, Tags: []string{"small", "multilingual"}},
	{Name: "qwen2:7b", Family: "qwen2", Quant: "Q4_K_M", DiskSizeMB: 4506, VRAMMinMB: 5632, RAMMinMB: 8192, Tags: []string{"multilingual"}},
	{Name: "qwen2:72b", Family: "qwen2", Quant: "Q4_K_M", DiskSizeMB: 41984, VRAMMinMB: 45056, RAMMinMB: 64000, Tags: []string{"multilingual"}},

	// ── Qwen 2.5 ──────────────────────────────────────────────────────────────
	{Name: "qwen2.5:0.5b", Family: "qwen2.5", Quant: "Q4_K_M", DiskSizeMB: 398, VRAMMinMB: 1024, RAMMinMB: 2048, Tags: []string{"small", "multilingual"}},
	{Name: "qwen2.5:1.5b", Family: "qwen2.5", Quant: "Q4_K_M", DiskSizeMB: 986, VRAMMinMB: 1536, RAMMinMB: 2048, Tags: []string{"small", "multilingual"}},
	{Name: "qwen2.5:3b", Family: "qwen2.5", Quant: "Q4_K_M", DiskSizeMB: 1946, VRAMMinMB: 2560, RAMMinMB: 4096, Tags: []string{"small", "multilingual"}},
	{Name: "qwen2.5:7b", Family: "qwen2.5", Quant: "Q4_K_M", DiskSizeMB: 4813, VRAMMinMB: 5632, RAMMinMB: 8192, Tags: []string{"multilingual"}},
	{Name: "qwen2.5:14b", Family: "qwen2.5", Quant: "Q4_K_M", DiskSizeMB: 9216, VRAMMinMB: 10240, RAMMinMB: 16384, Tags: []string{"multilingual"}},
	{Name: "qwen2.5:14b-instruct-q8_0", Family: "qwen2.5", Quant: "Q8_0", DiskSizeMB: 15360, VRAMMinMB: 16896, RAMMinMB: 24576, Tags: []string{"multilingual"}},
	{Name: "qwen2.5:32b", Family: "qwen2.5", Quant: "Q4_K_M", DiskSizeMB: 20480, VRAMMinMB: 22528, RAMMinMB: 32768, Tags: []string{"multilingual"}},
	{Name: "qwen2.5:72b", Family: "qwen2.5", Quant: "Q4_K_M", DiskSizeMB: 48128, VRAMMinMB: 51200, RAMMinMB: 80000, Tags: []string{"multilingual"}},

	// ── Qwen 2.5 Coder ────────────────────────────────────────────────────────
	{Name: "qwen2.5-coder:0.5b", Family: "qwen2.5-coder", Quant: "Q4_K_M", DiskSizeMB: 398, VRAMMinMB: 1024, RAMMinMB: 2048, Tags: []string{"code", "small"}},
	{Name: "qwen2.5-coder:1.5b", Family: "qwen2.5-coder", Quant: "Q4_K_M", DiskSizeMB: 986, VRAMMinMB: 1536, RAMMinMB: 2048, Tags: []string{"code", "small"}},
	{Name: "qwen2.5-coder:3b", Family: "qwen2.5-coder", Quant: "Q4_K_M", DiskSizeMB: 1946, VRAMMinMB: 2560, RAMMinMB: 4096, Tags: []string{"code", "small"}},
	{Name: "qwen2.5-coder:7b", Family: "qwen2.5-coder", Quant: "Q4_K_M", DiskSizeMB: 4813, VRAMMinMB: 5632, RAMMinMB: 8192, Tags: []string{"code"}},
	{Name: "qwen2.5-coder:14b", Family: "qwen2.5-coder", Quant: "Q4_K_M", DiskSizeMB: 9216, VRAMMinMB: 10240, RAMMinMB: 16384, Tags: []string{"code"}},
	{Name: "qwen2.5-coder:32b", Family: "qwen2.5-coder", Quant: "Q4_K_M", DiskSizeMB: 20480, VRAMMinMB: 22528, RAMMinMB: 32768, Tags: []string{"code"}},

	// ── Qwen 3 ────────────────────────────────────────────────────────────────
	{Name: "qwen3:0.6b", Family: "qwen3", Quant: "Q4_K_M", DiskSizeMB: 523, VRAMMinMB: 1024, RAMMinMB: 2048, Tags: []string{"small", "multilingual", "reasoning"}},
	{Name: "qwen3:1.7b", Family: "qwen3", Quant: "Q4_K_M", DiskSizeMB: 1434, VRAMMinMB: 2048, RAMMinMB: 3072, Tags: []string{"small", "multilingual", "reasoning"}},
	{Name: "qwen3:4b", Family: "qwen3", Quant: "Q4_K_M", DiskSizeMB: 2560, VRAMMinMB: 3584, RAMMinMB: 6144, Tags: []string{"small", "multilingual", "reasoning"}},
	{Name: "qwen3:8b", Family: "qwen3", Quant: "Q4_K_M", DiskSizeMB: 5325, VRAMMinMB: 6144, RAMMinMB: 10240, Tags: []string{"multilingual", "reasoning"}},
	{Name: "qwen3:14b", Family: "qwen3", Quant: "Q4_K_M", DiskSizeMB: 9523, VRAMMinMB: 10752, RAMMinMB: 16384, Tags: []string{"multilingual", "reasoning"}},
	{Name: "qwen3:32b", Family: "qwen3", Quant: "Q4_K_M", DiskSizeMB: 20480, VRAMMinMB: 22528, RAMMinMB: 32768, Tags: []string{"multilingual", "reasoning"}},
	{Name: "qwen3:30b-a3b", Family: "qwen3", Quant: "Q4_K_M", DiskSizeMB: 19456, VRAMMinMB: 6144, RAMMinMB: 24576, Tags: []string{"multilingual", "reasoning"}},
	{Name: "qwen3:235b-a22b", Family: "qwen3", Quant: "Q4_K_M", DiskSizeMB: 145408, VRAMMinMB: 28672, RAMMinMB: 192000, Tags: []string{"multilingual", "reasoning"}},

	// ── DeepSeek R1 ───────────────────────────────────────────────────────────
	{Name: "deepseek-r1:1.5b", Family: "deepseek-r1", Quant: "Q4_K_M", DiskSizeMB: 1127, VRAMMinMB: 2048, RAMMinMB: 3072, Tags: []string{"reasoning", "small"}},
	{Name: "deepseek-r1:7b", Family: "deepseek-r1", Quant: "Q4_K_M", DiskSizeMB: 4813, VRAMMinMB: 5632, RAMMinMB: 8192, Tags: []string{"reasoning"}},
	{Name: "deepseek-r1:8b", Family: "deepseek-r1", Quant: "Q4_K_M", DiskSizeMB: 5325, VRAMMinMB: 6144, RAMMinMB: 10240, Tags: []string{"reasoning"}},
	{Name: "deepseek-r1:14b", Family: "deepseek-r1", Quant: "Q4_K_M", DiskSizeMB: 9216, VRAMMinMB: 10240, RAMMinMB: 16384, Tags: []string{"reasoning"}},
	{Name: "deepseek-r1:32b", Family: "deepseek-r1", Quant: "Q4_K_M", DiskSizeMB: 20480, VRAMMinMB: 22528, RAMMinMB: 32768, Tags: []string{"reasoning"}},
	{Name: "deepseek-r1:70b", Family: "deepseek-r1", Quant: "Q4_K_M", DiskSizeMB: 44032, VRAMMinMB: 46080, RAMMinMB: 80000, Tags: []string{"reasoning"}},
	{Name: "deepseek-r1:671b", Family: "deepseek-r1", Quant: "Q4_K_M", DiskSizeMB: 413696, VRAMMinMB: 430080, RAMMinMB: 512000, Tags: []string{"reasoning"}},

	// ── DeepSeek V3 ───────────────────────────────────────────────────────────
	{Name: "deepseek-v3:671b", Family: "deepseek-v3", Quant: "Q4_K_M", DiskSizeMB: 413696, VRAMMinMB: 430080, RAMMinMB: 512000},

	// ── DeepSeek Coder ────────────────────────────────────────────────────────
	{Name: "deepseek-coder:1.3b", Family: "deepseek-coder", Quant: "Q4_K_M", DiskSizeMB: 776, VRAMMinMB: 1536, RAMMinMB: 2048, Tags: []string{"code", "small"}},
	{Name: "deepseek-coder:6.7b", Family: "deepseek-coder", Quant: "Q4_K_M", DiskSizeMB: 3891, VRAMMinMB: 5120, RAMMinMB: 8192, Tags: []string{"code"}},
	{Name: "deepseek-coder:33b", Family: "deepseek-coder", Quant: "Q4_K_M", DiskSizeMB: 19456, VRAMMinMB: 22528, RAMMinMB: 32768, Tags: []string{"code"}},

	// ── DeepSeek Coder V2 ─────────────────────────────────────────────────────
	{Name: "deepseek-coder-v2:16b", Family: "deepseek-coder-v2", Quant: "Q4_K_M", DiskSizeMB: 9114, VRAMMinMB: 10240, RAMMinMB: 16384, Tags: []string{"code"}},
	{Name: "deepseek-coder-v2:236b", Family: "deepseek-coder-v2", Quant: "Q4_K_M", DiskSizeMB: 136192, VRAMMinMB: 143360, RAMMinMB: 192000, Tags: []string{"code"}},

	// ── Code Llama ────────────────────────────────────────────────────────────
	{Name: "codellama:7b", Family: "codellama", Quant: "Q4_K_M", DiskSizeMB: 3891, VRAMMinMB: 5120, RAMMinMB: 8192, Tags: []string{"code"}},
	{Name: "codellama:13b", Family: "codellama", Quant: "Q4_K_M", DiskSizeMB: 7578, VRAMMinMB: 9216, RAMMinMB: 16384, Tags: []string{"code"}},
	{Name: "codellama:34b", Family: "codellama", Quant: "Q4_K_M", DiskSizeMB: 19456, VRAMMinMB: 22528, RAMMinMB: 32768, Tags: []string{"code"}},
	{Name: "codellama:70b", Family: "codellama", Quant: "Q4_K_M", DiskSizeMB: 39936, VRAMMinMB: 43008, RAMMinMB: 64000, Tags: []string{"code"}},

	// ── StarCoder 2 ───────────────────────────────────────────────────────────
	{Name: "starcoder2:3b", Family: "starcoder2", Quant: "Q4_K_M", DiskSizeMB: 1741, VRAMMinMB: 2560, RAMMinMB: 4096, Tags: []string{"code", "small"}},
	{Name: "starcoder2:7b", Family: "starcoder2", Quant: "Q4_K_M", DiskSizeMB: 4096, VRAMMinMB: 5120, RAMMinMB: 8192, Tags: []string{"code"}},
	{Name: "starcoder2:15b", Family: "starcoder2", Quant: "Q4_K_M", DiskSizeMB: 9318, VRAMMinMB: 10240, RAMMinMB: 16384, Tags: []string{"code"}},

	// ── CodeGemma ─────────────────────────────────────────────────────────────
	{Name: "codegemma:2b", Family: "codegemma", Quant: "Q4_K_M", DiskSizeMB: 1638, VRAMMinMB: 2560, RAMMinMB: 4096, Tags: []string{"code", "small"}},
	{Name: "codegemma:7b", Family: "codegemma", Quant: "Q4_K_M", DiskSizeMB: 5120, VRAMMinMB: 6144, RAMMinMB: 8192, Tags: []string{"code"}},

	// ── Codestral ─────────────────────────────────────────────────────────────
	{Name: "codestral:22b", Family: "codestral", Quant: "Q4_K_M", DiskSizeMB: 13312, VRAMMinMB: 14336, RAMMinMB: 24576, Tags: []string{"code"}},

	// ── Devstral ──────────────────────────────────────────────────────────────
	{Name: "devstral:24b", Family: "devstral", Quant: "Q4_K_M", DiskSizeMB: 14336, VRAMMinMB: 16384, RAMMinMB: 32768, Tags: []string{"code"}},

	// ── Granite Code ──────────────────────────────────────────────────────────
	{Name: "granite-code:3b", Family: "granite-code", Quant: "Q4_K_M", DiskSizeMB: 2048, VRAMMinMB: 2560, RAMMinMB: 4096, Tags: []string{"code", "small"}},
	{Name: "granite-code:8b", Family: "granite-code", Quant: "Q4_K_M", DiskSizeMB: 4710, VRAMMinMB: 5632, RAMMinMB: 8192, Tags: []string{"code"}},
	{Name: "granite-code:20b", Family: "granite-code", Quant: "Q4_K_M", DiskSizeMB: 12288, VRAMMinMB: 13312, RAMMinMB: 24576, Tags: []string{"code"}},
	{Name: "granite-code:34b", Family: "granite-code", Quant: "Q4_K_M", DiskSizeMB: 19456, VRAMMinMB: 22528, RAMMinMB: 32768, Tags: []string{"code"}},

	// ── LLaVA ─────────────────────────────────────────────────────────────────
	{Name: "llava:7b", Family: "llava", Quant: "Q4_K_M", DiskSizeMB: 4813, VRAMMinMB: 6144, RAMMinMB: 8192, Tags: []string{"vision"}},
	{Name: "llava:13b", Family: "llava", Quant: "Q4_K_M", DiskSizeMB: 8192, VRAMMinMB: 10240, RAMMinMB: 16384, Tags: []string{"vision"}},
	{Name: "llava:34b", Family: "llava", Quant: "Q4_K_M", DiskSizeMB: 20480, VRAMMinMB: 22528, RAMMinMB: 40960, Tags: []string{"vision"}},

	// ── LLaVA Llama 3 ────────────────────────────────────────────────────────
	{Name: "llava-llama3:8b", Family: "llava-llama3", Quant: "Q4_K_M", DiskSizeMB: 5632, VRAMMinMB: 6656, RAMMinMB: 10240, Tags: []string{"vision"}},

	// ── LLaVA Phi-3 ───────────────────────────────────────────────────────────
	{Name: "llava-phi3:3.8b", Family: "llava-phi3", Quant: "Q4_K_M", DiskSizeMB: 2970, VRAMMinMB: 4096, RAMMinMB: 6144, Tags: []string{"vision", "small"}},

	// ── BakLLaVA ──────────────────────────────────────────────────────────────
	{Name: "bakllava:7b", Family: "bakllava", Quant: "Q4_K_M", DiskSizeMB: 4813, VRAMMinMB: 6144, RAMMinMB: 8192, Tags: []string{"vision"}},

	// ── Moondream ─────────────────────────────────────────────────────────────
	{Name: "moondream:1.8b", Family: "moondream", Quant: "Q4_K_M", DiskSizeMB: 1741, VRAMMinMB: 2560, RAMMinMB: 4096, Tags: []string{"vision", "small"}},

	// ── Granite 3.2 Vision ────────────────────────────────────────────────────
	{Name: "granite3.2-vision:2b", Family: "granite3.2-vision", Quant: "Q4_K_M", DiskSizeMB: 2458, VRAMMinMB: 3584, RAMMinMB: 6144, Tags: []string{"vision", "small"}},

	// ── Nomic Embed Text ──────────────────────────────────────────────────────
	{Name: "nomic-embed-text:v1.5", Family: "nomic-embed-text", Quant: "F16", DiskSizeMB: 274, VRAMMinMB: 512, RAMMinMB: 1024, Tags: []string{"embed"}},

	// ── Mxbai Embed Large ────────────────────────────────────────────────────
	{Name: "mxbai-embed-large:335m", Family: "mxbai-embed-large", Quant: "F16", DiskSizeMB: 670, VRAMMinMB: 768, RAMMinMB: 1024, Tags: []string{"embed"}},

	// ── All-MiniLM ────────────────────────────────────────────────────────────
	{Name: "all-minilm:22m", Family: "all-minilm", Quant: "F16", DiskSizeMB: 46, VRAMMinMB: 256, RAMMinMB: 512, Tags: []string{"embed", "small"}},
	{Name: "all-minilm:33m", Family: "all-minilm", Quant: "F16", DiskSizeMB: 67, VRAMMinMB: 256, RAMMinMB: 512, Tags: []string{"embed", "small"}},

	// ── Snowflake Arctic Embed ────────────────────────────────────────────────
	{Name: "snowflake-arctic-embed:22m", Family: "snowflake-arctic-embed", Quant: "F16", DiskSizeMB: 46, VRAMMinMB: 256, RAMMinMB: 512, Tags: []string{"embed", "small"}},
	{Name: "snowflake-arctic-embed:110m", Family: "snowflake-arctic-embed", Quant: "F16", DiskSizeMB: 219, VRAMMinMB: 384, RAMMinMB: 512, Tags: []string{"embed", "small"}},
	{Name: "snowflake-arctic-embed:335m", Family: "snowflake-arctic-embed", Quant: "F16", DiskSizeMB: 669, VRAMMinMB: 768, RAMMinMB: 1024, Tags: []string{"embed"}},

	// ── Granite Embedding ─────────────────────────────────────────────────────
	{Name: "granite-embedding:30m", Family: "granite-embedding", Quant: "F16", DiskSizeMB: 63, VRAMMinMB: 256, RAMMinMB: 512, Tags: []string{"embed", "small"}},
	{Name: "granite-embedding:278m", Family: "granite-embedding", Quant: "F16", DiskSizeMB: 563, VRAMMinMB: 640, RAMMinMB: 1024, Tags: []string{"embed", "multilingual"}},

	// ── BGE-M3 ────────────────────────────────────────────────────────────────
	{Name: "bge-m3:567m", Family: "bge-m3", Quant: "F16", DiskSizeMB: 1140, VRAMMinMB: 1280, RAMMinMB: 2048, Tags: []string{"embed", "multilingual"}},

	// ── Granite 3 Dense ───────────────────────────────────────────────────────
	{Name: "granite3-dense:2b", Family: "granite3-dense", Quant: "Q4_K_M", DiskSizeMB: 1638, VRAMMinMB: 2560, RAMMinMB: 4096, Tags: []string{"small"}},
	{Name: "granite3-dense:8b", Family: "granite3-dense", Quant: "Q4_K_M", DiskSizeMB: 5018, VRAMMinMB: 6144, RAMMinMB: 10240},

	// ── Granite 3.1 Dense ─────────────────────────────────────────────────────
	{Name: "granite3.1-dense:2b", Family: "granite3.1-dense", Quant: "Q4_K_M", DiskSizeMB: 1638, VRAMMinMB: 2560, RAMMinMB: 4096, Tags: []string{"small"}},
	{Name: "granite3.1-dense:8b", Family: "granite3.1-dense", Quant: "Q4_K_M", DiskSizeMB: 5120, VRAMMinMB: 6144, RAMMinMB: 10240},

	// ── Granite 3.1 MoE ───────────────────────────────────────────────────────
	{Name: "granite3.1-moe:1b", Family: "granite3.1-moe", Quant: "Q4_K_M", DiskSizeMB: 1434, VRAMMinMB: 2048, RAMMinMB: 3072, Tags: []string{"small"}},
	{Name: "granite3.1-moe:3b", Family: "granite3.1-moe", Quant: "Q4_K_M", DiskSizeMB: 2048, VRAMMinMB: 3072, RAMMinMB: 4096, Tags: []string{"small"}},

	// ── Granite 3.2 ───────────────────────────────────────────────────────────
	{Name: "granite3.2:2b", Family: "granite3.2", Quant: "Q4_K_M", DiskSizeMB: 1536, VRAMMinMB: 2560, RAMMinMB: 4096, Tags: []string{"small", "reasoning"}},
	{Name: "granite3.2:8b", Family: "granite3.2", Quant: "Q4_K_M", DiskSizeMB: 5018, VRAMMinMB: 6144, RAMMinMB: 10240, Tags: []string{"reasoning"}},

	// ── Command R ─────────────────────────────────────────────────────────────
	{Name: "command-r:35b", Family: "command-r", Quant: "Q4_K_M", DiskSizeMB: 19456, VRAMMinMB: 22528, RAMMinMB: 40960, Tags: []string{"multilingual"}},

	// ── Command R+ ────────────────────────────────────────────────────────────
	{Name: "command-r-plus:104b", Family: "command-r-plus", Quant: "Q4_K_M", DiskSizeMB: 60416, VRAMMinMB: 65536, RAMMinMB: 128000, Tags: []string{"multilingual"}},

	// ── Aya ───────────────────────────────────────────────────────────────────
	{Name: "aya:8b", Family: "aya", Quant: "Q4_K_M", DiskSizeMB: 4915, VRAMMinMB: 6144, RAMMinMB: 10240, Tags: []string{"multilingual"}},
	{Name: "aya:35b", Family: "aya", Quant: "Q4_K_M", DiskSizeMB: 20480, VRAMMinMB: 22528, RAMMinMB: 40960, Tags: []string{"multilingual"}},

	// ── Solar ─────────────────────────────────────────────────────────────────
	{Name: "solar:10.7b", Family: "solar", Quant: "Q4_K_M", DiskSizeMB: 6246, VRAMMinMB: 7168, RAMMinMB: 12288},

	// ── SmolLM ────────────────────────────────────────────────────────────────
	{Name: "smollm:135m", Family: "smollm", Quant: "Q4_K_M", DiskSizeMB: 92, VRAMMinMB: 512, RAMMinMB: 1024, Tags: []string{"small"}},
	{Name: "smollm:360m", Family: "smollm", Quant: "Q4_K_M", DiskSizeMB: 229, VRAMMinMB: 512, RAMMinMB: 1024, Tags: []string{"small"}},
	{Name: "smollm:1.7b", Family: "smollm", Quant: "Q4_K_M", DiskSizeMB: 991, VRAMMinMB: 1536, RAMMinMB: 2048, Tags: []string{"small"}},

	// ── SmolLM 2 ──────────────────────────────────────────────────────────────
	{Name: "smollm2:135m", Family: "smollm2", Quant: "Q4_K_M", DiskSizeMB: 271, VRAMMinMB: 512, RAMMinMB: 1024, Tags: []string{"small"}},
	{Name: "smollm2:360m", Family: "smollm2", Quant: "Q4_K_M", DiskSizeMB: 726, VRAMMinMB: 768, RAMMinMB: 1024, Tags: []string{"small"}},
	{Name: "smollm2:1.7b", Family: "smollm2", Quant: "Q4_K_M", DiskSizeMB: 1843, VRAMMinMB: 2048, RAMMinMB: 3072, Tags: []string{"small"}},

	// ── TinyLlama ─────────────────────────────────────────────────────────────
	{Name: "tinyllama:1.1b", Family: "tinyllama", Quant: "Q4_K_M", DiskSizeMB: 638, VRAMMinMB: 1024, RAMMinMB: 2048, Tags: []string{"small"}},

	// ── StableLM 2 ────────────────────────────────────────────────────────────
	{Name: "stablelm2:1.6b", Family: "stablelm2", Quant: "Q4_K_M", DiskSizeMB: 983, VRAMMinMB: 1536, RAMMinMB: 2048, Tags: []string{"small", "multilingual"}},
	{Name: "stablelm2:12b", Family: "stablelm2", Quant: "Q4_K_M", DiskSizeMB: 7168, VRAMMinMB: 8192, RAMMinMB: 16384, Tags: []string{"multilingual"}},

	// ── Zephyr ────────────────────────────────────────────────────────────────
	{Name: "zephyr:7b", Family: "zephyr", Quant: "Q4_K_M", DiskSizeMB: 4198, VRAMMinMB: 5632, RAMMinMB: 8192},
	{Name: "zephyr:141b", Family: "zephyr", Quant: "Q4_K_M", DiskSizeMB: 81920, VRAMMinMB: 86016, RAMMinMB: 128000},

	// ── OpenChat ──────────────────────────────────────────────────────────────
	{Name: "openchat:7b", Family: "openchat", Quant: "Q4_K_M", DiskSizeMB: 4198, VRAMMinMB: 5632, RAMMinMB: 8192},

	// ── Neural Chat ───────────────────────────────────────────────────────────
	{Name: "neural-chat:7b", Family: "neural-chat", Quant: "Q4_K_M", DiskSizeMB: 4198, VRAMMinMB: 5632, RAMMinMB: 8192},

	// ── Dolphin variants ──────────────────────────────────────────────────────
	{Name: "dolphin-mistral:7b", Family: "dolphin-mistral", Quant: "Q4_K_M", DiskSizeMB: 4198, VRAMMinMB: 5632, RAMMinMB: 8192},
	{Name: "dolphin-llama3:8b", Family: "dolphin-llama3", Quant: "Q4_K_M", DiskSizeMB: 4813, VRAMMinMB: 5632, RAMMinMB: 8192},
	{Name: "dolphin-llama3:70b", Family: "dolphin-llama3", Quant: "Q4_K_M", DiskSizeMB: 40960, VRAMMinMB: 43008, RAMMinMB: 64000},
	{Name: "dolphin-phi:2.7b", Family: "dolphin-phi", Quant: "Q4_K_M", DiskSizeMB: 1638, VRAMMinMB: 2560, RAMMinMB: 4096, Tags: []string{"small"}},

	// ── Nous Hermes 2 ─────────────────────────────────────────────────────────
	{Name: "nous-hermes2:10.7b", Family: "nous-hermes2", Quant: "Q4_K_M", DiskSizeMB: 6246, VRAMMinMB: 7168, RAMMinMB: 12288},
	{Name: "nous-hermes2:34b", Family: "nous-hermes2", Quant: "Q4_K_M", DiskSizeMB: 19456, VRAMMinMB: 22528, RAMMinMB: 32768},

	// ── OpenHermes ────────────────────────────────────────────────────────────
	{Name: "openhermes:7b", Family: "openhermes", Quant: "Q4_K_M", DiskSizeMB: 4506, VRAMMinMB: 5632, RAMMinMB: 8192},

	// ── Wizard Vicuna Uncensored ──────────────────────────────────────────────
	{Name: "wizard-vicuna-uncensored:7b", Family: "wizard-vicuna-uncensored", Quant: "Q4_K_M", DiskSizeMB: 3891, VRAMMinMB: 5120, RAMMinMB: 8192},
	{Name: "wizard-vicuna-uncensored:13b", Family: "wizard-vicuna-uncensored", Quant: "Q4_K_M", DiskSizeMB: 7578, VRAMMinMB: 9216, RAMMinMB: 16384},
	{Name: "wizard-vicuna-uncensored:30b", Family: "wizard-vicuna-uncensored", Quant: "Q4_K_M", DiskSizeMB: 18432, VRAMMinMB: 20480, RAMMinMB: 32768},

	// ── Yi ────────────────────────────────────────────────────────────────────
	{Name: "yi:6b", Family: "yi", Quant: "Q4_K_M", DiskSizeMB: 3584, VRAMMinMB: 5120, RAMMinMB: 8192},
	{Name: "yi:9b", Family: "yi", Quant: "Q4_K_M", DiskSizeMB: 5120, VRAMMinMB: 6656, RAMMinMB: 10240},
	{Name: "yi:34b", Family: "yi", Quant: "Q4_K_M", DiskSizeMB: 19456, VRAMMinMB: 22528, RAMMinMB: 40960},

	// ── InternLM 2 ────────────────────────────────────────────────────────────
	{Name: "internlm2:1.8b", Family: "internlm2", Quant: "Q4_K_M", DiskSizeMB: 1127, VRAMMinMB: 2048, RAMMinMB: 3072, Tags: []string{"small"}},
	{Name: "internlm2:7b", Family: "internlm2", Quant: "Q4_K_M", DiskSizeMB: 4608, VRAMMinMB: 5632, RAMMinMB: 8192},
	{Name: "internlm2:20b", Family: "internlm2", Quant: "Q4_K_M", DiskSizeMB: 11264, VRAMMinMB: 12288, RAMMinMB: 24576},

	// ── Nemotron ──────────────────────────────────────────────────────────────
	{Name: "nemotron-mini:4b", Family: "nemotron-mini", Quant: "Q4_K_M", DiskSizeMB: 2765, VRAMMinMB: 3584, RAMMinMB: 6144, Tags: []string{"small"}},
	{Name: "nemotron:70b", Family: "nemotron", Quant: "Q4_K_M", DiskSizeMB: 44032, VRAMMinMB: 46080, RAMMinMB: 80000},

	// ── Llama Guard 3 ─────────────────────────────────────────────────────────
	{Name: "llama-guard3:1b", Family: "llama-guard3", Quant: "Q4_K_M", DiskSizeMB: 1638, VRAMMinMB: 2048, RAMMinMB: 3072, Tags: []string{"small"}},
	{Name: "llama-guard3:8b", Family: "llama-guard3", Quant: "Q4_K_M", DiskSizeMB: 5018, VRAMMinMB: 5632, RAMMinMB: 8192},

	// ── Falcon ────────────────────────────────────────────────────────────────
	{Name: "falcon:7b", Family: "falcon", Quant: "Q4_K_M", DiskSizeMB: 4301, VRAMMinMB: 5632, RAMMinMB: 8192},
	{Name: "falcon:40b", Family: "falcon", Quant: "Q4_K_M", DiskSizeMB: 24576, VRAMMinMB: 26624, RAMMinMB: 48000},

	// ── Falcon 3 ──────────────────────────────────────────────────────────────
	{Name: "falcon3:1b", Family: "falcon3", Quant: "Q4_K_M", DiskSizeMB: 1843, VRAMMinMB: 2048, RAMMinMB: 3072, Tags: []string{"small"}},
	{Name: "falcon3:3b", Family: "falcon3", Quant: "Q4_K_M", DiskSizeMB: 2048, VRAMMinMB: 2560, RAMMinMB: 4096, Tags: []string{"small"}},
	{Name: "falcon3:7b", Family: "falcon3", Quant: "Q4_K_M", DiskSizeMB: 4710, VRAMMinMB: 5632, RAMMinMB: 8192},
	{Name: "falcon3:10b", Family: "falcon3", Quant: "Q4_K_M", DiskSizeMB: 6451, VRAMMinMB: 7168, RAMMinMB: 12288},

	// ── QwQ ───────────────────────────────────────────────────────────────────
	{Name: "qwq:32b", Family: "qwq", Quant: "Q4_K_M", DiskSizeMB: 20480, VRAMMinMB: 22528, RAMMinMB: 32768, Tags: []string{"reasoning"}},

	// ── OLMo 2 ────────────────────────────────────────────────────────────────
	{Name: "olmo2:7b", Family: "olmo2", Quant: "Q4_K_M", DiskSizeMB: 4608, VRAMMinMB: 5632, RAMMinMB: 8192},
	{Name: "olmo2:13b", Family: "olmo2", Quant: "Q4_K_M", DiskSizeMB: 8602, VRAMMinMB: 9728, RAMMinMB: 16384},
	{Name: "nomic-embed-text", Tags: []string{"embedding"}},
	{Name: "gemma3:270m", Quant: "Q4_K_M", DiskSizeMB: 540, VRAMMinMB: 1052, RAMMinMB: 3100, Tags: []string{"vision", "cloud"}},
	{Name: "qwen3:30b", Quant: "Q4_K_M", DiskSizeMB: 18000, VRAMMinMB: 18512, RAMMinMB: 20560, Tags: []string{"tools", "thinking"}},
	{Name: "qwen3:235b", Quant: "Q4_K_M", DiskSizeMB: 141000, VRAMMinMB: 141512, RAMMinMB: 143560, Tags: []string{"tools", "thinking"}},
	{Name: "gpt-oss:20b", Quant: "Q4_K_M", DiskSizeMB: 12000, VRAMMinMB: 12512, RAMMinMB: 14560, Tags: []string{"tools", "thinking", "cloud"}},
	{Name: "gpt-oss:120b", Quant: "Q4_K_M", DiskSizeMB: 72000, VRAMMinMB: 72512, RAMMinMB: 74560, Tags: []string{"tools", "thinking", "cloud"}},
	{Name: "qwen:0.5b", Quant: "Q4_K_M", DiskSizeMB: 1000, VRAMMinMB: 1512, RAMMinMB: 3560, Tags: nil},
	{Name: "qwen:1.8b", Quant: "Q4_K_M", DiskSizeMB: 1080, VRAMMinMB: 1592, RAMMinMB: 3640, Tags: nil},
	{Name: "qwen:4b", Quant: "Q4_K_M", DiskSizeMB: 2400, VRAMMinMB: 2912, RAMMinMB: 4960, Tags: nil},
	{Name: "qwen:7b", Quant: "Q4_K_M", DiskSizeMB: 4200, VRAMMinMB: 4712, RAMMinMB: 6760, Tags: nil},
	{Name: "qwen:14b", Quant: "Q4_K_M", DiskSizeMB: 8400, VRAMMinMB: 8912, RAMMinMB: 10960, Tags: nil},
	{Name: "qwen:32b", Quant: "Q4_K_M", DiskSizeMB: 19200, VRAMMinMB: 19712, RAMMinMB: 21760, Tags: nil},
	{Name: "qwen:72b", Quant: "Q4_K_M", DiskSizeMB: 43200, VRAMMinMB: 43712, RAMMinMB: 45760, Tags: nil},
	{Name: "qwen:110b", Quant: "Q4_K_M", DiskSizeMB: 66000, VRAMMinMB: 66512, RAMMinMB: 68560, Tags: nil},
	{Name: "minicpm-v:8b", Quant: "Q4_K_M", DiskSizeMB: 4800, VRAMMinMB: 5312, RAMMinMB: 7360, Tags: []string{"vision"}},
	{Name: "dolphin3:8b", Quant: "Q4_K_M", DiskSizeMB: 4800, VRAMMinMB: 5312, RAMMinMB: 7360, Tags: nil},
	{Name: "qwen3-coder:30b", Quant: "Q4_K_M", DiskSizeMB: 18000, VRAMMinMB: 18512, RAMMinMB: 20560, Tags: []string{"tools", "cloud"}},
	{Name: "qwen3-coder:480b", Quant: "Q4_K_M", DiskSizeMB: 288000, VRAMMinMB: 288512, RAMMinMB: 290560, Tags: []string{"tools", "cloud"}},
	{Name: "snowflake-arctic-embed:33m", Quant: "Q4_K_M", DiskSizeMB: 66, VRAMMinMB: 578, RAMMinMB: 2626, Tags: []string{"embedding"}},
	{Name: "snowflake-arctic-embed:137m", Quant: "Q4_K_M", DiskSizeMB: 274, VRAMMinMB: 786, RAMMinMB: 2834, Tags: []string{"embedding"}},
	{Name: "orca-mini:3b", Quant: "Q4_K_M", DiskSizeMB: 1800, VRAMMinMB: 2312, RAMMinMB: 4360, Tags: nil},
	{Name: "orca-mini:7b", Quant: "Q4_K_M", DiskSizeMB: 4200, VRAMMinMB: 4712, RAMMinMB: 6760, Tags: nil},
	{Name: "orca-mini:13b", Quant: "Q4_K_M", DiskSizeMB: 7800, VRAMMinMB: 8312, RAMMinMB: 10360, Tags: nil},
	{Name: "orca-mini:70b", Quant: "Q4_K_M", DiskSizeMB: 42000, VRAMMinMB: 42512, RAMMinMB: 44560, Tags: nil},
	{Name: "qwen3-vl:2b", Quant: "Q4_K_M", DiskSizeMB: 1200, VRAMMinMB: 1712, RAMMinMB: 3760, Tags: []string{"vision", "tools", "thinking", "cloud"}},
	{Name: "qwen3-vl:4b", Quant: "Q4_K_M", DiskSizeMB: 2400, VRAMMinMB: 2912, RAMMinMB: 4960, Tags: []string{"vision", "tools", "thinking", "cloud"}},
	{Name: "qwen3-vl:8b", Quant: "Q4_K_M", DiskSizeMB: 4800, VRAMMinMB: 5312, RAMMinMB: 7360, Tags: []string{"vision", "tools", "thinking", "cloud"}},
	{Name: "qwen3-vl:30b", Quant: "Q4_K_M", DiskSizeMB: 18000, VRAMMinMB: 18512, RAMMinMB: 20560, Tags: []string{"vision", "tools", "thinking", "cloud"}},
	{Name: "qwen3-vl:32b", Quant: "Q4_K_M", DiskSizeMB: 19200, VRAMMinMB: 19712, RAMMinMB: 21760, Tags: []string{"vision", "tools", "thinking", "cloud"}},
	{Name: "qwen3-vl:235b", Quant: "Q4_K_M", DiskSizeMB: 141000, VRAMMinMB: 141512, RAMMinMB: 143560, Tags: []string{"vision", "tools", "thinking", "cloud"}},
	{Name: "llama2-uncensored:7b", Quant: "Q4_K_M", DiskSizeMB: 4200, VRAMMinMB: 4712, RAMMinMB: 6760, Tags: nil},
	{Name: "llama2-uncensored:70b", Quant: "Q4_K_M", DiskSizeMB: 42000, VRAMMinMB: 42512, RAMMinMB: 44560, Tags: nil},
	{Name: "cogito:3b", Quant: "Q4_K_M", DiskSizeMB: 1800, VRAMMinMB: 2312, RAMMinMB: 4360, Tags: []string{"tools"}},
	{Name: "cogito:8b", Quant: "Q4_K_M", DiskSizeMB: 4800, VRAMMinMB: 5312, RAMMinMB: 7360, Tags: []string{"tools"}},
	{Name: "cogito:14b", Quant: "Q4_K_M", DiskSizeMB: 8400, VRAMMinMB: 8912, RAMMinMB: 10960, Tags: []string{"tools"}},
	{Name: "cogito:32b", Quant: "Q4_K_M", DiskSizeMB: 19200, VRAMMinMB: 19712, RAMMinMB: 21760, Tags: []string{"tools"}},
	{Name: "cogito:70b", Quant: "Q4_K_M", DiskSizeMB: 42000, VRAMMinMB: 42512, RAMMinMB: 44560, Tags: []string{"tools"}},
	{Name: "qwen2.5vl:3b", Quant: "Q4_K_M", DiskSizeMB: 1800, VRAMMinMB: 2312, RAMMinMB: 4360, Tags: []string{"vision"}},
	{Name: "qwen2.5vl:7b", Quant: "Q4_K_M", DiskSizeMB: 4200, VRAMMinMB: 4712, RAMMinMB: 6760, Tags: []string{"vision"}},
	{Name: "qwen2.5vl:32b", Quant: "Q4_K_M", DiskSizeMB: 19200, VRAMMinMB: 19712, RAMMinMB: 21760, Tags: []string{"vision"}},
	{Name: "qwen2.5vl:72b", Quant: "Q4_K_M", DiskSizeMB: 43200, VRAMMinMB: 43712, RAMMinMB: 45760, Tags: []string{"vision"}},
	{Name: "mistral-small3.2:24b", Quant: "Q4_K_M", DiskSizeMB: 14400, VRAMMinMB: 14912, RAMMinMB: 16960, Tags: []string{"vision", "tools"}},
	{Name: "gemma3n", Tags: nil},
	{Name: "e2b", Tags: nil},
	{Name: "e4b", Tags: nil},
	{Name: "llama4:16x17b", Quant: "Q4_K_M", DiskSizeMB: 163200, VRAMMinMB: 163712, RAMMinMB: 165760, Tags: []string{"vision", "tools"}},
	{Name: "llama4:128x17b", Quant: "Q4_K_M", DiskSizeMB: 1305600, VRAMMinMB: 1306112, RAMMinMB: 1308160, Tags: []string{"vision", "tools"}},
	{Name: "phi4-reasoning:14b", Quant: "Q4_K_M", DiskSizeMB: 8400, VRAMMinMB: 8912, RAMMinMB: 10960, Tags: nil},
	{Name: "qwen3.5:0.8b", Quant: "Q4_K_M", DiskSizeMB: 1600, VRAMMinMB: 2112, RAMMinMB: 4160, Tags: []string{"vision", "tools", "thinking", "cloud"}},
	{Name: "qwen3.5:2b", Quant: "Q4_K_M", DiskSizeMB: 1200, VRAMMinMB: 1712, RAMMinMB: 3760, Tags: []string{"vision", "tools", "thinking", "cloud"}},
	{Name: "qwen3.5:4b", Quant: "Q4_K_M", DiskSizeMB: 2400, VRAMMinMB: 2912, RAMMinMB: 4960, Tags: []string{"vision", "tools", "thinking", "cloud"}},
	{Name: "qwen3.5:9b", Quant: "Q4_K_M", DiskSizeMB: 5400, VRAMMinMB: 5912, RAMMinMB: 7960, Tags: []string{"vision", "tools", "thinking", "cloud"}},
	{Name: "qwen3.5:27b", Quant: "Q4_K_M", DiskSizeMB: 16200, VRAMMinMB: 16712, RAMMinMB: 18760, Tags: []string{"vision", "tools", "thinking", "cloud"}},
	{Name: "qwen3.5:35b", Quant: "Q4_K_M", DiskSizeMB: 21000, VRAMMinMB: 21512, RAMMinMB: 23560, Tags: []string{"vision", "tools", "thinking", "cloud"}},
	{Name: "qwen3.5:122b", Quant: "Q4_K_M", DiskSizeMB: 73200, VRAMMinMB: 73712, RAMMinMB: 75760, Tags: []string{"vision", "tools", "thinking", "cloud"}},
	{Name: "magistral:24b", Quant: "Q4_K_M", DiskSizeMB: 14400, VRAMMinMB: 14912, RAMMinMB: 16960, Tags: []string{"tools", "thinking"}},
	{Name: "qwen3-embedding:0.6b", Quant: "Q4_K_M", DiskSizeMB: 1200, VRAMMinMB: 1712, RAMMinMB: 3760, Tags: []string{"embedding"}},
	{Name: "qwen3-embedding:4b", Quant: "Q4_K_M", DiskSizeMB: 2400, VRAMMinMB: 2912, RAMMinMB: 4960, Tags: []string{"embedding"}},
	{Name: "qwen3-embedding:8b", Quant: "Q4_K_M", DiskSizeMB: 4800, VRAMMinMB: 5312, RAMMinMB: 7360, Tags: []string{"embedding"}},
	{Name: "deepscaler:1.5b", Quant: "Q4_K_M", DiskSizeMB: 900, VRAMMinMB: 1412, RAMMinMB: 3460, Tags: nil},
	{Name: "dolphin-mixtral:8x7b", Quant: "Q4_K_M", DiskSizeMB: 33600, VRAMMinMB: 34112, RAMMinMB: 36160, Tags: nil},
	{Name: "dolphin-mixtral:8x22b", Quant: "Q4_K_M", DiskSizeMB: 105600, VRAMMinMB: 106112, RAMMinMB: 108160, Tags: nil},
	{Name: "phi:2.7b", Quant: "Q4_K_M", DiskSizeMB: 1620, VRAMMinMB: 2132, RAMMinMB: 4180, Tags: nil},
	{Name: "lfm2.5-thinking:1.2b", Quant: "Q4_K_M", DiskSizeMB: 720, VRAMMinMB: 1232, RAMMinMB: 3280, Tags: []string{"tools"}},
	{Name: "lfm2:24b", Quant: "Q4_K_M", DiskSizeMB: 14400, VRAMMinMB: 14912, RAMMinMB: 16960, Tags: []string{"tools"}},
	{Name: "granite3.3:2b", Quant: "Q4_K_M", DiskSizeMB: 1200, VRAMMinMB: 1712, RAMMinMB: 3760, Tags: []string{"tools"}},
	{Name: "granite3.3:8b", Quant: "Q4_K_M", DiskSizeMB: 4800, VRAMMinMB: 5312, RAMMinMB: 7360, Tags: []string{"tools"}},
	{Name: "openthinker:7b", Quant: "Q4_K_M", DiskSizeMB: 4200, VRAMMinMB: 4712, RAMMinMB: 6760, Tags: nil},
	{Name: "openthinker:32b", Quant: "Q4_K_M", DiskSizeMB: 19200, VRAMMinMB: 19712, RAMMinMB: 21760, Tags: nil},
	{Name: "granite4:350m", Quant: "Q4_K_M", DiskSizeMB: 700, VRAMMinMB: 1212, RAMMinMB: 3260, Tags: []string{"tools"}},
	{Name: "granite4:1b", Quant: "Q4_K_M", DiskSizeMB: 600, VRAMMinMB: 1112, RAMMinMB: 3160, Tags: []string{"tools"}},
	{Name: "granite4:3b", Quant: "Q4_K_M", DiskSizeMB: 1800, VRAMMinMB: 2312, RAMMinMB: 4360, Tags: []string{"tools"}},
	{Name: "qwen3-coder-next", Tags: []string{"tools", "cloud"}},
	{Name: "wizardlm2:7b", Quant: "Q4_K_M", DiskSizeMB: 4200, VRAMMinMB: 4712, RAMMinMB: 6760, Tags: nil},
	{Name: "wizardlm2:8x22b", Quant: "Q4_K_M", DiskSizeMB: 105600, VRAMMinMB: 106112, RAMMinMB: 108160, Tags: nil},
	{Name: "hermes3:3b", Quant: "Q4_K_M", DiskSizeMB: 1800, VRAMMinMB: 2312, RAMMinMB: 4360, Tags: []string{"tools"}},
	{Name: "hermes3:8b", Quant: "Q4_K_M", DiskSizeMB: 4800, VRAMMinMB: 5312, RAMMinMB: 7360, Tags: []string{"tools"}},
	{Name: "hermes3:70b", Quant: "Q4_K_M", DiskSizeMB: 42000, VRAMMinMB: 42512, RAMMinMB: 44560, Tags: []string{"tools"}},
	{Name: "hermes3:405b", Quant: "Q4_K_M", DiskSizeMB: 243000, VRAMMinMB: 243512, RAMMinMB: 245560, Tags: []string{"tools"}},
	{Name: "deepcoder:1.5b", Quant: "Q4_K_M", DiskSizeMB: 900, VRAMMinMB: 1412, RAMMinMB: 3460, Tags: nil},
	{Name: "deepcoder:14b", Quant: "Q4_K_M", DiskSizeMB: 8400, VRAMMinMB: 8912, RAMMinMB: 10960, Tags: nil},
	{Name: "mistral-small3.1:24b", Quant: "Q4_K_M", DiskSizeMB: 14400, VRAMMinMB: 14912, RAMMinMB: 16960, Tags: []string{"vision", "tools"}},
	{Name: "mistral-large:123b", Quant: "Q4_K_M", DiskSizeMB: 73800, VRAMMinMB: 74312, RAMMinMB: 76360, Tags: []string{"tools"}},
	{Name: "embeddinggemma:300m", Quant: "Q4_K_M", DiskSizeMB: 600, VRAMMinMB: 1112, RAMMinMB: 3160, Tags: []string{"embedding"}},
	{Name: "paraphrase-multilingual:278m", Quant: "Q4_K_M", DiskSizeMB: 556, VRAMMinMB: 1068, RAMMinMB: 3116, Tags: []string{"embedding"}},
	{Name: "ministral-3:3b", Quant: "Q4_K_M", DiskSizeMB: 1800, VRAMMinMB: 2312, RAMMinMB: 4360, Tags: []string{"vision", "tools", "cloud"}},
	{Name: "ministral-3:8b", Quant: "Q4_K_M", DiskSizeMB: 4800, VRAMMinMB: 5312, RAMMinMB: 7360, Tags: []string{"vision", "tools", "cloud"}},
	{Name: "ministral-3:14b", Quant: "Q4_K_M", DiskSizeMB: 8400, VRAMMinMB: 8912, RAMMinMB: 10960, Tags: []string{"vision", "tools", "cloud"}},
	{Name: "starcoder:1b", Quant: "Q4_K_M", DiskSizeMB: 600, VRAMMinMB: 1112, RAMMinMB: 3160, Tags: nil},
	{Name: "starcoder:3b", Quant: "Q4_K_M", DiskSizeMB: 1800, VRAMMinMB: 2312, RAMMinMB: 4360, Tags: nil},
	{Name: "starcoder:7b", Quant: "Q4_K_M", DiskSizeMB: 4200, VRAMMinMB: 4712, RAMMinMB: 6760, Tags: nil},
	{Name: "starcoder:15b", Quant: "Q4_K_M", DiskSizeMB: 9000, VRAMMinMB: 9512, RAMMinMB: 11560, Tags: nil},
	{Name: "translategemma:4b", Quant: "Q4_K_M", DiskSizeMB: 2400, VRAMMinMB: 2912, RAMMinMB: 4960, Tags: []string{"vision"}},
	{Name: "translategemma:12b", Quant: "Q4_K_M", DiskSizeMB: 7200, VRAMMinMB: 7712, RAMMinMB: 9760, Tags: []string{"vision"}},
	{Name: "translategemma:27b", Quant: "Q4_K_M", DiskSizeMB: 16200, VRAMMinMB: 16712, RAMMinMB: 18760, Tags: []string{"vision"}},
	{Name: "nous-hermes:7b", Quant: "Q4_K_M", DiskSizeMB: 4200, VRAMMinMB: 4712, RAMMinMB: 6760, Tags: nil},
	{Name: "nous-hermes:13b", Quant: "Q4_K_M", DiskSizeMB: 7800, VRAMMinMB: 8312, RAMMinMB: 10360, Tags: nil},
	{Name: "deepseek-llm:7b", Quant: "Q4_K_M", DiskSizeMB: 4200, VRAMMinMB: 4712, RAMMinMB: 6760, Tags: nil},
	{Name: "deepseek-llm:67b", Quant: "Q4_K_M", DiskSizeMB: 40200, VRAMMinMB: 40712, RAMMinMB: 42760, Tags: nil},
	{Name: "deepseek-v2:16b", Quant: "Q4_K_M", DiskSizeMB: 9600, VRAMMinMB: 10112, RAMMinMB: 12160, Tags: nil},
	{Name: "deepseek-v2:236b", Quant: "Q4_K_M", DiskSizeMB: 141600, VRAMMinMB: 142112, RAMMinMB: 144160, Tags: nil},
	{Name: "falcon:180b", Quant: "Q4_K_M", DiskSizeMB: 108000, VRAMMinMB: 108512, RAMMinMB: 110560, Tags: nil},
	{Name: "vicuna:7b", Quant: "Q4_K_M", DiskSizeMB: 4200, VRAMMinMB: 4712, RAMMinMB: 6760, Tags: nil},
	{Name: "vicuna:13b", Quant: "Q4_K_M", DiskSizeMB: 7800, VRAMMinMB: 8312, RAMMinMB: 10360, Tags: nil},
	{Name: "vicuna:33b", Quant: "Q4_K_M", DiskSizeMB: 19800, VRAMMinMB: 20312, RAMMinMB: 22360, Tags: nil},
	{Name: "glm4:9b", Quant: "Q4_K_M", DiskSizeMB: 5400, VRAMMinMB: 5912, RAMMinMB: 7960, Tags: nil},
	{Name: "exaone-deep:2.4b", Quant: "Q4_K_M", DiskSizeMB: 1440, VRAMMinMB: 1952, RAMMinMB: 4000, Tags: nil},
	{Name: "exaone-deep:7.8b", Quant: "Q4_K_M", DiskSizeMB: 4680, VRAMMinMB: 5192, RAMMinMB: 7240, Tags: nil},
	{Name: "exaone-deep:32b", Quant: "Q4_K_M", DiskSizeMB: 19200, VRAMMinMB: 19712, RAMMinMB: 21760, Tags: nil},
	{Name: "codeqwen:7b", Quant: "Q4_K_M", DiskSizeMB: 4200, VRAMMinMB: 4712, RAMMinMB: 6760, Tags: nil},
	{Name: "openhermes", Tags: nil},
	{Name: "qwen2-math:1.5b", Quant: "Q4_K_M", DiskSizeMB: 900, VRAMMinMB: 1412, RAMMinMB: 3460, Tags: nil},
	{Name: "qwen2-math:7b", Quant: "Q4_K_M", DiskSizeMB: 4200, VRAMMinMB: 4712, RAMMinMB: 6760, Tags: nil},
	{Name: "qwen2-math:72b", Quant: "Q4_K_M", DiskSizeMB: 43200, VRAMMinMB: 43712, RAMMinMB: 45760, Tags: nil},
	{Name: "llama2-chinese:7b", Quant: "Q4_K_M", DiskSizeMB: 4200, VRAMMinMB: 4712, RAMMinMB: 6760, Tags: nil},
	{Name: "llama2-chinese:13b", Quant: "Q4_K_M", DiskSizeMB: 7800, VRAMMinMB: 8312, RAMMinMB: 10360, Tags: nil},
	{Name: "stable-code:3b", Quant: "Q4_K_M", DiskSizeMB: 1800, VRAMMinMB: 2312, RAMMinMB: 4360, Tags: nil},
	{Name: "sqlcoder:7b", Quant: "Q4_K_M", DiskSizeMB: 4200, VRAMMinMB: 4712, RAMMinMB: 6760, Tags: nil},
	{Name: "sqlcoder:15b", Quant: "Q4_K_M", DiskSizeMB: 9000, VRAMMinMB: 9512, RAMMinMB: 11560, Tags: nil},
	{Name: "wizardcoder:33b", Quant: "Q4_K_M", DiskSizeMB: 19800, VRAMMinMB: 20312, RAMMinMB: 22360, Tags: nil},
	{Name: "yi-coder:1.5b", Quant: "Q4_K_M", DiskSizeMB: 900, VRAMMinMB: 1412, RAMMinMB: 3460, Tags: nil},
	{Name: "yi-coder:9b", Quant: "Q4_K_M", DiskSizeMB: 5400, VRAMMinMB: 5912, RAMMinMB: 7960, Tags: nil},
	{Name: "llama3-chatqa:8b", Quant: "Q4_K_M", DiskSizeMB: 4800, VRAMMinMB: 5312, RAMMinMB: 7360, Tags: nil},
	{Name: "llama3-chatqa:70b", Quant: "Q4_K_M", DiskSizeMB: 42000, VRAMMinMB: 42512, RAMMinMB: 44560, Tags: nil},
	{Name: "dolphincoder:7b", Quant: "Q4_K_M", DiskSizeMB: 4200, VRAMMinMB: 4712, RAMMinMB: 6760, Tags: nil},
	{Name: "dolphincoder:15b", Quant: "Q4_K_M", DiskSizeMB: 9000, VRAMMinMB: 9512, RAMMinMB: 11560, Tags: nil},
	{Name: "wizard-math:7b", Quant: "Q4_K_M", DiskSizeMB: 4200, VRAMMinMB: 4712, RAMMinMB: 6760, Tags: nil},
	{Name: "wizard-math:13b", Quant: "Q4_K_M", DiskSizeMB: 7800, VRAMMinMB: 8312, RAMMinMB: 10360, Tags: nil},
	{Name: "wizard-math:70b", Quant: "Q4_K_M", DiskSizeMB: 42000, VRAMMinMB: 42512, RAMMinMB: 44560, Tags: nil},
	{Name: "llama3-gradient:8b", Quant: "Q4_K_M", DiskSizeMB: 4800, VRAMMinMB: 5312, RAMMinMB: 7360, Tags: nil},
	{Name: "llama3-gradient:70b", Quant: "Q4_K_M", DiskSizeMB: 42000, VRAMMinMB: 42512, RAMMinMB: 44560, Tags: nil},
	{Name: "devstral-small-2:24b", Quant: "Q4_K_M", DiskSizeMB: 14400, VRAMMinMB: 14912, RAMMinMB: 16960, Tags: []string{"vision", "tools", "cloud"}},
	{Name: "samantha-mistral:7b", Quant: "Q4_K_M", DiskSizeMB: 4200, VRAMMinMB: 4712, RAMMinMB: 6760, Tags: nil},
	{Name: "opencoder:1.5b", Quant: "Q4_K_M", DiskSizeMB: 900, VRAMMinMB: 1412, RAMMinMB: 3460, Tags: nil},
	{Name: "opencoder:8b", Quant: "Q4_K_M", DiskSizeMB: 4800, VRAMMinMB: 5312, RAMMinMB: 7360, Tags: nil},
	{Name: "internlm2:1m", Quant: "Q4_K_M", DiskSizeMB: 2, VRAMMinMB: 514, RAMMinMB: 2562, Tags: nil},
	{Name: "llama3-groq-tool-use:8b", Quant: "Q4_K_M", DiskSizeMB: 4800, VRAMMinMB: 5312, RAMMinMB: 7360, Tags: []string{"tools"}},
	{Name: "llama3-groq-tool-use:70b", Quant: "Q4_K_M", DiskSizeMB: 42000, VRAMMinMB: 42512, RAMMinMB: 44560, Tags: []string{"tools"}},
	{Name: "starling-lm:7b", Quant: "Q4_K_M", DiskSizeMB: 4200, VRAMMinMB: 4712, RAMMinMB: 6760, Tags: nil},
	{Name: "phind-codellama:34b", Quant: "Q4_K_M", DiskSizeMB: 20400, VRAMMinMB: 20912, RAMMinMB: 22960, Tags: nil},
	{Name: "xwinlm:7b", Quant: "Q4_K_M", DiskSizeMB: 4200, VRAMMinMB: 4712, RAMMinMB: 6760, Tags: nil},
	{Name: "xwinlm:13b", Quant: "Q4_K_M", DiskSizeMB: 7800, VRAMMinMB: 8312, RAMMinMB: 10360, Tags: nil},
	{Name: "deepseek-v3.1:671b", Quant: "Q4_K_M", DiskSizeMB: 402600, VRAMMinMB: 403112, RAMMinMB: 405160, Tags: []string{"tools", "thinking", "cloud"}},
	{Name: "aya-expanse:8b", Quant: "Q4_K_M", DiskSizeMB: 4800, VRAMMinMB: 5312, RAMMinMB: 7360, Tags: []string{"tools"}},
	{Name: "aya-expanse:32b", Quant: "Q4_K_M", DiskSizeMB: 19200, VRAMMinMB: 19712, RAMMinMB: 21760, Tags: []string{"tools"}},
	{Name: "granite3-moe:1b", Quant: "Q4_K_M", DiskSizeMB: 600, VRAMMinMB: 1112, RAMMinMB: 3160, Tags: []string{"tools"}},
	{Name: "granite3-moe:3b", Quant: "Q4_K_M", DiskSizeMB: 1800, VRAMMinMB: 2312, RAMMinMB: 4360, Tags: []string{"tools"}},
	{Name: "yarn-llama2:7b", Quant: "Q4_K_M", DiskSizeMB: 4200, VRAMMinMB: 4712, RAMMinMB: 6760, Tags: nil},
	{Name: "yarn-llama2:13b", Quant: "Q4_K_M", DiskSizeMB: 7800, VRAMMinMB: 8312, RAMMinMB: 10360, Tags: nil},
	{Name: "glm-4.7-flash", Tags: []string{"tools", "thinking"}},
	{Name: "orca2:7b", Quant: "Q4_K_M", DiskSizeMB: 4200, VRAMMinMB: 4712, RAMMinMB: 6760, Tags: nil},
	{Name: "orca2:13b", Quant: "Q4_K_M", DiskSizeMB: 7800, VRAMMinMB: 8312, RAMMinMB: 10360, Tags: nil},
	{Name: "stable-beluga:7b", Quant: "Q4_K_M", DiskSizeMB: 4200, VRAMMinMB: 4712, RAMMinMB: 6760, Tags: nil},
	{Name: "stable-beluga:13b", Quant: "Q4_K_M", DiskSizeMB: 7800, VRAMMinMB: 8312, RAMMinMB: 10360, Tags: nil},
	{Name: "stable-beluga:70b", Quant: "Q4_K_M", DiskSizeMB: 42000, VRAMMinMB: 42512, RAMMinMB: 44560, Tags: nil},
	{Name: "reader-lm:0.5b", Quant: "Q4_K_M", DiskSizeMB: 1000, VRAMMinMB: 1512, RAMMinMB: 3560, Tags: nil},
	{Name: "reader-lm:1.5b", Quant: "Q4_K_M", DiskSizeMB: 900, VRAMMinMB: 1412, RAMMinMB: 3460, Tags: nil},
	{Name: "shieldgemma:2b", Quant: "Q4_K_M", DiskSizeMB: 1200, VRAMMinMB: 1712, RAMMinMB: 3760, Tags: nil},
	{Name: "shieldgemma:9b", Quant: "Q4_K_M", DiskSizeMB: 5400, VRAMMinMB: 5912, RAMMinMB: 7960, Tags: nil},
	{Name: "shieldgemma:27b", Quant: "Q4_K_M", DiskSizeMB: 16200, VRAMMinMB: 16712, RAMMinMB: 18760, Tags: nil},
	{Name: "tinydolphin:1.1b", Quant: "Q4_K_M", DiskSizeMB: 660, VRAMMinMB: 1172, RAMMinMB: 3220, Tags: nil},
	{Name: "codegeex4:9b", Quant: "Q4_K_M", DiskSizeMB: 5400, VRAMMinMB: 5912, RAMMinMB: 7960, Tags: nil},
	{Name: "llama-pro", Tags: nil},
	{Name: "mistral-openorca:7b", Quant: "Q4_K_M", DiskSizeMB: 4200, VRAMMinMB: 4712, RAMMinMB: 6760, Tags: nil},
	{Name: "yarn-mistral:7b", Quant: "Q4_K_M", DiskSizeMB: 4200, VRAMMinMB: 4712, RAMMinMB: 6760, Tags: nil},
	{Name: "nexusraven:13b", Quant: "Q4_K_M", DiskSizeMB: 7800, VRAMMinMB: 8312, RAMMinMB: 10360, Tags: nil},
	{Name: "wizardlm", Tags: nil},
	{Name: "qwen3-next:80b", Quant: "Q4_K_M", DiskSizeMB: 48000, VRAMMinMB: 48512, RAMMinMB: 50560, Tags: []string{"tools", "thinking", "cloud"}},
	{Name: "rnj-1:8b", Quant: "Q4_K_M", DiskSizeMB: 4800, VRAMMinMB: 5312, RAMMinMB: 7360, Tags: []string{"tools", "cloud"}},
	{Name: "meditron:7b", Quant: "Q4_K_M", DiskSizeMB: 4200, VRAMMinMB: 4712, RAMMinMB: 6760, Tags: nil},
	{Name: "meditron:70b", Quant: "Q4_K_M", DiskSizeMB: 42000, VRAMMinMB: 42512, RAMMinMB: 44560, Tags: nil},
	{Name: "deepseek-ocr:3b", Quant: "Q4_K_M", DiskSizeMB: 1800, VRAMMinMB: 2312, RAMMinMB: 4360, Tags: []string{"vision"}},
	{Name: "reflection:70b", Quant: "Q4_K_M", DiskSizeMB: 42000, VRAMMinMB: 42512, RAMMinMB: 44560, Tags: nil},
	{Name: "wizardlm-uncensored:13b", Quant: "Q4_K_M", DiskSizeMB: 7800, VRAMMinMB: 8312, RAMMinMB: 10360, Tags: nil},
	{Name: "athene-v2:72b", Quant: "Q4_K_M", DiskSizeMB: 43200, VRAMMinMB: 43712, RAMMinMB: 45760, Tags: []string{"tools"}},
	{Name: "exaone3.5:2.4b", Quant: "Q4_K_M", DiskSizeMB: 1440, VRAMMinMB: 1952, RAMMinMB: 4000, Tags: nil},
	{Name: "exaone3.5:7.8b", Quant: "Q4_K_M", DiskSizeMB: 4680, VRAMMinMB: 5192, RAMMinMB: 7240, Tags: nil},
	{Name: "exaone3.5:32b", Quant: "Q4_K_M", DiskSizeMB: 19200, VRAMMinMB: 19712, RAMMinMB: 21760, Tags: nil},
	{Name: "nous-hermes2-mixtral:8x7b", Quant: "Q4_K_M", DiskSizeMB: 33600, VRAMMinMB: 34112, RAMMinMB: 36160, Tags: nil},
	{Name: "medllama2:7b", Quant: "Q4_K_M", DiskSizeMB: 4200, VRAMMinMB: 4712, RAMMinMB: 6760, Tags: nil},
	{Name: "snowflake-arctic-embed2:568m", Quant: "Q4_K_M", DiskSizeMB: 1136, VRAMMinMB: 1648, RAMMinMB: 3696, Tags: []string{"embedding"}},
	{Name: "codeup:13b", Quant: "Q4_K_M", DiskSizeMB: 7800, VRAMMinMB: 8312, RAMMinMB: 10360, Tags: nil},
	{Name: "r1-1776:70b", Quant: "Q4_K_M", DiskSizeMB: 42000, VRAMMinMB: 42512, RAMMinMB: 44560, Tags: nil},
	{Name: "r1-1776:671b", Quant: "Q4_K_M", DiskSizeMB: 402600, VRAMMinMB: 403112, RAMMinMB: 405160, Tags: nil},
	{Name: "everythinglm:13b", Quant: "Q4_K_M", DiskSizeMB: 7800, VRAMMinMB: 8312, RAMMinMB: 10360, Tags: nil},
	{Name: "mathstral:7b", Quant: "Q4_K_M", DiskSizeMB: 4200, VRAMMinMB: 4712, RAMMinMB: 6760, Tags: nil},
	{Name: "solar-pro:22b", Quant: "Q4_K_M", DiskSizeMB: 13200, VRAMMinMB: 13712, RAMMinMB: 15760, Tags: nil},
	{Name: "magicoder:7b", Quant: "Q4_K_M", DiskSizeMB: 4200, VRAMMinMB: 4712, RAMMinMB: 6760, Tags: nil},
	{Name: "megadolphin:120b", Quant: "Q4_K_M", DiskSizeMB: 72000, VRAMMinMB: 72512, RAMMinMB: 74560, Tags: nil},
	{Name: "falcon2:11b", Quant: "Q4_K_M", DiskSizeMB: 6600, VRAMMinMB: 7112, RAMMinMB: 9160, Tags: nil},
	{Name: "stablelm-zephyr:3b", Quant: "Q4_K_M", DiskSizeMB: 1800, VRAMMinMB: 2312, RAMMinMB: 4360, Tags: nil},
	{Name: "duckdb-nsql:7b", Quant: "Q4_K_M", DiskSizeMB: 4200, VRAMMinMB: 4712, RAMMinMB: 6760, Tags: nil},
	{Name: "nuextract:3.8b", Quant: "Q4_K_M", DiskSizeMB: 2280, VRAMMinMB: 2792, RAMMinMB: 4840, Tags: nil},
	{Name: "mistrallite:7b", Quant: "Q4_K_M", DiskSizeMB: 4200, VRAMMinMB: 4712, RAMMinMB: 6760, Tags: nil},
	{Name: "bespoke-minicheck:7b", Quant: "Q4_K_M", DiskSizeMB: 4200, VRAMMinMB: 4712, RAMMinMB: 6760, Tags: nil},
	{Name: "notux:8x7b", Quant: "Q4_K_M", DiskSizeMB: 33600, VRAMMinMB: 34112, RAMMinMB: 36160, Tags: nil},
	{Name: "notus:7b", Quant: "Q4_K_M", DiskSizeMB: 4200, VRAMMinMB: 4712, RAMMinMB: 6760, Tags: nil},
	{Name: "wizard-vicuna:13b", Quant: "Q4_K_M", DiskSizeMB: 7800, VRAMMinMB: 8312, RAMMinMB: 10360, Tags: nil},
	{Name: "firefunction-v2:70b", Quant: "Q4_K_M", DiskSizeMB: 42000, VRAMMinMB: 42512, RAMMinMB: 44560, Tags: []string{"tools"}},
	{Name: "codebooga:34b", Quant: "Q4_K_M", DiskSizeMB: 20400, VRAMMinMB: 20912, RAMMinMB: 22960, Tags: nil},
	{Name: "open-orca-platypus2:13b", Quant: "Q4_K_M", DiskSizeMB: 7800, VRAMMinMB: 8312, RAMMinMB: 10360, Tags: nil},
	{Name: "tulu3:8b", Quant: "Q4_K_M", DiskSizeMB: 4800, VRAMMinMB: 5312, RAMMinMB: 7360, Tags: nil},
	{Name: "tulu3:70b", Quant: "Q4_K_M", DiskSizeMB: 42000, VRAMMinMB: 42512, RAMMinMB: 44560, Tags: nil},
	{Name: "goliath", Tags: nil},
	{Name: "bge-large:335m", Quant: "Q4_K_M", DiskSizeMB: 670, VRAMMinMB: 1182, RAMMinMB: 3230, Tags: []string{"embedding"}},
	{Name: "dbrx:132b", Quant: "Q4_K_M", DiskSizeMB: 79200, VRAMMinMB: 79712, RAMMinMB: 81760, Tags: nil},
	{Name: "nemotron-3-nano:30b", Quant: "Q4_K_M", DiskSizeMB: 18000, VRAMMinMB: 18512, RAMMinMB: 20560, Tags: []string{"tools", "thinking", "cloud"}},
	{Name: "olmo-3:7b", Quant: "Q4_K_M", DiskSizeMB: 4200, VRAMMinMB: 4712, RAMMinMB: 6760, Tags: nil},
	{Name: "olmo-3:32b", Quant: "Q4_K_M", DiskSizeMB: 19200, VRAMMinMB: 19712, RAMMinMB: 21760, Tags: nil},
	{Name: "sailor2:1b", Quant: "Q4_K_M", DiskSizeMB: 600, VRAMMinMB: 1112, RAMMinMB: 3160, Tags: nil},
	{Name: "sailor2:8b", Quant: "Q4_K_M", DiskSizeMB: 4800, VRAMMinMB: 5312, RAMMinMB: 7360, Tags: nil},
	{Name: "sailor2:20b", Quant: "Q4_K_M", DiskSizeMB: 12000, VRAMMinMB: 12512, RAMMinMB: 14560, Tags: nil},
	{Name: "command-r7b:7b", Quant: "Q4_K_M", DiskSizeMB: 4200, VRAMMinMB: 4712, RAMMinMB: 6760, Tags: []string{"tools"}},
	{Name: "deepseek-v2.5:236b", Quant: "Q4_K_M", DiskSizeMB: 141600, VRAMMinMB: 142112, RAMMinMB: 144160, Tags: nil},
	{Name: "phi4-mini-reasoning:3.8b", Quant: "Q4_K_M", DiskSizeMB: 2280, VRAMMinMB: 2792, RAMMinMB: 4840, Tags: nil},
	{Name: "granite3-guardian:2b", Quant: "Q4_K_M", DiskSizeMB: 1200, VRAMMinMB: 1712, RAMMinMB: 3760, Tags: nil},
	{Name: "granite3-guardian:8b", Quant: "Q4_K_M", DiskSizeMB: 4800, VRAMMinMB: 5312, RAMMinMB: 7360, Tags: nil},
	{Name: "smallthinker:3b", Quant: "Q4_K_M", DiskSizeMB: 1800, VRAMMinMB: 2312, RAMMinMB: 4360, Tags: nil},
	{Name: "command-a:111b", Quant: "Q4_K_M", DiskSizeMB: 66600, VRAMMinMB: 67112, RAMMinMB: 69160, Tags: []string{"tools"}},
	{Name: "kimi-k2.5", Tags: []string{"vision", "tools", "thinking", "cloud"}},
	{Name: "marco-o1:7b", Quant: "Q4_K_M", DiskSizeMB: 4200, VRAMMinMB: 4712, RAMMinMB: 6760, Tags: nil},
	{Name: "alfred:40b", Quant: "Q4_K_M", DiskSizeMB: 24000, VRAMMinMB: 24512, RAMMinMB: 26560, Tags: nil},
	{Name: "olmo-3.1:32b", Quant: "Q4_K_M", DiskSizeMB: 19200, VRAMMinMB: 19712, RAMMinMB: 21760, Tags: []string{"tools"}},
	{Name: "minimax-m2.5", Tags: []string{"tools", "thinking", "cloud"}},
	{Name: "devstral-2:123b", Quant: "Q4_K_M", DiskSizeMB: 73800, VRAMMinMB: 74312, RAMMinMB: 76360, Tags: []string{"tools", "cloud"}},
	{Name: "command-r7b-arabic:7b", Quant: "Q4_K_M", DiskSizeMB: 4200, VRAMMinMB: 4712, RAMMinMB: 6760, Tags: []string{"tools"}},
	{Name: "nomic-embed-text-v2-moe", Tags: []string{"embedding"}},
	{Name: "glm-5", Tags: []string{"tools", "thinking", "cloud"}},
	{Name: "cogito-2.1:671b", Quant: "Q4_K_M", DiskSizeMB: 402600, VRAMMinMB: 403112, RAMMinMB: 405160, Tags: []string{"cloud"}},
	{Name: "functiongemma:270m", Quant: "Q4_K_M", DiskSizeMB: 540, VRAMMinMB: 1052, RAMMinMB: 3100, Tags: []string{"tools"}},
	{Name: "gpt-oss-safeguard:20b", Quant: "Q4_K_M", DiskSizeMB: 12000, VRAMMinMB: 12512, RAMMinMB: 14560, Tags: []string{"tools", "thinking"}},
	{Name: "gpt-oss-safeguard:120b", Quant: "Q4_K_M", DiskSizeMB: 72000, VRAMMinMB: 72512, RAMMinMB: 74560, Tags: []string{"tools", "thinking"}},
	{Name: "glm-4.6", Tags: []string{"tools", "thinking", "cloud"}},
	{Name: "gemini-3-flash-preview", Tags: []string{"vision", "tools", "thinking", "cloud"}},
	{Name: "minimax-m2", Tags: []string{"tools", "thinking", "cloud"}},
	{Name: "glm-ocr", Tags: []string{"vision", "tools"}},
	{Name: "glm-4.7", Tags: []string{"tools", "thinking", "cloud"}},
	{Name: "deepseek-v3.2", Tags: []string{"tools", "thinking", "cloud"}},
	{Name: "kimi-k2", Tags: []string{"tools", "cloud"}},
	{Name: "kimi-k2-thinking", Tags: []string{"tools", "thinking", "cloud"}},
	{Name: "mistral-large-3", Tags: []string{"vision", "tools", "cloud"}},
	{Name: "minimax-m2.1", Tags: []string{"tools", "cloud"}},
}
