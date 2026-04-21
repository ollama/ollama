//go:build integration && migration

package integration

var compatibilityModelMatrix = []string{
	// Text / embedding families.
	"gpt-oss:20b",
	"gpt-oss:120b",
	"gpt-oss-safeguard:20b",
	"gpt-oss-safeguard:120b",
	"lfm2.5-thinking:1.2b-q4_K_M",
	"lfm2.5-thinking:1.2b-q8_0",
	"lfm2.5-thinking:1.2b-bf16",
	"embeddinggemma:300m",
	"embeddinggemma:300m-qat-q4_0",
	"embeddinggemma:300m-qat-q8_0",
	"snowflake-arctic-embed2:568m",
	"snowflake-arctic-embed2:568m-l-fp16",

	// Gemma compatibility families.
	"gemma3:1b",
	"gemma3:4b",
	"gemma3:12b",
	"gemma3:27b",
	"gemma3n:e2b",
	"gemma3n:e4b",
	"gemma4:e2b",
	"gemma4:e4b",
	"gemma4:26b",
	"gemma4:31b",

	// GLM / OCR families.
	"glm-4.7-flash:q4_K_M",
	"glm-4.7-flash:q8_0",
	"glm-4.7-flash:bf16",
	"deepseek-ocr:3b",
	"deepseek-ocr:3b-bf16",
	"glm-ocr:bf16",
	"glm-ocr:q8_0",

	// Qwen vision / MoE families.
	"qwen2.5vl:3b",
	"qwen2.5vl:7b",
	"qwen2.5vl:32b",
	"qwen2.5vl:72b",
	"qwen2.5vl:72b-q8_0",
	"qwen2.5vl:72b-fp16",
	"qwen3-vl:2b",
	"qwen3-vl:2b-instruct",
	"qwen3-vl:4b-instruct",
	"qwen3-vl:4b-thinking",
	"qwen3-vl:8b",
	"qwen3-vl:8b-instruct",
	"qwen3-vl:8b-thinking",
	"qwen3-vl:30b",
	"qwen3-vl:30b-a3b-instruct",
	"qwen3-vl:30b-a3b-thinking",
	"qwen3-vl:32b-instruct",
	"qwen3-vl:32b-thinking",
	"qwen3.5:2b",
	"qwen3.5:35b",
	"qwen3-coder-next:q4_K_M",

	// OLMo compatibility families.
	"olmo-3:7b-instruct",
	"olmo-3:7b-think",
	"olmo-3.1:32b",

	// Mistral / Pixtral-style multimodal families.
	"ministral-3:3b",
	"ministral-3:8b",
	"ministral-3:14b",
	"mistral-small3.1:24b",
	"mistral-small3.2:24b",

	// Legacy projector / Nemotron / Llama compatibility families.
	"bakllava:7b",
	"llava:7b",
	"llava:7b-v1.5-q4_K_M",
	"llava:13b",
	"llava:13b-v1.5-q4_K_M",
	"llava:34b",
	"llava-llama3:8b",
	"llava-phi3:3.8b",
	"nemotron-3-super:120b-a12b-q4_K_M",
	"nemotron3:33b",
	"dolphin-llama3:8b-v2.9-q4_K_M",
	"dolphin-llama3:8b-256k-v2.9-q4_K_M",
	"dolphin-llama3:70b-v2.9-q4_K_M",
	"llama3-chatqa:8b-v1.5-q4_K_M",
	"llama3-chatqa:70b-v1.5-q4_K_M",
	"llama3.2:1b",
	"llama4",
}

func compatibilityPublishedModelNames() []string {
	names := make([]string, 0, len(compatibilityModelMatrix))
	names = append(names, compatibilityModelMatrix...)
	return names
}

func compatibilityMigrationModelNames() []string {
	names := make([]string, 0, len(compatibilityModelMatrix))
	for _, name := range compatibilityModelMatrix {
		switch name {
		case "glm-4.7-flash:q8_0", "glm-4.7-flash:bf16":
			// These published variants already use llama.cpp-compatible
			// deepseek2 metadata, so there is no lazy conversion to validate.
			continue
		case "llama3.2:1b":
			// This current library artifact no longer triggers the Llama 3
			// metadata patch, so only published compatibility coverage applies.
			continue
		case "llava:7b", "llava:13b", "llava:34b", "llava-phi3:3.8b":
			// These current library artifacts already have projector metadata.
			// Keep it in published coverage, not lazy-conversion coverage.
			continue
		default:
			names = append(names, name)
		}
	}
	return names
}
