//go:build integration && release

package integration

var (
	releaseUnicodeInputModel    = integrationModel{Name: "deepseek-coder-v2:16b-lite-instruct-q2_K", MinVRAMGB: 12}
	releaseUnicodeOutputModel   = "gemma2:2b"
	releaseNumPredictModel      = "llama3.2:1b"
	releaseParallelHistoryModel = integrationModel{Name: "gpt-oss:20b", MinVRAMGB: 16}
	releaseChatModels           = []integrationModel{
		{Name: "gemma4", MinVRAMGB: 8},
		{Name: "gemma4:12b", MinVRAMGB: 16},
		{Name: "lfm2.5", MinVRAMGB: 6},
		{Name: "granite4.1:8b", MinVRAMGB: 6},
		{Name: "gpt-oss:20b", MinVRAMGB: 16},
		{Name: "qwen3.6:27b", MinVRAMGB: 20},
		{Name: "qwen3.5:2b", MinVRAMGB: 4},
		{Name: "qwen3.5:2b-nvfp4", MinVRAMGB: 4},
		{Name: "deepseek-r1:8b", MinVRAMGB: 6},
		{Name: "mistral-small3.2:latest", MinVRAMGB: 16},
		{Name: "llama3.2:latest"},
		{Name: "gemma4:e2b-nvfp4", MinVRAMGB: 8},
	}
	releaseEmbedModels = []string{
		"embeddinggemma",
		"nomic-embed-text",
		"all-minilm",
		"bge-large",
		"bge-m3",
		"granite-embedding",
		"mxbai-embed-large",
		"paraphrase-multilingual",
		"snowflake-arctic-embed",
		"snowflake-arctic-embed2",
		"qwen3-embedding",
	}
	releaseVisionModels = []string{
		"nemotron3:33b",
		"gemma4",
		"qwen3.6:27b",
		// "llama3.2-vision", // TODO: re-enable when llama.cpp supports mllama.
	}
	releaseVisionTextModels = []string{
		"gemma4",
		"qwen3.6:27b",
		"qwen3.5:2b",
		// "llama3.2-vision", // TODO: re-enable when llama.cpp supports mllama.
		"ministral-3:3b",
	}
	releaseToolsModels = []string{
		"lfm2.5",
		"nemotron3:33b",
		"gemma4",
		"gpt-oss:20b",
		"qwen3.6:27b",
	}
	releaseAudioModels = []string{
		"nemotron3:33b",
		"gemma4:e2b",
		"gemma4:e4b",
	}
)

const releaseSplitBatchVisionModel = "qwen3.5:2b"

func init() {
	// Fixed release regression cases
	registerIntegrationCases(
		integrationTestCase("api-generate", smol, runAPIGenerate),
		integrationTestCase("api-chat", smol, runAPIChat),
		integrationTestCase("api-list-models", "", runAPIListModels),
		integrationTestCase("api-show-model", "llama3.2", runAPIShowModel),
		integrationTestCase("generate-logprobs", smol, runAPIGenerateLogprobs),
		integrationTestCase("chat-logprobs", smol, runAPIChatLogprobs),

		integrationTestCase("blue-sky", smol, runBlueSky),
		integrationModelTestCase("unicode-input", releaseUnicodeInputModel.Name, runUnicode),
		integrationModelTestCase("unicode-output", releaseUnicodeOutputModel, runExtendedUnicodeOutput),
		integrationTestCase("unicode-model-dir", smol, runUnicodeModelDir),
		integrationModelTestCase("num-predict", releaseNumPredictModel, runNumPredict),

		integrationModelsTestCase("embed-correlation", releaseEmbedModels, runEmbedCosineDistanceCorrelation),
		integrationTestCase("embedding-api", "all-minilm", runAllMiniLMEmbeddings),
		integrationTestCase("embed-api", "all-minilm", runAllMiniLMEmbed),
		integrationTestCase("embed-api-batch", "all-minilm", runAllMiniLMBatchEmbed),
		integrationTestCase("embed-api-truncate", "all-minilm", runAllMiniLMEmbedTruncate),
		integrationModelsTestCase("embed-truncation", releaseEmbedModels, runEmbedTruncation),
		integrationModelsTestCase("embed-large-input", releaseEmbedModels, runEmbedLargeInput),
		integrationModelsTestCase("embed-status-code", releaseEmbedModels, runEmbedStatusCode),

		integrationModelsTestCase("vision-multiturn", releaseVisionModels, runVisionMultiTurn),
		integrationModelsTestCase("vision-count", releaseVisionModels, runVisionObjectCounting),
		integrationModelsTestCase("vision-scene", releaseVisionModels, runVisionSceneUnderstanding),
		integrationModelsTestCase("vision-spatial", releaseVisionModels, runVisionSpatialReasoning),
		integrationModelsTestCase("vision-detail", releaseVisionModels, runVisionDetailRecognition),
		integrationModelsTestCase("vision-multi-image", releaseVisionModels, runVisionMultiImage),
		integrationModelsTestCase("vision-description", releaseVisionModels, runVisionImageDescription),
		integrationModelTestCase("vision-split-batch", releaseSplitBatchVisionModel, runIntegrationSplitBatch),

		integrationModelsTestCase("audio-response", releaseAudioModels, runAudioResponse),
		integrationModelsTestCase("openai-audio-transcription", releaseAudioModels, runOpenAIAudioTranscription),
		integrationModelsTestCase("openai-chat-audio", releaseAudioModels, runOpenAIChatWithAudio),

		integrationTestCase("context-long-input", smol, runLongInputContext),
		integrationTestCase("context-exhaustion", smol, runContextExhaustion),
		integrationModelTestCase("parallel-generate-history", releaseParallelHistoryModel.Name, runParallelGenerateWithHistory),
		integrationTestCase("generate-history", smol, runGenerateWithHistory),
		integrationModelTestCase("parallel-chat-history", defaultTestModel(releaseParallelHistoryModel.Name), runParallelChatWithHistory),
		integrationTestCase("chat-history", smol, runChatWithHistory),
		integrationTestCase("concurrent-chat", smol, runConcurrentChat),
		integrationTestCase("scheduler-multimodel", "", runMultiModelStress),
		integrationTestCase("scheduler-max-queue", smol, runMaxQueue),

		integrationTestCase("thinking-enabled", smol, runThinkingEnabled),
		integrationTestCase("thinking-suppressed", smol, runThinkingSuppressed),

		integrationTestCase("create-safetensors", "", runCreateSafetensorsLLM),
		integrationTestCase("create-gguf", "", runCreateGGUF),
		integrationTestCase("quantization", "qwen2.5:0.5b-instruct-fp16", runQuantization),
		integrationTestCase("image-generation", "", runImageGeneration),
	)

	// Model-parametric cases
	registerModelMinVRAM([]integrationModel{releaseUnicodeInputModel, releaseParallelHistoryModel})
	registerModelMinVRAM(releaseChatModels)
	registerChatCases(testModels(modelNames(releaseChatModels)))
	registerEmbeddingCases(testModels(releaseEmbedModels))
	registerVisionTextCases(testModels(releaseVisionTextModels))
	registerToolCases(testModels(releaseToolsModels))
	registerToolStressCases(testModels(releaseToolsModels))
	registerAudioTranscriptionCases(testModels(releaseAudioModels))
}
