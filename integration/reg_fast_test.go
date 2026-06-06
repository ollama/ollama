//go:build integration && fast

package integration

var (
	fastNumPredictModel = "llama3.2:1b"
	fastChatModels      = []integrationModel{
		{Name: "gemma4", MinVRAMGB: 8},
		{Name: "gemma4:12b", MinVRAMGB: 16},
		{Name: "qwen3.5:2b-nvfp4", MinVRAMGB: 4},
	}
	fastEmbedModels       = []string{"qwen3-embedding"}
	fastVisionTextModels  = []string{"gemma4"}
	fastToolsModels       = []string{"qwen3.5:2b"}
	fastToolsStressModels = []string{"lfm2.5"}
	fastAudioModels       = []string{"gemma4:e2b"}
)

func init() {
	// API/basic/context/concurrency smoke cases
	registerIntegrationCases(
		integrationTestCase("api-generate", smol, runAPIGenerate),
		integrationTestCase("api-chat", smol, runAPIChat),
		integrationTestCase("api-list-models", "", runAPIListModels),
		integrationTestCase("api-show-model", "llama3.2", runAPIShowModel),
		integrationTestCase("generate-logprobs", smol, runAPIGenerateLogprobs),
		integrationTestCase("chat-logprobs", smol, runAPIChatLogprobs),
		integrationTestCase("blue-sky", smol, runBlueSky),
		integrationModelTestCase("num-predict", fastNumPredictModel, runNumPredict),
		integrationTestCase("embedding-api", "all-minilm", runAllMiniLMEmbeddings),
		integrationTestCase("embed-api-truncate", "all-minilm", runAllMiniLMEmbedTruncate),
		integrationTestCase("context-long-input", smol, runLongInputContext),
		integrationTestCase("context-exhaustion", smol, runContextExhaustion),
		integrationTestCase("generate-history", smol, runGenerateWithHistory),
		integrationTestCase("concurrent-chat", smol, runConcurrentChat),
	)

	// Model-parametric cases
	registerModelMinVRAM(fastChatModels)
	registerChatCases(testModels(modelNames(fastChatModels)))
	registerEmbeddingCases(testModels(fastEmbedModels))
	registerVisionTextCases(testModels(fastVisionTextModels))
	registerToolCases(testModels(fastToolsModels))
	registerToolStressCases(testModels(fastToolsStressModels))
	registerAudioTranscriptionCases(testModels(fastAudioModels))
}
