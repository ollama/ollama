//go:build integration && create

package integration

// The create scope exercises model-creation flows: importing a GGUF,
// importing a safetensors model, and quantizing an existing model.
func init() {
	registerIntegrationCases(
		integrationTestCase("create-safetensors", "", runCreateSafetensorsLLM),
		integrationTestCase("create-gguf", "", runCreateGGUF),
		integrationTestCase("quantization", "qwen2.5:0.5b-instruct-fp16", runQuantization),
	)
}
