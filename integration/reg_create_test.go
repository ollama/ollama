//go:build integration && create

package integration

// The create scope exercises model-creation flows: importing a safetensors
// model, importing a GGUF, its blob-transfer variant, and quantizing an
// existing model. These carry large client-side blob uploads, so they are
// kept out of the release scope and run in the dedicated go-create lane.
func init() {
	registerIntegrationCases(
		integrationTestCase("create-safetensors", "", runCreateSafetensorsLLM),
		integrationTestCase("create-gguf", "", runCreateGGUF),
		integrationTestCase("create-gguf-blob-transfer", "", runCreateGGUFBlobTransfer),
		integrationTestCase("quantization", "", runQuantization),
	)
}
