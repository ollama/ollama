package llm

import (
	"github.com/jmorganca/ollama/api"
)

func newDefaultExtServer(model string, adapters, projectors []string, numLayers int64, opts api.Options) (extServer, error) {
	// On windows we always load the llama.cpp libraries dynamically to avoid startup DLL dependencies
	// This ensures we can update the PATH at runtime to get everything loaded

	return newDynamicShimExtServer(AvailableShims["cpu"], model, adapters, projectors, numLayers, opts)
}
