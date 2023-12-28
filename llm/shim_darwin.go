package llm

import (
	"embed"
	"fmt"
	"log"
	"os"

	"github.com/jmorganca/ollama/api"
)

//go:embed llama.cpp/gguf/ggml-metal.metal
var libEmbed embed.FS

func newDynamicShimExtServer(library, model string, adapters, projectors []string, numLayers int64, opts api.Options) (extServer, error) {
	// should never happen...
	return nil, fmt.Errorf("Dynamic library loading not supported on Mac")
}

func nativeInit(workdir string) error {
	_, err := extractDynamicLibs(workdir, "llama.cpp/gguf/ggml-metal.metal")
	if err != nil {
		if err == payloadMissing {
			// TODO perhaps consider this a hard failure on arm macs?
			log.Printf("ggml-meta.metal payload missing")
			return nil
		}
		return err
	}
	os.Setenv("GGML_METAL_PATH_RESOURCES", workdir)
	return nil
}
