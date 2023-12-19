package llm

import (
	"embed"
	"fmt"
	"log"
	"os"

	"github.com/jmorganca/ollama/api"
)

//go:embed llama.cpp/gguf/build/*/bin/ggml-metal.metal
var libEmbed embed.FS

func newRocmShimExtServer(model string, adapters, projectors []string, numLayers int64, opts api.Options) (extServer, error) {
	// should never happen...
	return nil, fmt.Errorf("ROCM GPUs not supported on Mac")
}

func nativeInit(workdir string) error {
	err := extractLib(workdir, "llama.cpp/gguf/build/*/bin/ggml-metal.metal")
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
