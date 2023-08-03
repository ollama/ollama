package llm

import (
	"fmt"
	"log"
	"os"

	"github.com/jmorganca/ollama/api"
)

type LLM interface {
	Predict([]int, string, func(api.GenerateResponse)) error
	Embedding(string) ([]float64, error)
	Encode(string) []int
	Decode(...int) string
	SetOptions(api.Options)
	Close()
}

func New(model string, opts api.Options) (LLM, error) {
	if _, err := os.Stat(model); err != nil {
		return nil, err
	}

	f, err := os.Open(model)
	if err != nil {
		return nil, err
	}

	ggml, err := DecodeGGML(f, ModelFamilyLlama)
	if err != nil {
		return nil, err
	}

	switch ggml.FileType {
	case FileTypeF32, FileTypeF16, FileTypeQ5_0, FileTypeQ5_1, FileTypeQ8_0:
		if opts.NumGPU != 0 {
			// Q5_0, Q5_1, and Q8_0 do not support Metal API and will
			// cause the runner to segmentation fault so disable GPU
			log.Printf("WARNING: GPU disabled for F32, F16, Q5_0, Q5_1, and Q8_0")
			opts.NumGPU = 0
		}
	}

	switch ggml.ModelFamily {
	case ModelFamilyLlama:
		return newLlama(model, opts)
	default:
		return nil, fmt.Errorf("unknown ggml type: %s", ggml.ModelFamily)
	}
}
