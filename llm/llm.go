package llm

import (
	"fmt"
	"log"
	"os"

	"github.com/pbnjay/memory"

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

func New(model string, adapters []string, opts api.Options) (LLM, error) {
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

	totalResidentMemory := memory.TotalMemory()
	switch ggml.ModelType {
	case ModelType3B, ModelType7B:
		if totalResidentMemory < 8*1024*1024 {
			return nil, fmt.Errorf("model requires at least 8GB of memory")
		}
	case ModelType13B:
		if totalResidentMemory < 16*1024*1024 {
			return nil, fmt.Errorf("model requires at least 16GB of memory")
		}
	case ModelType30B:
		if totalResidentMemory < 32*1024*1024 {
			return nil, fmt.Errorf("model requires at least 32GB of memory")
		}
	case ModelType65B:
		if totalResidentMemory < 64*1024*1024 {
			return nil, fmt.Errorf("model requires at least 64GB of memory")
		}
	}

	switch ggml.ModelFamily {
	case ModelFamilyLlama:
		return newLlama(model, adapters, opts)
	default:
		return nil, fmt.Errorf("unknown ggml type: %s", ggml.ModelFamily)
	}
}
