package llm

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/pbnjay/memory"

	"github.com/jmorganca/ollama/api"
)

type LLM interface {
	Predict(context.Context, []int, string, func(api.GenerateResponse)) error
	Embedding(context.Context, string) ([]float64, error)
	Encode(context.Context, string) ([]int, error)
	Decode(context.Context, []int) (string, error)
	SetOptions(api.Options)
	Close()
	Ping(context.Context) error
}

func New(model string, adapters []string, opts api.Options) (LLM, error) {
	if _, err := os.Stat(model); err != nil {
		return nil, err
	}

	f, err := os.Open(model)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	ggml, err := DecodeGGML(f)
	if err != nil {
		return nil, err
	}

	switch ggml.FileType().String() {
	case "Q8_0":
		if ggml.Name() != "gguf" && opts.NumGPU != 0 {
			// GGML Q8_0 do not support Metal API and will
			// cause the runner to segmentation fault so disable GPU
			log.Printf("WARNING: GPU disabled for F32, Q5_0, Q5_1, and Q8_0")
			opts.NumGPU = 0
		}
	case "F32", "Q5_0", "Q5_1":
		if opts.NumGPU != 0 {
			// F32, Q5_0, Q5_1, and Q8_0 do not support Metal API and will
			// cause the runner to segmentation fault so disable GPU
			log.Printf("WARNING: GPU disabled for F32, Q5_0, Q5_1, and Q8_0")
			opts.NumGPU = 0
		}
	}

	totalResidentMemory := memory.TotalMemory()
	switch ggml.ModelType() {
	case ModelType3B, ModelType7B:
		if ggml.FileType().String() == "F16" && totalResidentMemory < 16*1024*1024 {
			return nil, fmt.Errorf("F16 model requires at least 16GB of memory")
		} else if totalResidentMemory < 8*1024*1024 {
			return nil, fmt.Errorf("model requires at least 8GB of memory")
		}
	case ModelType13B:
		if ggml.FileType().String() == "F16" && totalResidentMemory < 32*1024*1024 {
			return nil, fmt.Errorf("F16 model requires at least 32GB of memory")
		} else if totalResidentMemory < 16*1024*1024 {
			return nil, fmt.Errorf("model requires at least 16GB of memory")
		}
	case ModelType30B, ModelType34B:
		if ggml.FileType().String() == "F16" && totalResidentMemory < 64*1024*1024 {
			return nil, fmt.Errorf("F16 model requires at least 64GB of memory")
		} else if totalResidentMemory < 32*1024*1024 {
			return nil, fmt.Errorf("model requires at least 32GB of memory")
		}
	case ModelType65B:
		if ggml.FileType().String() == "F16" && totalResidentMemory < 128*1024*1024 {
			return nil, fmt.Errorf("F16 model requires at least 128GB of memory")
		} else if totalResidentMemory < 64*1024*1024 {
			return nil, fmt.Errorf("model requires at least 64GB of memory")
		}
	}

	switch ggml.Name() {
	case "gguf":
		opts.NumGQA = 0 // TODO: remove this when llama.cpp runners differ enough to need separate newLlama functions
		return newLlama(model, adapters, ggufRunner(), opts)
	case "ggml", "ggmf", "ggjt", "ggla":
		return newLlama(model, adapters, ggmlRunner(), opts)
	default:
		return nil, fmt.Errorf("unknown ggml type: %s", ggml.ModelFamily())
	}
}
