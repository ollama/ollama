package llm

import (
	"context"
	"fmt"
	"log"
	"os"
	"runtime"

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

func New(workDir, model string, adapters []string, opts api.Options) (LLM, error) {
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

	if runtime.GOOS == "darwin" {
		switch ggml.FileType() {
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
	}

	totalResidentMemory := memory.TotalMemory()
	switch ggml.ModelType() {
	case "3B", "7B":
		if ggml.FileType() == "F16" && totalResidentMemory < 16*1000*1000 {
			return nil, fmt.Errorf("F16 model requires at least 16 GB of memory")
		} else if totalResidentMemory < 8*1000*1000 {
			return nil, fmt.Errorf("model requires at least 8 GB of memory")
		}
	case "13B":
		if ggml.FileType() == "F16" && totalResidentMemory < 32*1000*1000 {
			return nil, fmt.Errorf("F16 model requires at least 32 GB of memory")
		} else if totalResidentMemory < 16*1000*1000 {
			return nil, fmt.Errorf("model requires at least 16 GB of memory")
		}
	case "30B", "34B", "40B":
		if ggml.FileType() == "F16" && totalResidentMemory < 64*1000*1000 {
			return nil, fmt.Errorf("F16 model requires at least 64 GB of memory")
		} else if totalResidentMemory < 32*1000*1000 {
			return nil, fmt.Errorf("model requires at least 32 GB of memory")
		}
	case "65B", "70B":
		if ggml.FileType() == "F16" && totalResidentMemory < 128*1000*1000 {
			return nil, fmt.Errorf("F16 model requires at least 128 GB of memory")
		} else if totalResidentMemory < 64*1000*1000 {
			return nil, fmt.Errorf("model requires at least 64 GB of memory")
		}
	case "180B":
		if ggml.FileType() == "F16" && totalResidentMemory < 512*1000*1000 {
			return nil, fmt.Errorf("F16 model requires at least 512GB of memory")
		} else if totalResidentMemory < 128*1000*1000 {
			return nil, fmt.Errorf("model requires at least 128GB of memory")
		}
	}

	switch ggml.Name() {
	case "gguf":
		opts.NumGQA = 0 // TODO: remove this when llama.cpp runners differ enough to need separate newLlama functions
		return newLlama(model, adapters, chooseRunners(workDir, "gguf"), ggml.NumLayers(), opts)
	case "ggml", "ggmf", "ggjt", "ggla":
		return newLlama(model, adapters, chooseRunners(workDir, "ggml"), ggml.NumLayers(), opts)
	default:
		return nil, fmt.Errorf("unknown ggml type: %s", ggml.ModelFamily())
	}
}
