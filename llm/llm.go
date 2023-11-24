package llm

import (
	"context"
	"fmt"
	"log"
	"os"
	"runtime"

	"github.com/pbnjay/memory"

	"github.com/jmorganca/ollama/api"
	"github.com/jmorganca/ollama/format"
)

type LLM interface {
	Predict(context.Context, []int, string, string, func(api.GenerateResponse)) error
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
		case "F32", "Q5_0", "Q5_1", "Q8_0":
			if ggml.Name() != "gguf" && opts.NumGPU != 0 {
				// GGML Q8_0 do not support Metal API and will
				// cause the runner to segmentation fault so disable GPU
				log.Printf("WARNING: GPU disabled for F32, Q5_0, Q5_1, and Q8_0")
				opts.NumGPU = 0
			}
		}

		var requiredMemory int64
		var f16Multiplier int64 = 2

		switch ggml.ModelType() {
		case "3B", "7B":
			requiredMemory = 8 * format.GigaByte
		case "13B":
			requiredMemory = 16 * format.GigaByte
		case "30B", "34B", "40B":
			requiredMemory = 32 * format.GigaByte
		case "65B", "70B":
			requiredMemory = 64 * format.GigaByte
		case "180B":
			requiredMemory = 128 * format.GigaByte
			f16Multiplier = 4
		}

		systemMemory := int64(memory.TotalMemory())

		if ggml.FileType() == "F16" && requiredMemory*f16Multiplier > systemMemory {
			return nil, fmt.Errorf("F16 model requires at least %s of total memory", format.HumanBytes(requiredMemory))
		} else if requiredMemory > systemMemory {
			return nil, fmt.Errorf("model requires at least %s of total memory", format.HumanBytes(requiredMemory))
		}
	}

	switch ggml.Name() {
	case "gguf":
		// TODO: gguf will load these options automatically from the model binary
		opts.NumGQA = 0
		opts.RopeFrequencyBase = 0.0
		opts.RopeFrequencyScale = 0.0
		return newLlama(model, adapters, chooseRunners(workDir, "gguf"), ggml.NumLayers(), opts)
	case "ggml", "ggmf", "ggjt", "ggla":
		return newLlama(model, adapters, chooseRunners(workDir, "ggml"), ggml.NumLayers(), opts)
	default:
		return nil, fmt.Errorf("unknown ggml type: %s", ggml.ModelFamily())
	}
}
