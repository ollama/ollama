package llm

import (
	"fmt"
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

	switch ggml.ModelFamily {
	case ModelFamilyLlama:
		return newLlama(model, opts)
	default:
		return nil, fmt.Errorf("unknown ggml type: %s", ggml.ModelFamily)
	}
}
