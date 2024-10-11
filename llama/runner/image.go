package main

import (
	"fmt"
	"sync"

	"github.com/ollama/ollama/llama"
)

// Locking: Mu must be held by the caller when generating image
// embeddings

type ImageContext struct {
	Mu sync.Mutex

	clip   *llama.ClipContext
	mllama *llama.MllamaContext
}

func NewImageContext(llamaContext *llama.Context, modelPath string) (*ImageContext, error) {
	arch, err := llama.GetModelArch(modelPath)
	if err != nil {
		return nil, fmt.Errorf("unable to determine vision architecture: %w (%s)", err, modelPath)
	}

	var c ImageContext
	if arch == "clip" {
		c.clip, err = llama.NewClipContext(llamaContext, modelPath)
	} else if arch == "mllama" {
		c.mllama, err = llama.NewMllamaContext(llamaContext, modelPath)
	} else {
		return nil, fmt.Errorf("unknown vision model architecture: %s", arch)
	}

	if err != nil {
		return nil, err
	}

	return &c, nil
}

func (c *ImageContext) Free(modelPath string) {
	if c.clip != nil {
		c.clip.Free()
	}
	if c.mllama != nil {
		c.mllama.Free()
	}
}

func (c *ImageContext) NewEmbed(llamaContext *llama.Context, data []byte, aspectRatioId int) [][]float32 {
	if c.mllama != nil {
		return c.mllama.NewEmbed(llamaContext, data, aspectRatioId)
	} else {
		return c.clip.NewEmbed(llamaContext, data)
	}
}

func (c *ImageContext) EmbedSize(llamaContext *llama.Context) int {
	if c != nil && c.mllama != nil {
		return c.mllama.EmbedSize(llamaContext)
	} else {
		return llamaContext.Model().NEmbd()
	}
}
