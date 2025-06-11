package cache

import (
	"fmt"
	"log/slog"
	"os"
	"slices"
	"sync"

	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/template"
	"github.com/ollama/ollama/thinking"
	"github.com/ollama/ollama/types/model"
)

// ggufCapabilities is a cache for gguf model capabilities
var ggufCapabilities = &sync.Map{}

// ModelInfo contains the minimal information needed to determine capabilities
type ModelInfo struct {
	ModelPath      string
	ProjectorPaths []string
	Template       *template.Template
}

// Capabilities returns the capabilities that the model supports
func Capabilities(info ModelInfo) []model.Capability {
	capabilities, err := ggufCapabilties(info.ModelPath)
	if err != nil {
		slog.Error("could not determine gguf capabilities", "error", err)
	}

	if info.Template == nil {
		return capabilities
	}

	// Check for tools capability
	if slices.Contains(info.Template.Vars(), "tools") {
		capabilities = append(capabilities, model.CapabilityTools)
	}

	// Check for insert capability
	if slices.Contains(info.Template.Vars(), "suffix") {
		capabilities = append(capabilities, model.CapabilityInsert)
	}

	// Check for vision capability in projector-based models
	if len(info.ProjectorPaths) > 0 {
		capabilities = append(capabilities, model.CapabilityVision)
	}

	// Check for thinking capability
	openingTag, closingTag := thinking.InferTags(info.Template.Template)
	if openingTag != "" && closingTag != "" {
		capabilities = append(capabilities, model.CapabilityThinking)
	}

	return capabilities
}

func ggufCapabilties(modelPath string) ([]model.Capability, error) {
	if ggufCapabilities, ok := ggufCapabilities.Load(modelPath); ok {
		capabilities := ggufCapabilities.([]model.Capability)
		return capabilities, nil
	}

	// If not cached, read the model file to determine capabilities
	capabilities := []model.Capability{}

	r, err := os.Open(modelPath)
	if err != nil {
		return nil, err
	}
	defer r.Close()

	f, err := ggml.Decode(r, 1024)
	if err != nil {
		return nil, err
	}

	if _, ok := f.KV()[fmt.Sprintf("%s.pooling_type", f.KV().Architecture())]; ok {
		capabilities = append(capabilities, model.CapabilityEmbedding)
	} else {
		capabilities = append(capabilities, model.CapabilityCompletion)
	}
	if _, ok := f.KV()[fmt.Sprintf("%s.vision.block_count", f.KV().Architecture())]; ok {
		capabilities = append(capabilities, model.CapabilityVision)
	}

	// Cache the capabilities for future use
	ggufCapabilities.Store(modelPath, capabilities)

	return capabilities, nil
}
