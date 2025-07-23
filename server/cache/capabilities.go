package cache

import (
	"fmt"
	"log/slog"
	"os"
	"slices"
	"sync"
	"time"

	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/template"
	"github.com/ollama/ollama/thinking"
	"github.com/ollama/ollama/types/model"
)

// cacheEntry stores capabilities and the modification time of the model file
type cacheEntry struct {
	capabilities []model.Capability
	modTime      time.Time
}

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
	// Get file info to check modification time
	fileInfo, err := os.Stat(modelPath)
	if err != nil {
		return nil, err
	}
	currentModTime := fileInfo.ModTime()

	// Check if we have a cached entry
	if cached, ok := ggufCapabilities.Load(modelPath); ok {
		entry := cached.(cacheEntry)
		// If the file hasn't been modified since we cached it, return the cached capabilities
		if entry.modTime.Equal(currentModTime) {
			return entry.capabilities, nil
		}
	}

	// If not cached or file was modified, read the model file to determine capabilities
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

	// Cache the capabilities with the modification time
	ggufCapabilities.Store(modelPath, cacheEntry{
		capabilities: capabilities,
		modTime:      currentModTime,
	})

	return capabilities, nil
}
