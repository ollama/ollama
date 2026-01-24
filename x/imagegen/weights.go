//go:build mlx

package imagegen

import (
	"fmt"
	"sort"
	"strings"

	"github.com/ollama/ollama/x/imagegen/mlx"
)

// ManifestWeights provides fast weight loading from tensor blobs.
// Uses native mmap loading with synthetic safetensors headers for zero-copy.
type ManifestWeights struct {
	manifest    *ModelManifest
	component   string
	tensors     map[string]ManifestLayer      // name -> layer
	cache       map[string]*mlx.Array         // name -> loaded array
	nativeCache []*mlx.SafetensorsFile        // keep native handles alive
}

// LoadWeightsFromManifest creates a weight loader for a component from manifest storage.
func LoadWeightsFromManifest(manifest *ModelManifest, component string) (*ManifestWeights, error) {
	layers := manifest.GetTensorLayers(component)
	if len(layers) == 0 {
		return nil, fmt.Errorf("no tensor layers found for component %q", component)
	}

	// Strip component prefix from tensor names for model loading
	// e.g., "text_encoder/model.embed_tokens.weight" -> "model.embed_tokens.weight"
	prefix := component + "/"
	tensors := make(map[string]ManifestLayer, len(layers))
	for _, layer := range layers {
		tensorName := strings.TrimPrefix(layer.Name, prefix)
		tensors[tensorName] = layer
	}

	return &ManifestWeights{
		manifest:  manifest,
		component: component,
		tensors:   tensors,
		cache:     make(map[string]*mlx.Array),
	}, nil
}

// LoadAllWeightsFromManifest creates a weight loader for all tensors without component filtering.
// Used for LLM models where tensors don't have a component prefix.
func LoadAllWeightsFromManifest(manifest *ModelManifest) (*ManifestWeights, error) {
	layers := manifest.GetAllTensorLayers()
	if len(layers) == 0 {
		return nil, fmt.Errorf("no tensor layers found in manifest")
	}

	tensors := make(map[string]ManifestLayer, len(layers))
	for _, layer := range layers {
		tensors[layer.Name] = layer
	}

	return &ManifestWeights{
		manifest: manifest,
		tensors:  tensors,
		cache:    make(map[string]*mlx.Array),
	}, nil
}

// Load loads all tensor blobs using native mmap (zero-copy).
// Blobs are stored in safetensors format for native mlx_load_safetensors mmap.
// If dtype is non-zero, tensors are converted to the specified dtype.
func (mw *ManifestWeights) Load(dtype mlx.Dtype) error {
	// Track native handles to free after batch eval
	nativeHandles := make([]*mlx.SafetensorsFile, 0, len(mw.tensors))
	arrays := make([]*mlx.Array, 0, len(mw.tensors))

	for name, layer := range mw.tensors {
		path := mw.manifest.BlobPath(layer.Digest)

		// Load blob as safetensors (native mmap, zero-copy)
		sf, err := mlx.LoadSafetensorsNative(path)
		if err != nil {
			// Free any handles we've accumulated
			for _, h := range nativeHandles {
				h.Free()
			}
			return fmt.Errorf("load %s: %w", name, err)
		}
		nativeHandles = append(nativeHandles, sf)

		// Blob contains single tensor named "data"
		arr := sf.Get("data")
		if arr == nil {
			for _, h := range nativeHandles {
				h.Free()
			}
			return fmt.Errorf("tensor 'data' not found in blob for %s", name)
		}

		// Convert dtype if needed
		if dtype != 0 && arr.Dtype() != dtype {
			arr = mlx.AsType(arr, dtype)
		}
		// Make contiguous copy to ensure independence from mmap
		arr = mlx.Contiguous(arr)
		mw.cache[name] = arr
		arrays = append(arrays, arr)
	}

	// Batch evaluate all tensors at once (much faster than one at a time)
	mlx.Eval(arrays...)

	// Now safe to free all native handles
	for _, sf := range nativeHandles {
		sf.Free()
	}

	return nil
}

// GetTensor returns a tensor from cache. Call Load() first.
func (mw *ManifestWeights) GetTensor(name string) (*mlx.Array, error) {
	if mw.cache == nil {
		return nil, fmt.Errorf("cache not initialized: call Load() first")
	}
	arr, ok := mw.cache[name]
	if !ok {
		return nil, fmt.Errorf("tensor %q not found", name)
	}
	return arr, nil
}

// ListTensors returns all tensor names in sorted order.
func (mw *ManifestWeights) ListTensors() []string {
	names := make([]string, 0, len(mw.tensors))
	for name := range mw.tensors {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

// HasTensor checks if a tensor exists.
func (mw *ManifestWeights) HasTensor(name string) bool {
	_, ok := mw.tensors[name]
	return ok
}

// Quantization returns the model's quantization type from model_index.json.
// Returns empty string if not quantized.
// Falls back to detecting from tensor names and shapes if not in config.
func (mw *ManifestWeights) Quantization() string {
	if mw.manifest == nil {
		return ""
	}

	// Try to read from model_index.json first
	var index struct {
		Quantization string `json:"quantization"`
	}
	if err := mw.manifest.ReadConfigJSON("model_index.json", &index); err == nil && index.Quantization != "" {
		return index.Quantization
	}

	// Fallback: detect from tensor names
	// Check if any tensors have _scale suffix (indicates quantization)
	hasScales := false
	hasQBias := false
	for name := range mw.tensors {
		if strings.HasSuffix(name, ".weight_scale") {
			hasScales = true
		}
		if strings.HasSuffix(name, ".weight_qbias") {
			hasQBias = true
		}
	}

	if !hasScales {
		// No scales = not quantized
		return ""
	}

	// Has scales but no qbias = NVFP4 (or other non-affine mode)
	if !hasQBias {
		return "NVFP4"
	}

	// Has both scales and qbias = affine mode
	// Need to determine FP4 vs FP8 from tensor shapes
	// FP4: weight last dim is 1/8 of scales last dim * group_size
	// FP8: weight last dim is 1/4 of scales last dim * group_size
	//
	// For affine mode with group_size=32:
	// - FP4 (4 bits): 8 elements packed per uint32, so weight_dim = orig_dim / 8
	// - FP8 (8 bits): 4 elements packed per uint32, so weight_dim = orig_dim / 4
	// scales_dim = orig_dim / group_size
	// So: weight_dim / scales_dim = group_size / pack_factor
	// FP4: ratio = 32/8 = 4
	// FP8: ratio = 32/4 = 8

	// Find a weight/scale pair to check the ratio
	for name := range mw.tensors {
		if !strings.HasSuffix(name, ".weight") || strings.Contains(name, "_scale") || strings.Contains(name, "_qbias") {
			continue
		}
		scaleName := name + "_scale"
		if _, ok := mw.tensors[scaleName]; !ok {
			continue
		}

		// Load both tensors to check shapes
		weightLayer := mw.tensors[name]
		scaleLayer := mw.tensors[scaleName]

		// Get shapes from manifest layer metadata if available
		// For now, default to FP4 since it's more common
		// The actual shape check would require loading the tensor

		// Simple heuristic: check if scale tensor is ~4x smaller than weight
		// FP4: weight is packed 8 per uint32, scales are 1 per group (32)
		// So scale size should be ~weight_size * 8 / 32 = weight_size / 4
		// FP8: weight is packed 4 per uint32, scales are 1 per group (32)
		// So scale size should be ~weight_size * 4 / 32 = weight_size / 8

		// Rough size heuristic (assuming float16 scales)
		// FP4: scale_bytes ≈ weight_bytes / 4 * 2 / 4 = weight_bytes / 8
		// FP8: scale_bytes ≈ weight_bytes / 8 * 2 / 4 = weight_bytes / 16
		ratio := float64(weightLayer.Size) / float64(scaleLayer.Size)
		if ratio < 12 {
			// Closer to 8 = FP4
			return "FP4"
		}
		// Closer to 16 = FP8
		return "FP8"
	}

	// Default to FP4 for affine mode (most common)
	return "FP4"
}

// ReleaseAll frees all native handles and clears the tensor cache.
func (mw *ManifestWeights) ReleaseAll() {
	for _, sf := range mw.nativeCache {
		sf.Free()
	}
	mw.nativeCache = nil
	mw.cache = nil
}
