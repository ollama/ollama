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

// Load loads all tensor blobs using native mmap (zero-copy).
// Blobs are stored in safetensors format for native mlx_load_safetensors mmap.
// If dtype is non-zero, tensors are converted to the specified dtype.
func (mw *ManifestWeights) Load(dtype mlx.Dtype) error {
	for name, layer := range mw.tensors {
		path := mw.manifest.BlobPath(layer.Digest)

		// Load blob as safetensors (native mmap, zero-copy)
		sf, err := mlx.LoadSafetensorsNative(path)
		if err != nil {
			return fmt.Errorf("load %s: %w", name, err)
		}

		// Blob contains single tensor named "data"
		arr := sf.Get("data")
		if arr == nil {
			sf.Free()
			return fmt.Errorf("tensor 'data' not found in blob for %s", name)
		}

		// Convert dtype if needed
		if dtype != 0 && arr.Dtype() != dtype {
			arr = mlx.AsType(arr, dtype)
		}
		// ALWAYS make a contiguous copy to ensure independence from mmap
		arr = mlx.Contiguous(arr)
		mlx.Eval(arr)
		mw.cache[name] = arr
		sf.Free() // Safe to free - arr is now an independent copy
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
// Returns empty string if not quantized or unknown.
func (mw *ManifestWeights) Quantization() string {
	if mw.manifest == nil {
		return ""
	}
	var index struct {
		Quantization string `json:"quantization"`
	}
	if err := mw.manifest.ReadConfigJSON("model_index.json", &index); err != nil {
		return ""
	}
	return index.Quantization
}

// ReleaseAll frees all native handles and clears the tensor cache.
func (mw *ManifestWeights) ReleaseAll() {
	for _, sf := range mw.nativeCache {
		sf.Free()
	}
	mw.nativeCache = nil
	mw.cache = nil
}
