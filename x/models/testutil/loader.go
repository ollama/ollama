package testutil

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/ollama/ollama/x/imagegen/manifest"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model"
	"github.com/ollama/ollama/x/mlxrunner/model/base"
)

// LoadModelFromDirOrErr is the *testing.T-free counterpart of
// LoadModelFromDir. It builds a synthetic manifest pointing at a HuggingFace-
// format model directory (config.json + tokenizer.json + *.safetensors), runs
// the architecture's registered factory, loads weights, and returns the
// initialized model. Symlinks are placed in tmpBlobDir; the caller owns
// cleanup of that directory.
func LoadModelFromDirOrErr(modelDir, tmpBlobDir string) (base.Model, error) {
	if err := mlx.CheckInit(); err != nil {
		return nil, fmt.Errorf("MLX not available: %w", err)
	}

	if _, err := os.Stat(modelDir); err != nil {
		return nil, fmt.Errorf("model dir %q: %w", modelDir, err)
	}

	if err := os.MkdirAll(tmpBlobDir, 0o755); err != nil {
		return nil, fmt.Errorf("create blob dir: %w", err)
	}

	var layers []manifest.ManifestLayer

	// Link config files (only the first two are required).
	for _, name := range []string{
		"config.json",
		"tokenizer.json",
		"generation_config.json",
		"tokenizer_config.json",
	} {
		src := filepath.Join(modelDir, name)
		if _, err := os.Stat(src); err != nil {
			if name == "config.json" || name == "tokenizer.json" {
				return nil, fmt.Errorf("required file missing: %s", src)
			}
			continue
		}
		digest := "sha256:" + name
		dst := filepath.Join(tmpBlobDir, "sha256-"+name)
		_ = os.Remove(dst) // tolerate stale symlinks from prior runs
		if err := os.Symlink(src, dst); err != nil {
			return nil, fmt.Errorf("symlink %s: %w", name, err)
		}
		layers = append(layers, manifest.ManifestLayer{
			MediaType: "application/vnd.ollama.image.json",
			Digest:    digest,
			Name:      name,
		})
	}

	// Link weight files.
	matches, _ := filepath.Glob(filepath.Join(modelDir, "*.safetensors"))
	sort.Strings(matches)
	for _, src := range matches {
		name := filepath.Base(src)
		digest := "sha256:" + name
		dst := filepath.Join(tmpBlobDir, "sha256-"+name)
		_ = os.Remove(dst)
		if err := os.Symlink(src, dst); err != nil {
			return nil, fmt.Errorf("symlink %s: %w", name, err)
		}
		layers = append(layers, manifest.ManifestLayer{
			MediaType: "application/vnd.ollama.image.tensor",
			Digest:    digest,
			Name:      name,
		})
	}

	mm := &manifest.ModelManifest{
		Manifest: &manifest.Manifest{
			SchemaVersion: 2,
			Layers:        layers,
		},
		BlobDir: tmpBlobDir,
	}
	root := &model.Root{Manifest: mm}

	m, err := base.New(root)
	if err != nil {
		return nil, fmt.Errorf("base.New: %w", err)
	}

	tensors := loadTensorsFromManifestPlain(root)
	if len(tensors) == 0 {
		return nil, fmt.Errorf("no tensors loaded from manifest")
	}

	if err := base.Weights(m)(tensors); err != nil {
		return nil, fmt.Errorf("LoadWeights: %w", err)
	}
	return m, nil
}

// LoadModelByNameOrErr is the *testing.T-free counterpart of LoadModelByName.
// It opens an ollama model from the local store by tag (e.g.
// "gemma4:e2b-base-mlx-bf16") and returns the initialized model. The caller
// is responsible for closing the underlying root via the returned closer.
func LoadModelByNameOrErr(modelName string) (base.Model, func(), error) {
	if err := mlx.CheckInit(); err != nil {
		return nil, func() {}, fmt.Errorf("MLX not available: %w", err)
	}
	if modelName == "" {
		return nil, func() {}, fmt.Errorf("model name is required")
	}

	root, err := model.Open(modelName)
	if err != nil {
		return nil, func() {}, fmt.Errorf("open model %q: %w", modelName, err)
	}
	cleanup := func() { root.Close() }

	m, err := base.New(root)
	if err != nil {
		cleanup()
		return nil, func() {}, fmt.Errorf("base.New(%s): %w", modelName, err)
	}

	tensors := loadTensorsFromManifestPlain(root)
	if len(tensors) == 0 {
		cleanup()
		return nil, func() {}, fmt.Errorf("no tensors loaded for %q", modelName)
	}

	if err := base.Weights(m)(tensors); err != nil {
		cleanup()
		return nil, func() {}, fmt.Errorf("LoadWeights(%s): %w", modelName, err)
	}
	return m, cleanup, nil
}

// loadTensorsFromManifestPlain mirrors loadTensorsFromManifest but does not
// require *testing.T. The two share the same key-remap algorithm so callers
// from CLI tools and tests behave identically.
func loadTensorsFromManifestPlain(root *model.Root) map[string]*mlx.Array {
	rawTensors := make(map[string]*mlx.Array)
	seen := make(map[string]bool)
	for _, layer := range root.Manifest.GetTensorLayers("") {
		if seen[layer.Digest] {
			continue
		}
		seen[layer.Digest] = true
		blobPath := root.Manifest.BlobPath(layer.Digest)
		for name, arr := range mlx.Load(blobPath) {
			rawTensors[name] = arr
		}
	}

	scaleBaseNames := make(map[string]bool)
	allTensors := make(map[string]*mlx.Array, len(rawTensors))
	for name, arr := range rawTensors {
		if strings.HasSuffix(name, ".scale") {
			baseName := strings.TrimSuffix(name, ".scale")
			allTensors[baseName+"_scale"] = arr
			scaleBaseNames[baseName] = true
		}
	}
	for name, arr := range rawTensors {
		if strings.HasSuffix(name, ".scale") {
			continue
		}
		if strings.HasSuffix(name, ".bias") && !strings.HasSuffix(name, ".weight_qbias") {
			baseName := strings.TrimSuffix(name, ".bias")
			if scaleBaseNames[baseName] {
				allTensors[baseName+"_qbias"] = arr
			} else {
				allTensors[name] = arr
			}
		} else {
			allTensors[name] = arr
		}
	}
	return allTensors
}
