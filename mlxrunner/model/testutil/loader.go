package testutil

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/model"
)

// LoadModelFromDirOrErr is the *testing.T-free counterpart of
// LoadModelFromDir. It builds a synthetic manifest pointing at a HuggingFace-
// format model directory (config.json + tokenizer.json + *.safetensors), runs
// the architecture's registered factory, loads weights, and returns the
// initialized model.
//
// The manifest package resolves blobs under OLLAMA_MODELS, so storeDir is
// used as a throwaway model store: every file is symlinked into
// storeDir/blobs and OLLAMA_MODELS points at storeDir for the duration of
// the load (restored before returning). The caller owns cleanup of storeDir.
// Must be called on the MLX thread.
//
// CAUTION: this bypasses the create-path import transform. Architectures
// whose runtime layout depends on it (renamed or synthesized tensors) load
// without error here but produce silently wrong outputs. Validate a new
// architecture through a created tag (LoadModelByNameOrErr) before trusting
// directory loading for it.
func LoadModelFromDirOrErr(modelDir, storeDir string) (model.Model, error) {
	if err := mlx.CheckInit(); err != nil {
		return nil, fmt.Errorf("MLX not available: %w", err)
	}

	if _, err := os.Stat(modelDir); err != nil {
		return nil, fmt.Errorf("model dir %q: %w", modelDir, err)
	}
	for _, name := range []string{"config.json", "tokenizer.json"} {
		if _, err := os.Stat(filepath.Join(modelDir, name)); err != nil {
			return nil, fmt.Errorf("required file missing: %s", filepath.Join(modelDir, name))
		}
	}

	restore, err := useModelStore(storeDir)
	if err != nil {
		return nil, err
	}
	defer restore()

	var layers []manifest.Layer

	// Link every top-level config file (config.json, tokenizer files,
	// generation/processor configs, chat templates); models read whichever
	// they need through Manifest.ReadConfig.
	var configs []string
	for _, pattern := range []string{"*.json", "*.jinja"} {
		matches, _ := filepath.Glob(filepath.Join(modelDir, pattern))
		configs = append(configs, matches...)
	}
	sort.Strings(configs)
	for _, src := range configs {
		name := filepath.Base(src)
		if strings.HasSuffix(name, ".safetensors.index.json") {
			continue
		}
		digest, err := linkBlob(src)
		if err != nil {
			return nil, err
		}
		layers = append(layers, manifest.Layer{
			MediaType: "application/vnd.ollama.image.json",
			Digest:    digest,
			Name:      name,
		})
	}

	// Link weight files.
	weights, _ := filepath.Glob(filepath.Join(modelDir, "*.safetensors"))
	sort.Strings(weights)
	for _, src := range weights {
		digest, err := linkBlob(src)
		if err != nil {
			return nil, err
		}
		layers = append(layers, manifest.Layer{
			MediaType: manifest.MediaTypeImageTensor,
			Digest:    digest,
			Name:      filepath.Base(src),
		})
	}

	root := &model.Root{Manifest: &manifest.Manifest{SchemaVersion: 2, Layers: layers}}
	return newModel(root, modelDir)
}

// LoadModelByNameOrErr is the *testing.T-free counterpart of LoadModelByName.
// It opens an ollama model from the local store by tag (e.g.
// "gemma4:e2b-base-mlx-bf16") and returns the initialized model. Must be
// called on the MLX thread. The returned closer is kept for API stability;
// the store needs no cleanup.
func LoadModelByNameOrErr(modelName string) (model.Model, func(), error) {
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

	m, err := newModel(root, modelName)
	if err != nil {
		return nil, func() {}, err
	}
	return m, func() {}, nil
}

// newModel constructs the registered architecture for root and loads its
// weights the way the runner does: inside a function scope that returns only
// the arrays the model kept (mlx.Collect), so manifest tensors the model did
// not consume are freed. The kept weights move to the caller's scope — the
// root scope for a CLI or a test body, i.e. alive for the process.
func newModel(root *model.Root, label string) (model.Model, error) {
	var (
		m   model.Model
		err error
	)
	weights := mlx.ScopedArrays(func() []*mlx.Array {
		m, err = model.New(root)
		if err != nil {
			err = fmt.Errorf("model.New(%s): %w", label, err)
			return nil
		}

		tensors, e := loadTensorsFromManifest(root)
		if e != nil {
			err = e
			return nil
		}
		if len(tensors) == 0 {
			err = fmt.Errorf("no tensors loaded for %s", label)
			return nil
		}

		if e := m.LoadWeights(tensors); e != nil {
			err = fmt.Errorf("LoadWeights(%s): %w", label, e)
			return nil
		}
		return mlx.Collect(m)
	})
	if err != nil {
		return nil, err
	}
	mlx.Eval(weights...)
	return m, nil
}

// useModelStore points OLLAMA_MODELS at storeDir and returns a func that
// restores the previous value.
func useModelStore(storeDir string) (func(), error) {
	if err := os.MkdirAll(filepath.Join(storeDir, "blobs"), 0o755); err != nil {
		return nil, fmt.Errorf("create model store: %w", err)
	}
	prev, had := os.LookupEnv("OLLAMA_MODELS")
	if err := os.Setenv("OLLAMA_MODELS", storeDir); err != nil {
		return nil, err
	}
	return func() {
		if had {
			os.Setenv("OLLAMA_MODELS", prev)
		} else {
			os.Unsetenv("OLLAMA_MODELS")
		}
	}, nil
}

// linkBlob symlinks src into the current model store under a synthetic
// digest derived from its absolute path (not its contents, which would mean
// hashing every weight file) and returns that digest.
func linkBlob(src string) (string, error) {
	abs, err := filepath.Abs(src)
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256([]byte(abs))
	digest := "sha256:" + hex.EncodeToString(sum[:])
	dst, err := manifest.BlobsPath(digest)
	if err != nil {
		return "", err
	}
	_ = os.Remove(dst) // tolerate stale symlinks from prior runs
	if err := os.Symlink(abs, dst); err != nil {
		return "", fmt.Errorf("symlink %s: %w", filepath.Base(src), err)
	}
	return digest, nil
}

// loadTensorsFromManifest loads all tensor blobs from a manifest,
// deduplicating by digest and remapping safetensors key suffixes
// (.scale -> _scale, .bias -> _qbias) the same way the runner does.
func loadTensorsFromManifest(root *model.Root) (map[string]*mlx.Array, error) {
	rawTensors := make(map[string]*mlx.Array)
	seen := make(map[string]bool)
	for _, layer := range root.Manifest.TensorLayers() {
		if seen[layer.Digest] {
			continue
		}
		seen[layer.Digest] = true
		blobPath, err := manifest.BlobsPath(layer.Digest)
		if err != nil {
			return nil, err
		}
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
	return allTensors, nil
}
