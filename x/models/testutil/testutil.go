// Package testutil provides reusable test utilities for validating MLX model
// implementations against Python reference outputs.
//
// Typical workflow:
//  1. Generate reference activations with x/models/scripts/dump_activations.py
//  2. Load reference data with LoadReference
//  3. Load model with LoadModelFromDir (BF16) or LoadModelByName (quantized)
//  4. Compare layer-by-layer with CompareArrays or CompareManyArrays
package testutil

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"

	"github.com/ollama/ollama/x/imagegen/manifest"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model"
	"github.com/ollama/ollama/x/mlxrunner/model/base"
)

// SkipIfNoMLX skips the test if the MLX dynamic library is not available.
// Centralizes the pattern currently duplicated across multiple test files.
func SkipIfNoMLX(t *testing.T) {
	t.Helper()
	if err := mlx.CheckInit(); err != nil {
		t.Skipf("MLX not available: %v", err)
	}
}

// DefaultRefDir returns /tmp/ollama_ref/<model>/ which matches the default
// output location of dump_activations.py. Override with OLLAMA_REF_DIR env var.
func DefaultRefDir(model string) string {
	if dir := os.Getenv("OLLAMA_REF_DIR"); dir != "" {
		return filepath.Join(dir, model)
	}
	return filepath.Join("/tmp", "ollama_ref", model)
}

// ModelDir returns a model directory path from an env var or default path.
// Skips the test if the directory does not exist.
func ModelDir(t *testing.T, envVar, defaultPath string) string {
	t.Helper()
	dir := os.Getenv(envVar)
	if dir == "" {
		dir = defaultPath
		if strings.HasPrefix(dir, "~/") {
			home, _ := os.UserHomeDir()
			dir = filepath.Join(home, dir[2:])
		}
	}
	if _, err := os.Stat(dir); err != nil {
		t.Skipf("model directory not available (%s=%s): %v", envVar, dir, err)
	}
	return dir
}

// LoadReference loads all tensors from a safetensors file.
// Skips the test if the file does not exist (reference not yet generated).
func LoadReference(t *testing.T, path string) map[string]*mlx.Array {
	t.Helper()
	return LoadReferenceFiltered(t, path, nil)
}

// LoadReferenceFiltered is like LoadReference but only retains tensors whose
// names are in the keep set. Tensors not in the set are dropped via Sweep
// after the file is loaded. If keep is nil or empty, every tensor is loaded
// (equivalent to LoadReference).
//
// Use this when an activation dump is large and you only need a subset of
// keys; loading the full file into pinned memory may exceed available unified
// memory on the test machine.
func LoadReferenceFiltered(t *testing.T, path string, keep map[string]bool) map[string]*mlx.Array {
	t.Helper()
	SkipIfNoMLX(t)

	if _, err := os.Stat(path); err != nil {
		t.Skipf("reference data not available: %s", path)
	}

	tensors := make(map[string]*mlx.Array)
	for name, arr := range mlx.Load(path) {
		if len(keep) > 0 && !keep[name] {
			continue
		}
		tensors[name] = arr
	}
	if len(tensors) == 0 {
		t.Fatalf("no tensors loaded from %s", path)
	}

	// Pin the kept arrays so the post-load Sweep doesn't free them, then
	// run Sweep to discard the unpinned ones produced by mlx.Load. Without
	// this, the dropped tensors would still occupy memory until the next
	// model-driven Sweep().
	arrays := make([]*mlx.Array, 0, len(tensors))
	for _, arr := range tensors {
		arrays = append(arrays, arr)
	}
	mlx.Pin(arrays...)
	mlx.Sweep()
	mlx.Eval(arrays...)

	return tensors
}

// CompareOption configures tensor comparison behavior.
type CompareOption func(*compareConfig)

type compareConfig struct {
	atol float32
	rtol float32
}

func defaultConfig() compareConfig {
	return compareConfig{atol: 1e-4, rtol: 1e-3}
}

// BFloat16Tol sets tolerances appropriate for BFloat16 precision.
func BFloat16Tol() CompareOption {
	return func(c *compareConfig) { c.atol = 5e-3; c.rtol = 5e-3 }
}

// Float32Tol sets tight tolerances for float32 precision.
func Float32Tol() CompareOption {
	return func(c *compareConfig) { c.atol = 1e-5; c.rtol = 1e-5 }
}

// QuantizedTol sets loose tolerances for quantized model comparison.
func QuantizedTol() CompareOption {
	return func(c *compareConfig) { c.atol = 1e-2; c.rtol = 1e-2 }
}

// WithTolerance sets custom absolute and relative tolerances.
func WithTolerance(atol, rtol float32) CompareOption {
	return func(c *compareConfig) { c.atol = atol; c.rtol = rtol }
}

// CompareEntry holds the result of comparing two tensors.
type CompareEntry struct {
	Name      string
	Passed    bool
	MaxDiff   float32
	MaxDiffAt int
	MeanDiff  float32
	CosineSim float32
	GotShape  []int
	WantShape []int
}

// CompareReport holds results from comparing multiple tensor pairs.
type CompareReport struct {
	Entries []CompareEntry
}

// AllPassed returns true if every comparison passed.
func (r CompareReport) AllPassed() bool {
	for _, e := range r.Entries {
		if !e.Passed {
			return false
		}
	}
	return true
}

// Summary logs a table of all comparisons to t.
func (r CompareReport) Summary(t *testing.T) {
	t.Helper()
	t.Logf("%-40s %-6s %12s %12s %10s", "Name", "Status", "MaxDiff", "MeanDiff", "CosineSim")
	t.Logf("%-40s %-6s %12s %12s %10s", strings.Repeat("-", 40), "------", "------------", "------------", "----------")
	for _, e := range r.Entries {
		status := "PASS"
		if !e.Passed {
			status = "FAIL"
		}
		t.Logf("%-40s %-6s %12.6f %12.6f %10.6f", e.Name, status, e.MaxDiff, e.MeanDiff, e.CosineSim)
	}
}

// compareArraysInner performs the actual comparison and returns the entry.
func compareArraysInner(name string, got, want *mlx.Array, cfg compareConfig) CompareEntry {
	entry := CompareEntry{
		Name:      name,
		GotShape:  got.Dims(),
		WantShape: want.Dims(),
	}

	// Shape check
	gotDims := got.Dims()
	wantDims := want.Dims()
	if len(gotDims) != len(wantDims) {
		return entry
	}
	for i := range gotDims {
		if gotDims[i] != wantDims[i] {
			return entry
		}
	}

	// Cast to float32 for comparison -- Floats() requires float32 data.
	gotF32 := got.AsType(mlx.DTypeFloat32)
	wantF32 := want.AsType(mlx.DTypeFloat32)
	mlx.Eval(gotF32, wantF32)
	gotData := gotF32.Floats()
	wantData := wantF32.Floats()

	if len(gotData) != len(wantData) {
		return entry
	}

	var maxDiff, sumDiff float32
	maxDiffAt := 0
	var dotProduct, gotNorm, wantNorm float64

	for i := range gotData {
		diff := float32(math.Abs(float64(gotData[i] - wantData[i])))
		if diff > maxDiff {
			maxDiff = diff
			maxDiffAt = i
		}
		sumDiff += diff
		dotProduct += float64(gotData[i]) * float64(wantData[i])
		gotNorm += float64(gotData[i]) * float64(gotData[i])
		wantNorm += float64(wantData[i]) * float64(wantData[i])
	}

	n := len(gotData)
	entry.MaxDiff = maxDiff
	entry.MaxDiffAt = maxDiffAt
	entry.MeanDiff = sumDiff / float32(n)

	denom := math.Sqrt(gotNorm) * math.Sqrt(wantNorm)
	if denom > 0 {
		entry.CosineSim = float32(dotProduct / denom)
	}

	// Check if within tolerance: |got - want| <= atol + rtol * |want|
	entry.Passed = true
	for i := range gotData {
		threshold := cfg.atol + cfg.rtol*float32(math.Abs(float64(wantData[i])))
		if float32(math.Abs(float64(gotData[i]-wantData[i]))) > threshold {
			entry.Passed = false
			break
		}
	}

	return entry
}

// CompareArrays compares two MLX arrays with detailed error reporting.
// Returns true if the arrays match within the configured tolerance.
func CompareArrays(t *testing.T, name string, got, want *mlx.Array, opts ...CompareOption) bool {
	t.Helper()
	cfg := defaultConfig()
	for _, opt := range opts {
		opt(&cfg)
	}

	entry := compareArraysInner(name, got, want, cfg)

	if !entry.Passed {
		t.Errorf("%s mismatch:", name)

		// Shape mismatch
		if fmt.Sprint(entry.GotShape) != fmt.Sprint(entry.WantShape) {
			t.Errorf("  shape: got %v, want %v", entry.GotShape, entry.WantShape)
			return false
		}

		t.Errorf("  shape:      %v", entry.GotShape)
		t.Errorf("  max_diff:   %.6f (at index %d)", entry.MaxDiff, entry.MaxDiffAt)
		t.Errorf("  mean_diff:  %.6f", entry.MeanDiff)
		t.Errorf("  cosine_sim: %.6f", entry.CosineSim)

		// Print first 5 values
		gf := got.AsType(mlx.DTypeFloat32)
		wf := want.AsType(mlx.DTypeFloat32)
		mlx.Eval(gf, wf)
		gotData := gf.Floats()
		wantData := wf.Floats()
		n := min(5, len(gotData))
		gotFirst := make([]string, n)
		wantFirst := make([]string, n)
		for i := range n {
			gotFirst[i] = fmt.Sprintf("%.4f", gotData[i])
			wantFirst[i] = fmt.Sprintf("%.4f", wantData[i])
		}
		t.Errorf("  got  first%d: [%s]", n, strings.Join(gotFirst, ", "))
		t.Errorf("  want first%d: [%s]", n, strings.Join(wantFirst, ", "))
	}

	return entry.Passed
}

// CompareArraysCosineSim asserts that the cosine similarity between got and
// want is at least minCosineSim. Prefer this for accumulated forward-pass
// outputs (anything that has been through several layers and a final
// per-channel scaled normalization): cosine similarity catches direction
// errors and ignores per-channel scale, where element-wise atol gives
// alarming false positives. Use CompareArrays with a tight tolerance for
// single-op tests where element-wise comparison is meaningful.
//
// Note: cosine similarity does not detect a uniform scale error.
func CompareArraysCosineSim(t *testing.T, name string, got, want *mlx.Array, minCosineSim float32) bool {
	t.Helper()
	entry := CompareEntry{
		Name:      name,
		GotShape:  got.Dims(),
		WantShape: want.Dims(),
	}

	// Shape mismatch is always a failure regardless of cosine sim.
	if fmt.Sprint(entry.GotShape) != fmt.Sprint(entry.WantShape) {
		t.Errorf("%s shape mismatch: got %v, want %v", name, entry.GotShape, entry.WantShape)
		return false
	}

	total := got.Size()
	if total == 0 {
		entry.Passed = true
		entry.CosineSim = 1
		return true
	}

	gotFlat := mlx.Reshape(got, 1, int32(total))
	wantFlat := mlx.Reshape(want, 1, int32(total))
	mlx.Pin(gotFlat, wantFlat)
	defer func() {
		mlx.Unpin(gotFlat, wantFlat)
		mlx.Sweep()
	}()

	const chunkElems = 1 << 20
	var maxDiff, sumDiff float32
	maxDiffAt := 0
	var dotProduct, gotNorm, wantNorm float64

	for start := 0; start < total; start += chunkElems {
		end := min(start+chunkElems, total)
		gotChunk := gotFlat.Slice(mlx.Slice(), mlx.Slice(start, end))
		wantChunk := wantFlat.Slice(mlx.Slice(), mlx.Slice(start, end))
		gotF32 := gotChunk.AsType(mlx.DTypeFloat32)
		wantF32 := wantChunk.AsType(mlx.DTypeFloat32)
		mlx.Eval(gotF32, wantF32)
		gotData := gotF32.Floats()
		wantData := wantF32.Floats()
		for i := range gotData {
			diff := float32(math.Abs(float64(gotData[i] - wantData[i])))
			if diff > maxDiff {
				maxDiff = diff
				maxDiffAt = start + i
			}
			sumDiff += diff
			dotProduct += float64(gotData[i]) * float64(wantData[i])
			gotNorm += float64(gotData[i]) * float64(gotData[i])
			wantNorm += float64(wantData[i]) * float64(wantData[i])
		}
		mlx.Sweep()
	}

	entry.MaxDiff = maxDiff
	entry.MaxDiffAt = maxDiffAt
	entry.MeanDiff = sumDiff / float32(total)
	denom := math.Sqrt(gotNorm) * math.Sqrt(wantNorm)
	if denom > 0 {
		entry.CosineSim = float32(dotProduct / denom)
	}
	if entry.CosineSim >= minCosineSim {
		return true
	}

	t.Errorf("%s cosine similarity check failed:", name)
	t.Errorf("  cosine_sim: %.6f (want >= %.6f)", entry.CosineSim, minCosineSim)
	t.Errorf("  shape:      %v", entry.GotShape)
	t.Errorf("  max_diff:   %.6f (at index %d)", entry.MaxDiff, entry.MaxDiffAt)
	t.Errorf("  mean_diff:  %.6f", entry.MeanDiff)
	return false
}

// CompareManyArrays compares all tensors present in both got and want maps.
// Keys are sorted so layers are compared in order. Returns a report.
func CompareManyArrays(t *testing.T, got, want map[string]*mlx.Array, opts ...CompareOption) CompareReport {
	t.Helper()
	cfg := defaultConfig()
	for _, opt := range opts {
		opt(&cfg)
	}

	// Find common keys
	var keys []string
	for k := range want {
		if _, ok := got[k]; ok {
			keys = append(keys, k)
		}
	}
	sort.Strings(keys)

	var report CompareReport
	for _, k := range keys {
		entry := compareArraysInner(k, got[k], want[k], cfg)
		report.Entries = append(report.Entries, entry)
	}

	return report
}

// LoadModelFromDir loads any registered model architecture from a
// HuggingFace-format directory containing config.json, tokenizer.json,
// and *.safetensors weight files. Suitable for BF16 reference models.
func LoadModelFromDir(t *testing.T, modelDir string) base.Model {
	t.Helper()
	SkipIfNoMLX(t)

	// Build a synthetic manifest pointing to the real files via symlinks.
	blobDir := t.TempDir()
	var layers []manifest.ManifestLayer

	// Link config files
	for _, name := range []string{
		"config.json",
		"tokenizer.json",
		"generation_config.json",
		"tokenizer_config.json",
	} {
		src := filepath.Join(modelDir, name)
		if _, err := os.Stat(src); err != nil {
			if name == "config.json" || name == "tokenizer.json" {
				t.Fatalf("required file missing: %s", src)
			}
			continue
		}
		digest := "sha256:" + name
		blobName := "sha256-" + name
		if err := os.Symlink(src, filepath.Join(blobDir, blobName)); err != nil {
			t.Fatalf("symlink %s: %v", name, err)
		}
		layers = append(layers, manifest.ManifestLayer{
			MediaType: "application/vnd.ollama.image.json",
			Digest:    digest,
			Name:      name,
		})
	}

	// Link safetensors files
	matches, _ := filepath.Glob(filepath.Join(modelDir, "*.safetensors"))
	sort.Strings(matches)
	for _, src := range matches {
		name := filepath.Base(src)
		digest := "sha256:" + name
		blobName := "sha256-" + name
		if err := os.Symlink(src, filepath.Join(blobDir, blobName)); err != nil {
			t.Fatalf("symlink %s: %v", name, err)
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
		BlobDir: blobDir,
	}

	root := &model.Root{Manifest: mm}
	t.Cleanup(root.Close)
	m, err := base.New(root)
	if err != nil {
		t.Fatalf("base.New: %v", err)
	}

	// Load all safetensors and remap keys for quantized models
	tensors := loadTensorsFromManifest(t, root)

	loadFn := base.Weights(m)
	if err := loadFn(tensors); err != nil {
		t.Fatalf("LoadWeights: %v", err)
	}

	return m
}

// LoadModelByName loads an ollama model by name (e.g., "gemma4:e2b-nvfp4").
// The model must already exist in the local ollama model store.
func LoadModelByName(t *testing.T, modelName string) base.Model {
	t.Helper()
	SkipIfNoMLX(t)

	if modelName == "" {
		t.Skip("no model name provided")
	}

	root, err := model.Open(modelName)
	if err != nil {
		t.Skipf("model %q not available: %v", modelName, err)
	}
	t.Cleanup(root.Close)

	m, err := base.New(root)
	if err != nil {
		t.Fatalf("base.New(%s): %v", modelName, err)
	}

	tensors := loadTensorsFromManifest(t, root)

	loadFn := base.Weights(m)
	if err := loadFn(tensors); err != nil {
		t.Fatalf("LoadWeights(%s): %v", modelName, err)
	}

	return m
}

// loadTensorsFromManifest loads all tensor blobs from a manifest, deduplicating
// by digest and remapping safetensors key suffixes (.scale → _scale, .bias → _qbias).
func loadTensorsFromManifest(t *testing.T, root *model.Root) map[string]*mlx.Array {
	t.Helper()

	// Phase 1: Load all tensors raw
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

	// Phase 2: Identify quantized base names (.scale entries)
	scaleBaseNames := make(map[string]bool)
	allTensors := make(map[string]*mlx.Array, len(rawTensors))
	for name, arr := range rawTensors {
		if strings.HasSuffix(name, ".scale") {
			baseName := strings.TrimSuffix(name, ".scale")
			allTensors[baseName+"_scale"] = arr
			scaleBaseNames[baseName] = true
		}
	}

	// Phase 3: Remap remaining tensors
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

	if len(allTensors) == 0 {
		t.Fatalf("no tensors loaded from manifest")
	}
	return allTensors
}
