//go:build mlx

package safetensors

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/ollama/ollama/x/imagegen/mlx"
)

// SafetensorHeader represents the JSON header of a safetensors file
type SafetensorHeader map[string]TensorInfo

// TensorInfo contains metadata about a tensor
type TensorInfo struct {
	Dtype       string  `json:"dtype"`
	Shape       []int32 `json:"shape"`
	DataOffsets [2]int  `json:"data_offsets"`
}

// parseSafetensorHeader reads only the JSON header from a safetensors file.
func parseSafetensorHeader(path string) (SafetensorHeader, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %w", err)
	}
	defer f.Close()

	var headerSize uint64
	if err := binary.Read(f, binary.LittleEndian, &headerSize); err != nil {
		return nil, fmt.Errorf("failed to read header size: %w", err)
	}

	headerBytes := make([]byte, headerSize)
	if _, err := f.Read(headerBytes); err != nil {
		return nil, fmt.Errorf("failed to read header: %w", err)
	}

	var header SafetensorHeader
	if err := json.Unmarshal(headerBytes, &header); err != nil {
		return nil, fmt.Errorf("failed to parse header: %w", err)
	}

	delete(header, "__metadata__")
	return header, nil
}

// dtypeFromString converts safetensors dtype string to mlx.Dtype
func dtypeFromString(s string) mlx.Dtype {
	switch strings.ToUpper(s) {
	case "F32", "FLOAT32":
		return mlx.DtypeFloat32
	case "F16", "FLOAT16":
		return mlx.DtypeFloat16
	case "BF16", "BFLOAT16":
		return mlx.DtypeBFloat16
	case "I32", "INT32":
		return mlx.DtypeInt32
	case "I64", "INT64":
		return mlx.DtypeInt64
	case "U8", "UINT8":
		return mlx.DtypeUint8
	default:
		return mlx.DtypeFloat32
	}
}

// ModelWeights manages weights from multiple safetensor files.
type ModelWeights struct {
	dir         string                          // Model directory
	tensorFiles map[string]string               // tensor name -> file path
	tensorInfo  map[string]TensorInfo           // tensor name -> metadata
	nativeCache map[string]*mlx.SafetensorsFile // file path -> loaded native handle
	cache       map[string]*mlx.Array           // tensor name -> array (after Load)
}

// LoadModelWeights scans safetensor files and builds a tensor index.
// This only reads JSON headers, not tensor data.
func LoadModelWeights(dir string) (*ModelWeights, error) {
	mw := &ModelWeights{
		dir:         dir,
		tensorFiles: make(map[string]string),
		tensorInfo:  make(map[string]TensorInfo),
		nativeCache: make(map[string]*mlx.SafetensorsFile),
	}

	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, fmt.Errorf("failed to read directory: %w", err)
	}

	for _, entry := range entries {
		if strings.HasSuffix(entry.Name(), ".safetensors") {
			path := filepath.Join(dir, entry.Name())

			header, err := parseSafetensorHeader(path)
			if err != nil {
				return nil, fmt.Errorf("failed to parse %s: %w", entry.Name(), err)
			}

			for name, info := range header {
				mw.tensorFiles[name] = path
				mw.tensorInfo[name] = info
			}
		}
	}

	if len(mw.tensorFiles) == 0 {
		return nil, fmt.Errorf("no safetensor files found in %s", dir)
	}

	return mw, nil
}

// LoadModelWeightsFromPaths loads weights from specific safetensor file paths.
// Used for loading from blob storage where files are not in a directory.
func LoadModelWeightsFromPaths(paths []string) (*ModelWeights, error) {
	mw := &ModelWeights{
		tensorFiles: make(map[string]string),
		tensorInfo:  make(map[string]TensorInfo),
		nativeCache: make(map[string]*mlx.SafetensorsFile),
	}

	for _, path := range paths {
		header, err := parseSafetensorHeader(path)
		if err != nil {
			return nil, fmt.Errorf("failed to parse %s: %w", path, err)
		}

		for name, info := range header {
			mw.tensorFiles[name] = path
			mw.tensorInfo[name] = info
		}
	}

	if len(mw.tensorFiles) == 0 {
		return nil, fmt.Errorf("no tensors found in provided paths")
	}

	return mw, nil
}

// Load loads all tensors into cache with the specified dtype.
// If dtype is 0, tensors are loaded in their original dtype.
// Automatically uses streaming (memory-efficient) when dtype conversion is needed,
// or native loading when tensors are already in the target dtype.
func (mw *ModelWeights) Load(dtype mlx.Dtype) error {
	if dtype == 0 {
		return mw.loadNative()
	}

	// Check if any tensor needs conversion
	needsConversion := false
	for name := range mw.tensorFiles {
		info := mw.tensorInfo[name]
		if dtypeFromString(info.Dtype) != dtype {
			needsConversion = true
			break
		}
	}

	if needsConversion {
		return mw.loadStreaming(dtype)
	}
	return mw.loadNative()
}

// loadNative loads all tensors using the native memory-mapped loader.
func (mw *ModelWeights) loadNative() error {
	mw.cache = make(map[string]*mlx.Array)

	fileToTensors := make(map[string][]string)
	for name, path := range mw.tensorFiles {
		fileToTensors[path] = append(fileToTensors[path], name)
	}

	for path, names := range fileToTensors {
		native, err := mlx.LoadSafetensorsNative(path)
		if err != nil {
			return fmt.Errorf("failed to load %s: %w", path, err)
		}

		for _, name := range names {
			arr := native.Get(name)
			if arr == nil {
				native.Free()
				return fmt.Errorf("tensor %q not found in %s", name, path)
			}
			mw.cache[name] = arr
		}

		mw.nativeCache[path] = native
	}

	return nil
}

// loadStreaming loads tensors with dtype conversion.
// Uses the same pattern as Python: replace each entry in the map after conversion,
// so the original tensor loses its reference and can be freed.
func (mw *ModelWeights) loadStreaming(dtype mlx.Dtype) error {
	mw.cache = make(map[string]*mlx.Array)

	fileToTensors := make(map[string][]string)
	for name, path := range mw.tensorFiles {
		fileToTensors[path] = append(fileToTensors[path], name)
	}

	for path, names := range fileToTensors {
		native, err := mlx.LoadSafetensorsNative(path)
		if err != nil {
			return fmt.Errorf("failed to load %s: %w", path, err)
		}

		for _, name := range names {
			src := native.Get(name)
			if src == nil {
				native.Free()
				return fmt.Errorf("tensor %q not found in %s", name, path)
			}

			dst := mlx.AsType(src, dtype)
			mlx.Eval(dst)
			native.Set(name, dst)
			mw.cache[name] = dst
		}

		native.Free()
	}

	return nil
}

// Get returns a tensor from cache. Call Load() first.
func (mw *ModelWeights) Get(name string) (*mlx.Array, error) {
	if mw.cache == nil {
		return nil, fmt.Errorf("cache not initialized: call Load() first")
	}
	arr, ok := mw.cache[name]
	if !ok {
		return nil, fmt.Errorf("tensor %q not found in cache", name)
	}
	return arr, nil
}

// GetTensor loads a tensor using the native loader without caching.
// For bulk loading, use Load() + Get() instead.
func (mw *ModelWeights) GetTensor(name string) (*mlx.Array, error) {
	if mw.cache != nil {
		if arr, ok := mw.cache[name]; ok {
			return arr, nil
		}
	}

	path, ok := mw.tensorFiles[name]
	if !ok {
		return nil, fmt.Errorf("tensor %q not found", name)
	}

	native, ok := mw.nativeCache[path]
	if !ok {
		var err error
		native, err = mlx.LoadSafetensorsNative(path)
		if err != nil {
			return nil, fmt.Errorf("failed to load %s: %w", path, err)
		}
		mw.nativeCache[path] = native
	}

	return native.Get(name), nil
}

// GetTensorInfo returns metadata about a tensor without loading it.
func (mw *ModelWeights) GetTensorInfo(name string) (TensorInfo, bool) {
	info, ok := mw.tensorInfo[name]
	return info, ok
}

// ListTensors returns all tensor names.
func (mw *ModelWeights) ListTensors() []string {
	names := make([]string, 0, len(mw.tensorFiles))
	for name := range mw.tensorFiles {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

// HasTensor checks if a tensor exists.
func (mw *ModelWeights) HasTensor(name string) bool {
	_, ok := mw.tensorFiles[name]
	return ok
}

// Quantization returns empty string for directory-based weights (not quantized).
func (mw *ModelWeights) Quantization() string {
	return ""
}

// ReleaseAll releases all cached native file handles.
func (mw *ModelWeights) ReleaseAll() {
	for path, native := range mw.nativeCache {
		native.Free()
		delete(mw.nativeCache, path)
	}
}

