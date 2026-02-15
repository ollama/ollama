//go:build mlx

package client

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"

	"github.com/ollama/ollama/x/create"
	"github.com/ollama/ollama/x/imagegen/mlx"
)

// quantizeParams maps quantization type names to MLX quantize parameters.
var quantizeParams = map[string]struct {
	groupSize int
	bits      int
	mode      string
}{
	"int4":  {32, 4, "affine"},
	"nvfp4": {16, 4, "nvfp4"},
	"int8":  {64, 8, "affine"},
	"mxfp8": {32, 8, "mxfp8"},
}

// loadAndQuantizeArray writes a safetensors reader to a temp file, loads it with MLX,
// quantizes the tensor, and appends the resulting arrays (weight, scale, optional bias)
// to the provided maps. If quantize is empty, the tensor is kept as-is.
// Returns any temp file paths created (caller must clean up) and arrays needing eval.
func loadAndQuantizeArray(r io.Reader, name, quantize string, arrays map[string]*mlx.Array) (tmpPath string, toEval []*mlx.Array, nativeHandle *mlx.SafetensorsFile, err error) {
	tmpDir := ensureTempDir()

	tmpFile, err := os.CreateTemp(tmpDir, "quant-*.safetensors")
	if err != nil {
		return "", nil, nil, fmt.Errorf("failed to create temp file: %w", err)
	}
	tmpPath = tmpFile.Name()

	if _, err := io.Copy(tmpFile, r); err != nil {
		tmpFile.Close()
		return tmpPath, nil, nil, fmt.Errorf("failed to write temp file for %s: %w", name, err)
	}
	tmpFile.Close()

	st, err := mlx.LoadSafetensorsNative(tmpPath)
	if err != nil {
		return tmpPath, nil, nil, fmt.Errorf("failed to load safetensors for %s: %w", name, err)
	}

	// Find the tensor key (may differ from name for single-tensor blobs)
	inputKey, err := findSafetensorsKey(tmpPath)
	if err != nil {
		st.Free()
		return tmpPath, nil, nil, fmt.Errorf("failed to read blob header for %s: %w", name, err)
	}

	arr := st.Get(inputKey)
	if arr == nil {
		st.Free()
		return tmpPath, nil, nil, fmt.Errorf("tensor %q not found in safetensors", inputKey)
	}

	if quantize == "" {
		arr = mlx.Contiguous(arr)
		arrays[name] = arr
		return tmpPath, []*mlx.Array{arr}, st, nil
	}

	// Convert to float type if needed (quantize expects float)
	if arr.Dtype() != mlx.DtypeBFloat16 && arr.Dtype() != mlx.DtypeFloat32 && arr.Dtype() != mlx.DtypeFloat16 {
		arr = mlx.AsType(arr, mlx.DtypeBFloat16)
		mlx.Eval(arr)
	}

	params, ok := quantizeParams[quantize]
	if !ok {
		st.Free()
		return tmpPath, nil, nil, fmt.Errorf("unsupported quantization type: %s", quantize)
	}

	qweight, scales, qbiases := mlx.Quantize(arr, params.groupSize, params.bits, params.mode)

	qweight = mlx.Contiguous(qweight)
	scales = mlx.Contiguous(scales)
	arrays[name] = qweight
	arrays[name+".scale"] = scales
	toEval = append(toEval, qweight, scales)

	if qbiases != nil {
		qbiases = mlx.Contiguous(qbiases)
		arrays[name+".bias"] = qbiases
		toEval = append(toEval, qbiases)
	}

	return tmpPath, toEval, st, nil
}

// quantizeTensor loads a tensor from safetensors format, quantizes it,
// and returns a single combined safetensors blob with the quantized weight, scale, and optional bias.
// Tensor keys use the original tensor name: name, name.scale, name.bias.
// The blob includes __metadata__ with quant_type and group_size.
// Supported quantization types: "int4", "nvfp4", "int8", "mxfp8".
func quantizeTensor(r io.Reader, tensorName, dtype string, shape []int32, quantize string) (blobData []byte, err error) {
	arrays := make(map[string]*mlx.Array)
	tmpPath, toEval, st, err := loadAndQuantizeArray(r, tensorName, quantize, arrays)
	if tmpPath != "" {
		defer os.Remove(tmpPath)
	}
	if st != nil {
		defer st.Free()
	}
	if err != nil {
		return nil, err
	}

	mlx.Eval(toEval...)

	// Build metadata for single-tensor blobs
	params := quantizeParams[quantize]
	metadata := map[string]string{
		"quant_type": quantize,
		"group_size": strconv.Itoa(params.groupSize),
	}

	tmpDir := ensureTempDir()
	outPath := filepath.Join(tmpDir, "combined.safetensors")
	defer os.Remove(outPath)
	if err := mlx.SaveSafetensorsWithMetadata(outPath, arrays, metadata); err != nil {
		return nil, fmt.Errorf("failed to save combined blob: %w", err)
	}
	return os.ReadFile(outPath)
}

// quantizePackedGroup quantizes multiple tensors and saves them all into a single
// combined safetensors blob. Used for packing expert groups.
// Each tensor may have a different quantization type (mixed-precision).
// Returns the blob bytes. No __metadata__ is added because different tensors
// may use different quantization types.
func quantizePackedGroup(inputs []create.PackedTensorInput) ([]byte, error) {
	allArrays := make(map[string]*mlx.Array)
	var allToEval []*mlx.Array
	var tmpPaths []string
	var handles []*mlx.SafetensorsFile

	for _, input := range inputs {
		tmpPath, toEval, st, err := loadAndQuantizeArray(input.Reader, input.Name, input.Quantize, allArrays)
		if tmpPath != "" {
			tmpPaths = append(tmpPaths, tmpPath)
		}
		if st != nil {
			handles = append(handles, st)
		}
		if err != nil {
			// Cleanup on error
			for _, h := range handles {
				h.Free()
			}
			for _, p := range tmpPaths {
				os.Remove(p)
			}
			return nil, err
		}
		allToEval = append(allToEval, toEval...)
	}

	mlx.Eval(allToEval...)

	// Free native handles after eval
	for _, h := range handles {
		h.Free()
	}

	// Save combined blob (no global metadata for mixed-precision packed blobs)
	tmpDir := ensureTempDir()
	outPath := filepath.Join(tmpDir, "packed-combined.safetensors")
	defer os.Remove(outPath)
	if err := mlx.SaveSafetensorsWithMetadata(outPath, allArrays, nil); err != nil {
		return nil, fmt.Errorf("failed to save packed blob: %w", err)
	}

	blobData, err := os.ReadFile(outPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read packed blob: %w", err)
	}

	for _, p := range tmpPaths {
		os.Remove(p)
	}

	return blobData, nil
}

// QuantizeSupported returns true if quantization is supported (MLX build)
func QuantizeSupported() bool {
	return true
}

// ensureTempDir creates the temp directory for quantization if it doesn't exist
func ensureTempDir() string {
	tmpDir := filepath.Join(os.TempDir(), "ollama-quantize")
	os.MkdirAll(tmpDir, 0755)
	return tmpDir
}

// findSafetensorsKey reads the first non-metadata tensor key from a safetensors file.
func findSafetensorsKey(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()

	var headerSize uint64
	if err := binary.Read(f, binary.LittleEndian, &headerSize); err != nil {
		return "", err
	}
	headerBytes := make([]byte, headerSize)
	if _, err := io.ReadFull(f, headerBytes); err != nil {
		return "", err
	}

	var header map[string]json.RawMessage
	if err := json.Unmarshal(headerBytes, &header); err != nil {
		return "", err
	}

	for k := range header {
		if k != "__metadata__" {
			return k, nil
		}
	}
	return "", fmt.Errorf("no tensor found in safetensors header")
}
