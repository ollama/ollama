package client

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"github.com/ollama/ollama/x/create"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model"
)

// loadAndQuantizeArray writes a safetensors reader to a temp file, loads it with MLX,
// quantizes the tensor, and appends the resulting arrays (weight, scale, optional bias)
// to the provided maps. If quantize is empty, the tensor is kept as-is.
// Returns any temp file paths created (caller must clean up) and arrays needing eval.
func loadAndQuantizeArray(r io.Reader, name, quantize string, arrays map[string]*mlx.Array) (tmpPath string, toEval []*mlx.Array, nativeHandle *mlx.SafetensorsFile, err error) {
	if quantize != "" {
		if gs, _, _ := model.QuantizationParams(quantize); gs == 0 {
			return "", nil, nil, fmt.Errorf("unsupported quantization type: %s", quantize)
		}
	}

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
	header, err := readSafetensorsHeader(tmpPath)
	if err != nil {
		st.Free()
		return tmpPath, nil, nil, fmt.Errorf("failed to read blob header for %s: %w", name, err)
	}
	inputKey, err := safetensorsKey(name, header)
	if err != nil {
		st.Free()
		return tmpPath, nil, nil, fmt.Errorf("failed to resolve tensor key for %s: %w", name, err)
	}

	arr := st.Get(inputKey)
	if arr == nil {
		st.Free()
		return tmpPath, nil, nil, fmt.Errorf("tensor %q not found in safetensors", inputKey)
	}

	// Decode FP8 source encoding before checking quantize, so that callers
	// requesting decode-only (quantize="") receive usable float data.
	if info, ok := header[inputKey]; ok && info.Dtype == "F8_E4M3" {
		scaleKey := inputKey + ".scale_inv"
		scaleInv := st.Get(scaleKey)
		if scaleInv == nil {
			scaleKey = inputKey + ".scale"
			scaleInv = st.Get(scaleKey)
		}
		if scaleInv == nil {
			st.Free()
			return tmpPath, nil, nil, fmt.Errorf("missing companion tensor %q or %q for fp8 source tensor %q", inputKey+".scale_inv", inputKey+".scale", inputKey)
		}
		arr, err = decodeSourceFP8Tensor(arr, scaleInv)
		if err != nil {
			st.Free()
			return tmpPath, nil, nil, fmt.Errorf("failed to decode fp8 tensor %s: %w", inputKey, err)
		}
		mlx.Eval(arr)
	}

	if quantize == "" {
		arr = mlx.Contiguous(arr, false)
		arrays[name] = arr
		return tmpPath, []*mlx.Array{arr}, st, nil
	}

	if arr.DType() != mlx.DTypeBFloat16 && arr.DType() != mlx.DTypeFloat32 && arr.DType() != mlx.DTypeFloat16 {
		// Convert to float type if needed (quantize expects float)
		arr = arr.AsType(mlx.DTypeBFloat16)
		mlx.Eval(arr)
	}

	groupSize, bits, mode := model.QuantizationParams(quantize)
	qweight, scales, qbiases := mlx.Quantize(arr, groupSize, bits, mode)

	// Validate quantization produced non-empty output. MLX quantize may return
	// empty arrays for unsupported mode/bits combinations without raising an error.
	mlx.Eval(qweight, scales)
	if len(qweight.Dims()) == 0 || qweight.Dims()[0] == 0 {
		st.Free()
		return tmpPath, nil, nil, fmt.Errorf("mlx.Quantize produced empty weight for %s (quantize=%s, groupSize=%d, bits=%d, mode=%s)",
			name, quantize, groupSize, bits, mode)
	}
	if len(scales.Dims()) == 0 || scales.Dims()[0] == 0 {
		st.Free()
		return tmpPath, nil, nil, fmt.Errorf("mlx.Quantize produced empty scales for %s (quantize=%s, groupSize=%d, bits=%d, mode=%s)",
			name, quantize, groupSize, bits, mode)
	}

	qweight = mlx.Contiguous(qweight, false)
	scales = mlx.Contiguous(scales, false)
	arrays[name] = qweight
	arrays[name+".scale"] = scales
	toEval = append(toEval, qweight, scales)

	if qbiases != nil {
		qbiases = mlx.Contiguous(qbiases, false)
		arrays[name+".bias"] = qbiases
		toEval = append(toEval, qbiases)
	}

	return tmpPath, toEval, st, nil
}

// quantizeTensor loads a tensor from safetensors format, quantizes it,
// and returns a single combined safetensors blob with the quantized weight, scale, and optional bias.
// Tensor keys use the original tensor name: name, name.scale, name.bias.
// The blob includes __metadata__ with quant_type and group_size.
// Supported quantization types: "int4", "nvfp4", "mxfp4", "int8", "mxfp8".
func quantizeTensor(r io.Reader, tensorName, dtype string, shape []int32, quantize string) (blobData []byte, err error) {
	arrays := make(map[string]*mlx.Array)
	tmpPath, toEval, st, err := loadAndQuantizeArray(r, tensorName, quantize, arrays)
	if tmpPath != "" {
		defer os.Remove(tmpPath)
	}
	if err != nil {
		return nil, err
	}

	finalArrays := make([]*mlx.Array, 0, len(arrays))
	for _, arr := range arrays {
		if arr != nil {
			finalArrays = append(finalArrays, arr)
		}
	}
	mlx.Pin(finalArrays...)
	defer func() {
		if st != nil {
			st.Free()
		}
		mlx.Unpin(finalArrays...)
		mlx.Sweep()
	}()

	mlx.Eval(toEval...)
	mlx.Sweep()
	// Free early to release mmap; defer guard handles error paths
	if st != nil {
		st.Free()
		st = nil
	}

	// Build metadata for single-tensor blobs
	groupSize, _, _ := model.QuantizationParams(quantize)
	metadata := map[string]string{
		"quant_type": quantize,
		"group_size": strconv.Itoa(groupSize),
	}

	tmpDir := ensureTempDir()
	outPath := filepath.Join(tmpDir, "combined.safetensors")
	defer os.Remove(outPath)
	if err := mlx.SaveSafetensorsWithMetadata(outPath, arrays, metadata); err != nil {
		return nil, fmt.Errorf("failed to save combined blob: %w", err)
	}
	return os.ReadFile(outPath)
}

// quantizePackedGroup quantizes multiple tensors and packs them into a single
// combined safetensors blob. Each tensor may use its own quantization type
// (mixed-precision). Returns the blob bytes.
//
// Per-expert source tensors are stacked into 3-D switch_mlp form upstream by
// create.StackPerExpertGroup, so this function only sees stacked tensors.
func quantizePackedGroup(inputs []create.PackedTensorInput) ([]byte, error) {
	allArrays := make(map[string]*mlx.Array)
	var pinned []*mlx.Array

	var metadata map[string]string
	uniformQuantize := ""
	hasQuantized := false
	mixedQuantize := false
	for _, input := range inputs {
		if input.Quantize == "" {
			if hasQuantized {
				mixedQuantize = true
			}
			continue
		}
		if !hasQuantized {
			hasQuantized = true
			uniformQuantize = input.Quantize
			continue
		}
		if input.Quantize != uniformQuantize {
			mixedQuantize = true
		}
	}
	if hasQuantized && !mixedQuantize {
		if groupSize, _, _ := model.QuantizationParams(uniformQuantize); groupSize > 0 {
			metadata = map[string]string{
				"quant_type": uniformQuantize,
				"group_size": strconv.Itoa(groupSize),
			}
		}
	}

	for _, input := range inputs {
		tmpPath, toEval, st, err := loadAndQuantizeArray(input.Reader, input.Name, input.Quantize, allArrays)
		if err != nil {
			mlx.Unpin(pinned...)
			mlx.Sweep()
			return nil, err
		}

		mlx.Eval(toEval...)

		finalArrays := arraysForPackedInput(allArrays, input)
		mlx.Pin(finalArrays...)
		pinned = append(pinned, finalArrays...)

		// Record per-tensor quant type so the model can resolve params at load time.
		if input.Quantize != "" {
			if groupSize, _, _ := model.QuantizationParams(input.Quantize); groupSize > 0 {
				if metadata == nil {
					metadata = make(map[string]string)
				}
				metadata[input.Name+".quant_type"] = input.Quantize
				metadata[input.Name+".group_size"] = strconv.Itoa(groupSize)
			}
		}

		if st != nil {
			st.Free()
		}
		if tmpPath != "" {
			os.Remove(tmpPath)
		}
		mlx.Sweep()
	}
	defer func() {
		mlx.Unpin(pinned...)
		mlx.Sweep()
	}()

	// Save combined blob. Add global metadata only when every packed tensor uses
	// the same quantization mode and group size.
	tmpDir := ensureTempDir()
	outPath := filepath.Join(tmpDir, "packed-combined.safetensors")
	defer os.Remove(outPath)
	if err := mlx.SaveSafetensorsWithMetadata(outPath, allArrays, metadata); err != nil {
		return nil, fmt.Errorf("failed to save packed blob: %w", err)
	}

	blobData, err := os.ReadFile(outPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read packed blob: %w", err)
	}

	return blobData, nil
}

func arraysForPackedInput(allArrays map[string]*mlx.Array, input create.PackedTensorInput) []*mlx.Array {
	keys := []string{input.Name}
	if input.Quantize != "" {
		keys = append(keys, input.Name+".scale", input.Name+".bias")
	}

	out := make([]*mlx.Array, 0, len(keys))
	for _, key := range keys {
		if arr := allArrays[key]; arr != nil {
			out = append(out, arr)
		}
	}
	return out
}

// QuantizeSupported returns true if quantization is supported (MLX library available)
func QuantizeSupported() bool {
	return mlx.CheckInit() == nil
}

// ensureTempDir creates the temp directory for quantization if it doesn't exist
func ensureTempDir() string {
	tmpDir := filepath.Join(os.TempDir(), "ollama-quantize")
	os.MkdirAll(tmpDir, 0755)
	return tmpDir
}

type safetensorsHeaderEntry struct {
	Dtype string  `json:"dtype"`
	Shape []int32 `json:"shape"`
}

func readSafetensorsHeader(path string) (map[string]safetensorsHeaderEntry, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var headerSize uint64
	if err := binary.Read(f, binary.LittleEndian, &headerSize); err != nil {
		return nil, err
	}
	headerBytes := make([]byte, headerSize)
	if _, err := io.ReadFull(f, headerBytes); err != nil {
		return nil, err
	}

	var header map[string]safetensorsHeaderEntry
	if err := json.Unmarshal(headerBytes, &header); err != nil {
		return nil, err
	}
	return header, nil
}

// safetensorsKey resolves the primary tensor key from a header.
func safetensorsKey(preferred string, header map[string]safetensorsHeaderEntry) (string, error) {
	if preferred != "" {
		if _, ok := header[preferred]; ok {
			return preferred, nil
		}
	}

	keys := make([]string, 0, len(header))
	for k := range header {
		if k == "__metadata__" || strings.HasSuffix(k, ".scale_inv") {
			continue
		}
		keys = append(keys, k)
	}
	sort.Strings(keys)
	if len(keys) == 0 {
		return "", fmt.Errorf("no tensor found in safetensors header")
	}
	return keys[0], nil
}

func decodeSourceFP8Tensor(weight, scale *mlx.Array) (*mlx.Array, error) {
	if weight == nil || scale == nil {
		return nil, fmt.Errorf("fp8 weight and scale tensors are required")
	}

	weightShape := weight.Dims()
	scaleShape := scale.Dims()
	if len(weightShape) != 2 || len(scaleShape) != 2 {
		return nil, fmt.Errorf("expected 2D fp8 weight and scale tensors, got %v and %v", weightShape, scaleShape)
	}

	// These must match the block size validated by resolveEffectiveQuantization
	// in create.go, which rejects any source model with a different block size.
	const blockRows = 128
	const blockCols = 128
	rows, cols := weightShape[0], weightShape[1]
	expectedScaleRows := (rows + blockRows - 1) / blockRows
	expectedScaleCols := (cols + blockCols - 1) / blockCols
	if scaleShape[0] != expectedScaleRows || scaleShape[1] != expectedScaleCols {
		return nil, fmt.Errorf(
			"unexpected fp8 scale shape %v for weight shape %v; want [%d %d]",
			scaleShape,
			weightShape,
			expectedScaleRows,
			expectedScaleCols,
		)
	}

	decoded := mlx.FromFP8(weight, mlx.DTypeBFloat16)
	padBottom := blockRows*scaleShape[0] - rows
	padSide := blockCols*scaleShape[1] - cols
	if padBottom > 0 || padSide > 0 {
		decoded = mlx.PadConstant(decoded, []int{0, 1}, []int{0, 0}, []int{padBottom, padSide})
	}

	decoded = mlx.Reshape(decoded, int32(scaleShape[0]), int32(blockRows), int32(scaleShape[1]), int32(blockCols))
	decoded = mlx.Mul(decoded, mlx.ExpandDims(mlx.ExpandDims(scale, 1), 3))
	decoded = mlx.Reshape(decoded, int32(rows+padBottom), int32(cols+padSide))
	if padBottom > 0 || padSide > 0 {
		decoded = mlx.SliceStartStop(decoded, []int32{0, 0}, []int32{int32(rows), int32(cols)})
	}

	return decoded, nil
}
