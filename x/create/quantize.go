package create

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strconv"

	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/quant"
)

// QuantizeSupported reports whether MLX (and thus quantization) is available.
func QuantizeSupported() bool {
	return mlx.CheckInit() == nil
}

// quantizeItem is one tensor going into a (possibly multi-tensor) quantized
// blob: its output name, the quantization to apply (or "" to decode/keep at
// source precision), a safetensors-wrapped reader for its input bytes (keyed by
// name), and whether the input is a block-FP8 weight to decode before use.
type quantizeItem struct {
	name      string
	quantize  string
	reader    io.Reader
	decodeFP8 bool
}

// quantizeBlob loads, optionally quantizes, and packs the given tensors into a
// single safetensors blob (weight + scale + optional bias per quantized
// tensor). All MLX work runs on the pinned MLX thread.
func quantizeBlob(items []quantizeItem) ([]byte, error) {
	var blob []byte
	err := runOnMLXThread(func() error {
		var err error
		blob, err = quantizeBlobLocked(items)
		return err
	})
	return blob, err
}

func quantizeBlobLocked(items []quantizeItem) ([]byte, error) {
	allArrays := make(map[string]*mlx.Array)
	var pinned []*mlx.Array
	defer func() {
		mlx.Unpin(pinned...)
		mlx.Sweep()
	}()

	tmpDir, err := os.MkdirTemp("", "ollama-quantize-*")
	if err != nil {
		return nil, fmt.Errorf("failed to create temp dir: %w", err)
	}
	defer os.RemoveAll(tmpDir)

	// Blob metadata: a single quant_type/group_size when every quantized
	// tensor matches, otherwise per-tensor entries.
	uniform, mixed, hasQuant := "", false, false
	for _, it := range items {
		if it.quantize == "" {
			if hasQuant {
				mixed = true
			}
			continue
		}
		if !hasQuant {
			hasQuant, uniform = true, it.quantize
			continue
		}
		if it.quantize != uniform {
			mixed = true
		}
	}
	var metadata map[string]string
	if hasQuant && !mixed {
		if gs, _, _ := quant.Params(uniform); gs > 0 {
			metadata = map[string]string{"quant_type": uniform, "group_size": strconv.Itoa(gs)}
		}
	}

	for _, it := range items {
		if err := func() error {
			defer mlx.Sweep()
			tmpPath, toEval, st, err := loadAndQuantizeArray(it.reader, it.name, it.quantize, it.decodeFP8, allArrays, tmpDir)
			if tmpPath != "" {
				defer os.Remove(tmpPath)
			}
			if err != nil {
				return err
			}
			if st != nil {
				defer st.Free()
			}
			mlx.Eval(toEval...)
			final := arraysForItem(allArrays, it)
			mlx.Pin(final...)
			pinned = append(pinned, final...)

			if mixed && it.quantize != "" {
				if gs, _, _ := quant.Params(it.quantize); gs > 0 {
					if metadata == nil {
						metadata = make(map[string]string)
					}
					metadata[it.name+".quant_type"] = it.quantize
					metadata[it.name+".group_size"] = strconv.Itoa(gs)
				}
			}
			return nil
		}(); err != nil {
			return nil, err
		}
	}

	outPath := filepath.Join(tmpDir, "blob.safetensors")
	if err := mlx.SaveSafetensorsWithMetadata(outPath, allArrays, metadata); err != nil {
		return nil, fmt.Errorf("failed to save blob: %w", err)
	}
	return os.ReadFile(outPath)
}

func arraysForItem(all map[string]*mlx.Array, it quantizeItem) []*mlx.Array {
	keys := []string{it.name}
	if it.quantize != "" {
		keys = append(keys, it.name+".scale", it.name+".bias")
	}
	out := make([]*mlx.Array, 0, len(keys))
	for _, k := range keys {
		if a := all[k]; a != nil {
			out = append(out, a)
		}
	}
	return out
}

// loadAndQuantizeArray writes a safetensors reader to a temp file, loads it
// with MLX, decodes a block-FP8 source tensor if present, optionally
// quantizes, and adds the resulting arrays (weight, scale, optional bias) to
// arrays keyed by name. With quantize == "" the (decoded) tensor is kept as-is.
// It must be called on the MLX thread.
//
// TODO: MLX's safetensors loader takes a file path, so we spill each tensor to a
// temp file. Wiring a streaming mlx_load_safetensors_reader into the CGO wrapper
// would let us load from the reader directly and drop the temp files.
func loadAndQuantizeArray(r io.Reader, name, quantize string, decodeFP8 bool, arrays map[string]*mlx.Array, tmpDir string) (tmpPath string, toEval []*mlx.Array, nativeHandle *mlx.SafetensorsFile, err error) {
	if quantize != "" && quant.Canonical(quantize) == "" {
		return "", nil, nil, fmt.Errorf("unsupported quantization type: %s", quantize)
	}

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

	arr := st.Get(name)
	if arr == nil {
		st.Free()
		return tmpPath, nil, nil, fmt.Errorf("tensor %q not found in safetensors", name)
	}

	// Decode an FP8 source tensor (using its block scale) before quantizing,
	// so a decode-only request (quantize == "") still yields usable float data.
	if decodeFP8 {
		scaleKey := name + ".scale_inv"
		scaleInv := st.Get(scaleKey)
		if scaleInv == nil {
			scaleKey = name + ".scale"
			scaleInv = st.Get(scaleKey)
		}
		if scaleInv == nil {
			st.Free()
			return tmpPath, nil, nil, fmt.Errorf("missing companion tensor %q or %q for fp8 source tensor %q", name+".scale_inv", name+".scale", name)
		}
		arr, err = decodeSourceFP8Tensor(arr, scaleInv)
		if err != nil {
			st.Free()
			return tmpPath, nil, nil, fmt.Errorf("failed to decode fp8 tensor %s: %w", name, err)
		}
		mlx.Eval(arr)
	}

	if quantize == "" {
		arr = mlx.Contiguous(arr, false)
		arrays[name] = arr
		return tmpPath, []*mlx.Array{arr}, st, nil
	}

	if arr.DType() != mlx.DTypeBFloat16 && arr.DType() != mlx.DTypeFloat32 && arr.DType() != mlx.DTypeFloat16 {
		arr = arr.AsType(mlx.DTypeBFloat16)
		mlx.Eval(arr)
	}

	groupSize, bits, mode := quant.Params(quantize)
	qweight, scales, qbiases := mlx.Quantize(arr, groupSize, bits, mode)
	mlx.Eval(qweight, scales)
	if len(qweight.Dims()) == 0 || qweight.Dims()[0] == 0 {
		st.Free()
		return tmpPath, nil, nil, fmt.Errorf("mlx.Quantize produced empty weight for %s (quantize=%s, groupSize=%d, bits=%d, mode=%s)", name, quantize, groupSize, bits, mode)
	}
	if len(scales.Dims()) == 0 || scales.Dims()[0] == 0 {
		st.Free()
		return tmpPath, nil, nil, fmt.Errorf("mlx.Quantize produced empty scales for %s (quantize=%s, groupSize=%d, bits=%d, mode=%s)", name, quantize, groupSize, bits, mode)
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

// decodeSourceFP8Tensor dequantizes a 128x128 block-FP8 weight using its block
// scale, returning a BF16 tensor. The weight is either 2D [rows, cols] with a 2D
// scale [ceil(rows/128), ceil(cols/128)], or a stacked 3D expert tensor
// [experts, rows, cols] with a 3D scale [experts, ceil(rows/128), ceil(cols/128)];
// the leading expert axis (if present) is decoded block-wise per expert.
func decodeSourceFP8Tensor(weight, scale *mlx.Array) (*mlx.Array, error) {
	if weight == nil || scale == nil {
		return nil, fmt.Errorf("fp8 weight and scale tensors are required")
	}
	weightShape := weight.Dims()
	scaleShape := scale.Dims()
	rank := len(weightShape)
	if (rank != 2 && rank != 3) || len(scaleShape) != rank {
		return nil, fmt.Errorf("expected matching 2D or 3D fp8 weight and scale tensors, got %v and %v", weightShape, scaleShape)
	}

	const blockRows = 128
	const blockCols = 128

	// The 128x128 blocks tile the trailing [rows, cols]; a 3D weight carries a
	// leading expert axis that broadcasts over those blocks one expert at a time.
	lead := weightShape[:rank-2]
	rows, cols := weightShape[rank-2], weightShape[rank-1]
	sr := (rows + blockRows - 1) / blockRows
	sc := (cols + blockCols - 1) / blockCols
	wantScale := append(append([]int(nil), lead...), sr, sc)
	if !slices.Equal(scaleShape, wantScale) {
		return nil, fmt.Errorf("unexpected fp8 scale shape %v for weight shape %v; want %v", scaleShape, weightShape, wantScale)
	}

	leadI32 := make([]int32, len(lead))
	for i, d := range lead {
		leadI32[i] = int32(d)
	}

	decoded := mlx.FromFP8(weight, mlx.DTypeBFloat16)
	dtype := decoded.DType()
	padBottom := blockRows*sr - rows
	padSide := blockCols*sc - cols
	if padBottom > 0 || padSide > 0 {
		// Pad the bottom/right of the trailing [rows, cols] only.
		decoded = mlx.PadConstant(decoded, []int{rank - 2, rank - 1}, []int{0, 0}, []int{padBottom, padSide})
	}

	// Split each 128x128 block into its own axis pair, scale every block by its
	// per-block factor (broadcast across the block interior), then restore.
	blocked := append(append([]int32(nil), leadI32...), int32(sr), blockRows, int32(sc), blockCols)
	decoded = mlx.Reshape(decoded, blocked...)
	// scale [..., sr, sc] -> [..., sr, 1, sc, 1]
	scaleB := mlx.ExpandDims(mlx.ExpandDims(scale, len(lead)+1), len(lead)+3)
	// Multiplying by an F32 scale promotes the result; keep the decoded dtype.
	decoded = mlx.Mul(decoded, scaleB).AsType(dtype)
	padded := append(append([]int32(nil), leadI32...), int32(rows+padBottom), int32(cols+padSide))
	decoded = mlx.Reshape(decoded, padded...)
	if padBottom > 0 || padSide > 0 {
		stops := append(append([]int32(nil), leadI32...), int32(rows), int32(cols))
		decoded = mlx.SliceStartStop(decoded, make([]int32, rank), stops)
	}
	return decoded, nil
}
