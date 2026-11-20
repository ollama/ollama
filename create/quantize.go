package create

import (
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/quant"
)

// quantizeItem is one tensor going into a (possibly multi-tensor) quantized
// blob: its output name, the quantization to apply (or "" to decode/keep at
// source precision), a safetensors-wrapped reader for its input bytes (keyed by
// name), and whether the input is an FP8 weight to widen to BF16 first.
type quantizeItem struct {
	name      string
	quantize  string
	reader    io.Reader
	decodeFP8 bool
}

// quantizeBlob loads, optionally quantizes, and packs the given tensors into a
// single safetensors blob (weight + scale + optional bias per quantized
// tensor). All MLX work runs on the pinned MLX thread.
func quantizeBlob(ctx context.Context, items []quantizeItem) ([]byte, error) {
	var blob []byte
	err := runOnMLXThread(ctx, func() error {
		var err error
		blob, err = quantizeBlobLocked(items)
		return err
	})
	return blob, err
}

func quantizeBlobLocked(items []quantizeItem) ([]byte, error) {
	allArrays := make(map[string]*mlx.Array)
	held := mlx.NewScope()
	defer held.Close()

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
		if err := quantizeItemArrays(it, allArrays, tmpDir, held); err != nil {
			return nil, err
		}
		// The item's intermediates are free; hand their buffers back before
		// the next item, which may never reuse those sizes.
		mlx.ClearCache()
		if mixed && it.quantize != "" {
			if gs, _, _ := quant.Params(it.quantize); gs > 0 {
				if metadata == nil {
					metadata = make(map[string]string)
				}
				metadata[it.name+".quant_type"] = it.quantize
				metadata[it.name+".group_size"] = strconv.Itoa(gs)
			}
		}
	}

	outPath := filepath.Join(tmpDir, "blob.safetensors")
	if err := mlx.SaveSafetensorsWithMetadata(outPath, allArrays, metadata); err != nil {
		return nil, fmt.Errorf("failed to save blob: %w", err)
	}
	return os.ReadFile(outPath)
}

// quantizeItemArrays loads and quantizes one item into arrays and holds its
// finished arrays in held.
func quantizeItemArrays(it quantizeItem, arrays map[string]*mlx.Array, tmpDir string, held *mlx.Scope) error {
	tmpPath, toEval, st, err := loadAndQuantizeArray(it.reader, it.name, it.quantize, it.decodeFP8, arrays, tmpDir)
	if tmpPath != "" {
		defer os.Remove(tmpPath)
	}
	if err != nil {
		return err
	}
	defer st.Free()
	mlx.Eval(toEval...)
	held.Attach(toEval...)
	return nil
}

// loadAndQuantizeArray writes a safetensors reader to a temp file, loads it
// with MLX, widens an FP8 source tensor if present, optionally quantizes, and
// adds the resulting arrays (weight, scale, optional bias) to
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
		_ = tmpFile.Close()
		return tmpPath, nil, nil, fmt.Errorf("failed to write temp file for %s: %w", name, err)
	}
	if err := tmpFile.Close(); err != nil {
		return tmpPath, nil, nil, fmt.Errorf("failed to close temp file for %s: %w", name, err)
	}

	st, err := mlx.LoadSafetensorsNative(tmpPath)
	if err != nil {
		return tmpPath, nil, nil, fmt.Errorf("failed to load safetensors for %s: %w", name, err)
	}

	toEval = mlx.ScopedArrays(func() []*mlx.Array {
		arr := st.Get(name)
		if arr == nil {
			err = fmt.Errorf("tensor %q not found in safetensors", name)
			return nil
		}

		if decodeFP8 {
			scaleKey := name + ".scale_inv"
			scaleInv := st.Get(scaleKey)
			if scaleInv == nil {
				scaleKey = name + ".scale"
				scaleInv = st.Get(scaleKey)
			}
			if scaleInv == nil {
				err = fmt.Errorf("missing companion tensor %q or %q for fp8 source tensor %q", name+".scale_inv", name+".scale", name)
				return nil
			}
			arr, err = decodeSourceFP8Tensor(arr, scaleInv)
			if err != nil {
				err = fmt.Errorf("failed to decode fp8 tensor %s: %w", name, err)
				return nil
			}
		}

		if quantize == "" {
			arr = mlx.Contiguous(arr, false)
			arrays[name] = arr
			return []*mlx.Array{arr}
		}

		if arr.DType() != mlx.DTypeBFloat16 && arr.DType() != mlx.DTypeFloat32 && arr.DType() != mlx.DTypeFloat16 {
			arr = arr.AsType(mlx.DTypeBFloat16)
		}

		groupSize, bits, mode := quant.Params(quantize)
		qweight, scales, qbiases := mlx.Quantize(arr, groupSize, bits, mode)
		if len(qweight.Dims()) == 0 || qweight.Dims()[0] == 0 {
			err = fmt.Errorf("mlx.Quantize produced empty weight for %s (quantize=%s, groupSize=%d, bits=%d, mode=%s)", name, quantize, groupSize, bits, mode)
			return nil
		}
		if len(scales.Dims()) == 0 || scales.Dims()[0] == 0 {
			err = fmt.Errorf("mlx.Quantize produced empty scales for %s (quantize=%s, groupSize=%d, bits=%d, mode=%s)", name, quantize, groupSize, bits, mode)
			return nil
		}

		qweight = mlx.Contiguous(qweight, false)
		scales = mlx.Contiguous(scales, false)
		arrays[name] = qweight
		arrays[name+".scale"] = scales
		out := []*mlx.Array{qweight, scales}
		if qbiases != nil {
			qbiases = mlx.Contiguous(qbiases, false)
			arrays[name+".bias"] = qbiases
			out = append(out, qbiases)
		}
		return out
	})
	if err != nil {
		st.Free()
		return tmpPath, nil, nil, err
	}
	return tmpPath, toEval, st, nil
}

// decodeSourceFP8Tensor widens an FP8 weight with its float scale to a BF16
// tensor. The weight is 2D [rows, cols] or a stacked expert bank [experts,
// rows, cols]. The scale holds one value per tensor, per expert for a bank.
func decodeSourceFP8Tensor(weight, scale *mlx.Array) (*mlx.Array, error) {
	if weight == nil || scale == nil {
		return nil, fmt.Errorf("fp8 weight and scale tensors are required")
	}
	weightShape := weight.Dims()
	rank := len(weightShape)
	if rank != 2 && rank != 3 {
		return nil, fmt.Errorf("expected a 2D or 3D fp8 weight, got %v", weightShape)
	}
	// The scales broadcast over the trailing [rows, cols] once reshaped to [..., 1, 1].
	shape := []int32{1, 1}
	experts := 1
	if rank == 3 {
		experts = weightShape[0]
		shape = []int32{int32(experts), 1, 1}
	}
	if scale.Size() != experts {
		return nil, fmt.Errorf("unexpected fp8 scale shape %v for weight shape %v", scale.Dims(), weightShape)
	}

	decoded := mlx.FromFP8(weight, mlx.DTypeBFloat16)
	// Multiplying by an F32 scale promotes the result; keep the decoded dtype.
	return mlx.Mul(decoded, mlx.Reshape(scale, shape...)).AsType(decoded.DType()), nil
}
