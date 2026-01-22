//go:build mlx

package client

import (
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/ollama/ollama/x/imagegen/mlx"
)

// quantizeTensor loads a tensor from safetensors format, quantizes it,
// and returns safetensors data for the quantized weights, scales, and biases.
// Supported quantization types: "fp8" (affine 8-bit)
// Uses MLX's native SaveSafetensors to ensure correct dtype handling (especially uint32 for quantized weights).
func quantizeTensor(r io.Reader, name, dtype string, shape []int32, quantize string) (qweightData, scalesData, qbiasData []byte, qweightShape, scalesShape, qbiasShape []int32, err error) {
	tmpDir := ensureTempDir()

	// Read safetensors data to a temp file (LoadSafetensorsNative needs a path)
	tmpFile, err := os.CreateTemp(tmpDir, "quant-input-*.safetensors")
	if err != nil {
		return nil, nil, nil, nil, nil, nil, fmt.Errorf("failed to create temp file: %w", err)
	}
	tmpPath := tmpFile.Name()
	defer os.Remove(tmpPath)

	if _, err := io.Copy(tmpFile, r); err != nil {
		tmpFile.Close()
		return nil, nil, nil, nil, nil, nil, fmt.Errorf("failed to write temp file: %w", err)
	}
	tmpFile.Close()

	// Load the tensor using MLX's native loader
	st, err := mlx.LoadSafetensorsNative(tmpPath)
	if err != nil {
		return nil, nil, nil, nil, nil, nil, fmt.Errorf("failed to load safetensors: %w", err)
	}
	defer st.Free()

	// Get the tensor (it's stored as "data" in our minimal safetensors format)
	arr := st.Get("data")
	if arr == nil {
		return nil, nil, nil, nil, nil, nil, fmt.Errorf("tensor 'data' not found in safetensors")
	}

	// Convert to BFloat16 if needed (quantize expects float type)
	if arr.Dtype() != mlx.DtypeBFloat16 && arr.Dtype() != mlx.DtypeFloat32 && arr.Dtype() != mlx.DtypeFloat16 {
		arr = mlx.AsType(arr, mlx.DtypeBFloat16)
		mlx.Eval(arr)
	}

	// Quantize based on quantization type
	var qweight, scales, qbiases *mlx.Array
	switch quantize {
	case "fp4":
		// affine mode: group_size=32, bits=4
		qweight, scales, qbiases = mlx.Quantize(arr, 32, 4, "affine")
	case "fp8":
		// affine mode: group_size=32, bits=8
		qweight, scales, qbiases = mlx.Quantize(arr, 32, 8, "affine")
	default:
		return nil, nil, nil, nil, nil, nil, fmt.Errorf("unsupported quantization type: %s", quantize)
	}

	// Eval and make contiguous for data access
	qweight = mlx.Contiguous(qweight)
	scales = mlx.Contiguous(scales)
	if qbiases != nil {
		qbiases = mlx.Contiguous(qbiases)
		mlx.Eval(qweight, scales, qbiases)
	} else {
		mlx.Eval(qweight, scales)
	}

	// Get shapes
	qweightShape = qweight.Shape()
	scalesShape = scales.Shape()

	// Save quantized weight using MLX's native safetensors (correctly handles uint32 dtype)
	qweightPath := filepath.Join(tmpDir, "qweight.safetensors")
	defer os.Remove(qweightPath)
	if err := mlx.SaveSafetensors(qweightPath, map[string]*mlx.Array{"data": qweight}); err != nil {
		return nil, nil, nil, nil, nil, nil, fmt.Errorf("failed to save quantized weight: %w", err)
	}
	qweightData, err = os.ReadFile(qweightPath)
	if err != nil {
		return nil, nil, nil, nil, nil, nil, fmt.Errorf("failed to read quantized weight: %w", err)
	}

	// Save scales using MLX's native safetensors
	scalesPath := filepath.Join(tmpDir, "scales.safetensors")
	defer os.Remove(scalesPath)
	if err := mlx.SaveSafetensors(scalesPath, map[string]*mlx.Array{"data": scales}); err != nil {
		return nil, nil, nil, nil, nil, nil, fmt.Errorf("failed to save scales: %w", err)
	}
	scalesData, err = os.ReadFile(scalesPath)
	if err != nil {
		return nil, nil, nil, nil, nil, nil, fmt.Errorf("failed to read scales: %w", err)
	}

	// Affine mode returns qbiases for zero-point offset
	if qbiases != nil {
		qbiasShape = qbiases.Shape()
		qbiasPath := filepath.Join(tmpDir, "qbias.safetensors")
		defer os.Remove(qbiasPath)
		if err := mlx.SaveSafetensors(qbiasPath, map[string]*mlx.Array{"data": qbiases}); err != nil {
			return nil, nil, nil, nil, nil, nil, fmt.Errorf("failed to save qbiases: %w", err)
		}
		qbiasData, err = os.ReadFile(qbiasPath)
		if err != nil {
			return nil, nil, nil, nil, nil, nil, fmt.Errorf("failed to read qbiases: %w", err)
		}
	}

	return qweightData, scalesData, qbiasData, qweightShape, scalesShape, qbiasShape, nil
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
