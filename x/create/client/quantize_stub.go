//go:build !mlx

package client

import (
	"fmt"
	"io"

	"github.com/ollama/ollama/x/create"
)

// quantizeTensor is not available without MLX
func quantizeTensor(r io.Reader, tensorName, dtype string, shape []int32, quantize string) (blobData []byte, err error) {
	return nil, fmt.Errorf("quantization requires MLX support (build with mlx tag)")
}

// quantizePackedGroup is not available without MLX
func quantizePackedGroup(inputs []create.PackedTensorInput) ([]byte, error) {
	return nil, fmt.Errorf("quantization requires MLX support (build with mlx tag)")
}

// QuantizeSupported returns false when MLX is not available
func QuantizeSupported() bool {
	return false
}
