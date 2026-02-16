//go:build !mlx

package client

import (
	"fmt"
	"io"
)

// quantizeTensor is not available without MLX
func quantizeTensor(r io.Reader, name, dtype string, shape []int32, quantize string) (qweightData, scalesData, qbiasData []byte, qweightShape, scalesShape, qbiasShape []int32, err error) {
	return nil, nil, nil, nil, nil, nil, fmt.Errorf("quantization requires MLX support (build with mlx tag)")
}

// QuantizeSupported returns false when MLX is not available
func QuantizeSupported() bool {
	return false
}
