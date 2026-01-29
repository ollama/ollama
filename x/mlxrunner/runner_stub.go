//go:build !mlx

package mlxrunner

import "errors"

// Execute returns an error when not built with MLX support.
func Execute(args []string) error {
	return errors.New("MLX runner not available: build with mlx tag")
}
