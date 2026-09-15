//go:build !mlx

package runner

import "errors"

// Execute returns an error when not built with MLX support.
func Execute(args []string) error {
	return errors.New("image generation not available: build with mlx tag")
}
