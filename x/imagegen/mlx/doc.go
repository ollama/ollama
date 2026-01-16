//go:build mlx

// Package mlx provides Go bindings for the MLX-C library with dynamic loading support.
//
//go:generate go run generate_wrappers.go ../../../build/_deps/mlx-c-src/mlx/c mlx.h mlx.c
package mlx
