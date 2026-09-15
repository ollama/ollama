package mlxrunner

import (
	"github.com/ollama/ollama/x/mlxrunner/model/base"
)

// SupportsArchitecture reports whether the MLX runner has a constructor for arch.
func SupportsArchitecture(arch string) bool {
	return base.SupportsArchitecture(arch)
}

// SupportsDraftArchitecture reports whether the MLX runner has a draft constructor for arch.
func SupportsDraftArchitecture(arch string) bool {
	return base.SupportsDraftArchitecture(arch)
}
