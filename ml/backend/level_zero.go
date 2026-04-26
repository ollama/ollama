// SPDX-License-Identifier: MIT
//go:build level_zero

// Package backend provides the Level Zero backend registration for the Ollama
// native ML engine.  This file is compiled only when the level_zero build tag
// is active, which happens when cmake builds with -DGGML_LEVEL_ZERO=ON and
// the Go linker flags include -tags level_zero.
//
// INFO-2 COMPLIANCE: ml.DeviceInfo.Variant does not exist in the frozen ABI
// (verified absent in ml/backend.go and ml/backend/backend.go).  GPU/NPU
// distinction is encoded exclusively via the Library field string values
// "level-zero-gpu" and "level-zero-npu".
package backend

import (
	// Blank import registers the Level Zero ggml dynamic backend so that
	// ggml_backend_dev_get() picks it up alongside CUDA, ROCm, and Vulkan.
	// The actual registration macro is GGML_BACKEND_DL_IMPL in ggml-level-zero.cpp.
	_ "github.com/ollama/ollama/ml/backend/ggml"
)
