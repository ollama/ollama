// SPDX-License-Identifier: MIT
// Package envconfig registers Intel Level Zero environment variable knobs.
// All Level Zero / oneAPI env vars are single-sourced here per the envconfig
// package contract: no other package reads these env vars directly.
package envconfig

import (
	"strconv"
)

// L0DeviceIndex returns the zero-based index of the Intel Level Zero device
// that Ollama should use for inference.  -1 (the default) means "let the
// scheduler choose the best available device".  Set OLLAMA_L0_DEVICE_INDEX
// to a non-negative integer to pin a specific device.
func L0DeviceIndex() int {
	s := Var("OLLAMA_L0_DEVICE_INDEX")
	if s == "" {
		return -1
	}
	if n, err := strconv.ParseInt(s, 10, 64); err == nil {
		return int(n)
	}
	return -1
}

// L0NPUEnable reports whether the NPU (VPU) device class should be included
// during Level Zero enumeration.  Defaults to false.  Set OLLAMA_L0_NPU_ENABLE
// to "1" or "true" to enable NPU devices.
func L0NPUEnable() bool {
	return Bool("OLLAMA_L0_NPU_ENABLE")()
}

// L0AffinityMask returns the ZE_AFFINITY_MASK environment variable value that
// should be forwarded to the runner subprocess.  An empty string means no mask
// is applied.  This mirrors how CUDA_VISIBLE_DEVICES works.
func L0AffinityMask() string {
	return String("ZE_AFFINITY_MASK")()
}
