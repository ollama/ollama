package mlx

// #include "generated.h"
import "C"

import (
	"strings"
	"sync"
)

// MLX exposes the backend device name and architecture that select its
// native CUDA and Metal reductions.
var kernelDevice = sync.OnceValues(func() (string, string) {
	info := C.mlx_device_info_new()
	if mlxError(info) != nil {
		return "", ""
	}
	defer freeDeviceInfo(info)
	if mlxError(C.mlx_device_info_get(&info, DefaultDevice().ctx)) != nil {
		return "", ""
	}
	name, err := deviceInfoString(info, "device_name")
	if err != nil {
		return "", ""
	}
	arch, err := deviceInfoString(info, "architecture")
	if err != nil {
		return "", ""
	}
	return name, arch
})

func isM5Pro(name, architecture string) bool {
	return name == "Apple M5 Pro" && architecture == "applegpu_g17s"
}

func isM5MatrixDevice(name, architecture string) bool {
	return (name == "Apple M5 Pro" || name == "Apple M5 Max") && architecture == "applegpu_g17s" ||
		name == "Apple M5 Ultra" && architecture == "applegpu_g17d"
}

// MLX JITs custom shaders as Metal 4 on macOS 26 and later, with either
// bundled dylib. Earlier drivers cannot compile the TensorOps header.
var hasMetalTensorOps = sync.OnceValue(func() bool {
	return macOSMajorVersion() >= 26 && isAppleGPU()
})

func isAppleGPU() bool {
	_, arch := kernelDevice()
	return strings.HasPrefix(arch, "applegpu_")
}

type metalMatmulMode uint8

const (
	metalMatmulUnsupported metalMatmulMode = iota
	metalMatmulScalar
	metalMatmulTensor
)

// A stock grouped dot distinguishes the active reduction, including when TF32
// is disabled on Metal 4. Device names alone cannot distinguish the two modes.
var metalMatmul = sync.OnceValue(func() metalMatmulMode {
	if !isAppleGPU() {
		return metalMatmulUnsupported
	}

	mode := metalMatmulUnsupported
	Scoped(func() {
		// After the leading product, sixteen 1/16 terms round away in a
		// scalar FP32 sum. Their tensor-core subtotal adds exactly one.
		input := [32]float32{1536}
		for i := 16; i < len(input); i++ {
			input[i] = 0.125
		}
		x := BroadcastTo(FromValues(input[:], 1, 1, 1, 32), 16, 1, 1, 32)
		w := FromValues([]uint32{7, 0, 0x11111111, 0x11111111}, 1, 1, 4)
		s := FromValues([]uint8{0x7e, 0x38}, 1, 1, 2)
		ids := Zeros(DTypeInt32, 16, 1)
		y := GatherQMM(x, w, s, nil, nil, ids, true, 16, 4, "nvfp4", nil, true)
		switch y.Flatten(0, -1).Slice(Slice(0, 1)).Float() {
		case 4128768:
			mode = metalMatmulScalar
		case 4128769:
			mode = metalMatmulTensor
		}
	})
	return mode
})

// Dense and grouped products can select different reductions. In this dense
// dot, TF32 rounds the only nonzero input down to one. Full FP32 preserves it.
var usesBF16TensorMatmul = sync.OnceValue(func() bool {
	if !hasMetalTensorOps() {
		return false
	}
	match := false
	Scoped(func() {
		const rows, width, columns = 8, 16, 32
		const fraction = 1 + 1.0/4096
		input := [rows * width]float32{fraction}
		weight := [columns * width]float32{1}
		x := FromValues(input[:], rows, width)
		w := FromValues(weight[:], columns, width)
		tensor := matmulBF16WeightTensor(x, w.AsType(DTypeBFloat16), rows, width, columns)
		if tensor == nil {
			return
		}
		stock := x.Matmul(w.Transpose(1, 0))
		cell := Slice(0, 1)
		got, want := tensor.Slice(cell, cell), stock.Slice(cell, cell)
		Eval(got, want)
		native := want.Float()
		match = (native == 1 || native == fraction) && got.Float() == native
	})
	return match
})

// UsesM5MatrixKernels identifies Metal devices qualified for the short projection kernels.
func UsesM5MatrixKernels() bool {
	return isM5MatrixDevice(kernelDevice())
}

// SupportsPackedController reports where a combined gain/forget projection
// matches the two separate projections. M3 Ultra was verified on Metal 3.
func SupportsPackedController() bool {
	name, arch := kernelDevice()
	return isM5MatrixDevice(name, arch) ||
		name == "Apple M3 Ultra" && arch == "applegpu_g15d" && !hasMetalTensorOps()
}

// SupportsBF16WeightMatmul lets the loader retain compact weights only for a known reduction.
func SupportsBF16WeightMatmul() bool {
	return usesBF16TensorMatmul() || UsesM5MatrixKernels() && metalMatmul() == metalMatmulScalar
}
