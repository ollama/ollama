package mlx

// #include <stdlib.h>
// #include "generated.h"
import "C"

import (
	"log/slog"
	"sync"
	"unsafe"
)

// gpuSource is one backend's implementation of a kernel.
type gpuSource struct {
	source string
	header string
}

// gpuKernel is a custom kernel with per-backend sources and a graph
// fallback. Either backend may be absent; a backend that cannot be created
// or launched disables itself permanently. Contract checks belong to the
// caller, before run: run itself cannot fail.
type gpuKernel struct {
	name    string
	inputs  []string
	outputs []string
	metal   gpuSource
	cuda    gpuSource

	// fallback computes the same outputs with graph ops when no GPU
	// backend can run the launch.
	fallback func(launch gpuLaunch) []*Array

	metalOnce     sync.Once
	metalKernel   C.mlx_fast_metal_kernel
	metalDisabled bool

	cudaOnce     sync.Once
	cudaKernel   C.mlx_fast_cuda_kernel
	cudaDisabled bool
}

// gpuDTypeArg and gpuIntArg name template arguments for one launch.
type gpuDTypeArg struct {
	name  string
	dtype DType
}

type gpuIntArg struct {
	name  string
	value int
}

// gpuOutputSpec declares one kernel output buffer.
type gpuOutputSpec struct {
	name  string
	shape []int32
	dtype DType
}

// gpuLaunch is the per-call configuration for gpuKernel.run. Grid and
// thread-group units are shared across backends.
type gpuLaunch struct {
	dtypes      []gpuDTypeArg
	ints        []gpuIntArg
	outputs     []gpuOutputSpec
	grid        [3]int
	threadGroup [3]int
	inputs      []*Array
}

func cStringVector(values []string) (C.mlx_vector_string, func(), bool) {
	vec := C.mlx_vector_string_new()
	ok := true
	for _, s := range values {
		cs := C.CString(s)
		if C.mlx_vector_string_append_value(vec, cs) != 0 {
			ok = false
		}
		C.free(unsafe.Pointer(cs))
		if !ok {
			break
		}
	}
	cleanup := func() {
		C.mlx_vector_string_free(vec)
	}
	return vec, cleanup, ok
}

// run executes the kernel with the first backend that works, in CUDA,
// Metal, fallback order. It panics if no variant can run the launch.
func (k *gpuKernel) run(launch gpuLaunch) []*Array {
	if outs, ok := k.applyCUDA(launch); ok {
		return outs
	}
	if outs, ok := k.applyMetal(launch); ok {
		return outs
	}
	if k.fallback == nil {
		panic("mlx: kernel " + k.name + " has no usable implementation")
	}
	outs := k.fallback(launch)
	if len(outs) != len(k.outputs) {
		panic("mlx: kernel " + k.name + " fallback returned wrong output count")
	}
	return outs
}

func (k *gpuKernel) disableMetal(reason string) {
	k.metalDisabled = true
	slog.Warn("custom GPU kernel backend disabled", "kernel", k.name, "backend", "metal", "reason", reason)
}

func (k *gpuKernel) disableCUDA(reason string) {
	k.cudaDisabled = true
	slog.Warn("custom GPU kernel backend disabled", "kernel", k.name, "backend", "cuda", "reason", reason)
}

func (k *gpuKernel) getMetal() (C.mlx_fast_metal_kernel, bool) {
	k.metalOnce.Do(func() {
		if !MetalIsAvailable() {
			k.metalDisabled = true
			return
		}

		if k.metal.source == "" {
			k.disableMetal("no source")
			return
		}

		inputs, freeInputs, ok := cStringVector(k.inputs)
		if !ok {
			freeInputs()
			k.disableMetal("creating input names failed")
			return
		}
		defer freeInputs()

		outputs, freeOutputs, ok := cStringVector(k.outputs)
		if !ok {
			freeOutputs()
			k.disableMetal("creating output names failed")
			return
		}
		defer freeOutputs()

		cName := C.CString(k.name)
		defer C.free(unsafe.Pointer(cName))
		cSource := C.CString(k.metal.source)
		defer C.free(unsafe.Pointer(cSource))
		cHeader := C.CString(k.metal.header)
		defer C.free(unsafe.Pointer(cHeader))

		k.metalKernel = C.mlx_fast_metal_kernel_new(
			cName,
			inputs,
			outputs,
			cSource,
			cHeader,
			// ensure_row_contiguous, so kernels can index inputs linearly.
			C.bool(true),
			C.bool(false),
		)
		if k.metalKernel.ctx == nil {
			k.disableMetal("creating kernel failed")
		}
	})
	return k.metalKernel, !k.metalDisabled
}

func (k *gpuKernel) applyMetal(launch gpuLaunch) ([]*Array, bool) {
	if k.metalDisabled {
		return nil, false
	}
	kernel, ok := k.getMetal()
	if !ok {
		return nil, false
	}

	cfg := C.mlx_fast_metal_kernel_config_new()
	defer C.mlx_fast_metal_kernel_config_free(cfg)
	for _, arg := range launch.dtypes {
		name := C.CString(arg.name)
		rc := C.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, name, C.mlx_dtype(arg.dtype))
		C.free(unsafe.Pointer(name))
		if rc != 0 {
			k.disableMetal("setting dtype template arg failed")
			return nil, false
		}
	}
	for _, arg := range launch.ints {
		name := C.CString(arg.name)
		rc := C.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, name, C.int(arg.value))
		C.free(unsafe.Pointer(name))
		if rc != 0 {
			k.disableMetal("setting int template arg failed")
			return nil, false
		}
	}
	for _, out := range launch.outputs {
		shape := make([]C.int, len(out.shape))
		for i, d := range out.shape {
			shape[i] = C.int(d)
		}
		if C.mlx_fast_metal_kernel_config_add_output_arg(cfg, unsafe.SliceData(shape), C.size_t(len(shape)), C.mlx_dtype(out.dtype)) != 0 {
			k.disableMetal("adding output failed")
			return nil, false
		}
	}
	if C.mlx_fast_metal_kernel_config_set_grid(cfg, C.int(launch.grid[0]), C.int(launch.grid[1]), C.int(launch.grid[2])) != 0 ||
		C.mlx_fast_metal_kernel_config_set_thread_group(cfg, C.int(launch.threadGroup[0]), C.int(launch.threadGroup[1]), C.int(launch.threadGroup[2])) != 0 {
		k.disableMetal("setting grid failed")
		return nil, false
	}

	inputs := make([]C.mlx_array, len(launch.inputs))
	for i, in := range launch.inputs {
		inputs[i] = in.ctx
	}
	inVec := C.mlx_vector_array_new_data(unsafe.SliceData(inputs), C.size_t(len(inputs)))
	defer C.mlx_vector_array_free(inVec)
	outVec := C.mlx_vector_array_new()
	defer C.mlx_vector_array_free(outVec)
	if C.mlx_fast_metal_kernel_apply(&outVec, kernel, inVec, cfg, DefaultStream().ctx) != 0 {
		k.disableMetal("launching failed")
		return nil, false
	}
	if int(C.mlx_vector_array_size(outVec)) < len(launch.outputs) {
		return nil, false
	}

	outs := make([]*Array, len(launch.outputs))
	for i, out := range launch.outputs {
		outs[i] = New(out.name)
		C.mlx_vector_array_get(&outs[i].ctx, outVec, C.size_t(i))
	}
	return outs, true
}

func (k *gpuKernel) getCUDA() (C.mlx_fast_cuda_kernel, bool) {
	k.cudaOnce.Do(func() {
		if !CUDAIsAvailable() {
			k.cudaDisabled = true
			return
		}

		if k.cuda.source == "" {
			k.disableCUDA("no source")
			return
		}

		inputs, freeInputs, ok := cStringVector(k.inputs)
		if !ok {
			freeInputs()
			k.disableCUDA("creating input names failed")
			return
		}
		defer freeInputs()

		outputs, freeOutputs, ok := cStringVector(k.outputs)
		if !ok {
			freeOutputs()
			k.disableCUDA("creating output names failed")
			return
		}
		defer freeOutputs()

		cName := C.CString(k.name)
		defer C.free(unsafe.Pointer(cName))
		cSource := C.CString(k.cuda.source)
		defer C.free(unsafe.Pointer(cSource))
		cHeader := C.CString(k.cuda.header)
		defer C.free(unsafe.Pointer(cHeader))

		k.cudaKernel = C.mlx_fast_cuda_kernel_new(
			cName,
			inputs,
			outputs,
			cSource,
			cHeader,
			C.bool(true),
			C.int(0),
		)
		if k.cudaKernel.ctx == nil {
			k.disableCUDA("creating kernel failed")
		}
	})
	return k.cudaKernel, !k.cudaDisabled
}

func (k *gpuKernel) applyCUDA(launch gpuLaunch) ([]*Array, bool) {
	if k.cudaDisabled {
		return nil, false
	}
	kernel, ok := k.getCUDA()
	if !ok {
		return nil, false
	}

	cfg := C.mlx_fast_cuda_kernel_config_new()
	defer C.mlx_fast_cuda_kernel_config_free(cfg)
	for _, arg := range launch.dtypes {
		name := C.CString(arg.name)
		rc := C.mlx_fast_cuda_kernel_config_add_template_arg_dtype(cfg, name, C.mlx_dtype(arg.dtype))
		C.free(unsafe.Pointer(name))
		if rc != 0 {
			k.disableCUDA("setting dtype template arg failed")
			return nil, false
		}
	}
	for _, arg := range launch.ints {
		name := C.CString(arg.name)
		rc := C.mlx_fast_cuda_kernel_config_add_template_arg_int(cfg, name, C.int(arg.value))
		C.free(unsafe.Pointer(name))
		if rc != 0 {
			k.disableCUDA("setting int template arg failed")
			return nil, false
		}
	}
	for _, out := range launch.outputs {
		shape := make([]C.int, len(out.shape))
		for i, d := range out.shape {
			shape[i] = C.int(d)
		}
		if C.mlx_fast_cuda_kernel_config_add_output_arg(cfg, unsafe.SliceData(shape), C.size_t(len(shape)), C.mlx_dtype(out.dtype)) != 0 {
			k.disableCUDA("adding output failed")
			return nil, false
		}
	}
	if C.mlx_fast_cuda_kernel_config_set_grid(cfg, C.int(launch.grid[0]), C.int(launch.grid[1]), C.int(launch.grid[2])) != 0 ||
		C.mlx_fast_cuda_kernel_config_set_thread_group(cfg, C.int(launch.threadGroup[0]), C.int(launch.threadGroup[1]), C.int(launch.threadGroup[2])) != 0 {
		k.disableCUDA("setting grid failed")
		return nil, false
	}

	inputs := make([]C.mlx_array, len(launch.inputs))
	for i, in := range launch.inputs {
		inputs[i] = in.ctx
	}
	inVec := C.mlx_vector_array_new_data(unsafe.SliceData(inputs), C.size_t(len(inputs)))
	defer C.mlx_vector_array_free(inVec)
	outVec := C.mlx_vector_array_new()
	defer C.mlx_vector_array_free(outVec)
	if C.mlx_fast_cuda_kernel_apply(&outVec, kernel, inVec, cfg, DefaultStream().ctx) != 0 {
		k.disableCUDA("launching failed")
		return nil, false
	}
	if int(C.mlx_vector_array_size(outVec)) < len(launch.outputs) {
		return nil, false
	}

	outs := make([]*Array, len(launch.outputs))
	for i, out := range launch.outputs {
		outs[i] = New(out.name)
		C.mlx_vector_array_get(&outs[i].ctx, outVec, C.size_t(i))
	}
	return outs, true
}
