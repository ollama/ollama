package mlx

// #include "generated.h"
// #include <stdlib.h>
import "C"

import (
	"fmt"
	"slices"
	"strconv"
	"strings"
	"unsafe"
)

type cudaCompilationTarget struct {
	virtual    bool
	capability int
	qualifier  byte
}

// cudaFamilyCompatibility follows NVIDIA's family-specific compiler target
// compatibility table.
var cudaFamilyCompatibility = map[int][]int{
	100: {100, 103, 107},
	103: {103, 107},
	107: {107},
	110: {110},
	120: {120, 121},
	121: {121},
}

func parseCUDADeviceArchitecture(architecture string) (int, error) {
	value, ok := strings.CutPrefix(architecture, "sm_")
	if !ok {
		return 0, fmt.Errorf("invalid CUDA device architecture %q", architecture)
	}

	capability, err := strconv.Atoi(value)
	if err != nil || capability < 10 {
		return 0, fmt.Errorf("invalid CUDA device architecture %q", architecture)
	}
	return capability, nil
}

func parseCUDACompilationTarget(value string) (cudaCompilationTarget, error) {
	target := cudaCompilationTarget{}
	architecture, ok := strings.CutPrefix(value, "sm_")
	if !ok {
		architecture, ok = strings.CutPrefix(value, "compute_")
		target.virtual = ok
	}
	if !ok {
		return target, fmt.Errorf("invalid CUDA compilation target %q", value)
	}

	if strings.HasSuffix(architecture, "a") || strings.HasSuffix(architecture, "f") {
		target.qualifier = architecture[len(architecture)-1]
		architecture = architecture[:len(architecture)-1]
	}

	capability, err := strconv.Atoi(architecture)
	if err != nil || capability < 10 {
		return target, fmt.Errorf("invalid CUDA compilation target %q", value)
	}
	target.capability = capability
	return target, nil
}

func (target cudaCompilationTarget) supports(capability int) (bool, error) {
	switch target.qualifier {
	case 'a':
		return target.capability == capability, nil
	case 'f':
		if target.capability == capability {
			return true, nil
		}
		family, ok := cudaFamilyCompatibility[target.capability]
		if !ok {
			return false, fmt.Errorf("unknown CUDA family-specific target %d", target.capability)
		}
		return slices.Contains(family, capability), nil
	}

	if target.virtual {
		return capability >= target.capability, nil
	}
	return capability/10 == target.capability/10 && capability >= target.capability, nil
}

func cudaArchitectureSupported(architecture string, compilationTargets []string) (bool, error) {
	capability, err := parseCUDADeviceArchitecture(architecture)
	if err != nil {
		return false, err
	}

	targets := make([]cudaCompilationTarget, 0, len(compilationTargets))
	for _, value := range compilationTargets {
		target, err := parseCUDACompilationTarget(value)
		if err != nil {
			return false, err
		}
		targets = append(targets, target)
	}

	var familyErr error
	for _, target := range targets {
		supported, err := target.supports(capability)
		if err != nil {
			familyErr = err
			continue
		}
		if supported {
			return true, nil
		}
	}
	return false, familyErr
}

func cudaCompilationTargets() ([]string, error) {
	targets := mlxCheck(C.mlx_vector_string_new())
	defer freeVectorString(targets)
	if err := mlxError(C.mlx_cuda_get_arch_list(&targets)); err != nil {
		return nil, err
	}

	result := make([]string, int(mlxCheck(C.mlx_vector_string_size(targets))))
	for i := range result {
		var value *C.char
		if err := mlxError(C.mlx_vector_string_get(&value, targets, C.size_t(i))); err != nil {
			return nil, err
		}
		result[i] = C.GoString(value)
	}
	return result, nil
}

func deviceInfoString(info C.mlx_device_info, key string) (string, error) {
	cKey := C.CString(key)
	defer C.free(unsafe.Pointer(cKey))

	var exists C.bool
	if err := mlxError(C.mlx_device_info_has_key(&exists, info, cKey)); err != nil {
		return "", err
	}
	if !bool(exists) {
		return "", fmt.Errorf("MLX device info does not contain %q", key)
	}

	var value *C.char
	if err := mlxError(C.mlx_device_info_get_string(&value, info, cKey)); err != nil {
		return "", err
	}
	return C.GoString(value), nil
}

// CheckCUDADeviceSupport verifies that this MLX build contains code compatible
// with the first CUDA device.
func CheckCUDADeviceSupport() error {
	device := mlxCheck(C.mlx_device_new_type(C.MLX_GPU, 0))
	defer freeDevice(device)

	info := mlxCheck(C.mlx_device_info_new())
	defer freeDeviceInfo(info)
	if err := mlxError(C.mlx_device_info_get(&info, device)); err != nil {
		return fmt.Errorf("get CUDA device info: %w", err)
	}

	architecture, err := deviceInfoString(info, "architecture")
	if err != nil {
		return err
	}
	targets, err := cudaCompilationTargets()
	if err != nil {
		return fmt.Errorf("get MLX CUDA compilation targets: %w", err)
	}
	if len(targets) == 0 {
		return fmt.Errorf("MLX CUDA build reported no compilation targets")
	}

	supported, err := cudaArchitectureSupported(architecture, targets)
	if err != nil {
		return err
	}
	if !supported {
		return fmt.Errorf(
			"CUDA device %s is not supported by this MLX build (compilation targets: %s)",
			architecture,
			strings.Join(targets, ", "),
		)
	}
	return nil
}
