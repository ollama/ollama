//go:build dragonfly || freebsd || netbsd || openbsd

package discover

import "github.com/ollama/ollama/format"
//import sysctl "github.com/lorenzosaino/go-sysctl" // sysctl: this is Linux-only, see https://github.com/lorenzosaino/go-sysctl/issues/7
import sysctl "github.com/blabber/go-freebsd-sysctl/sysctl" // sysctl: this is FreeBSD-only basic library
import (
	"log/slog"
	"syscall"
)

/*
#cgo CFLAGS: -I/usr/local/include
#cgo LDFLAGS: -L/usr/local/lib -lvulkan

#include <stdbool.h>
#include <unistd.h>
#include <vulkan/vulkan.h>

bool hasVulkanSupport(uint64_t *memSize) {
  VkInstance instance;

	VkApplicationInfo appInfo = { VK_STRUCTURE_TYPE_APPLICATION_INFO };
	appInfo.pApplicationName = "Ollama";
	appInfo.apiVersion = VK_API_VERSION_1_0;

	VkInstanceCreateInfo createInfo = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
	createInfo.pApplicationInfo = &appInfo;

	// Create a Vulkan instance
	if (vkCreateInstance(&createInfo, NULL, &instance) != VK_SUCCESS)
		return false;

	// Fetch the first physical Vulkan device. Note that numDevices is overwritten with the number of devices found
	uint32_t numDevices = 1;
	VkPhysicalDevice device;
	vkEnumeratePhysicalDevices(instance, &numDevices, &device);
	if (numDevices == 0) {
		vkDestroyInstance(instance, NULL);
		return false;
	}

	// Fetch the memory information for this device.
	VkPhysicalDeviceMemoryProperties memProperties;
	vkGetPhysicalDeviceMemoryProperties(device, &memProperties);

	// Add up all the heaps.
	VkDeviceSize totalMemory = 0;
	for (uint32_t i = 0; i < memProperties.memoryHeapCount; ++i) {
		if (memProperties.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
			*memSize += memProperties.memoryHeaps[i].size;
		}
	}

	vkDestroyInstance(instance, NULL);
	return true;
}
*/
import "C"

func GetGPUInfo() GpuInfoList {
	var gpuMem C.uint64_t
	if C.hasVulkanSupport(&gpuMem) {
		// Vulkan supported
		return []GpuInfo{
			{
				Library: 				"vulkan",
				ID:							"0",
				MinimumMemory: 	512 * format.MebiByte,
				memInfo: 	memInfo{
					FreeMemory: uint64(gpuMem),
					TotalMemory: uint64(gpuMem),
				},
			},
		}
	}

	// CPU fallback
	cpuMem, _ := GetCPUMem()
	return []GpuInfo{
		{
			Library: "cpu",
			memInfo: cpuMem,
		},
	}
}

func GetCPUInfo() GpuInfoList {
	mem, _ := GetCPUMem()
	return []GpuInfo{
		{
			Library: "cpu",
			memInfo: mem,
		},
	}
}

func GetCPUMem() (memInfo, error) {
	// all involved sysctl variables
	sysctl_vm_page_size, _ := sysctl.GetInt64("vm.stats.vm.v_page_size") // memory page size
	sysctl_hw_physmem, _ := sysctl.GetInt64("hw.physmem") // physical memory in bytes
	sysctl_vm_free_count, _ := sysctl.GetInt64("vm.stats.vm.v_free_count") // free page count
	sysctl_vm_swap_total, _ := sysctl.GetInt64("vm.swap_total") // total swap size in bytes

	// individual values
	total_memory := uint64(sysctl_hw_physmem)
	free_memory := uint64(sysctl_vm_free_count) * uint64(sysctl_vm_page_size)
	free_swap := uint64(sysctl_vm_swap_total) // wrong to use the total swap size here, should be vm.swap_free, see https://bugs.freebsd.org/bugzilla/show_bug.cgi?id=280909

	slog.Debug("gpu_bsd.go::GetCPUMem::GetCPUMem", "total_memory", total_memory, "free_memory", free_memory, "free_swap", free_swap)

	return memInfo{
		TotalMemory: uint64(total_memory),
		FreeMemory: uint64(free_memory),
		FreeSwap: uint64(free_swap),
	}, nil
}

func (l GpuInfoList) GetVisibleDevicesEnv() (string, string) {
	// No-op on darwin
	return "", ""
}

func GetSystemInfo() SystemInfo {
	mem, _ := GetCPUMem()
	perfCores := uint32(0) // TODO what is the correct sysctl?

	query := "kern.smp.cores"
	efficiencyCores, _ := syscall.SysctlUint32(query) // On x86 xeon this wont return data

	// Determine thread count
	query = "hw.ncpu"
	logicalCores, _ := syscall.SysctlUint32(query)

	return SystemInfo{
		System: CPUInfo{
			GpuInfo: GpuInfo{
				memInfo: mem,
			},
			CPUs: []CPU{
				{
					CoreCount:           int(perfCores + efficiencyCores),
					EfficiencyCoreCount: int(efficiencyCores),
					ThreadCount:         int(logicalCores),
				},
			},
		},
		GPUs: GetGPUInfo(),
	}
}
