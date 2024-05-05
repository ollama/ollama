//go:build dragonfly || freebsd || netbsd || openbsd

package gpu

import "github.com/ollama/ollama/format"

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

	// Create a Vulkan instance.
	if (vkCreateInstance(&createInfo, NULL, &instance) != VK_SUCCESS) {
		return false;
	}

	// Fetch the first physical Vulkan device. Note that numDevices is overwritten
	// with the number of devices found.
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

	// Check if there is hardware support for Vulkan.
	if C.hasVulkanSupport(&gpuMem) {
		return []GpuInfo{
			{
				Library:       "vulkan",
				ID:            "0",
				MinimumMemory: 512 * format.MebiByte,
				memInfo: memInfo{
					FreeMemory:  uint64(gpuMem),
					TotalMemory: uint64(gpuMem),
				},
			},
		}
	}

	// If there is no Vulkan support, default back to CPU.
	cpuMem, _ := GetCPUMem()
	return []GpuInfo{
		{
			Library: "cpu",
			Variant: GetCPUVariant(),
			memInfo: cpuMem,
		},
	}
}

func GetCPUMem() (memInfo, error) {
	size := C.sysconf(C._SC_PHYS_PAGES) * C.sysconf(C._SC_PAGE_SIZE)
	return memInfo{TotalMemory: uint64(size)}, nil
}

func (l GpuInfoList) GetVisibleDevicesEnv() (string, string) {
	return "", ""
}
