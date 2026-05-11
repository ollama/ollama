package discover

import (
	"fmt"
	"syscall"
	"unsafe"
)

const (
	vkSuccess                           = 0
	vkStructureTypeInstanceCreateInfo   = 1
	vkPhysicalDeviceTypeIntegratedGPU   = 1
	vkMaxPhysicalDeviceNameSize         = 256
	vkPhysicalDevicePropertiesByteCount = 4096
)

type vkInstanceCreateInfo struct {
	SType                   uint32
	PNext                   uintptr
	Flags                   uint32
	PApplicationInfo        uintptr
	EnabledLayerCount       uint32
	PpEnabledLayerNames     uintptr
	EnabledExtensionCount   uint32
	PpEnabledExtensionNames uintptr
}

var (
	vulkanDLL                     = syscall.NewLazyDLL("vulkan-1.dll")
	vkCreateInstanceProc          = vulkanDLL.NewProc("vkCreateInstance")
	vkDestroyInstanceProc         = vulkanDLL.NewProc("vkDestroyInstance")
	vkEnumeratePhysicalDevices    = vulkanDLL.NewProc("vkEnumeratePhysicalDevices")
	vkGetPhysicalDeviceProperties = vulkanDLL.NewProc("vkGetPhysicalDeviceProperties")
)

func init() {
	probeLlamaServerVulkanDevices = windowsVulkanPhysicalDevices
}

func windowsVulkanPhysicalDevices() ([]vulkanPhysicalDevice, error) {
	if err := vkCreateInstanceProc.Find(); err != nil {
		return nil, fmt.Errorf("vkCreateInstance unavailable: %w", err)
	}
	if err := vkDestroyInstanceProc.Find(); err != nil {
		return nil, fmt.Errorf("vkDestroyInstance unavailable: %w", err)
	}
	if err := vkEnumeratePhysicalDevices.Find(); err != nil {
		return nil, fmt.Errorf("vkEnumeratePhysicalDevices unavailable: %w", err)
	}
	if err := vkGetPhysicalDeviceProperties.Find(); err != nil {
		return nil, fmt.Errorf("vkGetPhysicalDeviceProperties unavailable: %w", err)
	}

	createInfo := vkInstanceCreateInfo{SType: vkStructureTypeInstanceCreateInfo}
	var instance uintptr
	result, _, err := vkCreateInstanceProc.Call(
		uintptr(unsafe.Pointer(&createInfo)),
		0,
		uintptr(unsafe.Pointer(&instance)),
	)
	if result != vkSuccess {
		return nil, fmt.Errorf("vkCreateInstance failed: result=%d error=%w", result, err)
	}
	defer vkDestroyInstanceProc.Call(instance, 0)

	var count uint32
	result, _, err = vkEnumeratePhysicalDevices.Call(
		instance,
		uintptr(unsafe.Pointer(&count)),
		0,
	)
	if result != vkSuccess {
		return nil, fmt.Errorf("vkEnumeratePhysicalDevices count failed: result=%d error=%w", result, err)
	}
	if count == 0 {
		return nil, nil
	}

	physicalDevices := make([]uintptr, int(count))
	result, _, err = vkEnumeratePhysicalDevices.Call(
		instance,
		uintptr(unsafe.Pointer(&count)),
		uintptr(unsafe.Pointer(&physicalDevices[0])),
	)
	if result != vkSuccess {
		return nil, fmt.Errorf("vkEnumeratePhysicalDevices failed: result=%d error=%w", result, err)
	}

	devices := make([]vulkanPhysicalDevice, 0, count)
	for _, physicalDevice := range physicalDevices[:int(count)] {
		properties := make([]byte, vkPhysicalDevicePropertiesByteCount)
		vkGetPhysicalDeviceProperties.Call(
			physicalDevice,
			uintptr(unsafe.Pointer(&properties[0])),
		)
		deviceType := *(*uint32)(unsafe.Pointer(&properties[16]))
		deviceNameBytes := properties[20 : 20+vkMaxPhysicalDeviceNameSize]
		devices = append(devices, vulkanPhysicalDevice{
			Name:       nulTerminatedString(deviceNameBytes),
			Integrated: deviceType == vkPhysicalDeviceTypeIntegratedGPU,
		})
	}

	return devices, nil
}

func nulTerminatedString(data []byte) string {
	for i, b := range data {
		if b == 0 {
			return string(data[:i])
		}
	}
	return string(data)
}
