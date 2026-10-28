//go:build windows

package discover

import (
	"fmt"
	"unsafe"

	"github.com/ollama/ollama/llm"
	"golang.org/x/sys/windows"
)

// Vulkan values are defined locally so VK_INTEL_device_info (headers 1.4.363+)
// can be queried regardless of the Vulkan SDK used at build time.
const (
	intelVendorID = 0x8086

	vkIntelDeviceInfoExtensionName = "VK_INTEL_device_info"
	vkEXTPCIBusInfoExtensionName   = "VK_EXT_pci_bus_info"

	vkStructureTypeApplicationInfo                   = 0
	vkStructureTypePhysicalDeviceProperties2         = 1000059001
	vkStructureTypePhysicalDevicePCIBusInfoEXT       = 1000212000
	vkStructureTypePhysicalDeviceInfoPropertiesINTEL = 1000708000

	vkAPIVersion11 = 1<<22 | 1<<12

	vkExtensionPropertiesByteCount = 260 // char[256] + uint32
)

type vkApplicationInfo struct {
	SType              uint32
	PNext              uintptr
	PApplicationName   uintptr
	ApplicationVersion uint32
	PEngineName        uintptr
	EngineVersion      uint32
	APIVersion         uint32
}

// Nested structures live in one allocation so pointers between them stay valid
// for the duration of the Vulkan call.
type intelVulkanInstanceInfo struct {
	create vkInstanceCreateInfo
	app    vkApplicationInfo
}

type vkPhysicalDeviceProperties2 struct {
	SType      uint32
	PNext      uintptr
	Properties [vkPhysicalDevicePropertiesByteCount]byte
}

type vkPhysicalDeviceInfoPropertiesINTEL struct {
	SType                   uint32
	PNext                   uintptr
	DeviceIPVersionArch     uint32
	DeviceIPVersionRelease  uint32
	DeviceIPVersionRevision uint32
}

type vkPhysicalDevicePCIBusInfoPropertiesEXT struct {
	SType       uint32
	PNext       uintptr
	PCIDomain   uint32
	PCIBus      uint32
	PCIDevice   uint32
	PCIFunction uint32
}

type intelVulkanPropertiesQuery struct {
	props2 vkPhysicalDeviceProperties2
	intel  vkPhysicalDeviceInfoPropertiesINTEL
	pci    vkPhysicalDevicePCIBusInfoPropertiesEXT
}

func probeIntelVulkanDevices(libDirs []string) ([]intelVulkanDevice, error) {
	vulkanPath, err := llm.WindowsVulkanRuntimeDLLPath(libDirs)
	if err != nil {
		return nil, err
	}
	vulkanDLL, err := loadDLLFromPath(vulkanPath)
	if err != nil {
		return nil, err
	}

	procs := map[string]*windows.Proc{}
	for _, name := range []string{
		"vkEnumerateInstanceVersion",
		"vkCreateInstance",
		"vkDestroyInstance",
		"vkEnumeratePhysicalDevices",
		"vkGetPhysicalDeviceProperties",
		"vkGetPhysicalDeviceProperties2",
		"vkEnumerateDeviceExtensionProperties",
	} {
		proc, err := findProc(vulkanDLL, name)
		if err != nil {
			// Loaders older than Vulkan 1.1 lack vkEnumerateInstanceVersion and vkGetPhysicalDeviceProperties2.
			return nil, fmt.Errorf("%s unavailable: %w", name, err)
		}
		procs[name] = proc
	}

	var loaderVersion uint32
	if result, _, _ := procs["vkEnumerateInstanceVersion"].Call(uintptr(unsafe.Pointer(&loaderVersion))); int32(result) != vkSuccess {
		return nil, fmt.Errorf("vkEnumerateInstanceVersion failed: result=%d", int32(result))
	}
	if loaderVersion < vkAPIVersion11 {
		return nil, fmt.Errorf("vulkan loader version %#x older than 1.1", loaderVersion)
	}

	info := &intelVulkanInstanceInfo{}
	info.app = vkApplicationInfo{SType: vkStructureTypeApplicationInfo, APIVersion: vkAPIVersion11}
	info.create = vkInstanceCreateInfo{
		SType:            vkStructureTypeInstanceCreateInfo,
		PApplicationInfo: uintptr(unsafe.Pointer(&info.app)),
	}
	var instance uintptr
	result, _, callErr := procs["vkCreateInstance"].Call(
		uintptr(unsafe.Pointer(&info.create)),
		0,
		uintptr(unsafe.Pointer(&instance)),
	)
	if int32(result) != vkSuccess {
		return nil, fmt.Errorf("vkCreateInstance failed: result=%d error=%w", int32(result), callErr)
	}
	defer procs["vkDestroyInstance"].Call(instance, 0)

	var count uint32
	if result, _, _ := procs["vkEnumeratePhysicalDevices"].Call(instance, uintptr(unsafe.Pointer(&count)), 0); int32(result) != vkSuccess {
		return nil, fmt.Errorf("vkEnumeratePhysicalDevices count failed: result=%d", int32(result))
	}
	if count == 0 {
		return nil, nil
	}
	physicalDevices := make([]uintptr, int(count))
	if result, _, _ := procs["vkEnumeratePhysicalDevices"].Call(instance, uintptr(unsafe.Pointer(&count)), uintptr(unsafe.Pointer(&physicalDevices[0]))); int32(result) < vkSuccess {
		return nil, fmt.Errorf("vkEnumeratePhysicalDevices failed: result=%d", int32(result))
	}

	var devices []intelVulkanDevice
	for _, physicalDevice := range physicalDevices[:int(count)] {
		properties := make([]byte, vkPhysicalDevicePropertiesByteCount)
		procs["vkGetPhysicalDeviceProperties"].Call(physicalDevice, uintptr(unsafe.Pointer(&properties[0])))
		apiVersion := *(*uint32)(unsafe.Pointer(&properties[0]))
		vendorID := *(*uint32)(unsafe.Pointer(&properties[8]))
		deviceType := *(*uint32)(unsafe.Pointer(&properties[16]))
		if vendorID != intelVendorID || apiVersion < vkAPIVersion11 {
			continue
		}

		device := intelVulkanDevice{
			Name:       nulTerminatedString(properties[20 : 20+vkMaxPhysicalDeviceNameSize]),
			Integrated: deviceType == vkPhysicalDeviceTypeIntegratedGPU,
		}

		extensions := windowsVulkanDeviceExtensions(procs["vkEnumerateDeviceExtensionProperties"], physicalDevice)
		hasIntelInfo := extensions[vkIntelDeviceInfoExtensionName]
		hasPCIBusInfo := extensions[vkEXTPCIBusInfoExtensionName]

		q := &intelVulkanPropertiesQuery{}
		q.props2.SType = vkStructureTypePhysicalDeviceProperties2
		q.intel.SType = vkStructureTypePhysicalDeviceInfoPropertiesINTEL
		q.pci.SType = vkStructureTypePhysicalDevicePCIBusInfoEXT
		next := &q.props2.PNext
		if hasIntelInfo {
			*next = uintptr(unsafe.Pointer(&q.intel))
			next = &q.intel.PNext
		}
		if hasPCIBusInfo {
			*next = uintptr(unsafe.Pointer(&q.pci))
		}
		procs["vkGetPhysicalDeviceProperties2"].Call(physicalDevice, uintptr(unsafe.Pointer(&q.props2)))

		if hasIntelInfo {
			device.IPVersion = packIntelIPVersion(q.intel.DeviceIPVersionArch, q.intel.DeviceIPVersionRelease, q.intel.DeviceIPVersionRevision)
		}
		if hasPCIBusInfo {
			device.PCIID = vulkanPCIBusID(q.pci.PCIDomain, q.pci.PCIBus, q.pci.PCIDevice, q.pci.PCIFunction)
		}
		devices = append(devices, device)
	}

	return devices, nil
}

func windowsVulkanDeviceExtensions(enumerate *windows.Proc, physicalDevice uintptr) map[string]bool {
	var count uint32
	if result, _, _ := enumerate.Call(physicalDevice, 0, uintptr(unsafe.Pointer(&count)), 0); int32(result) != vkSuccess || count == 0 {
		return nil
	}
	buf := make([]byte, int(count)*vkExtensionPropertiesByteCount)
	if result, _, _ := enumerate.Call(physicalDevice, 0, uintptr(unsafe.Pointer(&count)), uintptr(unsafe.Pointer(&buf[0]))); int32(result) < vkSuccess {
		return nil
	}

	extensions := make(map[string]bool, int(count))
	for i := range int(count) {
		offset := i * vkExtensionPropertiesByteCount
		extensions[nulTerminatedString(buf[offset:offset+256])] = true
	}
	return extensions
}
