//go:build linux && cgo

package discover

/*
#include <stdint.h>
#include <stdlib.h>

typedef struct {
	uint32_t sType;
	const void * pNext;
	const char * pApplicationName;
	uint32_t applicationVersion;
	const char * pEngineName;
	uint32_t engineVersion;
	uint32_t apiVersion;
} ollama_vk_application_info;

typedef struct {
	uint32_t sType;
	const void * pNext;
	uint32_t flags;
	const ollama_vk_application_info * pApplicationInfo;
	uint32_t enabledLayerCount;
	const char * const * ppEnabledLayerNames;
	uint32_t enabledExtensionCount;
	const char * const * ppEnabledExtensionNames;
} ollama_vk_instance_create_info;

typedef struct {
	char extensionName[256];
	uint32_t specVersion;
} ollama_vk_extension_properties;

// Leading fields of VkPhysicalDeviceProperties; the padding covers the rest of the struct.
typedef struct {
	uint32_t apiVersion;
	uint32_t driverVersion;
	uint32_t vendorID;
	uint32_t deviceID;
	uint32_t deviceType;
	char deviceName[256];
	uint64_t padding[512];
} ollama_vk_physical_device_properties;

typedef struct {
	uint32_t sType;
	void * pNext;
	ollama_vk_physical_device_properties properties;
} ollama_vk_physical_device_properties2;

typedef struct {
	uint32_t sType;
	void * pNext;
	uint32_t deviceIpVersionArch;
	uint32_t deviceIpVersionRelease;
	uint32_t deviceIpVersionRevision;
} ollama_vk_physical_device_info_properties_intel;

typedef struct {
	uint32_t sType;
	void * pNext;
	uint32_t pciDomain;
	uint32_t pciBus;
	uint32_t pciDevice;
	uint32_t pciFunction;
} ollama_vk_physical_device_pci_bus_info_properties;

// Chained structures share one allocation so pNext pointers stay in C memory.
typedef struct {
	ollama_vk_physical_device_properties2 props2;
	ollama_vk_physical_device_info_properties_intel intel;
	ollama_vk_physical_device_pci_bus_info_properties pci;
} ollama_intel_vk_properties_query;

typedef void * (*ollama_vk_get_instance_proc_addr_fn)(void *, const char *);
typedef int32_t (*ollama_vk_enumerate_instance_version_fn)(uint32_t *);
typedef int32_t (*ollama_vk_create_instance_fn)(const ollama_vk_instance_create_info *, const void *, void **);
typedef void (*ollama_vk_destroy_instance_fn)(void *, const void *);
typedef int32_t (*ollama_vk_enumerate_physical_devices_fn)(void *, uint32_t *, void **);
typedef void (*ollama_vk_get_physical_device_properties_fn)(void *, ollama_vk_physical_device_properties *);
typedef void (*ollama_vk_get_physical_device_properties2_fn)(void *, ollama_vk_physical_device_properties2 *);
typedef int32_t (*ollama_vk_enumerate_device_extension_properties_fn)(void *, const char *, uint32_t *, ollama_vk_extension_properties *);

static void * ollama_call_vk_get_instance_proc_addr(void * fn, void * instance, const char * name) {
	return ((ollama_vk_get_instance_proc_addr_fn) fn)(instance, name);
}

static int32_t ollama_call_vk_enumerate_instance_version(void * fn, uint32_t * version) {
	return ((ollama_vk_enumerate_instance_version_fn) fn)(version);
}

static int32_t ollama_call_vk_create_instance(void * fn, const ollama_vk_instance_create_info * info, void ** instance) {
	return ((ollama_vk_create_instance_fn) fn)(info, NULL, instance);
}

static void ollama_call_vk_destroy_instance(void * fn, void * instance) {
	((ollama_vk_destroy_instance_fn) fn)(instance, NULL);
}

static int32_t ollama_call_vk_enumerate_physical_devices(void * fn, void * instance, uint32_t * count, void ** devices) {
	return ((ollama_vk_enumerate_physical_devices_fn) fn)(instance, count, devices);
}

static void ollama_call_vk_get_physical_device_properties(void * fn, void * device, ollama_vk_physical_device_properties * props) {
	((ollama_vk_get_physical_device_properties_fn) fn)(device, props);
}

static void ollama_call_vk_get_physical_device_properties2(void * fn, void * device, ollama_vk_physical_device_properties2 * props) {
	((ollama_vk_get_physical_device_properties2_fn) fn)(device, props);
}

static int32_t ollama_call_vk_enumerate_device_extension_properties(void * fn, void * device, uint32_t * count, ollama_vk_extension_properties * props) {
	return ((ollama_vk_enumerate_device_extension_properties_fn) fn)(device, NULL, count, props);
}
*/
import "C"

import (
	"errors"
	"fmt"
	"unsafe"
)

func probeIntelVulkanDevices([]string) ([]intelVulkanDevice, error) {
	vulkan, err := dlopenFirst([]string{"libvulkan.so.1", "libvulkan.so"}, false)
	if err != nil {
		return nil, err
	}
	getInstanceProcAddr, err := dlsym(vulkan, "vkGetInstanceProcAddr")
	if err != nil {
		return nil, err
	}

	// Loaders older than Vulkan 1.1 lack vkEnumerateInstanceVersion and vkGetPhysicalDeviceProperties2.
	enumerateInstanceVersion := vkProcAddr(getInstanceProcAddr, nil, "vkEnumerateInstanceVersion")
	if enumerateInstanceVersion == nil {
		return nil, errors.New("vulkan loader older than 1.1")
	}
	var loaderVersion C.uint32_t
	if ret := C.ollama_call_vk_enumerate_instance_version(enumerateInstanceVersion, &loaderVersion); ret != vkSuccess {
		return nil, fmt.Errorf("vkEnumerateInstanceVersion failed: %d", int(ret))
	}
	if loaderVersion < vkAPIVersion11 {
		return nil, fmt.Errorf("vulkan loader version %#x older than 1.1", uint32(loaderVersion))
	}

	createInstance := vkProcAddr(getInstanceProcAddr, nil, "vkCreateInstance")
	if createInstance == nil {
		return nil, errors.New("vkCreateInstance unavailable")
	}

	appInfo := (*C.ollama_vk_application_info)(C.calloc(1, C.sizeof_ollama_vk_application_info))
	if appInfo == nil {
		return nil, errors.New("out of memory")
	}
	defer C.free(unsafe.Pointer(appInfo))
	appInfo.sType = vkStructureTypeApplicationInfo
	appInfo.apiVersion = vkAPIVersion11

	createInfo := (*C.ollama_vk_instance_create_info)(C.calloc(1, C.sizeof_ollama_vk_instance_create_info))
	if createInfo == nil {
		return nil, errors.New("out of memory")
	}
	defer C.free(unsafe.Pointer(createInfo))
	createInfo.sType = vkStructureTypeInstanceCreateInfo
	createInfo.pApplicationInfo = appInfo

	var instance unsafe.Pointer
	if ret := C.ollama_call_vk_create_instance(createInstance, createInfo, &instance); ret != vkSuccess || instance == nil {
		return nil, fmt.Errorf("vkCreateInstance failed: %d", int(ret))
	}

	destroyInstance := vkProcAddr(getInstanceProcAddr, instance, "vkDestroyInstance")
	if destroyInstance == nil {
		return nil, errors.New("vkDestroyInstance unavailable")
	}
	defer C.ollama_call_vk_destroy_instance(destroyInstance, instance)

	procs := map[string]unsafe.Pointer{}
	for _, name := range []string{
		"vkEnumeratePhysicalDevices",
		"vkGetPhysicalDeviceProperties",
		"vkGetPhysicalDeviceProperties2",
		"vkEnumerateDeviceExtensionProperties",
	} {
		proc := vkProcAddr(getInstanceProcAddr, instance, name)
		if proc == nil {
			return nil, fmt.Errorf("%s unavailable", name)
		}
		procs[name] = proc
	}

	var count C.uint32_t
	if ret := C.ollama_call_vk_enumerate_physical_devices(procs["vkEnumeratePhysicalDevices"], instance, &count, nil); ret != vkSuccess {
		return nil, fmt.Errorf("vkEnumeratePhysicalDevices count failed: %d", int(ret))
	}
	if count == 0 {
		return nil, nil
	}
	physicalDevicesPtr := (*unsafe.Pointer)(C.calloc(C.size_t(count), C.size_t(unsafe.Sizeof(unsafe.Pointer(nil)))))
	if physicalDevicesPtr == nil {
		return nil, errors.New("out of memory")
	}
	defer C.free(unsafe.Pointer(physicalDevicesPtr))
	if ret := C.ollama_call_vk_enumerate_physical_devices(procs["vkEnumeratePhysicalDevices"], instance, &count, physicalDevicesPtr); ret < vkSuccess {
		return nil, fmt.Errorf("vkEnumeratePhysicalDevices failed: %d", int(ret))
	}

	props := (*C.ollama_vk_physical_device_properties)(C.calloc(1, C.sizeof_ollama_vk_physical_device_properties))
	if props == nil {
		return nil, errors.New("out of memory")
	}
	defer C.free(unsafe.Pointer(props))
	query := (*C.ollama_intel_vk_properties_query)(C.calloc(1, C.sizeof_ollama_intel_vk_properties_query))
	if query == nil {
		return nil, errors.New("out of memory")
	}
	defer C.free(unsafe.Pointer(query))

	var devices []intelVulkanDevice
	for _, physicalDevice := range unsafe.Slice(physicalDevicesPtr, int(count)) {
		C.ollama_call_vk_get_physical_device_properties(procs["vkGetPhysicalDeviceProperties"], physicalDevice, props)
		if props.vendorID != intelVendorID || props.apiVersion < vkAPIVersion11 {
			continue
		}

		device := intelVulkanDevice{
			Name:       C.GoString(&props.deviceName[0]),
			Integrated: props.deviceType == vkPhysicalDeviceTypeIntegratedGPU,
		}

		extensions := linuxVulkanDeviceExtensions(procs["vkEnumerateDeviceExtensionProperties"], physicalDevice)
		hasIntelInfo := extensions[vkIntelDeviceInfoExtensionName]
		hasPCIBusInfo := extensions[vkEXTPCIBusInfoExtensionName]

		*query = C.ollama_intel_vk_properties_query{}
		query.props2.sType = vkStructureTypePhysicalDeviceProperties2
		query.intel.sType = vkStructureTypePhysicalDeviceInfoPropertiesINTEL
		query.pci.sType = vkStructureTypePhysicalDevicePCIBusInfoEXT
		next := &query.props2.pNext
		if hasIntelInfo {
			*next = unsafe.Pointer(&query.intel)
			next = &query.intel.pNext
		}
		if hasPCIBusInfo {
			*next = unsafe.Pointer(&query.pci)
		}
		C.ollama_call_vk_get_physical_device_properties2(procs["vkGetPhysicalDeviceProperties2"], physicalDevice, &query.props2)

		if hasIntelInfo {
			device.IPVersion = packIntelIPVersion(uint32(query.intel.deviceIpVersionArch), uint32(query.intel.deviceIpVersionRelease), uint32(query.intel.deviceIpVersionRevision))
		}
		if hasPCIBusInfo {
			device.PCIID = vulkanPCIBusID(uint32(query.pci.pciDomain), uint32(query.pci.pciBus), uint32(query.pci.pciDevice), uint32(query.pci.pciFunction))
		}
		devices = append(devices, device)
	}

	return devices, nil
}

func vkProcAddr(getInstanceProcAddr, instance unsafe.Pointer, name string) unsafe.Pointer {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	return C.ollama_call_vk_get_instance_proc_addr(getInstanceProcAddr, instance, cname)
}

func linuxVulkanDeviceExtensions(fn, physicalDevice unsafe.Pointer) map[string]bool {
	var count C.uint32_t
	if ret := C.ollama_call_vk_enumerate_device_extension_properties(fn, physicalDevice, &count, nil); ret != vkSuccess || count == 0 {
		return nil
	}
	props := (*C.ollama_vk_extension_properties)(C.calloc(C.size_t(count), C.sizeof_ollama_vk_extension_properties))
	if props == nil {
		return nil
	}
	defer C.free(unsafe.Pointer(props))
	if ret := C.ollama_call_vk_enumerate_device_extension_properties(fn, physicalDevice, &count, props); ret < vkSuccess {
		return nil
	}

	extensions := make(map[string]bool, int(count))
	all := unsafe.Slice(props, int(count))
	for i := range all {
		extensions[C.GoString(&all[i].extensionName[0])] = true
	}
	return extensions
}
