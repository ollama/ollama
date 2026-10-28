//go:build linux && cgo

package discover

/*
#cgo linux LDFLAGS: -ldl

#include <dlfcn.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Minimal Vulkan definitions so VK_INTEL_device_info (headers 1.4.363+) can be
// queried without depending on the Vulkan headers available at build time.
#define OLLAMA_VK_SUCCESS 0
#define OLLAMA_VK_API_VERSION_1_1 ((1u << 22) | (1u << 12))
#define OLLAMA_VK_STRUCTURE_TYPE_APPLICATION_INFO 0
#define OLLAMA_VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO 1
#define OLLAMA_VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 1000059001
#define OLLAMA_VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PCI_BUS_INFO_PROPERTIES_EXT 1000212000
#define OLLAMA_VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INFO_PROPERTIES_INTEL 1000708000
#define OLLAMA_VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU 1
#define OLLAMA_VK_INTEL_VENDOR_ID 0x8086
#define OLLAMA_VK_MAX_PHYSICAL_DEVICE_NAME_SIZE 256
#define OLLAMA_VK_MAX_EXTENSION_NAME_SIZE 256
#define OLLAMA_INTEL_VK_MAX_DEVICES 16

typedef int32_t ollama_vk_result;
typedef void * ollama_vk_handle;
typedef void (*ollama_vk_void_fn)(void);

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
	char extensionName[OLLAMA_VK_MAX_EXTENSION_NAME_SIZE];
	uint32_t specVersion;
} ollama_vk_extension_properties;

// Leading fields of VkPhysicalDeviceProperties; the padding covers the full struct.
typedef struct {
	uint32_t apiVersion;
	uint32_t driverVersion;
	uint32_t vendorID;
	uint32_t deviceID;
	uint32_t deviceType;
	char deviceName[OLLAMA_VK_MAX_PHYSICAL_DEVICE_NAME_SIZE];
	uint64_t _rest[512];
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

typedef ollama_vk_void_fn (*ollama_vk_get_instance_proc_addr_fn)(ollama_vk_handle, const char *);
typedef ollama_vk_result (*ollama_vk_enumerate_instance_version_fn)(uint32_t *);
typedef ollama_vk_result (*ollama_vk_create_instance_fn)(const ollama_vk_instance_create_info *, const void *, ollama_vk_handle *);
typedef void (*ollama_vk_destroy_instance_fn)(ollama_vk_handle, const void *);
typedef ollama_vk_result (*ollama_vk_enumerate_physical_devices_fn)(ollama_vk_handle, uint32_t *, ollama_vk_handle *);
typedef void (*ollama_vk_get_physical_device_properties_fn)(ollama_vk_handle, ollama_vk_physical_device_properties *);
typedef void (*ollama_vk_get_physical_device_properties2_fn)(ollama_vk_handle, ollama_vk_physical_device_properties2 *);
typedef ollama_vk_result (*ollama_vk_enumerate_device_extension_properties_fn)(ollama_vk_handle, const char *, uint32_t *, ollama_vk_extension_properties *);

typedef struct {
	char name[OLLAMA_VK_MAX_PHYSICAL_DEVICE_NAME_SIZE];
	int integrated;
	int has_ip_version;
	uint32_t ip_arch;
	uint32_t ip_release;
	uint32_t ip_revision;
	int has_pci;
	uint32_t pci_domain;
	uint32_t pci_bus;
	uint32_t pci_device;
	uint32_t pci_function;
} ollama_intel_vk_device;

static int ollama_intel_vk_has_extension(ollama_vk_enumerate_device_extension_properties_fn enumerate, ollama_vk_handle device, const char * name) {
	uint32_t count = 0;
	if (enumerate(device, NULL, &count, NULL) != OLLAMA_VK_SUCCESS || count == 0) {
		return 0;
	}
	ollama_vk_extension_properties * props = calloc(count, sizeof(*props));
	if (props == NULL) {
		return 0;
	}
	int found = 0;
	if (enumerate(device, NULL, &count, props) >= OLLAMA_VK_SUCCESS) {
		for (uint32_t i = 0; i < count; i++) {
			if (strncmp(props[i].extensionName, name, OLLAMA_VK_MAX_EXTENSION_NAME_SIZE) == 0) {
				found = 1;
				break;
			}
		}
	}
	free(props);
	return found;
}

// Returns the number of Intel devices written to out, or -1 with err populated.
static int ollama_intel_vk_probe(ollama_intel_vk_device * out, int max_devices, char * err, size_t err_len) {
	void * lib = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
	if (lib == NULL) {
		lib = dlopen("libvulkan.so", RTLD_NOW | RTLD_LOCAL);
	}
	if (lib == NULL) {
		snprintf(err, err_len, "unable to load vulkan loader: %s", dlerror());
		return -1;
	}

	ollama_vk_get_instance_proc_addr_fn gipa = (ollama_vk_get_instance_proc_addr_fn) dlsym(lib, "vkGetInstanceProcAddr");
	if (gipa == NULL) {
		snprintf(err, err_len, "vkGetInstanceProcAddr unavailable");
		return -1;
	}

	// Loaders older than Vulkan 1.1 do not provide vkEnumerateInstanceVersion.
	ollama_vk_enumerate_instance_version_fn enumerate_version = (ollama_vk_enumerate_instance_version_fn) gipa(NULL, "vkEnumerateInstanceVersion");
	uint32_t loader_version = 0;
	if (enumerate_version == NULL || enumerate_version(&loader_version) != OLLAMA_VK_SUCCESS || loader_version < OLLAMA_VK_API_VERSION_1_1) {
		snprintf(err, err_len, "vulkan loader older than 1.1");
		return -1;
	}

	ollama_vk_create_instance_fn create_instance = (ollama_vk_create_instance_fn) gipa(NULL, "vkCreateInstance");
	if (create_instance == NULL) {
		snprintf(err, err_len, "vkCreateInstance unavailable");
		return -1;
	}

	ollama_vk_application_info app = {0};
	app.sType = OLLAMA_VK_STRUCTURE_TYPE_APPLICATION_INFO;
	app.apiVersion = OLLAMA_VK_API_VERSION_1_1;
	ollama_vk_instance_create_info create = {0};
	create.sType = OLLAMA_VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
	create.pApplicationInfo = &app;

	ollama_vk_handle instance = NULL;
	ollama_vk_result result = create_instance(&create, NULL, &instance);
	if (result != OLLAMA_VK_SUCCESS || instance == NULL) {
		snprintf(err, err_len, "vkCreateInstance failed: %d", (int) result);
		return -1;
	}

	ollama_vk_destroy_instance_fn destroy_instance = (ollama_vk_destroy_instance_fn) gipa(instance, "vkDestroyInstance");
	ollama_vk_enumerate_physical_devices_fn enumerate_devices = (ollama_vk_enumerate_physical_devices_fn) gipa(instance, "vkEnumeratePhysicalDevices");
	ollama_vk_get_physical_device_properties_fn get_props = (ollama_vk_get_physical_device_properties_fn) gipa(instance, "vkGetPhysicalDeviceProperties");
	ollama_vk_get_physical_device_properties2_fn get_props2 = (ollama_vk_get_physical_device_properties2_fn) gipa(instance, "vkGetPhysicalDeviceProperties2");
	ollama_vk_enumerate_device_extension_properties_fn enumerate_extensions = (ollama_vk_enumerate_device_extension_properties_fn) gipa(instance, "vkEnumerateDeviceExtensionProperties");
	if (destroy_instance == NULL || enumerate_devices == NULL || get_props == NULL || get_props2 == NULL || enumerate_extensions == NULL) {
		snprintf(err, err_len, "required vulkan entry points unavailable");
		if (destroy_instance != NULL) {
			destroy_instance(instance, NULL);
		}
		return -1;
	}

	uint32_t count = 0;
	if (enumerate_devices(instance, &count, NULL) != OLLAMA_VK_SUCCESS) {
		snprintf(err, err_len, "vkEnumeratePhysicalDevices failed");
		destroy_instance(instance, NULL);
		return -1;
	}
	if (count == 0) {
		destroy_instance(instance, NULL);
		return 0;
	}

	ollama_vk_handle * devices = calloc(count, sizeof(*devices));
	ollama_vk_physical_device_properties * props = calloc(1, sizeof(*props));
	ollama_vk_physical_device_properties2 * props2 = calloc(1, sizeof(*props2));
	if (devices == NULL || props == NULL || props2 == NULL) {
		snprintf(err, err_len, "out of memory");
		free(devices);
		free(props);
		free(props2);
		destroy_instance(instance, NULL);
		return -1;
	}
	if (enumerate_devices(instance, &count, devices) < OLLAMA_VK_SUCCESS) {
		snprintf(err, err_len, "vkEnumeratePhysicalDevices failed");
		free(devices);
		free(props);
		free(props2);
		destroy_instance(instance, NULL);
		return -1;
	}

	int n = 0;
	for (uint32_t i = 0; i < count && n < max_devices; i++) {
		memset(props, 0, sizeof(*props));
		get_props(devices[i], props);
		if (props->vendorID != OLLAMA_VK_INTEL_VENDOR_ID || props->apiVersion < OLLAMA_VK_API_VERSION_1_1) {
			continue;
		}

		ollama_intel_vk_device * dev = &out[n++];
		memset(dev, 0, sizeof(*dev));
		memcpy(dev->name, props->deviceName, sizeof(dev->name));
		dev->name[sizeof(dev->name) - 1] = '\0';
		dev->integrated = props->deviceType == OLLAMA_VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU;

		int has_intel_info = ollama_intel_vk_has_extension(enumerate_extensions, devices[i], "VK_INTEL_device_info");
		int has_pci = ollama_intel_vk_has_extension(enumerate_extensions, devices[i], "VK_EXT_pci_bus_info");

		ollama_vk_physical_device_info_properties_intel intel_info = {0};
		intel_info.sType = OLLAMA_VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INFO_PROPERTIES_INTEL;
		ollama_vk_physical_device_pci_bus_info_properties pci_info = {0};
		pci_info.sType = OLLAMA_VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PCI_BUS_INFO_PROPERTIES_EXT;

		memset(props2, 0, sizeof(*props2));
		props2->sType = OLLAMA_VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
		void ** next = &props2->pNext;
		if (has_intel_info) {
			*next = &intel_info;
			next = &intel_info.pNext;
		}
		if (has_pci) {
			*next = &pci_info;
		}
		get_props2(devices[i], props2);

		if (has_intel_info) {
			dev->has_ip_version = 1;
			dev->ip_arch = intel_info.deviceIpVersionArch;
			dev->ip_release = intel_info.deviceIpVersionRelease;
			dev->ip_revision = intel_info.deviceIpVersionRevision;
		}
		if (has_pci) {
			dev->has_pci = 1;
			dev->pci_domain = pci_info.pciDomain;
			dev->pci_bus = pci_info.pciBus;
			dev->pci_device = pci_info.pciDevice;
			dev->pci_function = pci_info.pciFunction;
		}
	}

	free(devices);
	free(props);
	free(props2);
	destroy_instance(instance, NULL);
	return n;
}

static int ollama_intel_vk_max_devices(void) {
	return OLLAMA_INTEL_VK_MAX_DEVICES;
}
*/
import "C"

import (
	"errors"
	"unsafe"
)

func probeIntelVulkanDevices([]string) ([]intelVulkanDevice, error) {
	maxDevices := int(C.ollama_intel_vk_max_devices())
	out := (*C.ollama_intel_vk_device)(C.calloc(C.size_t(maxDevices), C.size_t(unsafe.Sizeof(C.ollama_intel_vk_device{}))))
	if out == nil {
		return nil, errors.New("out of memory")
	}
	defer C.free(unsafe.Pointer(out))

	errBuf := (*C.char)(C.calloc(256, 1))
	if errBuf == nil {
		return nil, errors.New("out of memory")
	}
	defer C.free(unsafe.Pointer(errBuf))

	n := int(C.ollama_intel_vk_probe(out, C.int(maxDevices), errBuf, 256))
	if n < 0 {
		return nil, errors.New(C.GoString(errBuf))
	}

	results := unsafe.Slice(out, maxDevices)[:n]
	devices := make([]intelVulkanDevice, 0, n)
	for _, r := range results {
		device := intelVulkanDevice{
			Name:       C.GoString(&r.name[0]),
			Integrated: r.integrated != 0,
		}
		if r.has_ip_version != 0 {
			device.IPVersion = packIntelIPVersion(uint32(r.ip_arch), uint32(r.ip_release), uint32(r.ip_revision))
		}
		if r.has_pci != 0 {
			device.PCIID = vulkanPCIBusID(uint32(r.pci_domain), uint32(r.pci_bus), uint32(r.pci_device), uint32(r.pci_function))
		}
		devices = append(devices, device)
	}
	return devices, nil
}
