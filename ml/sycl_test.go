package ml

import "testing"

func TestDevicePreferredLibrarySYCLOverVulkan(t *testing.T) {
	sycl := DeviceInfo{DeviceID: DeviceID{Library: "SYCL"}, FilterID: "0"}
	vulkan := DeviceInfo{DeviceID: DeviceID{Library: "Vulkan"}, FilterID: "0"}

	if !sycl.PreferredLibrary(vulkan) {
		t.Fatal("expected SYCL to be preferred over Vulkan")
	}
	if vulkan.PreferredLibrary(sycl) {
		t.Fatal("expected Vulkan not to be preferred over SYCL")
	}
}
