package ml

import (
	"slices"
	"testing"
)

func cuda(id, name string) DeviceInfo {
	return DeviceInfo{DeviceID: DeviceID{ID: id, Library: "CUDA"}, Name: name}
}

func TestRunnerDeviceNames(t *testing.T) {
	cases := []struct {
		name    string
		devices []DeviceInfo
		want    []string
	}{
		{
			// the common case, and the one that hid this: selecting the host's
			// first device gives a child whose numbering happens to match
			name:    "first device only",
			devices: []DeviceInfo{cuda("0", "CUDA0")},
			want:    []string{"CUDA0"},
		},
		{
			// a model placed on the second device runs in a child that can see
			// only that device, and calls it CUDA0
			name:    "second device only",
			devices: []DeviceInfo{cuda("1", "CUDA1")},
			want:    []string{"CUDA0"},
		},
		{
			name:    "both devices",
			devices: []DeviceInfo{cuda("0", "CUDA0"), cuda("1", "CUDA1")},
			want:    []string{"CUDA0", "CUDA1"},
		},
		{
			// discovery orders devices by scheduling preference, so the set can
			// be a permutation of the host order; the child numbers them in the
			// order it was given
			name:    "devices in preference order",
			devices: []DeviceInfo{cuda("1", "CUDA1"), cuda("0", "CUDA0")},
			want:    []string{"CUDA0", "CUDA1"},
		},
		{
			name:    "third device only",
			devices: []DeviceInfo{cuda("2", "CUDA2")},
			want:    []string{"CUDA0"},
		},
		{
			// a single device of any library is filtered
			name:    "single metal device",
			devices: []DeviceInfo{{DeviceID: DeviceID{ID: "0", Library: "Metal"}, Name: "MTL0"}},
			want:    []string{"MTL0"},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := RunnerDeviceNames(tc.devices); !slices.Equal(got, tc.want) {
				t.Errorf("got %v, want %v", got, tc.want)
			}
		})
	}
}

// A mixed-vendor set is not filtered for CUDA, so the child sees every device
// and keeps the host numbering.
func TestRunnerDeviceNamesMixedVendorUnfiltered(t *testing.T) {
	devices := []DeviceInfo{
		cuda("1", "CUDA1"),
		{DeviceID: DeviceID{ID: "0", Library: "Vulkan"}, Name: "Vulkan0"},
	}

	got := RunnerDeviceNames(devices)
	if want := []string{"CUDA1", "Vulkan0"}; !slices.Equal(got, want) {
		t.Errorf("got %v, want %v", got, want)
	}
}
