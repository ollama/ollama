package ml

import "testing"

func TestNeedsInitValidation(t *testing.T) {
	tests := []struct {
		name        string
		device      DeviceInfo
		wantValidation bool
	}{
		{
			name: "regular CUDA device needs validation",
			device: DeviceInfo{
				DeviceID:    DeviceID{Library: "CUDA"},
				Description: "NVIDIA GeForce RTX 3080",
			},
			wantValidation: true,
		},
		{
			name: "ROCm device needs validation",
			device: DeviceInfo{
				DeviceID:    DeviceID{Library: "ROCm"},
				Description: "AMD Radeon RX 7900 XTX",
			},
			wantValidation: true,
		},
		{
			name: "Metal device does not need validation",
			device: DeviceInfo{
				DeviceID:    DeviceID{Library: "Metal"},
				Description: "Apple M1 Pro",
			},
			wantValidation: false,
		},
		{
			name: "CPU does not need validation",
			device: DeviceInfo{
				DeviceID:    DeviceID{Library: "cpu"},
				Description: "CPU",
			},
			wantValidation: false,
		},
		{
			name: "MIG device does not need validation",
			device: DeviceInfo{
				DeviceID:    DeviceID{Library: "CUDA"},
				Description: "NVIDIA A100 80GB PCIe MIG 7g.80gb",
			},
			wantValidation: false,
		},
		{
			name: "MIG device with different partition does not need validation",
			device: DeviceInfo{
				DeviceID:    DeviceID{Library: "CUDA"},
				Description: "NVIDIA A100-SXM4-40GB MIG 3g.20gb",
			},
			wantValidation: false,
		},
		{
			name: "H100 MIG device does not need validation",
			device: DeviceInfo{
				DeviceID:    DeviceID{Library: "CUDA"},
				Description: "NVIDIA H100 PCIe MIG 1g.10gb",
			},
			wantValidation: false,
		},
		{
			name: "regular A100 without MIG needs validation",
			device: DeviceInfo{
				DeviceID:    DeviceID{Library: "CUDA"},
				Description: "NVIDIA A100 80GB PCIe",
			},
			wantValidation: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.device.NeedsInitValidation()
			if got != tt.wantValidation {
				t.Errorf("NeedsInitValidation() = %v, want %v", got, tt.wantValidation)
			}
		})
	}
}
