package ggml

import "testing"

func TestGGMLDeviceIdentityPreservesBackendWithExternalLibraryPath(t *testing.T) {
	tests := []struct {
		name    string
		library string
		id      string
	}{
		{name: "CUDA", library: "CUDA", id: "GPU-123"},
		{name: "ROCm", library: "ROCm", id: "0"},
		{name: "Vulkan", library: "Vulkan", id: "0000:03:00.0"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := ggmlDeviceIdentity(ggmlDeviceProps{library: tt.library, id: tt.id}, "SYCL0", "Intel GPU")
			if got.library != tt.library {
				t.Fatalf("library = %q, want %q", got.library, tt.library)
			}
			if got.id != tt.id {
				t.Fatalf("id = %q, want %q", got.id, tt.id)
			}
		})
	}
}

func TestGGMLDeviceIdentityReportsSYCLFromProps(t *testing.T) {
	got := ggmlDeviceIdentity(ggmlDeviceProps{library: "SYCL", id: "1"}, "CUDA0", "NVIDIA GPU")
	if got.library != "SYCL" {
		t.Fatalf("library = %q, want SYCL", got.library)
	}
	if got.id != "1" {
		t.Fatalf("id = %q, want 1", got.id)
	}
}

func TestGGMLDeviceIdentityMissingMetadataDoesNotOverwriteKnownBackend(t *testing.T) {
	got := ggmlDeviceIdentity(ggmlDeviceProps{name: "CUDA0", description: "NVIDIA GPU", library: "CUDA", id: "GPU-123"}, "SYCL0", "Intel GPU")
	if got.library != "CUDA" {
		t.Fatalf("library = %q, want CUDA", got.library)
	}
	if got.id != "GPU-123" {
		t.Fatalf("id = %q, want GPU-123", got.id)
	}
}
