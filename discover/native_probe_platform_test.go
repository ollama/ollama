//go:build (linux && cgo) || windows

package discover

import (
	"path/filepath"
	"runtime"
	"testing"
)

// TestGGMLProbeLibraryNamePreservesIdentity verifies that each backend keeps its
// own identity: the library label is derived from the ggml backend registration
// name, so CUDA stays CUDA, ROCm stays ROCm, Vulkan stays Vulkan, and SYCL stays
// SYCL. Unknown names pass through unchanged.
func TestGGMLProbeLibraryNamePreservesIdentity(t *testing.T) {
	cases := []struct {
		name string
		in   string
		want string
	}{
		{"cuda", "cuda", "CUDA"},
		{"cuda mixed case", "CUDA", "CUDA"},
		{"hip", "hip", "ROCm"},
		{"rocm", "rocm", "ROCm"},
		{"vulkan", "vulkan", "Vulkan"},
		{"vulkan mixed case", "Vulkan", "Vulkan"},
		{"metal", "metal", "Metal"},
		{"sycl", "sycl", "SYCL"},
		{"sycl mixed case", "SYCL", "SYCL"},
		{"unknown passthrough", "madeupbackend", "madeupbackend"},
	}
	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			if got := ggmlProbeLibraryName(tt.in); got != tt.want {
				t.Fatalf("ggmlProbeLibraryName(%q) = %q, want %q", tt.in, got, tt.want)
			}
		})
	}
}

// TestNativeProbeBackendPatterns verifies that the SYCL backend module is probed
// in addition to the existing CUDA, HIP, and Vulkan modules. Without a matching
// pattern here a discovered libggml-sycl module would never be loaded.
func TestNativeProbeBackendPatterns(t *testing.T) {
	dir := filepath.Join("opt", "ollama", "lib")
	patterns := nativeProbeBackendPatterns(dir)

	got := make(map[string]bool, len(patterns))
	for _, p := range patterns {
		got[p] = true
	}

	var wantNames []string
	if runtime.GOOS == "windows" {
		wantNames = []string{"ggml-cuda.dll", "ggml-hip.dll", "ggml-vulkan.dll", "ggml-sycl.dll"}
	} else {
		wantNames = []string{"libggml-cuda.so", "libggml-hip.so", "libggml-vulkan.so", "libggml-sycl.so"}
	}

	for _, name := range wantNames {
		want := filepath.Join(dir, name)
		if !got[want] {
			t.Errorf("nativeProbeBackendPatterns(%q) missing %q; got %v", dir, want, patterns)
		}
	}
}
