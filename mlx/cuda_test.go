package mlx

import (
	"testing"

	"github.com/ollama/ollama/mlx/mlxthread/mlxthreadtest"
)

func TestCUDAArchitectureSupported(t *testing.T) {
	tests := []struct {
		name         string
		architecture string
		targets      []string
		want         bool
		wantErr      bool
	}{
		{name: "exact binary", architecture: "sm_120", targets: []string{"sm_120"}, want: true},
		{name: "newer minor binary", architecture: "sm_89", targets: []string{"sm_80"}, want: true},
		{name: "older minor binary", architecture: "sm_80", targets: []string{"sm_89"}},
		{name: "different major binary", architecture: "sm_90", targets: []string{"sm_80"}},
		{name: "newer PTX", architecture: "sm_120", targets: []string{"compute_80"}, want: true},
		{name: "older than PTX", architecture: "sm_89", targets: []string{"compute_90"}},
		{name: "exact architecture specific", architecture: "sm_120", targets: []string{"sm_120a"}, want: true},
		{name: "newer architecture specific", architecture: "sm_121", targets: []string{"sm_120a"}},
		{name: "exact architecture-specific PTX", architecture: "sm_120", targets: []string{"compute_120a"}, want: true},
		{name: "architecture-specific PTX is not forward compatible", architecture: "sm_121", targets: []string{"compute_120a"}},
		{name: "family exact", architecture: "sm_120", targets: []string{"sm_120f"}, want: true},
		{name: "family forward", architecture: "sm_121", targets: []string{"sm_120f"}, want: true},
		{name: "family backward", architecture: "sm_120", targets: []string{"sm_121f"}},
		{name: "family forward 100 to 103", architecture: "sm_103", targets: []string{"sm_100f"}, want: true},
		{name: "family forward 100 to 107", architecture: "sm_107", targets: []string{"compute_100f"}, want: true},
		{name: "family exception forward", architecture: "sm_107", targets: []string{"compute_103f"}, want: true},
		{name: "different family same major", architecture: "sm_110", targets: []string{"compute_100f"}},
		{name: "several targets", architecture: "sm_120", targets: []string{"sm_75", "sm_80", "sm_120a", "compute_120"}, want: true},
		{name: "unsupported", architecture: "sm_61", targets: []string{"sm_75", "sm_80", "sm_120a", "compute_120"}},
		{name: "empty targets", architecture: "sm_120"},
		{name: "invalid architecture", architecture: "gfx1200", targets: []string{"sm_120"}, wantErr: true},
		{name: "invalid target", architecture: "sm_120", targets: []string{"120a-real"}, wantErr: true},
		{name: "future family exact", architecture: "sm_130", targets: []string{"sm_130f"}, want: true},
		{name: "unknown future family", architecture: "sm_131", targets: []string{"sm_130f"}, wantErr: true},
		{name: "unknown family with baseline fallback", architecture: "sm_131", targets: []string{"sm_130f", "compute_120"}, want: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := cudaArchitectureSupported(tt.architecture, tt.targets)
			if (err != nil) != tt.wantErr {
				t.Fatalf("cudaArchitectureSupported(%q, %q) error = %v, wantErr %v", tt.architecture, tt.targets, err, tt.wantErr)
			}
			if got != tt.want {
				t.Errorf("cudaArchitectureSupported(%q, %q) = %v, want %v", tt.architecture, tt.targets, got, tt.want)
			}
		})
	}
}

func TestCUDACompilationTargets(t *testing.T) {
	withMLXThread(t, func(t *mlxthreadtest.T) {
		if !CUDAIsAvailable() {
			t.Skip("MLX CUDA not available")
		}

		targets, err := cudaCompilationTargets()
		if err != nil {
			t.Fatal(err)
		}
		if len(targets) == 0 {
			t.Fatal("MLX CUDA build reported no compilation targets")
		}
		t.Logf("MLX CUDA compilation targets: %v", targets)

		if err := CheckCUDADeviceSupport(); err != nil {
			t.Fatal(err)
		}
	})
}
