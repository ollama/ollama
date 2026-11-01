package llm

import (
	"slices"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/ml"
)

func TestNormalizeTensorSplit(t *testing.T) {
	tests := []struct {
		name    string
		in      string
		want    string
		wantErr bool
	}{
		{name: "two devices", in: "0.3,0.7", want: "0.3,0.7"},
		{name: "whitespace tolerated", in: " 0.3 , 0.7 ", want: "0.3,0.7"},
		{name: "unnormalized proportions", in: "3,7", want: "3,7"},
		{name: "three devices", in: "1,1,1", want: "1,1,1"},
		{name: "single device", in: "1", want: "1"},
		{name: "zero for a device is allowed", in: "0,1", want: "0,1"},
		{name: "integers", in: "30,70", want: "30,70"},

		{name: "all zeros rejected", in: "0,0", wantErr: true},
		{name: "negative rejected", in: "0.5,-0.5", wantErr: true},
		{name: "non-numeric rejected", in: "half,rest", wantErr: true},
		{name: "empty field rejected", in: "0.5,,0.5", wantErr: true},
		{name: "trailing comma rejected", in: "0.5,", wantErr: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := normalizeTensorSplit(tt.in)
			if tt.wantErr {
				if err == nil {
					t.Fatalf("normalizeTensorSplit(%q) = %q, want error", tt.in, got)
				}
				return
			}
			if err != nil {
				t.Fatalf("normalizeTensorSplit(%q) unexpected error: %v", tt.in, err)
			}
			if got != tt.want {
				t.Errorf("normalizeTensorSplit(%q) = %q, want %q", tt.in, got, tt.want)
			}
		})
	}
}

func TestAppendTensorSplitArgs(t *testing.T) {
	mainGPU := 0

	// Mirrors a discrete + integrated pair: llama.cpp would prune the iGPU from
	// its own device selection, which is why --device has to be emitted too.
	twoGPUs := []ml.DeviceInfo{
		{Name: "Vulkan0", DeviceID: ml.DeviceID{ID: "0", Library: "Vulkan"}},
		{Name: "Vulkan1", DeviceID: ml.DeviceID{ID: "1", Library: "Vulkan"}, Integrated: true},
	}

	tests := []struct {
		name string
		opts api.Options
		gpus []ml.DeviceInfo
		env  string
		want []string
	}{
		{
			name: "unset adds nothing",
			opts: api.Options{},
			gpus: twoGPUs,
			want: nil,
		},
		{
			name: "option names the devices and the split",
			opts: api.Options{Runner: api.Runner{TensorSplit: "0.3,0.7"}},
			gpus: twoGPUs,
			want: []string{"--device", "Vulkan0,Vulkan1", "--split-mode", "layer", "--tensor-split", "0.3,0.7"},
		},
		{
			name: "env var is used when option is unset",
			opts: api.Options{},
			gpus: twoGPUs,
			env:  "0.25,0.75",
			want: []string{"--device", "Vulkan0,Vulkan1", "--split-mode", "layer", "--tensor-split", "0.25,0.75"},
		},
		{
			name: "option overrides env var",
			opts: api.Options{Runner: api.Runner{TensorSplit: "0.9,0.1"}},
			gpus: twoGPUs,
			env:  "0.25,0.75",
			want: []string{"--device", "Vulkan0,Vulkan1", "--split-mode", "layer", "--tensor-split", "0.9,0.1"},
		},
		{
			// --main-gpu emits --split-mode none; emitting a split too would be
			// contradictory, so the split is dropped with a warning.
			name: "main_gpu suppresses the split",
			opts: api.Options{Runner: api.Runner{TensorSplit: "0.3,0.7", MainGPU: &mainGPU}},
			gpus: twoGPUs,
			want: nil,
		},
		{
			name: "invalid value is ignored rather than fatal",
			opts: api.Options{Runner: api.Runner{TensorSplit: "not,numbers"}},
			gpus: twoGPUs,
			want: nil,
		},
		{
			// Without devices to name there is nothing to split across, and
			// emitting --tensor-split alone would be a silent no-op.
			name: "no gpus drops the split",
			opts: api.Options{Runner: api.Runner{TensorSplit: "0.3,0.7"}},
			gpus: nil,
			want: nil,
		},
		{
			// Mismatched counts warn but still apply -- llama.cpp tolerates it.
			name: "count mismatch still applies",
			opts: api.Options{Runner: api.Runner{TensorSplit: "0.3,0.3,0.4"}},
			gpus: twoGPUs,
			want: []string{"--device", "Vulkan0,Vulkan1", "--split-mode", "layer", "--tensor-split", "0.3,0.3,0.4"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Setenv("OLLAMA_TENSOR_SPLIT", tt.env)

			got := appendTensorSplitArgs(nil, tt.opts, tt.gpus)
			if !slices.Equal(got, tt.want) {
				t.Fatalf("appendTensorSplitArgs() = %v, want %v", got, tt.want)
			}
		})
	}
}
