package server

import (
	"context"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/ml"
)

func TestAssessManifestFit(t *testing.T) {
	t.Setenv("OLLAMA_GPU_OVERHEAD", "0")

	sched := &Scheduler{
		getGpuFn: func(context.Context, []ml.FilteredRunnerDiscovery) []ml.DeviceInfo {
			return []ml.DeviceInfo{{
				DeviceID:    ml.DeviceID{Library: "Metal"},
				TotalMemory: 2 << 30,
				FreeMemory:  1,
			}}
		},
		getSystemInfoFn: func() ml.SystemInfo {
			return ml.SystemInfo{TotalMemory: 8 << 30}
		},
	}

	t.Run("does not reject based on live GPU pressure", func(t *testing.T) {
		mf := &manifest.Manifest{Layers: []manifest.Layer{
			{MediaType: manifest.MediaTypeImageTensor, Size: 1 << 30},
		}}
		got := sched.assessManifestFit(t.Context(), mf)
		if got.Status != modelFitUnknown {
			t.Fatalf("fit status = %v, want modelFitUnknown", got.Status)
		}
	})

	t.Run("rejects a definite non-fit", func(t *testing.T) {
		mf := &manifest.Manifest{Layers: []manifest.Layer{
			{MediaType: manifest.MediaTypeImageTensor, Size: 2 << 30},
		}}
		got := sched.assessManifestFit(t.Context(), mf)
		if got.Status != modelDoesNotFit {
			t.Fatalf("fit status = %v, want modelDoesNotFit", got.Status)
		}
	})

	t.Run("leaves CPU-only systems unknown", func(t *testing.T) {
		cpuOnly := &Scheduler{
			getGpuFn: func(context.Context, []ml.FilteredRunnerDiscovery) []ml.DeviceInfo {
				return nil
			},
		}
		mf := &manifest.Manifest{Layers: []manifest.Layer{
			{MediaType: manifest.MediaTypeImageTensor, Size: 2 << 30},
		}}
		got := cpuOnly.assessManifestFit(t.Context(), mf)
		if got.Status != modelFitUnknown {
			t.Fatalf("fit status = %v, want modelFitUnknown", got.Status)
		}
	})

	t.Run("uses idle host capacity for an integrated GPU", func(t *testing.T) {
		integrated := &Scheduler{
			getGpuFn: func(context.Context, []ml.FilteredRunnerDiscovery) []ml.DeviceInfo {
				return []ml.DeviceInfo{{
					DeviceID:    ml.DeviceID{Library: "Metal"},
					Integrated:  true,
					TotalMemory: 4 << 30,
				}}
			},
			getSystemInfoFn: func() ml.SystemInfo {
				return ml.SystemInfo{TotalMemory: 1 << 30, FreeMemory: 1}
			},
		}
		mf := &manifest.Manifest{Layers: []manifest.Layer{
			{MediaType: manifest.MediaTypeImageTensor, Size: 768 << 20},
		}}
		got := integrated.assessManifestFit(t.Context(), mf)
		if got.Status != modelDoesNotFit {
			t.Fatalf("fit status = %v, want modelDoesNotFit", got.Status)
		}
	})

	t.Run("leaves GGUF unknown", func(t *testing.T) {
		mf := &manifest.Manifest{Layers: []manifest.Layer{
			{MediaType: manifest.MediaTypeImageModel, Size: 4 << 30},
		}}
		got := sched.assessManifestFit(t.Context(), mf)
		if got.Status != modelFitUnknown {
			t.Fatalf("fit status = %v, want modelFitUnknown", got.Status)
		}
	})
}

func TestModelDoesNotFitError(t *testing.T) {
	t.Setenv("OLLAMA_GPU_OVERHEAD", "0")
	setTestHome(t, t.TempDir())

	// 16 GB of idle GPU capacity.
	assessment := modelFitAssessment{
		Status: modelDoesNotFit,
		gpus: []ml.DeviceInfo{{
			DeviceID:   ml.DeviceID{Library: "Metal"},
			FreeMemory: 16 * format.GigaByte,
		}},
	}
	// Cloud and local models interleave in recommendation order.
	recs := []api.ModelRecommendation{
		{Model: "big", VRAMBytes: 64 * format.GigaByte},
		{Model: "generic", VRAMBytes: 4 * format.GigaByte},
		{Model: "kimi-k2.6:cloud", RequiredPlan: "pro"},
		{Model: "medium", VRAMBytes: 10 * format.GigaByte},
		{Model: "glm-5.1:cloud"},
		{Model: "unknown-size"},
		{Model: "qwen3.5:9b", VRAMBytes: 6 * format.GigaByte},
		// Listed late but the largest that fits, so suggested first.
		{Model: "largest-fit", VRAMBytes: 12 * format.GigaByte},
		{Model: "too-big", VRAMBytes: 17 * format.GigaByte},
		{Model: "qwen3.5:cloud"},
	}

	tests := []struct {
		name    string
		model   string
		noCloud bool
		recs    []api.ModelRecommendation
		want    string
	}{
		{
			name:  "largest local models that fit and the first cloud model",
			model: "qwen3.5:122b",
			recs:  recs,
			want:  "qwen3.5:122b may not fit on your system. You can try largest-fit, medium, or qwen3.5:9b locally, or consider kimi-k2.6:cloud. Use --force to pull qwen3.5:122b anyway.",
		},
		{
			name:  "skips the requested model",
			model: "generic",
			recs:  recs[:4],
			want:  "generic may not fit on your system. You can try medium locally, or consider kimi-k2.6:cloud. Use --force to pull generic anyway.",
		},
		{
			name:  "smaller model advice with cloud when no local model fits",
			model: "qwen3.5:122b",
			recs:  []api.ModelRecommendation{recs[0], recs[2], recs[8]},
			want:  "qwen3.5:122b may not fit on your system. Try a smaller model, or consider kimi-k2.6:cloud. Use --force to pull qwen3.5:122b anyway.",
		},
		{
			name:  "local only without a cloud recommendation",
			model: "qwen3.5:122b",
			recs:  recs[:2],
			want:  "qwen3.5:122b may not fit on your system. You can try generic, or use --force to pull qwen3.5:122b anyway.",
		},
		{
			name:  "generic advice without recommendations",
			model: "qwen3.5:122b",
			want:  "qwen3.5:122b may not fit on your system. Try a smaller model, or use --force to pull qwen3.5:122b anyway.",
		},
		{
			name:    "cloud disabled ignores recommendations",
			model:   "qwen3.5:122b",
			noCloud: true,
			recs:    recs,
			want:    "qwen3.5:122b may not fit on your system. Try a smaller model, or use --force to pull qwen3.5:122b anyway.",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.noCloud {
				t.Setenv("OLLAMA_NO_CLOUD", "1")
			} else {
				t.Setenv("OLLAMA_NO_CLOUD", "")
			}
			err := modelDoesNotFitError(tt.model, assessment, tt.recs)
			if err.Error() != tt.want {
				t.Fatalf("error = %q\n     want %q", err, tt.want)
			}
		})
	}
}
