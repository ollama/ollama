package server

import (
	"context"
	"testing"

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
		err := sched.checkManifestFit(t.Context(), "too-big", mf)
		want := "too-big is too large to run on this system - Try a smaller local model, consider a cloud model, or --force to pull anyway"
		if err == nil || err.Error() != want {
			t.Fatalf("checkManifestFit error = %q, want %q", err, want)
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
		if got.Backend != modelFitBackendMLX || got.Status != modelFitUnknown {
			t.Fatalf("assessment = %#v, want MLX with unknown fit", got)
		}
	})

	t.Run("leaves GGUF plumbed but unknown", func(t *testing.T) {
		mf := &manifest.Manifest{Layers: []manifest.Layer{
			{MediaType: "application/vnd.ollama.image.model", Size: 4 << 30},
		}}
		got := sched.assessManifestFit(t.Context(), mf)
		if got.Backend != modelFitBackendLlama || got.Status != modelFitUnknown {
			t.Fatalf("assessment = %#v, want llama-server with unknown fit", got)
		}
	})
}
