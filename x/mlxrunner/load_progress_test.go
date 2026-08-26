package mlxrunner

import (
	"math"
	"testing"

	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/x/imagegen/manifest"
	"github.com/ollama/ollama/x/mlxrunner/model"
)

func TestLoadStateStartsLoading(t *testing.T) {
	state := newLoadState()

	if got := state.Status(); got != llm.ServerStatusLoadingModel {
		t.Errorf("Status() = %v, want %v", got, llm.ServerStatusLoadingModel)
	}
	if got := state.Progress(); got != 0 {
		t.Errorf("Progress() = %v, want 0", got)
	}
	if got := state.ContextLength(); got != 0 {
		t.Errorf("ContextLength() = %d, want 0", got)
	}
}

func TestLoadStateReportsProgress(t *testing.T) {
	state := newLoadState()
	state.SetProgress(0.25)

	if got := state.Progress(); got != 0.25 {
		t.Errorf("Progress() = %v, want 0.25", got)
	}
	if got := state.Status(); got != llm.ServerStatusLoadingModel {
		t.Errorf("Status() = %v, want %v", got, llm.ServerStatusLoadingModel)
	}
}

func TestLoadStateProgressIsMonotonic(t *testing.T) {
	state := newLoadState()
	state.SetProgress(0.75)
	state.SetProgress(0.25)

	if got := state.Progress(); got != 0.75 {
		t.Errorf("Progress() = %v, want 0.75", got)
	}
}

func TestLoadStateMarkReady(t *testing.T) {
	state := newLoadState()
	state.SetProgress(0.5)
	state.MarkReady(4096)

	if got := state.Status(); got != llm.ServerStatusReady {
		t.Errorf("Status() = %v, want %v", got, llm.ServerStatusReady)
	}
	if got := state.Progress(); got != 1 {
		t.Errorf("Progress() = %v, want 1", got)
	}
	if got := state.ContextLength(); got != 4096 {
		t.Errorf("ContextLength() = %d, want 4096", got)
	}
}

func TestLoadProgressReporterUsesUniqueTensorLayerBytes(t *testing.T) {
	root := &model.Root{Manifest: &manifest.ModelManifest{Manifest: &manifest.Manifest{
		Layers: []manifest.ManifestLayer{
			{MediaType: "application/vnd.ollama.image.tensor", Digest: "sha256:a", Size: 100},
			{MediaType: "application/vnd.ollama.image.tensor", Digest: "sha256:a", Size: 100},
			{MediaType: "application/vnd.ollama.image.tensor", Digest: "sha256:b", Size: 300},
			{MediaType: "application/vnd.ollama.image.json", Digest: "sha256:c", Size: 1000},
		},
	}}}

	var got []float32
	report := newLoadProgressReporter(root, func(progress float32) {
		got = append(got, progress)
	})

	report(100)
	report(100)

	if len(got) != 2 {
		t.Fatalf("progress reports = %d, want 2", len(got))
	}
	if got[0] != 0.25 {
		t.Errorf("first progress = %v, want 0.25", got[0])
	}
	if got[1] != 0.5 {
		t.Errorf("second progress = %v, want 0.5", got[1])
	}
}

func TestLoadProgressReporterReservesOneForReady(t *testing.T) {
	root := &model.Root{Manifest: &manifest.ModelManifest{Manifest: &manifest.Manifest{
		Layers: []manifest.ManifestLayer{
			{MediaType: "application/vnd.ollama.image.tensor", Digest: "sha256:a", Size: 100},
		},
	}}}

	var got float32
	report := newLoadProgressReporter(root, func(progress float32) {
		got = progress
	})

	report(100)

	if got != math.Nextafter32(1, 0) {
		t.Errorf("progress = %v, want nextafter(1, 0)", got)
	}
}
