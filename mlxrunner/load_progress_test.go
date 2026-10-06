package mlxrunner

import (
	"testing"

	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/mlxrunner/model"
)

func TestLoadState(t *testing.T) {
	state := newLoadState()
	if got := state.Status(); got != llm.ServerStatusLoadingModel {
		t.Fatalf("initial status = %v, want %v", got, llm.ServerStatusLoadingModel)
	}

	state.SetProgress(0.75)
	state.SetProgress(0.25)
	if got := state.Progress(); got != 0.75 {
		t.Fatalf("progress after regression = %v, want 0.75", got)
	}

	state.MarkReady(4096)
	if got := state.Status(); got != llm.ServerStatusReady {
		t.Errorf("ready status = %v, want %v", got, llm.ServerStatusReady)
	}
	if got := state.Progress(); got != 1 {
		t.Errorf("ready progress = %v, want 1", got)
	}
	if got := state.ContextLength(); got != 4096 {
		t.Errorf("context length = %d, want 4096", got)
	}
}

func TestLoadProgressReporterUsesUniqueTensorLayerBytes(t *testing.T) {
	root := &model.Root{Manifest: &manifest.Manifest{
		Layers: []manifest.Layer{
			{MediaType: manifest.MediaTypeImageTensor, Digest: "sha256:a", Size: 100},
			{MediaType: manifest.MediaTypeImageTensor, Digest: "sha256:a", Size: 100},
			{MediaType: manifest.MediaTypeImageTensor, Digest: "sha256:b", Size: 300},
			{MediaType: "application/vnd.ollama.image.json", Digest: "sha256:c", Size: 1000},
		},
	}}

	var got []float32
	report := newLoadProgressReporter(root, func(progress float32) {
		got = append(got, progress)
	})
	report(100)
	report(300)

	if len(got) != 2 {
		t.Fatalf("progress reports = %d, want 2", len(got))
	}
	if got[0] != 0.25 || got[1] != 1 {
		t.Fatalf("progress = %v, want [0.25 1]", got)
	}
}
