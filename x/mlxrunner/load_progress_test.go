package mlxrunner

import (
	"testing"

	"github.com/ollama/ollama/llm"
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

func TestEvalChunksCoverEveryArrayOnce(t *testing.T) {
	cases := []struct {
		name  string
		sizes []int
	}{
		{"single small array", []int{1024}},
		{"many small arrays", []int{1024, 2048, 4096, 8192, 16384}},
		{"one array larger than a chunk", []int{4 * evalChunkBytes}},
		{"mixed sizes spanning chunks", []int{
			evalChunkBytes / 2, evalChunkBytes / 2, 1024,
			2 * evalChunkBytes, 512, evalChunkBytes,
		}},
		{"exact chunk boundary", []int{evalChunkBytes, evalChunkBytes}},
		{"zero-byte arrays", []int{0, 0, 0}},
	}

	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			chunks := evalChunks(tt.sizes)
			if len(chunks) == 0 {
				t.Fatal("no chunks emitted, arrays would never be evaluated")
			}

			next := 0
			var last float32
			for i, chunk := range chunks {
				if chunk.start != next {
					t.Fatalf("chunk %d starts at %d, want %d (gap or overlap)", i, chunk.start, next)
				}
				if chunk.end <= chunk.start {
					t.Fatalf("chunk %d is empty: [%d,%d)", i, chunk.start, chunk.end)
				}
				if chunk.progress < last {
					t.Errorf("chunk %d progress = %v, went backwards from %v", i, chunk.progress, last)
				}
				last = chunk.progress
				next = chunk.end
			}

			if next != len(tt.sizes) {
				t.Errorf("chunks cover %d arrays, want %d", next, len(tt.sizes))
			}
			if last != 1 {
				t.Errorf("final progress = %v, want 1", last)
			}
		})
	}
}

func TestEvalChunksSplitsLargeModels(t *testing.T) {
	// A model several chunks wide must report progress more than once,
	// otherwise the parent cannot tell a slow load from a stalled one.
	sizes := make([]int, 64)
	for i := range sizes {
		sizes[i] = evalChunkBytes / 4
	}

	if got := len(evalChunks(sizes)); got != 16 {
		t.Errorf("got %d chunks for a 16-chunk model, want 16", got)
	}
}

func TestEvalChunksHandlesNoArrays(t *testing.T) {
	if got := evalChunks(nil); len(got) != 0 {
		t.Errorf("got %d chunks for an empty model, want 0", len(got))
	}
}

func TestEvalWithProgressToleratesNoArrays(t *testing.T) {
	// Guards against a divide-by-zero or nil-callback panic on a model that
	// collects no arrays.
	evalWithProgress(nil, nil)
}
