package mlxrunner

import (
	"math"
	"sync/atomic"

	"github.com/ollama/ollama/llm"
)

// loadState carries the runner's load status to the /v1/status handler. The
// HTTP server starts before the model is loaded so the parent can tell a slow
// load apart from a stalled one, which means handler goroutines read these
// fields while the load goroutine writes them.
type loadState struct {
	status        atomic.Int32
	progress      atomic.Uint32 // float32 bits
	contextLength atomic.Int64
}

func newLoadState() *loadState {
	s := &loadState{}
	s.status.Store(int32(llm.ServerStatusLoadingModel))
	return s
}

func (s *loadState) Status() llm.ServerStatus {
	return llm.ServerStatus(s.status.Load())
}

func (s *loadState) Progress() float32 {
	return math.Float32frombits(s.progress.Load())
}

func (s *loadState) ContextLength() int {
	return int(s.contextLength.Load())
}

// SetProgress records how much of the model has been materialized, as a
// fraction between 0 and 1.
func (s *loadState) SetProgress(progress float32) {
	if math.IsNaN(float64(progress)) || progress <= 0 {
		return
	}
	if progress > 1 {
		progress = 1
	}

	next := math.Float32bits(progress)
	for {
		current := s.progress.Load()
		if progress <= math.Float32frombits(current) {
			return
		}
		if s.progress.CompareAndSwap(current, next) {
			return
		}
	}
}

// MarkReady publishes the loaded model's context length and flips the runner
// to ready. Callers must not report progress afterwards.
//
// The status store stays last: it is the release edge publishing everything the
// load wrote to handlers, which acquire it by gating on Status.
func (s *loadState) MarkReady(contextLength int) {
	s.contextLength.Store(int64(contextLength))
	s.progress.Store(math.Float32bits(1))
	s.status.Store(int32(llm.ServerStatusReady))
}
