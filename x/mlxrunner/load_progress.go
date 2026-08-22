package mlxrunner

import (
	"math"
	"sync/atomic"

	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

// Bytes per mlx.Eval; smaller chunks report more often but cost load time.
const evalChunkBytes = 1 << 30

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
	s.progress.Store(math.Float32bits(progress))
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

// evalChunk is a half-open range of arrays to materialize in one eval, along
// with the fraction of the model loaded once it completes.
type evalChunk struct {
	start, end int
	progress   float32
}

// evalChunks splits arrays of the given byte sizes into chunks of at least
// evalChunkBytes, the last chunk taking whatever remains. Progress is weighted
// by bytes rather than array count because array sizes vary by orders of
// magnitude.
func evalChunks(sizes []int) []evalChunk {
	var total int
	for _, size := range sizes {
		total += size
	}

	var chunks []evalChunk
	var done, pending, start int
	for i, size := range sizes {
		pending += size
		if pending < evalChunkBytes && i < len(sizes)-1 {
			continue
		}

		done += pending
		progress := float32(1)
		if total > 0 {
			progress = float32(done) / float32(total)
		}

		chunks = append(chunks, evalChunk{start: start, end: i + 1, progress: progress})
		pending, start = 0, i+1
	}

	return chunks
}

// evalWithProgress materializes arrays a chunk at a time, reporting the
// fraction completed after each one. Chunking does not cost read concurrency:
// MLX parallelizes reads within a single tensor rather than across tensors.
func evalWithProgress(arrays []*mlx.Array, progress func(float32)) {
	sizes := make([]int, len(arrays))
	for i, arr := range arrays {
		sizes[i] = arr.NumBytes()
	}

	for _, chunk := range evalChunks(sizes) {
		mlx.Eval(arrays[chunk.start:chunk.end]...)
		if progress != nil {
			progress(chunk.progress)
		}
	}
}
