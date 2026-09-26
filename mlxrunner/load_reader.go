package mlxrunner

import (
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"runtime"
	"sync"
	"sync/atomic"

	"github.com/ollama/ollama/mlxrunner/model"
)

// loadReaderBatchSize is 32 MiB, matching MLX's parallel file reader.
const loadReaderBatchSize = 1 << 25

// Let slow storage refresh the stall timer during large reads.
const loadReaderProgressSize = 1 << 22

var loadReaderTokens = make(chan struct{}, loadReaderParallelism())

func loadReaderParallelism() int {
	return min(max(runtime.NumCPU()/2, 4), 16)
}

type fileDescriptor struct {
	mu   sync.Mutex // guards file and serializes one active read per descriptor
	file *os.File
}

// parallelFileReader mirrors MLX's bounded batched reads while giving each
// worker its own file description. A worker reads one contiguous range so
// network filesystems and the kernel can maintain sequential readahead.
type parallelFileReader struct {
	path        string
	size        int64
	progress    func(int64)
	descriptors []fileDescriptor
	next        atomic.Uint32
	closed      atomic.Bool
}

func newParallelFileReader(path string, progress func(int64)) (*parallelFileReader, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	info, err := file.Stat()
	if err != nil {
		file.Close()
		return nil, err
	}

	descriptors := make([]fileDescriptor, loadReaderParallelism())
	descriptors[0].file = file
	return &parallelFileReader{
		path:        path,
		size:        info.Size(),
		progress:    progress,
		descriptors: descriptors,
	}, nil
}

func (r *parallelFileReader) Size() int64 {
	return r.size
}

func (r *parallelFileReader) Name() string {
	return r.path
}

func (r *parallelFileReader) ReadAt(p []byte, off int64) (int, error) {
	if len(p) == 0 {
		return 0, nil
	}
	if off < 0 || off > math.MaxInt64-int64(len(p)) {
		return 0, fmt.Errorf("%s: invalid read range at offset %d with length %d", r.path, off, len(p))
	}
	if r.closed.Load() {
		return 0, os.ErrClosed
	}

	batches := (len(p) + loadReaderBatchSize - 1) / loadReaderBatchSize
	workers := min(batches, len(r.descriptors))
	if workers == 1 {
		// MLX submits small tensor reads concurrently. Spread them across
		// descriptions so one mutex does not serialize the load.
		index := int(r.next.Add(1)-1) % len(r.descriptors)
		return r.readRange(index, p, off)
	}

	type result struct {
		n   int
		err error
	}

	// Group adjacent MLX-sized batches into one range per descriptor. Unlike a
	// work queue, this keeps every descriptor's access pattern sequential.
	batchesPerRange := (batches + workers - 1) / workers
	rangeSize := batchesPerRange * loadReaderBatchSize
	results := make(chan result, workers)
	launched := 0
	for i := range workers {
		start := i * rangeSize
		if start >= len(p) {
			break
		}
		end := min(start+rangeSize, len(p))
		launched++
		go func(i, start, end int) {
			loadReaderTokens <- struct{}{}
			defer func() { <-loadReaderTokens }()
			n, err := r.readRange(i, p[start:end], off+int64(start))
			results <- result{n: n, err: err}
		}(i, start, end)
	}

	var firstErr error
	total := 0
	for range launched {
		result := <-results
		total += result.n
		if firstErr == nil && result.err != nil {
			firstErr = result.err
		}
	}
	if total != len(p) && firstErr == nil {
		firstErr = io.ErrUnexpectedEOF
	}
	return total, firstErr
}

func (r *parallelFileReader) readRange(index int, p []byte, off int64) (int, error) {
	descriptor := &r.descriptors[index]
	descriptor.mu.Lock()
	defer descriptor.mu.Unlock()

	if r.closed.Load() {
		return 0, os.ErrClosed
	}
	if descriptor.file == nil {
		file, err := os.Open(r.path)
		if err != nil {
			return 0, err
		}
		descriptor.file = file
	}

	total := 0
	for total < len(p) {
		chunk := p[total : total+min(loadReaderProgressSize, len(p)-total)]
		n, err := descriptor.file.ReadAt(chunk, off+int64(total))
		total += n
		if n > 0 && r.progress != nil {
			r.progress(int64(n))
		}
		if n != len(chunk) {
			if err == nil {
				err = io.ErrUnexpectedEOF
			}
			return total, fmt.Errorf("%s: %w", r.path, err)
		}
	}
	return total, nil
}

func (r *parallelFileReader) Close() error {
	if r == nil || r.closed.Swap(true) {
		return nil
	}

	var errs []error
	for i := range r.descriptors {
		descriptor := &r.descriptors[i]
		descriptor.mu.Lock()
		if descriptor.file != nil {
			if err := descriptor.file.Close(); err != nil {
				errs = append(errs, err)
			}
			descriptor.file = nil
		}
		descriptor.mu.Unlock()
	}
	return errors.Join(errs...)
}

func newLoadProgressReporter(root *model.Root, progress func(float32)) func(int64) {
	if progress == nil || root == nil || root.Manifest == nil {
		return nil
	}

	total := uniqueTensorLayerSize(root)
	if total <= 0 {
		return nil
	}

	var completed atomic.Int64
	return func(n int64) {
		if n <= 0 {
			return
		}
		fraction := min(float32(completed.Add(n))/float32(total), 1)
		progress(fraction)
	}
}

func uniqueTensorLayerSize(root *model.Root) int64 {
	var total int64
	seen := make(map[string]bool)
	for _, layer := range root.Manifest.TensorLayers() {
		if seen[layer.Digest] {
			continue
		}
		seen[layer.Digest] = true
		total += layer.Size
	}
	return total
}
