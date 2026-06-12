package ggml

import (
	"bytes"
	"fmt"
	"io"
	"math/rand/v2"
	"os"
	"runtime"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
)

func TestWriteGGUF(t *testing.T) {
	tensorData := make([]byte, 2*3*4) // 6 F32 elements = 24 bytes
	for range 8 {
		t.Run("shuffle", func(t *testing.T) {
			t.Parallel()

			ts := []*Tensor{
				{Name: "token_embd.weight", Shape: []uint64{2, 3}, WriterTo: bytes.NewReader(tensorData)},
				{Name: "blk.0.ffn_norm.weight", Shape: []uint64{2, 3}, WriterTo: bytes.NewReader(tensorData)},
				{Name: "blk.0.attn_norm.weight", Shape: []uint64{2, 3}, WriterTo: bytes.NewReader(tensorData)},
				{Name: "blk.1.ffn_up.weight", Shape: []uint64{2, 3}, WriterTo: bytes.NewReader(tensorData)},
				{Name: "blk.2.ffn_norm.weight", Shape: []uint64{2, 3}, WriterTo: bytes.NewReader(tensorData)},
				{Name: "blk.1.ffn_down.weight", Shape: []uint64{2, 3}, WriterTo: bytes.NewReader(tensorData)},
				{Name: "blk.0.attn_k.weight", Shape: []uint64{2, 3}, WriterTo: bytes.NewReader(tensorData)},
				{Name: "output_norm.weight", Shape: []uint64{3, 2}, WriterTo: bytes.NewReader(tensorData)},
				{Name: "output.weight", Shape: []uint64{3, 2}, WriterTo: bytes.NewReader(tensorData)},
			}

			rand.Shuffle(len(ts), func(i, j int) {
				ts[i], ts[j] = ts[j], ts[i]
			})

			w, err := os.CreateTemp(t.TempDir(), strings.ReplaceAll(t.Name(), "/", "_")+"*.bin")
			if err != nil {
				t.Fatal(err)
			}
			defer w.Close()

			if err := WriteGGUF(w, KV{
				"general.architecture": "test",
				"general.alignment":    uint32(16),
				"test.key":             "value",
				"test.bytes":           []uint8{1, 2, 3, 255},
				"test.int32_key":       int32(-42),
				"test.int64_key":       int64(-9223372036854775808),
				"test.int32_array":     []int32{-1, 0, 1, 2147483647, -2147483648},
				"test.int64_array":     []int64{-1, 0, 1, 9223372036854775807, -9223372036854775808},
				"attention.key":        "value2",
				"tokenizer.key":        "value3",
				"adapter.key":          "value4",
			}, ts); err != nil {
				t.Fatal(err)
			}

			r, err := os.Open(w.Name())
			if err != nil {
				t.Fatal(err)
			}
			defer r.Close()

			ff, err := Decode(r, -1)
			if err != nil {
				t.Fatal(err)
			}

			if diff := cmp.Diff(KV{
				"general.architecture":    "test",
				"general.alignment":       uint32(16),
				"general.parameter_count": uint64(54),
				"test.key":                "value",
				"test.bytes":              &array[uint8]{size: 4, values: []uint8{1, 2, 3, 255}},
				"test.int32_key":          int32(-42),
				"test.int64_key":          int64(-9223372036854775808),
				"test.int32_array":        &array[int32]{size: 5, values: []int32{-1, 0, 1, 2147483647, -2147483648}},
				"test.int64_array":        &array[int64]{size: 5, values: []int64{-1, 0, 1, 9223372036854775807, -9223372036854775808}},
				"test.attention.key":      "value2",
				"tokenizer.key":           "value3",
				"adapter.key":             "value4",
			}, ff.KV(), cmp.AllowUnexported(array[uint8]{}, array[int32]{}, array[int64]{})); diff != "" {
				t.Errorf("Mismatch (-want +got):\n%s", diff)
			}

			if diff := cmp.Diff(Tensors{
				Offset: 1040,
				items: []*Tensor{
					{Name: "blk.0.attn_k.weight", Offset: 0, Shape: []uint64{2, 3}},
					{Name: "blk.0.attn_norm.weight", Offset: 32, Shape: []uint64{2, 3}},
					{Name: "blk.0.ffn_norm.weight", Offset: 64, Shape: []uint64{2, 3}},
					{Name: "blk.1.ffn_down.weight", Offset: 96, Shape: []uint64{2, 3}},
					{Name: "blk.1.ffn_up.weight", Offset: 128, Shape: []uint64{2, 3}},
					{Name: "blk.2.ffn_norm.weight", Offset: 160, Shape: []uint64{2, 3}},
					{Name: "output.weight", Offset: 192, Shape: []uint64{3, 2}},
					{Name: "output_norm.weight", Offset: 224, Shape: []uint64{3, 2}},
					{Name: "token_embd.weight", Offset: 256, Shape: []uint64{2, 3}},
				},
			}, ff.Tensors(), cmp.AllowUnexported(Tensors{})); diff != "" {
				t.Errorf("Mismatch (-want +got):\n%s", diff)
			}
		})
	}

	t.Run("truncated_tensor_data", func(t *testing.T) {
		t.Parallel()

		ts := []*Tensor{
			{Name: "blk.0.attn.weight", Kind: 0, Shape: []uint64{512, 2}, WriterTo: bytes.NewBuffer(make([]byte, 32))},
		}

		w, err := os.CreateTemp(t.TempDir(), "truncated_*.bin")
		if err != nil {
			t.Fatal(err)
		}
		defer w.Close()

		if err := WriteGGUF(w, KV{"general.architecture": "test"}, ts); err != nil {
			t.Fatal(err)
		}

		r, err := os.Open(w.Name())
		if err != nil {
			t.Fatal(err)
		}
		defer r.Close()

		if _, err := Decode(r, -1); err == nil {
			t.Error("Decode should reject GGUF files where tensor data extends beyond file size")
		}
	})
}

type blockingCountingWriterTo struct {
	active  *atomic.Int32
	max     *atomic.Int32
	entered chan<- struct{}
	release <-chan struct{}
}

func (w blockingCountingWriterTo) WriteTo(io.Writer) (int64, error) {
	active := w.active.Add(1)
	for {
		maxActive := w.max.Load()
		if active <= maxActive || w.max.CompareAndSwap(maxActive, active) {
			break
		}
	}
	w.entered <- struct{}{}
	<-w.release
	w.active.Add(-1)
	return 0, nil
}

func TestWriteGGUFLimitsLargeTensorConcurrency(t *testing.T) {
	old := runtime.GOMAXPROCS(4)
	t.Cleanup(func() { runtime.GOMAXPROCS(old) })

	var active, maxActive atomic.Int32
	entered := make(chan struct{}, 3)
	release := make(chan struct{})
	ts := []*Tensor{
		{Name: "blk.0.large.weight", Shape: []uint64{512 << 20}, WriterTo: blockingCountingWriterTo{active: &active, max: &maxActive, entered: entered, release: release}},
		{Name: "blk.1.large.weight", Shape: []uint64{512 << 20}, WriterTo: blockingCountingWriterTo{active: &active, max: &maxActive, entered: entered, release: release}},
		{Name: "blk.2.large.weight", Shape: []uint64{512 << 20}, WriterTo: blockingCountingWriterTo{active: &active, max: &maxActive, entered: entered, release: release}},
	}

	w, err := os.CreateTemp(t.TempDir(), "large-concurrency-*.bin")
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()

	done := make(chan error, 1)
	go func() {
		done <- WriteGGUF(w, KV{"general.architecture": "test"}, ts)
	}()

	for range 2 {
		select {
		case <-entered:
		case <-time.After(2 * time.Second):
			close(release)
			select {
			case err := <-done:
				if err != nil {
					t.Fatal(err)
				}
			case <-time.After(2 * time.Second):
				t.Fatal("WriteGGUF did not finish after releasing blocked writers")
			}
			t.Fatalf("max concurrent writers = %d, want 2", maxActive.Load())
		}
	}
	close(release)

	if err := <-done; err != nil {
		t.Fatal(err)
	}
	if maxActive.Load() != 2 {
		t.Fatalf("max concurrent writers = %d, want 2", maxActive.Load())
	}
}

func TestWriteGGUFConcurrencyLimitCapsGOMAXPROCS(t *testing.T) {
	old := runtime.GOMAXPROCS(8)
	t.Cleanup(func() { runtime.GOMAXPROCS(old) })

	if got := ggufWriteConcurrencyLimit(); got != 4 {
		t.Fatalf("ggufWriteConcurrencyLimit() = %d, want 4", got)
	}
}

func TestWriteGGUFConcurrencyLimitUsesLowerGOMAXPROCS(t *testing.T) {
	old := runtime.GOMAXPROCS(2)
	t.Cleanup(func() { runtime.GOMAXPROCS(old) })

	if got := ggufWriteConcurrencyLimit(); got != 2 {
		t.Fatalf("ggufWriteConcurrencyLimit() = %d, want 2", got)
	}
}

func TestWriteGGUFConcurrencyLimitUsesMemoryBudget(t *testing.T) {
	old := runtime.GOMAXPROCS(8)
	t.Cleanup(func() { runtime.GOMAXPROCS(old) })

	if got := ggufWriteConcurrencyLimitForBudget(1); got != 8 {
		t.Fatalf("ggufWriteConcurrencyLimitForBudget() = %d, want 8", got)
	}
}

func TestGGUFWriteMemoryBudgetKeepsFractionAndPad(t *testing.T) {
	tests := []struct {
		name      string
		available uint64
		want      uint64
	}{
		{name: "unknown", available: 0, want: 0},
		{name: "below pad", available: 4 << 30, want: 3 << 30},
		{name: "pad", available: 20 << 30, want: 12 << 30},
		{name: "fraction", available: 64 << 30, want: 48 << 30},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := ggufWriteMemoryBudget(tt.available); got != tt.want {
				t.Fatalf("ggufWriteMemoryBudget(%d) = %d, want %d", tt.available, got, tt.want)
			}
		})
	}
}

func TestGGUFWriteSemaphoreLimitSaturates(t *testing.T) {
	if got := ggufWriteSemaphoreLimit(4, 0); got != 4 {
		t.Fatalf("ggufWriteSemaphoreLimit(4, 0) = %d, want 4", got)
	}
	if got := ggufWriteSemaphoreLimit(4, 64<<30); got != 64<<30 {
		t.Fatalf("ggufWriteSemaphoreLimit(4, 64GiB) = %d, want %d", got, int64(64<<30))
	}
	if got := ggufWriteSemaphoreLimit(4, maxInt64Uint+1); got != int64(maxInt64Uint) {
		t.Fatalf("ggufWriteSemaphoreLimit() = %d, want %d", got, int64(maxInt64Uint))
	}
}

func TestGGUFTensorWriteWeightUsesQuantizedWorkingSet(t *testing.T) {
	tensor := &Tensor{
		Name:     "blk.0.quantized.weight",
		Kind:     uint32(TensorTypeQ4_K),
		Shape:    []uint64{257 << 20},
		WriterTo: bytes.NewReader(nil),
	}
	if tensor.Size() >= 1<<30 {
		t.Fatalf("test tensor output size = %d, want less than exclusive threshold", tensor.Size())
	}
	if got := ggufTensorWriteWeight(tensor, 4, 0); got != 2 {
		t.Fatalf("ggufTensorWriteWeight() = %d, want 2", got)
	}
}

type memoryEstimateWriterTo struct {
	estimate uint64
}

func (w memoryEstimateWriterTo) WriteTo(io.Writer) (int64, error) {
	return 0, nil
}

func (w memoryEstimateWriterTo) GGUFWriteMemoryEstimate() uint64 {
	return w.estimate
}

func TestGGUFTensorWriteWeightUsesWriterEstimate(t *testing.T) {
	tensor := &Tensor{
		Name:     "blk.0.estimated.weight",
		Kind:     uint32(TensorTypeF16),
		Shape:    []uint64{1024},
		WriterTo: memoryEstimateWriterTo{estimate: 1 << 30},
	}
	if got := ggufTensorWriteWeight(tensor, 4, 0); got != 2 {
		t.Fatalf("ggufTensorWriteWeight() = %d, want 2", got)
	}
}

func TestGGUFTensorWriteEstimateOverridesQuantizedFallback(t *testing.T) {
	tensor := &Tensor{
		Name:     "blk.0.copied_quantized.weight",
		Kind:     uint32(TensorTypeQ4_K),
		Shape:    []uint64{1 << 30},
		WriterTo: memoryEstimateWriterTo{estimate: 32 << 10},
	}
	if got := ggufTensorWriteWeight(tensor, 4, 0); got != 1 {
		t.Fatalf("ggufTensorWriteWeight() = %d, want 1", got)
	}
}

func TestWriteGGUFWithOptionsReturnsTensorWriteError(t *testing.T) {
	wantErr := fmt.Errorf("write failed")
	ts := []*Tensor{
		{Name: "blk.0.bad.weight", Shape: []uint64{1}, WriterTo: failingWriterTo{err: wantErr}},
	}

	w, err := os.CreateTemp(t.TempDir(), "write-error-*.bin")
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()

	err = WriteGGUFWithOptions(w, KV{"general.architecture": "test"}, ts, WriteGGUFOptions{AvailableMemory: 64 << 30})
	if err != wantErr {
		t.Fatalf("WriteGGUFWithOptions() error = %v, want %v", err, wantErr)
	}
}

type failingWriterTo struct {
	err error
}

func (w failingWriterTo) WriteTo(io.Writer) (int64, error) {
	return 0, w.err
}
