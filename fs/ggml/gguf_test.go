package ggml

import (
	"bytes"
	"encoding/binary"
	"errors"
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
				"test.int32_key":          int32(-42),
				"test.int64_key":          int64(-9223372036854775808),
				"test.int32_array":        &array[int32]{size: 5, values: []int32{-1, 0, 1, 2147483647, -2147483648}},
				"test.int64_array":        &array[int64]{size: 5, values: []int64{-1, 0, 1, 9223372036854775807, -9223372036854775808}},
				"test.attention.key":      "value2",
				"tokenizer.key":           "value3",
				"adapter.key":             "value4",
			}, ff.KV(), cmp.AllowUnexported(array[int32]{}, array[int64]{})); diff != "" {
				t.Errorf("Mismatch (-want +got):\n%s", diff)
			}

			if diff := cmp.Diff(Tensors{
				Offset: 992,
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

func minimalGGUFHeader(numTensors, numKV uint64) *bytes.Buffer {
	var b bytes.Buffer
	binaryWrite := func(v any) {
		_ = binary.Write(&b, binary.LittleEndian, v)
	}
	binaryWrite(uint32(FILE_MAGIC_GGUF_LE))
	binaryWrite(uint32(3))
	binaryWrite(numTensors)
	binaryWrite(numKV)
	return &b
}

func writeRawGGUFString(b *bytes.Buffer, s string) {
	_ = binary.Write(b, binary.LittleEndian, uint64(len(s)))
	_, _ = b.WriteString(s)
}

func TestDecodeGGUFRejectsExcessiveMetadataCounts(t *testing.T) {
	tests := []struct {
		name       string
		numTensors uint64
		numKV      uint64
	}{
		{name: "kv count", numKV: maxGGUFKeyValueCount + 1},
		{name: "tensor count", numTensors: maxGGUFTensorCount + 1},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := Decode(bytes.NewReader(minimalGGUFHeader(tt.numTensors, tt.numKV).Bytes()), -1); err == nil {
				t.Fatal("Decode succeeded, want metadata count error")
			}
		})
	}
}

func TestDecodeGGUFRejectsExcessiveStringAndArrayLengths(t *testing.T) {
	t.Run("string", func(t *testing.T) {
		b := minimalGGUFHeader(0, 1)
		if err := binary.Write(b, binary.LittleEndian, uint64(maxGGUFStringLength+1)); err != nil {
			t.Fatal(err)
		}

		if _, err := Decode(bytes.NewReader(b.Bytes()), -1); err == nil {
			t.Fatal("Decode succeeded, want string length error")
		}
	})

	t.Run("array", func(t *testing.T) {
		b := minimalGGUFHeader(0, 1)
		writeRawGGUFString(b, "test.array")
		if err := binary.Write(b, binary.LittleEndian, ggufTypeArray); err != nil {
			t.Fatal(err)
		}
		if err := binary.Write(b, binary.LittleEndian, ggufTypeUint32); err != nil {
			t.Fatal(err)
		}
		if err := binary.Write(b, binary.LittleEndian, uint64(maxGGUFArrayLength+1)); err != nil {
			t.Fatal(err)
		}

		if _, err := Decode(bytes.NewReader(b.Bytes()), -1); err == nil {
			t.Fatal("Decode succeeded, want array length error")
		}
	})

	t.Run("discarded string array element", func(t *testing.T) {
		b := minimalGGUFHeader(0, 1)
		writeRawGGUFString(b, "test.strings")
		if err := binary.Write(b, binary.LittleEndian, uint32(ggufTypeArray)); err != nil {
			t.Fatal(err)
		}
		if err := binary.Write(b, binary.LittleEndian, uint32(ggufTypeString)); err != nil {
			t.Fatal(err)
		}
		if err := binary.Write(b, binary.LittleEndian, uint64(1)); err != nil {
			t.Fatal(err)
		}
		if err := binary.Write(b, binary.LittleEndian, uint64(maxGGUFStringLength+1)); err != nil {
			t.Fatal(err)
		}

		if _, err := Decode(bytes.NewReader(b.Bytes()), 0); err == nil {
			t.Fatal("Decode succeeded, want discarded string length error")
		}
	})
}

func TestDecodeGGUFRejectsInvalidTensorMetadata(t *testing.T) {
	tests := []struct {
		name  string
		write func(*bytes.Buffer)
	}{
		{
			name: "too many dimensions",
			write: func(b *bytes.Buffer) {
				writeRawGGUFString(b, "bad.weight")
				_ = binary.Write(b, binary.LittleEndian, uint32(maxGGUFTensorDims+1))
			},
		},
		{
			name: "unsupported kind",
			write: func(b *bytes.Buffer) {
				writeRawGGUFString(b, "bad.weight")
				_ = binary.Write(b, binary.LittleEndian, uint32(1))
				_ = binary.Write(b, binary.LittleEndian, uint64(1))
				_ = binary.Write(b, binary.LittleEndian, uint32(999))
			},
		},
		{
			name: "shape overflow",
			write: func(b *bytes.Buffer) {
				writeRawGGUFString(b, "bad.weight")
				_ = binary.Write(b, binary.LittleEndian, uint32(2))
				_ = binary.Write(b, binary.LittleEndian, uint64(^uint64(0)))
				_ = binary.Write(b, binary.LittleEndian, uint64(2))
				_ = binary.Write(b, binary.LittleEndian, uint32(TensorTypeF32))
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			b := minimalGGUFHeader(1, 0)
			tt.write(b)

			if _, err := Decode(bytes.NewReader(b.Bytes()), -1); err == nil {
				t.Fatal("Decode succeeded, want invalid tensor metadata error")
			}
		})
	}
}

type countingWriterTo struct {
	active *atomic.Int32
	max    *atomic.Int32
}

func (w countingWriterTo) WriteTo(dst io.Writer) (int64, error) {
	active := w.active.Add(1)
	for {
		maxActive := w.max.Load()
		if active <= maxActive || w.max.CompareAndSwap(maxActive, active) {
			break
		}
	}
	time.Sleep(10 * time.Millisecond)
	w.active.Add(-1)
	return 0, nil
}

func TestWriteGGUFLimitsLargeTensorConcurrency(t *testing.T) {
	var active, maxActive atomic.Int32
	ts := []*Tensor{
		{Name: "blk.0.large.weight", Shape: []uint64{512 << 20}, WriterTo: countingWriterTo{active: &active, max: &maxActive}},
		{Name: "blk.1.large.weight", Shape: []uint64{512 << 20}, WriterTo: countingWriterTo{active: &active, max: &maxActive}},
		{Name: "blk.2.large.weight", Shape: []uint64{512 << 20}, WriterTo: countingWriterTo{active: &active, max: &maxActive}},
	}

	w, err := os.CreateTemp(t.TempDir(), "large-concurrency-*.bin")
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()

	if err := WriteGGUF(w, KV{"general.architecture": "test"}, ts); err != nil {
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

func TestGGUFWriteMemoryBudgetFallsBackForImplausiblyLargeFreeMemory(t *testing.T) {
	old := runtime.GOMAXPROCS(8)
	t.Cleanup(func() { runtime.GOMAXPROCS(old) })

	budget := ggufWriteMemoryBudget(^uint64(0))
	if budget != 0 {
		t.Fatalf("ggufWriteMemoryBudget(max uint64) = %d, want 0", budget)
	}
	if got := ggufWriteConcurrencyLimitForBudget(budget); got != 4 {
		t.Fatalf("ggufWriteConcurrencyLimitForBudget(%d) = %d, want 4", budget, got)
	}
}

func TestGGUFWriteMemoryBudgetUsesLargeRealMemoryEstimate(t *testing.T) {
	old := runtime.GOMAXPROCS(16)
	t.Cleanup(func() { runtime.GOMAXPROCS(old) })

	available := uint64(1 << 40) // 1 TiB
	wantBudget := uint64(768 << 30)
	budget := ggufWriteMemoryBudget(available)
	if budget != wantBudget {
		t.Fatalf("ggufWriteMemoryBudget(1TiB) = %d, want %d", budget, wantBudget)
	}
	if got := ggufWriteConcurrencyLimitForBudget(budget); got != 16 {
		t.Fatalf("ggufWriteConcurrencyLimitForBudget(%d) = %d, want 16", budget, got)
	}
	if got := ggufWriteSemaphoreLimit(16, budget); got != int64(wantBudget) {
		t.Fatalf("ggufWriteSemaphoreLimit(16, %d) = %d, want %d", budget, got, int64(wantBudget))
	}
}

func TestGGUFWriteMemoryBudgetAllowsProgressWithVeryLowFreeMemory(t *testing.T) {
	budget := ggufWriteMemoryBudget(1)
	if budget != 1 {
		t.Fatalf("ggufWriteMemoryBudget(1) = %d, want 1", budget)
	}
	if got := ggufWriteSemaphoreLimit(4, budget); got != 1 {
		t.Fatalf("ggufWriteSemaphoreLimit(4, %d) = %d, want 1", budget, got)
	}

	tensor := &Tensor{
		Name:     "blk.0.large.weight",
		Kind:     uint32(TensorTypeF16),
		Shape:    []uint64{1 << 30},
		WriterTo: bytes.NewReader(nil),
	}
	if got := ggufTensorWriteMemoryWeight(tensor, budget); got != 1 {
		t.Fatalf("ggufTensorWriteMemoryWeight() = %d, want 1", got)
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

func TestWriteGGUFWithAvailableMemoryAllowsMoreParallelism(t *testing.T) {
	old := runtime.GOMAXPROCS(8)
	t.Cleanup(func() { runtime.GOMAXPROCS(old) })

	var active, maxActive atomic.Int32
	var ts []*Tensor
	for i := range 8 {
		ts = append(ts, &Tensor{
			Name:     fmt.Sprintf("blk.%d.small.weight", i),
			Shape:    []uint64{1 << 20},
			WriterTo: countingWriterTo{active: &active, max: &maxActive},
		})
	}

	w, err := os.CreateTemp(t.TempDir(), "budget-concurrency-*.bin")
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()

	opts := WriteGGUFOptions{AvailableMemory: 64 << 30}
	if err := WriteGGUFWithOptions(w, KV{"general.architecture": "test"}, ts, opts); err != nil {
		t.Fatal(err)
	}
	if maxActive.Load() <= 4 {
		t.Fatalf("max concurrent writers = %d, want more than fallback cap", maxActive.Load())
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

func TestGGUFTensorWriteWeightUsesUnitWeightWhenLimitIsOne(t *testing.T) {
	tensor := &Tensor{
		Name:     "blk.0.huge.weight",
		Kind:     uint32(TensorTypeF16),
		Shape:    []uint64{4 << 30},
		WriterTo: bytes.NewReader(nil),
	}
	if got := ggufTensorWriteWeight(tensor, 1, 0); got != 1 {
		t.Fatalf("ggufTensorWriteWeight() = %d, want 1", got)
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

func TestGGUFTensorWriteWeightMakesHugeWritersExclusive(t *testing.T) {
	tensor := &Tensor{
		Name:     "blk.0.huge.weight",
		Kind:     uint32(TensorTypeF16),
		Shape:    []uint64{1024},
		WriterTo: memoryEstimateWriterTo{estimate: 8 << 30},
	}
	if got := ggufTensorWriteWeight(tensor, 4, 0); got != 4 {
		t.Fatalf("ggufTensorWriteWeight() = %d, want 4", got)
	}
}

func TestGGUFTensorWriteMemoryWeightHandlesUnknownAndSaturatedEstimates(t *testing.T) {
	unknown := &Tensor{
		Name:     "blk.0.empty.weight",
		Kind:     uint32(TensorTypeF32),
		Shape:    []uint64{0},
		WriterTo: bytes.NewReader(nil),
	}
	if got := ggufTensorWriteMemoryWeight(unknown, 64<<30); got != 1 {
		t.Fatalf("ggufTensorWriteMemoryWeight(unknown) = %d, want 1", got)
	}

	huge := &Tensor{
		Name:     "blk.0.huge.weight",
		Kind:     uint32(TensorTypeF16),
		Shape:    []uint64{1024},
		WriterTo: memoryEstimateWriterTo{estimate: ^uint64(0)},
	}
	if got := ggufTensorWriteMemoryWeight(huge, maxInt64Uint+1); got != int64(maxInt64Uint) {
		t.Fatalf("ggufTensorWriteMemoryWeight(huge) = %d, want %d", got, int64(maxInt64Uint))
	}
}

func TestSaturatingMul(t *testing.T) {
	if got := saturatingMul(3, 4); got != 12 {
		t.Fatalf("saturatingMul(3, 4) = %d, want 12", got)
	}
	if got := saturatingMul(^uint64(0), 2); got != ^uint64(0) {
		t.Fatalf("saturatingMul(max, 2) = %d, want %d", got, ^uint64(0))
	}
}

type classifiedWriterTo struct {
	weight      int
	activeTotal *atomic.Int32
	activeLarge *atomic.Int32
	activeHuge  *atomic.Int32
	activeSmall *atomic.Int32
	maxSmall    *atomic.Int32
	maxLarge    *atomic.Int32
	overlap     *atomic.Bool
}

func (w classifiedWriterTo) WriteTo(dst io.Writer) (int64, error) {
	total := w.activeTotal.Add(1)
	if w.weight > 1 {
		large := w.activeLarge.Add(1)
		for {
			maxLarge := w.maxLarge.Load()
			if large <= maxLarge || w.maxLarge.CompareAndSwap(maxLarge, large) {
				break
			}
		}
		if w.weight == 4 {
			w.activeHuge.Add(1)
			if large != 1 || total != 1 {
				w.overlap.Store(true)
			}
		} else if w.activeHuge.Load() != 0 {
			w.overlap.Store(true)
		}
	} else {
		small := w.activeSmall.Add(1)
		for {
			maxSmall := w.maxSmall.Load()
			if small <= maxSmall || w.maxSmall.CompareAndSwap(maxSmall, small) {
				break
			}
		}
		if w.activeHuge.Load() != 0 {
			w.overlap.Store(true)
		}
	}

	time.Sleep(10 * time.Millisecond)

	if w.weight > 1 {
		if w.weight == 4 {
			w.activeHuge.Add(-1)
		}
		w.activeLarge.Add(-1)
	} else {
		w.activeSmall.Add(-1)
	}
	w.activeTotal.Add(-1)
	return 0, nil
}

func TestWriteGGUFWeightsLargeWritersAndMakesHugeWritersExclusive(t *testing.T) {
	old := runtime.GOMAXPROCS(4)
	t.Cleanup(func() { runtime.GOMAXPROCS(old) })

	var activeTotal, activeLarge, activeHuge, activeSmall, maxSmall, maxLarge atomic.Int32
	var overlap atomic.Bool
	small := func(name string) *Tensor {
		return &Tensor{
			Name:     name,
			Shape:    []uint64{1 << 20},
			WriterTo: classifiedWriterTo{weight: 1, activeTotal: &activeTotal, activeLarge: &activeLarge, activeHuge: &activeHuge, activeSmall: &activeSmall, maxSmall: &maxSmall, maxLarge: &maxLarge, overlap: &overlap},
		}
	}
	large := func(name string) *Tensor {
		return &Tensor{
			Name:     name,
			Shape:    []uint64{512 << 20},
			WriterTo: classifiedWriterTo{weight: 2, activeTotal: &activeTotal, activeLarge: &activeLarge, activeHuge: &activeHuge, activeSmall: &activeSmall, maxSmall: &maxSmall, maxLarge: &maxLarge, overlap: &overlap},
		}
	}
	huge := func(name string) *Tensor {
		return &Tensor{
			Name:     name,
			Shape:    []uint64{8 << 30},
			WriterTo: classifiedWriterTo{weight: 4, activeTotal: &activeTotal, activeLarge: &activeLarge, activeHuge: &activeHuge, activeSmall: &activeSmall, maxSmall: &maxSmall, maxLarge: &maxLarge, overlap: &overlap},
		}
	}
	ts := []*Tensor{
		small("blk.0.small.weight"),
		small("blk.1.small.weight"),
		large("blk.2.large.weight"),
		small("blk.3.small.weight"),
		small("blk.4.small.weight"),
		large("blk.5.large.weight"),
		huge("blk.6.huge.weight"),
	}

	w, err := os.CreateTemp(t.TempDir(), "mixed-concurrency-*.bin")
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()

	if err := WriteGGUF(w, KV{"general.architecture": "test"}, ts); err != nil {
		t.Fatal(err)
	}
	if overlap.Load() {
		t.Fatal("huge writer overlapped another tensor write")
	}
	if maxSmall.Load() < 2 {
		t.Fatalf("max concurrent small writers = %d, want at least 2", maxSmall.Load())
	}
}

type errorWriterTo struct {
	err error
}

func (w errorWriterTo) WriteTo(io.Writer) (int64, error) {
	return 0, w.err
}

func TestWriteGGUFWithOptionsReturnsTensorWriteError(t *testing.T) {
	wantErr := errors.New("write failed")
	ts := []*Tensor{
		{Name: "blk.0.fail.weight", Shape: []uint64{1}, WriterTo: errorWriterTo{err: wantErr}},
	}

	w, err := os.CreateTemp(t.TempDir(), "write-error-*.bin")
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()

	err = WriteGGUFWithOptions(w, KV{"general.architecture": "test"}, ts, WriteGGUFOptions{AvailableMemory: 64 << 30})
	if !errors.Is(err, wantErr) {
		t.Fatalf("WriteGGUFWithOptions() error = %v, want %v", err, wantErr)
	}
}
