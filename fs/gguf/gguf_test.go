package gguf_test

import (
	"bytes"
	"encoding/binary"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/fs/gguf"
)

func createBinFile(tb testing.TB) string {
	tb.Helper()
	f, err := os.CreateTemp(tb.TempDir(), "")
	if err != nil {
		tb.Fatal(err)
	}
	defer f.Close()

	kv := ggml.KV{
		"general.architecture":                   "llama",
		"llama.block_count":                      uint32(8),
		"llama.embedding_length":                 uint32(3),
		"llama.attention.head_count":             uint32(2),
		"llama.attention.head_count_kv":          uint32(2),
		"llama.attention.key_length":             uint32(3),
		"llama.rope.dimension_count":             uint32(4),
		"llama.rope.freq_base":                   float32(10000.0),
		"llama.rope.freq_scale":                  float32(1.0),
		"llama.attention.layer_norm_rms_epsilon": float32(1e-6),
		"tokenizer.ggml.eos_token_id":            uint32(0),
		"tokenizer.ggml.eos_token_ids":           []int32{1, 2, 3},
		"tokenizer.ggml.tokens":                  []string{"hello", "world"},
		"tokenizer.ggml.scores":                  []float32{0, 1},
	}

	tensors := []*ggml.Tensor{
		{
			Name:     "token_embd.weight",
			Kind:     0,
			Shape:    []uint64{2, 3},
			WriterTo: bytes.NewBuffer(make([]byte, 4*2*3)),
		},
		{
			Name:     "output.weight",
			Kind:     0,
			Shape:    []uint64{3, 2},
			WriterTo: bytes.NewBuffer(make([]byte, 4*3*2)),
		},
	}

	for i := range 8 {
		tensors = append(tensors, &ggml.Tensor{
			Name:     "blk." + strconv.Itoa(i) + ".attn_q.weight",
			Kind:     0,
			Shape:    []uint64{3, 3},
			WriterTo: bytes.NewBuffer(make([]byte, 4*3*3)),
		}, &ggml.Tensor{
			Name:     "blk." + strconv.Itoa(i) + ".attn_k.weight",
			Kind:     0,
			Shape:    []uint64{3, 3},
			WriterTo: bytes.NewBuffer(make([]byte, 4*3*3)),
		}, &ggml.Tensor{
			Name:     "blk." + strconv.Itoa(i) + ".attn_v.weight",
			Kind:     0,
			Shape:    []uint64{3, 3},
			WriterTo: bytes.NewBuffer(make([]byte, 4*3*3)),
		}, &ggml.Tensor{
			Name:     "blk." + strconv.Itoa(i) + ".attn_output.weight",
			Kind:     0,
			Shape:    []uint64{3, 3},
			WriterTo: bytes.NewBuffer(make([]byte, 4*3*3)),
		})
	}

	if err := ggml.WriteGGUF(f, kv, tensors); err != nil {
		tb.Fatal(err)
	}

	return f.Name()
}

func TestRead(t *testing.T) {
	f, err := gguf.Open(createBinFile(t))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	if got := f.KeyValue("does.not.exist").Valid(); got {
		t.Errorf(`KeyValue("does.not.exist").Exists() = %v, want false`, got)
	}

	if got := f.KeyValue("general.architecture").String(); got != "llama" {
		t.Errorf(`KeyValue("general.architecture").String() = %q, want %q`, got, "llama")
	}

	if got := f.TensorInfo("token_embd.weight"); got.Name != "token_embd.weight" {
		t.Errorf(`TensorInfo("token_embd.weight").Name = %q, want %q`, got.Name, "token_embd.weight")
	} else if diff := cmp.Diff(got.Shape, []uint64{2, 3}); diff != "" {
		t.Errorf(`TensorInfo("token_embd.weight").Shape mismatch (-got +want):\n%s`, diff)
	} else if got.Type != gguf.TensorTypeF32 {
		t.Errorf(`TensorInfo("token_embd.weight").Type = %d, want %d`, got.Type, gguf.TensorTypeF32)
	}

	if got := f.KeyValue("block_count").Uint(); got != 8 {
		t.Errorf(`KeyValue("block_count").Uint() = %d, want %d`, got, 8)
	}

	if diff := cmp.Diff(f.KeyValue("tokenizer.ggml.tokens").Strings(), []string{"hello", "world"}); diff != "" {
		t.Errorf("KeyValue(\"tokenizer.ggml.tokens\").Strings() mismatch (-got +want):\n%s", diff)
	}

	if diff := cmp.Diff(f.KeyValue("tokenizer.ggml.scores").Floats(), []float64{0, 1}); diff != "" {
		t.Errorf("KeyValue(\"tokenizer.ggml.scores\").Ints() mismatch (-got +want):\n%s", diff)
	}

	var kvs []string
	for _, kv := range f.KeyValues() {
		if !kv.Valid() {
			t.Error("found invalid key-value pair:", kv)
		}

		kvs = append(kvs, kv.Key)
	}

	if len(kvs) != f.NumKeyValues() {
		t.Errorf("iterated key count = %d, want %d", len(kvs), f.NumKeyValues())
	}

	if diff := cmp.Diff(kvs, []string{
		"general.architecture",
		"llama.block_count",
		"llama.embedding_length",
		"llama.attention.head_count",
		"llama.attention.head_count_kv",
		"llama.attention.key_length",
		"llama.rope.dimension_count",
		"llama.rope.freq_base",
		"llama.rope.freq_scale",
		"llama.attention.layer_norm_rms_epsilon",
		"tokenizer.ggml.eos_token_id",
		"tokenizer.ggml.eos_token_ids",
		"tokenizer.ggml.tokens",
		"tokenizer.ggml.scores",
	}, cmpopts.SortSlices(strings.Compare)); diff != "" {
		t.Errorf("KeyValues() mismatch (-got +want):\n%s", diff)
	}

	var tis []string
	for _, ti := range f.TensorInfos() {
		if !ti.Valid() {
			t.Error("found invalid tensor info:", ti)
		}

		tis = append(tis, ti.Name)
	}

	if len(tis) != f.NumTensors() {
		t.Errorf("iterated tensor count = %d, want %d", len(tis), f.NumTensors())
	}

	if diff := cmp.Diff(tis, []string{
		"token_embd.weight",
		"output.weight",
		"blk.0.attn_q.weight",
		"blk.0.attn_k.weight",
		"blk.0.attn_v.weight",
		"blk.0.attn_output.weight",
		"blk.1.attn_q.weight",
		"blk.1.attn_k.weight",
		"blk.1.attn_v.weight",
		"blk.1.attn_output.weight",
		"blk.2.attn_q.weight",
		"blk.2.attn_k.weight",
		"blk.2.attn_v.weight",
		"blk.2.attn_output.weight",
		"blk.3.attn_q.weight",
		"blk.3.attn_k.weight",
		"blk.3.attn_v.weight",
		"blk.3.attn_output.weight",
		"blk.4.attn_q.weight",
		"blk.4.attn_k.weight",
		"blk.4.attn_v.weight",
		"blk.4.attn_output.weight",
		"blk.5.attn_q.weight",
		"blk.5.attn_k.weight",
		"blk.5.attn_v.weight",
		"blk.5.attn_output.weight",
		"blk.6.attn_q.weight",
		"blk.6.attn_k.weight",
		"blk.6.attn_v.weight",
		"blk.6.attn_output.weight",
		"blk.7.attn_q.weight",
		"blk.7.attn_k.weight",
		"blk.7.attn_v.weight",
		"blk.7.attn_output.weight",
	}, cmpopts.SortSlices(strings.Compare)); diff != "" {
		t.Errorf("TensorInfos() mismatch (-got +want):\n%s", diff)
	}

	ti, r, err := f.TensorReader("output.weight")
	if err != nil {
		t.Fatalf(`TensorReader("output.weight") error: %v`, err)
	}

	if ti.Name != "output.weight" {
		t.Errorf(`TensorReader("output.weight").Name = %q, want %q`, ti.Name, "output.weight")
	} else if diff := cmp.Diff(ti.Shape, []uint64{3, 2}); diff != "" {
		t.Errorf(`TensorReader("output.weight").Shape mismatch (-got +want):\n%s`, diff)
	} else if ti.Type != gguf.TensorTypeF32 {
		t.Errorf(`TensorReader("output.weight").Type = %d, want %d`, ti.Type, gguf.TensorTypeF32)
	}

	var b bytes.Buffer
	if _, err := b.ReadFrom(r); err != nil {
		t.Fatalf(`ReadFrom TensorReader("output.weight") error: %v`, err)
	}

	if b.Len() != int(ti.NumBytes()) {
		t.Errorf(`ReadFrom TensorReader("output.weight") length = %d, want %d`, b.Len(), ti.NumBytes())
	}
}

func TestScanMetadata(t *testing.T) {
	info, err := gguf.ScanMetadata(createBinFile(t))
	if err != nil {
		t.Fatal(err)
	}

	if info.Architecture != "llama" {
		t.Fatalf("architecture = %q, want llama", info.Architecture)
	}
	if info.EmbeddingLength != 3 {
		t.Fatalf("embedding length = %d, want 3", info.EmbeddingLength)
	}
	if info.HasFileType {
		t.Fatalf("has file type = true, want false: %+v", info)
	}
	if info.HasEmbedding || info.HasVision || info.HasAudio {
		t.Fatalf("unexpected capability metadata: %+v", info)
	}
}

func TestScanMetadataCorrupt(t *testing.T) {
	le32 := func(v uint32) []byte {
		var b [4]byte
		binary.LittleEndian.PutUint32(b[:], v)
		return b[:]
	}
	le64 := func(v uint64) []byte {
		var b [8]byte
		binary.LittleEndian.PutUint64(b[:], v)
		return b[:]
	}
	str := func(s string) []byte {
		return append(le64(uint64(len(s))), s...)
	}
	// GGUF v3 little-endian header with zero tensors and one metadata entry.
	header := func(entry []byte) []byte {
		var b []byte
		b = append(b, le32(0x46554747)...) // magic
		b = append(b, le32(3)...)          // version
		b = append(b, le64(0)...)          // tensor count
		b = append(b, le64(1)...)          // kv count
		return append(b, entry...)
	}

	// Value type ids from the GGUF spec: 8=string, 9=array, 10=uint64.
	cases := map[string][]byte{
		// Read path: a string value whose declared length would allocate
		// exabytes must fail the MaxStringLength check, not make([]byte, n).
		"huge string value": header(append(append(str("general.architecture"), le32(8)...), le64(1<<62)...)),
		// Skip path: an array whose count*size multiply would overflow into a
		// negative skip must fail the MaxArraySize check, not silently no-op.
		"huge array count": header(append(append(append(str("tokenizer.ggml.tokens"), le32(9)...), le32(10)...), le64(1<<61)...)),
		// Skip path: a skipped string with an absurd length must not desync.
		"huge skipped string": header(append(append(str("some.ignored.key"), le32(8)...), le64(1<<62)...)),
	}

	for name, data := range cases {
		t.Run(name, func(t *testing.T) {
			p := filepath.Join(t.TempDir(), "corrupt.gguf")
			if err := os.WriteFile(p, data, 0o644); err != nil {
				t.Fatal(err)
			}
			if _, err := gguf.ScanMetadata(p); err == nil {
				t.Fatal("expected error scanning corrupt metadata, got nil")
			}
		})
	}
}

func TestOpenDoesNotPreallocateDeclaredArray(t *testing.T) {
	const allocationLimit = 1 << 20

	for _, elementType := range []uint32{0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12} {
		t.Run(strconv.FormatUint(uint64(elementType), 10), func(t *testing.T) {
			var data []byte
			data = append(data, 'G', 'G', 'U', 'F')
			data = binary.LittleEndian.AppendUint32(data, 3)
			data = binary.LittleEndian.AppendUint64(data, 0)
			data = binary.LittleEndian.AppendUint64(data, 1)
			data = binary.LittleEndian.AppendUint64(data, 3)
			data = append(data, "big"...)
			data = binary.LittleEndian.AppendUint32(data, 9)
			data = binary.LittleEndian.AppendUint32(data, elementType)
			data = binary.LittleEndian.AppendUint64(data, 8_000_000)

			runtime.GC()
			var before, after runtime.MemStats
			runtime.ReadMemStats(&before)
			g, err := gguf.Open(createFile(t, data))
			if err != nil {
				t.Fatal(err)
			}
			for range g.KeyValues() {
			}
			if err := g.Close(); err != nil {
				t.Fatal(err)
			}
			runtime.ReadMemStats(&after)

			if allocated := after.TotalAlloc - before.TotalAlloc; allocated > allocationLimit {
				t.Fatalf("declared array allocated %d bytes, want at most %d", allocated, allocationLimit)
			}
		})
	}
}

func FuzzOpen(f *testing.F) {
	validPath := createBinFile(f)
	validData, err := os.ReadFile(validPath)
	if err != nil {
		f.Fatal(err)
	}
	f.Add(validData)
	f.Add([]byte{})
	f.Add([]byte{'G', 'G', 'U', 'F'})
	f.Add([]byte{
		'G', 'G', 'U', 'F',
		3, 0, 0, 0,
		1, 0, 0, 0, 0, 0, 0, 0,
	})
	f.Add([]byte{
		'G', 'G', 'U', 'F',
		3, 0, 0, 0,
		0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
	})

	f.Fuzz(func(t *testing.T, data []byte) {
		if len(data) > 1<<20 {
			t.Skip("bounded fuzz input")
		}

		g, err := gguf.Open(createFile(t, data))
		if err != nil {
			return
		}
		defer g.Close()

		for i, kv := range g.KeyValues() {
			if i > 4096 {
				break
			}
			_ = kv.Valid()
		}
		for i, tensor := range g.TensorInfos() {
			if i > 4096 {
				break
			}
			_ = tensor.Valid()
		}
		_ = g.Err()
	})
}

func createFile(tb testing.TB, b []byte) string {
	tb.Helper()

	p := filepath.Join(tb.TempDir(), "model.gguf")
	if err := os.WriteFile(p, b, 0o600); err != nil {
		tb.Fatal(err)
	}

	return p
}

func BenchmarkRead(b *testing.B) {
	b.ReportAllocs()

	p := createBinFile(b)
	for b.Loop() {
		f, err := gguf.Open(p)
		if err != nil {
			b.Fatal(err)
		}

		if got := f.KeyValue("general.architecture").String(); got != "llama" {
			b.Errorf("got = %q, want %q", got, "llama")
		}

		// Iterate through some tensors
		for range f.TensorInfos() {
		}

		f.Close()
	}
}
