package llm

import (
	"bufio"
	"io"
	"os"
	"testing"

	"github.com/ollama/ollama/fs/gguf"
	gguftest "github.com/ollama/ollama/internal/testutil/gguf"
)

// writeSparseTestGGUF writes a small GGUF and extends it to size bytes so
// aggregate file-size accounting can be exercised without large files.
func writeSparseTestGGUF(t *testing.T, kv gguftest.KV, tensor string, size int64) string {
	t.Helper()
	path, _ := writeTestGGUF(t, kv, []*gguftest.Tensor{
		testGGUFTensor(tensor, gguf.TensorTypeF32, []uint64{4}),
	})
	if err := os.Truncate(path, size); err != nil {
		t.Fatal(err)
	}
	return path
}

// TestMemorySizeSplitGGUFDeepSeek41 replays the buffer lines llama-server
// printed when loading the 7-shard DeepSeek-V4.1-Flash-1017 UD-Q2_K_XL GGUF
// (334,793,544,895 bytes) with its F16 mmproj on an M3 Ultra. Before summing
// per-file buffers, /api/ps reported 1.4 GiB for this ~313 GiB load.
func TestMemorySizeSplitGGUFDeepSeek41(t *testing.T) {
	const primarySize = 5_780_927
	const restSize = 334_793_544_895 - primarySize

	primary := writeSparseTestGGUF(t, gguftest.KV{
		"general.architecture": "deepseek41",
		"split.no":             uint16(0),
		"split.count":          uint16(2),
	}, "blk.0.attn_q.weight", primarySize)
	shard := writeSparseTestGGUF(t, gguftest.KV{
		"general.architecture": "deepseek41",
		"split.no":             uint16(1),
		"split.count":          uint16(2),
	}, "blk.1.attn_q.weight", restSize)
	model, err := LoadModel(primary, 0, shard)
	if err != nil {
		t.Fatal(err)
	}
	if got := model.FileSize(); got != 334_793_544_895 {
		t.Fatalf("FileSize() = %d, want all shards", got)
	}

	runner := &llamaServerRunner{
		vramByDevice: make(map[string]uint64),
		modelPath:    primary,
		metadata:     model,
	}
	w := &memoryParsingWriter{inner: io.Discard, runner: runner}

	f, err := os.Open("testdata/deepseek41-split-load.log")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		if _, err := w.Write(append(sc.Bytes(), '\n')); err != nil {
			t.Fatal(err)
		}
	}
	if err := sc.Err(); err != nil {
		t.Fatal(err)
	}

	const mib = 1 << 20
	// MTL0 model 189697.37 + KV 75 + compute 873.06 + 97.45 + mmproj 926.
	vramMiB, totalMiB := 191669.19, 321354.21
	wantVRAM := uint64(vramMiB * mib)
	// + CPU_Mapped 670.70 + 128910.09 (Engram) + output 0.49 + compute 98.54 + 5.20.
	wantTotal := uint64(totalMiB * mib)

	total, vram := runner.MemorySize()
	within := func(got, want uint64) bool {
		d := int64(got) - int64(want)
		return d > -mib && d < mib
	}
	if !within(vram, wantVRAM) {
		t.Errorf("vram = %.2f MiB, want %.2f MiB", float64(vram)/mib, float64(wantVRAM)/mib)
	}
	if !within(total, wantTotal) {
		t.Errorf("total = %.2f MiB, want %.2f MiB", float64(total)/mib, float64(wantTotal)/mib)
	}
}

func TestMemoryParsingSumsConsecutiveBuffers(t *testing.T) {
	runner := &llamaServerRunner{vramByDevice: make(map[string]uint64)}
	w := &memoryParsingWriter{inner: io.Discard, runner: runner}
	for _, line := range []string{
		// fit probe
		"load_tensors:  MTL0_Mapped model buffer size =   100.00 MiB\n",
		"llama_kv_cache:       MTL0 KV buffer size =    10.00 MiB\n",
		// real load: one buffer per shard, two KV caches
		"load_tensors:  MTL0_Mapped model buffer size =  1000.00 MiB\n",
		"load_tensors:  MTL0_Mapped model buffer size =     0.00 MiB\n",
		"load_tensors:  MTL0_Mapped model buffer size =  2000.00 MiB\n",
		"llama_kv_cache:       MTL0 KV buffer size =    50.00 MiB\n",
		"llama_kv_cache:       MTL0 KV buffer size =    15.00 MiB\n",
		"sched_reserve:       MTL0 compute buffer size =   20.00 MiB\n",
	} {
		w.Write([]byte(line))
	}
	const mib = 1 << 20
	if want := uint64((3000 + 65 + 20) * mib); runner.memGPU != want {
		t.Fatalf("memGPU = %d MiB, want %d MiB", runner.memGPU/mib, want/mib)
	}
}

func TestMemorySizeFullOffloadKeepsLargeHostWeights(t *testing.T) {
	const mib = 1 << 20
	tests := []struct {
		name      string
		lines     []string
		wantTotal uint64
	}{
		{
			name: "small host embeddings stay masked",
			lines: []string{
				"load_tensors: offloaded 33/33 layers to GPU\n",
				"load_tensors:   CPU_Mapped model buffer size =   300.00 MiB\n",
				"load_tensors:  MTL0_Mapped model buffer size =  8000.00 MiB\n",
			},
			wantTotal: 8000 * mib,
		},
		{
			name: "large host weights are counted",
			lines: []string{
				"load_tensors: offloaded 41/41 layers to GPU\n",
				"load_tensors:   CPU_Mapped model buffer size =   600.00 MiB\n",
				"load_tensors:  MTL0_Mapped model buffer size =  9000.00 MiB\n",
				"load_tensors:   CPU_Mapped model buffer size =  6000.00 MiB\n",
			},
			wantTotal: 15600 * mib,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			runner := &llamaServerRunner{vramByDevice: make(map[string]uint64)}
			w := &memoryParsingWriter{inner: io.Discard, runner: runner}
			for _, line := range tt.lines {
				w.Write([]byte(line))
			}
			if total, _ := runner.MemorySize(); total != tt.wantTotal {
				t.Fatalf("total = %d MiB, want %d MiB", total/mib, tt.wantTotal/mib)
			}
		})
	}
}

func TestPredictServerVRAMSplitWithProjector(t *testing.T) {
	primary := writeSparseTestGGUF(t, gguftest.KV{"general.architecture": "llama"}, "blk.0.attn_q.weight", 1<<20)
	shard := writeSparseTestGGUF(t, gguftest.KV{"general.architecture": "llama"}, "blk.1.attn_q.weight", 3<<20)
	model, err := LoadModel(primary, 0, shard)
	if err != nil {
		t.Fatal(err)
	}
	projector, _ := writeTestGGUF(t, gguftest.KV{"general.architecture": "clip"}, []*gguftest.Tensor{
		testGGUFTensor("v.patch_embd.weight", gguf.TensorTypeF16, []uint64{1024}),
	})

	base := PredictServerVRAM(primary, model, 0)
	if base != 4<<20 {
		t.Fatalf("weights = %d, want sum of shards %d", base, 4<<20)
	}
	if got := PredictServerVRAM(primary, model, 0, projector); got != base+2048 {
		t.Fatalf("with projector = %d, want %d", got, base+2048)
	}
	if got := PredictServerVRAM(primary, model, 0, primary); got != base {
		t.Fatalf("inline projector = %d, want %d (already in file size)", got, base)
	}
}
