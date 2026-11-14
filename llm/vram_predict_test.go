package llm

import (
	"math"
	"slices"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/fs/gguf"
	gguftest "github.com/ollama/ollama/internal/testutil/gguf"
	"github.com/ollama/ollama/ml"
)

// Metadata below is copied from the real GGUFs (qwen3.6-27B "qwen35",
// glimmer, gemma4 26B-A4B). Expected sizes are llama-server's logged buffers
// for those models (build b11963, q8_0 KV, flash attention on, CUDA).

const mib = 1 << 20

func repeatPattern[T any](pattern []T, n int) []T {
	out := make([]T, 0, n)
	for len(out) < n {
		out = append(out, pattern...)
	}
	return out[:n]
}

func qwen35KV() gguftest.KV {
	return gguftest.KV{
		"general.architecture":    "qwen35",
		"block_count":             uint32(65),
		"embedding_length":        uint32(5120),
		"attention.head_count":    uint32(24),
		"attention.head_count_kv": uint32(4),
		"attention.key_length":    uint32(256),
		"attention.value_length":  uint32(256),
		"full_attention_interval": uint32(4),
		"ssm.inner_size":          uint32(6144),
		"ssm.state_size":          uint32(128),
		"ssm.group_count":         uint32(16),
		"ssm.conv_kernel":         uint32(4),
		"nextn_predict_layers":    uint32(1),
	}
}

func glimmerKV() gguftest.KV {
	return gguftest.KV{
		"general.architecture":             "muse-glimmer",
		"block_count":                      uint32(52),
		"embedding_length":                 uint32(6656),
		"attention.head_count":             uint32(32),
		"attention.head_count_kv":          uint32(2),
		"attention.key_length":             uint32(128),
		"attention.value_length":           uint32(128),
		"attention.sliding_window":         uint32(2048),
		"attention.sliding_window_pattern": repeatPattern([]bool{true, true, true, false}, 52),
	}
}

func gemma4KV() gguftest.KV {
	return gguftest.KV{
		"general.architecture":             "gemma4",
		"block_count":                      uint32(30),
		"embedding_length":                 uint32(2816),
		"attention.head_count":             uint32(16),
		"attention.head_count_kv":          repeatPattern([]uint32{8, 8, 8, 8, 8, 2}, 30),
		"attention.key_length":             uint32(512),
		"attention.value_length":           uint32(512),
		"attention.key_length_swa":         uint32(256),
		"attention.value_length_swa":       uint32(256),
		"attention.sliding_window":         uint32(1024),
		"attention.sliding_window_pattern": repeatPattern([]bool{true, true, true, true, true, false}, 30),
	}
}

func testLayout(t *testing.T, kv gguftest.KV) vramLayout {
	t.Helper()
	path, f := writeTestGGUF(t, kv, nil)
	return newVRAMLayout(path, f, nil)
}

func approxMiB(t *testing.T, what string, got uint64, wantMiB float64) {
	t.Helper()
	if d := math.Abs(float64(got)/mib - wantMiB); d > 0.01 {
		t.Errorf("%s = %.2f MiB, want %.2f MiB (logged)", what, float64(got)/mib, wantMiB)
	}
}

// TestNewVRAMLayoutFromMetadata parses real model metadata and checks KV and
// recurrent-state sizes against llama-server's logged buffers.
func TestNewVRAMLayoutFromMetadata(t *testing.T) {
	t.Run("qwen35", func(t *testing.T) {
		l := testLayout(t, qwen35KV())
		var attn int
		for _, layer := range l.layers {
			if layer.kvElementsPerToken > 0 {
				attn++
			}
		}
		if len(l.layers) != 64 || attn != 16 || len(l.nextnLayers) != 1 {
			t.Fatalf("layers=%d attention=%d nextn=%d, want 64/16/1", len(l.layers), attn, len(l.nextnLayers))
		}

		// 64k with MTP draft 4: "KV 2176.00 MiB", draft "KV 256.00 MiB",
		// "RS 748.12 MiB (... 4 rs_seq)".
		b := l.breakdown(vramPredictParams{numCtx: 65536, numParallel: 1, draftN: 4, cacheType: "q8_0", flashAttn: true})
		approxMiB(t, "kv", b.kv, 2176)
		approxMiB(t, "draft kv", b.draftKV, 256)
		approxMiB(t, "rs", b.rs, 748.125)

		// 262k without a draft: "RS 77.93 + 71.70 MiB (0 rs_seq)", KV 4352+4352.
		b = l.breakdown(vramPredictParams{numCtx: 262144, numParallel: 1, cacheType: "q8_0", flashAttn: true})
		approxMiB(t, "kv", b.kv, 8704)
		approxMiB(t, "rs", b.rs, 149.625)
		if b.draftKV != 0 || b.draftCompute != 0 {
			t.Errorf("draft counted without draft: kv=%d compute=%d", b.draftKV, b.draftCompute)
		}
	})

	t.Run("glimmer", func(t *testing.T) {
		l := testLayout(t, glimmerKV())
		// 16k ubatch 512: non-SWA 110.50 MiB, SWA 2560 cells 51.80 MiB.
		b := l.breakdown(vramPredictParams{numCtx: 16384, numParallel: 1, cacheType: "q8_0", flashAttn: true})
		approxMiB(t, "kv", b.kv, 110.5+51.796875)
		// 128k ubatch 2048: 884.00 MiB + SWA 4096 cells 82.88 MiB.
		b = l.breakdown(vramPredictParams{numCtx: 131072, numParallel: 1, numUBatch: 2048, cacheType: "q8_0", flashAttn: true})
		approxMiB(t, "kv", b.kv, 884+82.875)
	})

	t.Run("gemma4", func(t *testing.T) {
		l := testLayout(t, gemma4KV())
		var swa int
		for _, layer := range l.layers {
			if layer.window > 0 {
				swa++
			}
		}
		if swa != 25 || len(l.layers) != 30 {
			t.Fatalf("swa=%d layers=%d, want 25/30", swa, len(l.layers))
		}
		// 32k ubatch 1024: non-SWA 340.00 MiB, SWA 2048 cells 212.50 MiB.
		b := l.breakdown(vramPredictParams{numCtx: 32768, numParallel: 1, numUBatch: 1024, cacheType: "q8_0", flashAttn: true})
		approxMiB(t, "kv", b.kv, 340+212.5)
	})
}

// TestComputeBytesBoundsLoggedBuffers checks the compute model against every
// logged sched_reserve buffer for these models: it must never be below the
// log, and no more than one extra base allowance above it.
func TestComputeBytesBoundsLoggedBuffers(t *testing.T) {
	layouts := map[string]vramLayout{
		"qwen35":  testLayout(t, qwen35KV()),
		"glimmer": testLayout(t, glimmerKV()),
		"gemma4":  testLayout(t, gemma4KV()),
	}

	tests := []struct {
		model     string
		ctx, ub   int
		devices   int
		draft     bool    // check the MTP draft context's buffer
		loggedMiB float64 // largest per-device compute buffer
	}{
		{"qwen35", 2048, 512, 1, false, 144.29},
		{"qwen35", 16384, 512, 1, false, 180.29},
		{"qwen35", 32768, 512, 1, false, 260.29},
		{"qwen35", 65536, 512, 1, false, 420.29},
		{"qwen35", 98304, 512, 1, false, 580.29},
		{"qwen35", 114688, 512, 1, false, 660.29},
		{"qwen35", 16384, 512, 1, true, 148.02},
		{"qwen35", 114688, 512, 1, true, 180.02},
		{"qwen35", 98304, 512, 2, false, 979.14},
		{"qwen35", 114688, 512, 2, false, 1107.14},
		{"qwen35", 262144, 512, 2, false, 2259.14},
		{"qwen35", 16384, 1024, 2, false, 633.22},
		{"qwen35", 65536, 1024, 2, false, 1189.22},
		{"qwen35", 98304, 1024, 2, false, 1573.22},
		{"qwen35", 40960, 2048, 2, false, 1649.38},
		{"qwen35", 98304, 2048, 2, false, 2761.38},
		{"qwen35", 98304, 2048, 2, true, 2344.25},
		{"glimmer", 16384, 512, 1, false, 187.68},
		{"glimmer", 65536, 512, 1, false, 235.68},
		{"glimmer", 262144, 512, 1, false, 446.68},
		{"glimmer", 16384, 1024, 1, false, 376.2},
		{"glimmer", 131072, 512, 2, false, 861.71},
		{"glimmer", 131072, 2048, 2, false, 3360.97},
		{"gemma4", 32768, 1024, 1, false, 355.33},
		{"gemma4", 131072, 512, 1, false, 706.43},
		{"gemma4", 262144, 2048, 2, false, 5680.0},
	}
	for _, tt := range tests {
		l := layouts[tt.model]
		draftN := 0
		if tt.draft {
			draftN = 4
		}
		b := l.breakdown(vramPredictParams{numCtx: tt.ctx, numParallel: 1, numUBatch: tt.ub, draftN: draftN, cacheType: "q8_0", flashAttn: true, devices: tt.devices})
		got := float64(b.compute) / mib
		if tt.draft {
			got = float64(b.draftCompute) / mib
		}
		slack := 256 + float64(tt.ub)*predictComputeBytesPerUBatch/mib
		if got < tt.loggedMiB || got > tt.loggedMiB+slack {
			t.Errorf("%s ctx=%d ub=%d devices=%d draft=%v: predicted %.2f MiB, logged %.2f MiB (allowed +%.0f)",
				tt.model, tt.ctx, tt.ub, tt.devices, tt.draft, got, tt.loggedMiB, slack)
		}
	}
}

// TestPredictionBoundsMeasuredUse checks whole-model predictions against
// nvidia-smi per-process usage, with device weights taken from the logged
// model buffers and the fit target llama-server ran with.
func TestPredictionBoundsMeasuredUse(t *testing.T) {
	tests := []struct {
		name       string
		kv         gguftest.KV
		weightsMiB float64 // logged CUDA model buffers
		projMiB    float64 // mmprojMemoryRequirement
		ctx, ub    int
		draftN     int
		devices    int
		actualMiB  []float64 // nvidia-smi per GPU
	}{
		{"qwen35 64k", qwen35KV(), 15339.44, 888, 65536, 512, 4, 1, []float64{20514}},
		{"qwen35 98k", qwen35KV(), 15339.44, 888, 98304, 512, 4, 1, []float64{21922}},
		{"glimmer 16k", glimmerKV(), 15246.49, 1335.4, 16384, 512, 0, 1, []float64{17476}},
		{"gemma4 32k", gemma4KV(), 13755.42, 1139.5, 32768, 1024, 0, 1, []float64{16260}},
		{"qwen35 q8_0 262k split", qwen35KV(), 12730.5 + 13241.79, 888, 262144, 512, 0, 2, []float64{20872, 20146}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			l := testLayout(t, tt.kv)
			l.weights = uint64(tt.weightsMiB * mib)
			l.projector = uint64(tt.projMiB * mib)
			p := l.predict(vramPredictParams{
				numCtx: tt.ctx, numParallel: 1, numUBatch: tt.ub, draftN: tt.draftN,
				cacheType: "q8_0", flashAttn: true, devices: tt.devices,
				fitTarget: predictFitTarget(l.projector),
			})
			var actual float64
			for _, a := range tt.actualMiB {
				actual += a
			}
			total := float64(p.Total()) / mib
			// The fit margin is kept on every device, so allow 1.5 GiB each.
			if total < actual || total > actual+1536*float64(tt.devices) {
				t.Errorf("total %.0f MiB, measured %.0f MiB", total, actual)
			}
			share := float64(p.Share(1, uint64(tt.devices))) / mib
			if worst := slices.Max(tt.actualMiB); share < worst {
				t.Errorf("per-device share %.0f MiB below measured %.0f MiB", share, worst)
			}
		})
	}
}

func TestVRAMPredictionShare(t *testing.T) {
	p := VRAMPrediction{Shared: 30 * mib, PerDevice: 2 * mib, Devices: 2}
	if got := p.Total(); got != 34*mib {
		t.Errorf("Total = %d", got)
	}
	if got := p.Share(1, 3); got != 12*mib {
		t.Errorf("Share(1,3) = %d, want 12 MiB", got)
	}
	if got := p.Share(0, 0); got != p.Total() {
		t.Errorf("Share(0,0) = %d, want Total", got)
	}
	// Byte-sized fractions must not overflow.
	big := VRAMPrediction{Shared: 20 << 30, Devices: 2}
	if got := big.Share(6<<30, 26<<30); got != (20<<30)*6/26 {
		t.Errorf("Share overflowed: %d", got)
	}
}

func TestDeviceWeights(t *testing.T) {
	embd := testGGUFTensor("token_embd.weight", gguf.TensorTypeF32, []uint64{64, 32})
	out := testGGUFTensor("output.weight", gguf.TensorTypeF32, []uint64{64, 32})
	blk := testGGUFTensor("blk.0.attn_q.weight", gguf.TensorTypeF32, []uint64{64, 64})

	path, f := writeTestGGUF(t, qwen35KV(), []*gguftest.Tensor{embd, out, blk})
	size := modelFileSize(path, f)
	if got, want := deviceWeights(size, f), size-embd.Size(); got != want {
		t.Errorf("untied: device weights %d, want file %d - token_embd %d", got, size, embd.Size())
	}
	if got := hostResidentWeights(f); got != embd.Size() {
		t.Errorf("host weights %d, want %d", got, embd.Size())
	}

	// GPT-2/BERT-style input layers also keep position and token-type
	// embeddings on host.
	embd = testGGUFTensor("token_embd.weight", gguf.TensorTypeF32, []uint64{64, 32})
	out = testGGUFTensor("output.weight", gguf.TensorTypeF32, []uint64{64, 32})
	pos := testGGUFTensor("position_embd.weight", gguf.TensorTypeF32, []uint64{64, 128})
	types := testGGUFTensor("token_types.weight", gguf.TensorTypeF32, []uint64{64, 2})
	blk = testGGUFTensor("blk.0.attn_q.weight", gguf.TensorTypeF32, []uint64{64, 64})
	_, f = writeTestGGUF(t, qwen35KV(), []*gguftest.Tensor{embd, out, pos, types, blk})
	if got, want := hostResidentWeights(f), embd.Size()+pos.Size()+types.Size(); got != want {
		t.Errorf("input layer host weights %d, want %d", got, want)
	}

	// Tied output: llama.cpp keeps a device copy of token_embd.
	embd = testGGUFTensor("token_embd.weight", gguf.TensorTypeF32, []uint64{64, 32})
	blk = testGGUFTensor("blk.0.attn_q.weight", gguf.TensorTypeF32, []uint64{64, 64})
	path, f = writeTestGGUF(t, gemma4KV(), []*gguftest.Tensor{embd, blk})
	if got, want := deviceWeights(modelFileSize(path, f), f), modelFileSize(path, f); got != want {
		t.Errorf("tied: device weights %d, want whole file %d", got, want)
	}
}

func TestPredictKVCacheType(t *testing.T) {
	t.Setenv("OLLAMA_KV_CACHE_TYPE", "q8_0")
	for fa, want := range map[ml.FlashAttentionType]string{
		ml.FlashAttentionDisabled: "f16",
		ml.FlashAttentionEnabled:  "q8_0",
		ml.FlashAttentionAuto:     "q8_0",
	} {
		if got := predictKVCacheType(fa); got != want {
			t.Errorf("predictKVCacheType(%v) = %q, want %q", fa, got, want)
		}
	}

	// FA off also switches compute to the f32 KQ matrix and drops the
	// dequantized K/V copy.
	l := testLayout(t, qwen35KV())
	on := l.breakdown(vramPredictParams{numCtx: 32768, cacheType: "q8_0", flashAttn: true})
	off := l.breakdown(vramPredictParams{numCtx: 32768, cacheType: "f16", flashAttn: false})
	if off.kv != 2*on.kv*32/34 || off.compute <= on.compute {
		t.Errorf("FA off: kv %d (q8 %d), compute %d (FA %d)", off.kv, on.kv, off.compute, on.compute)
	}
}

func TestPredictFitTarget(t *testing.T) {
	if got := predictFitTarget(0); got != 1024*mib {
		t.Errorf("default = %d", got)
	}
	if got := predictFitTarget(888 * mib); got != (1024+888)*mib {
		t.Errorf("with projector = %d", got)
	}
	t.Setenv("LLAMA_ARG_FIT_TARGET", "2048,512")
	if got := predictFitTarget(0); got != 2048*mib {
		t.Errorf("env = %d", got)
	}
}

// TestPredictDraftMatchesLauncher: the MTP draft is predicted exactly when the
// launcher passes --spec-type mtp.
func TestPredictDraftMatchesLauncher(t *testing.T) {
	path, f := writeTestGGUF(t, qwen35KV(), nil)
	for _, tt := range []struct {
		draftN    int
		draftPath string
	}{{0, ""}, {4, ""}, {4, "/external/draft.gguf"}} {
		opts := api.DefaultOptions()
		opts.DraftNumPredict = tt.draftN
		draftType := ""
		if tt.draftPath == "" && hasMTPDraft(f) {
			draftType = draftTypeMTP
		}
		args := appendDraftArgs(nil, draftType, tt.draftPath, opts)
		launches := slices.Contains(args, draftTypeMTP)
		if got := mtpDraftActive(f, tt.draftPath, opts); got != launches {
			t.Errorf("draftN=%d draft=%q: predicted MTP %v, launcher %v (%v)", tt.draftN, tt.draftPath, got, launches, args)
		}
		pred := PredictServerVRAM(path, f, nil, opts, VRAMPredictConfig{NumCtx: 4096, NumParallel: 1, FlashAttention: ml.FlashAttentionEnabled})
		if pred.Total() == 0 {
			t.Errorf("zero prediction")
		}
	}
}

func TestKVCacheBytesPerElement(t *testing.T) {
	for cacheType, want := range map[string]float64{
		"":     2,
		"f16":  2,
		"Q8_0": 1.0625,
		"q4_0": 0.5625,
	} {
		if got := kvCacheBytesPerElement(cacheType); got != want {
			t.Errorf("kvCacheBytesPerElement(%q) = %v, want %v", cacheType, got, want)
		}
	}
}
