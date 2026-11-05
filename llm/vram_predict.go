package llm

import (
	"math/bits"
	"os"
	"strconv"
	"strings"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/fs/gguf"
	"github.com/ollama/ollama/ml"
)

// The constants below are fitted to llama-server's logged buffers on CUDA
// (b11963). See the comments on each for the data they were checked against.
const (
	// llama.cpp's n_ubatch when -ub is not passed.
	predictDefaultUBatch = 512
	// SWA cache size is padded to this many cells.
	predictSWAPad = 256
	// Compute buffer bytes per ubatch token not tied to the KV length.
	// Observed 0.16–0.32 MiB/token (qwen35, glimmer, gemma4).
	predictComputeBytesPerUBatch = 384 << 10
	// llama.cpp's default --fit-target: free memory kept on every device.
	predictFitTargetDefault = 1024 << 20
	// Pipeline-parallel graph copies llama.cpp uses for multi-GPU splits
	// (logged "sched copies = 4").
	predictPipelineCopies = 4
)

// kvCacheBytesPerElement: quantized types use 32-element blocks plus an f16
// scale, so q8_0 is 34/32 bytes.
func kvCacheBytesPerElement(cacheType string) float64 {
	switch strings.ToLower(cacheType) {
	case "f32":
		return 4
	case "q8_0":
		return 34.0 / 32
	case "q5_1":
		return 24.0 / 32
	case "q5_0":
		return 22.0 / 32
	case "q4_1":
		return 20.0 / 32
	case "q4_0":
		return 18.0 / 32
	case "iq4_nl":
		return 18.0 / 32
	default: // f16, bf16, unset
		return 2
	}
}

// predictKVCacheType returns the KV type llama-server will allocate. Without
// flash attention a quantized cache is not used, so predict f16.
func predictKVCacheType(fa ml.FlashAttentionType) string {
	if fa == ml.FlashAttentionDisabled {
		return "f16"
	}
	return strings.ToLower(envconfig.KvCacheType())
}

func isQuantizedKV(cacheType string) bool {
	return kvCacheBytesPerElement(cacheType) < 2
}

type vramLayer struct {
	kvElementsPerToken uint64 // kv_heads*(key_len+value_len); 0 for recurrent layers
	window             uint64 // sliding window; 0 for full attention
}

// vramLayout is the metadata that sizes llama-server's VRAM, split from GGUF
// parsing for testing.
type vramLayout struct {
	weights             uint64 // device-resident weights
	projector           uint64
	headCount           uint64
	layers              []vramLayer
	nextnLayers         []vramLayer // MTP draft layers; own f16 KV when draft is on
	recurrentStateBytes uint64      // f32 conv+SSM state, all layers, one slot
}

type vramPredictParams struct {
	numCtx      int // total context across all parallel sequences
	numParallel int
	numUBatch   int // physical batch (-ub); <= 0 means llama.cpp's default
	draftN      int // speculative draft tokens; 0 disables the MTP draft
	cacheType   string
	flashAttn   bool
	devices     int    // GPUs the model is split across
	fitTarget   uint64 // per-device --fit-target, including any mmproj pad
}

// VRAMPrediction is llama-server's predicted device memory, excluding the
// user's OLLAMA_GPU_OVERHEAD. A measured prediction (PredictLlamaServer)
// has the requirement of each device in ByDevice. An estimate has Shared,
// split across devices, and PerDevice, needed on every device.
type VRAMPrediction struct {
	Shared    uint64 // weights, KV cache, recurrent state
	PerDevice uint64 // compute buffers + --fit-target margin (covers mmproj and CUDA context)
	Devices   int

	ByDevice []uint64 // measured: required on each device, in placement order
	Measured bool
	// MinByDevice is a hard lower bound per device (measured weights, KV
	// and recurrent state, draft KV, projector weights): a placement with
	// less can't hold every layer.
	MinByDevice []uint64
}

// MinTotal is the sum of MinByDevice.
func (p VRAMPrediction) MinTotal() uint64 {
	var total uint64
	for _, b := range p.MinByDevice {
		total += b
	}
	return total
}

// Total is the memory required across all devices.
func (p VRAMPrediction) Total() uint64 {
	if p.ByDevice != nil {
		var total uint64
		for _, b := range p.ByDevice {
			total += b
		}
		return total
	}
	return p.Shared + p.PerDevice*uint64(max(p.Devices, 1))
}

// Device is the memory required on device i of a placement whose devices
// have the given free memory. An estimate splits Shared in proportion to
// free memory, as llama.cpp does by default.
func (p VRAMPrediction) Device(i int, free []uint64) uint64 {
	if p.ByDevice != nil {
		if i < len(p.ByDevice) {
			return p.ByDevice[i]
		}
		return p.Total()
	}
	var total uint64
	for _, f := range free {
		total += f
	}
	if i >= len(free) {
		return p.Total()
	}
	return p.Share(free[i], total)
}

// Share is the memory required on a device holding fraction num/den of the
// shared part.
func (p VRAMPrediction) Share(num, den uint64) uint64 {
	if den == 0 || num > den {
		return p.Total()
	}
	// 128-bit intermediate: byte counts times byte counts overflow uint64.
	hi, lo := bits.Mul64(p.Shared, num)
	q, _ := bits.Div64(hi, lo, den)
	return q + p.PerDevice
}

// vramBreakdown is predict's result by component, for tests and logging.
type vramBreakdown struct {
	weights, kv, draftKV, rs, compute, draftCompute uint64
}

func (l vramLayout) predict(p vramPredictParams) VRAMPrediction {
	b := l.breakdown(p)
	return VRAMPrediction{
		Shared:    b.weights + b.kv + b.draftKV + b.rs,
		PerDevice: b.compute + b.draftCompute + p.fitTarget,
		Devices:   max(p.devices, 1),
	}
}

func (l vramLayout) breakdown(p vramPredictParams) vramBreakdown {
	ctx := uint64(max(p.numCtx, 0))
	parallel := uint64(max(p.numParallel, 1))
	devices := uint64(max(p.devices, 1))
	bpe := kvCacheBytesPerElement(p.cacheType)
	ubatch := uint64(p.numUBatch)
	if p.numUBatch <= 0 {
		ubatch = predictDefaultUBatch
	}
	if ctx > 0 {
		ubatch = min(ubatch, ctx)
	}

	// SWA cells = pad(window*parallel + n_ubatch), capped at ctx; matches
	// logged 2560 (2048+512), 4096 (2048+2048), 2048 (1024+1024).
	cells := func(layer vramLayer) uint64 {
		if layer.window == 0 {
			return ctx
		}
		n := layer.window*parallel + ubatch
		n = (n + predictSWAPad - 1) / predictSWAPad * predictSWAPad
		return min(ctx, n)
	}

	var kv, draftKV float64
	var maxKV uint64
	for _, layer := range l.layers {
		kv += float64(layer.kvElementsPerToken*cells(layer)) * bpe
		if layer.window == 0 {
			maxKV = max(maxKV, layer.kvElementsPerToken)
		}
	}
	if maxKV == 0 { // all-SWA model
		for _, layer := range l.layers {
			maxKV = max(maxKV, layer.kvElementsPerToken)
		}
	}

	draftActive := p.draftN > 0 && len(l.nextnLayers) > 0
	if draftActive {
		for _, layer := range l.nextnLayers {
			draftKV += float64(layer.kvElementsPerToken*cells(layer)) * kvCacheBytesPerElement("f16")
		}
	}

	// One RS slot per sequence, plus one per draft token for rollback.
	rsSlots := parallel
	if draftActive {
		rsSlots *= uint64(1 + p.draftN)
	}

	copies := uint64(1)
	if devices > 1 {
		copies = predictPipelineCopies
	}
	b := vramBreakdown{
		weights: l.weights,
		kv:      uint64(kv),
		draftKV: uint64(draftKV),
		rs:      l.recurrentStateBytes * rsSlots,
		compute: computeBytes(ctx, ubatch, copies, maxKV, l.headCount, isQuantizedKV(p.cacheType), p.flashAttn),
	}
	if draftActive {
		// The draft context's KV is f16, so no dequantized copy.
		b.draftCompute = computeBytes(ctx, ubatch, copies, 0, l.headCount, false, p.flashAttn)
	}
	return b
}

// computeBytes is one llama context's compute buffer on one device:
//
//	base:      ubatch * predictComputeBytesPerUBatch, doubled for pipeline copies
//	per cell:  copies * 2*ubatch   (f16 KQ mask, one per pipeline copy)
//	         + 2*maxKV             (f16 copy of one layer's quantized K/V)
//
// The per-cell term matches qwen35's logged slope (+160 MiB from 64k to 98k);
// the base over-estimates, so all 26 logged buffers are bounded with 36-976
// MiB to spare. Without flash attention the f32 KQ matrix (n_head per cell
// per token) dominates.
func computeBytes(ctx, ubatch, copies, maxKV, headCount uint64, quantKV, fa bool) uint64 {
	base := ubatch * predictComputeBytesPerUBatch * max(copies/2, 1)
	if !fa {
		return base + ctx*ubatch*4*(max(headCount, 1)+1)
	}
	perCell := copies * 2 * ubatch
	if quantKV {
		perCell += 2 * maxKV
	}
	return base + ctx*perCell
}

// layerValues reads a per-layer key stored as a scalar or an array.
func layerValues(kv *gguf.Metadata, key string, n int, fallback uint64) []uint64 {
	out := make([]uint64, n)
	value := kv.KeyValue(key)
	values := value.Uints()
	if len(values) == 0 {
		for _, v := range value.Ints() {
			values = append(values, uint64(max(v, 0)))
		}
	}
	scalar, scalarOK := kv.UintOK(key)
	for i := range out {
		switch {
		case i < len(values):
			out[i] = values[i]
		case scalarOK && len(values) == 0:
			out[i] = scalar
		default:
			out[i] = fallback
		}
	}
	return out
}

// deviceWeights is the model file minus tensors llama.cpp keeps in host
// memory: the input token embeddings, unless the output projection is tied
// to them (then a device copy exists anyway). Matches the logged CPU_Mapped
// buffer for qwen35, glimmer and gemma4 exactly.
func deviceWeights(fileSize uint64, f *gguf.Model) uint64 {
	tensors := f.Tensors()
	if len(tensors.Items("output.weight")) == 0 {
		return fileSize
	}
	host := hostResidentWeights(f)
	if host >= fileSize {
		return fileSize
	}
	return fileSize - host
}

// hostResidentWeights is the weight bytes llama.cpp keeps in host memory
// even with full offload: its input-layer tensors (token, per-layer,
// position and token-type embeddings).
func hostResidentWeights(f *gguf.Model) uint64 {
	return f.Tensors().Size("token_embd.weight", "per_layer_token_embd.weight",
		"position_embd.weight", "token_types.weight", "token_embd_norm.weight", "token_embd_norm.bias")
}

func newVRAMLayout(modelPath string, f *gguf.Model, projectors []string) vramLayout {
	kv := f.KV()
	n := int(kv.BlockCount())
	nextn := min(int(kv.Uint("nextn_predict_layers")), n)

	headDim := uint64(0)
	if heads := kv.HeadCountMax(); heads > 0 {
		headDim = kv.EmbeddingLength() / heads
	}
	keyLen := kv.Uint("attention.key_length", headDim)
	valueLen := kv.Uint("attention.value_length", headDim)
	keyLenSWA := kv.Uint("attention.key_length_swa", keyLen)
	valueLenSWA := kv.Uint("attention.value_length_swa", valueLen)
	kvHeads := layerValues(kv, "attention.head_count_kv", n, kv.HeadCountKVMin())

	window := kv.Uint("attention.sliding_window")
	swaPattern := kv.KeyValue("attention.sliding_window_pattern").Bools()

	// Hybrid models (qwen35, qwen3next): attention every Nth layer.
	interval := kv.Uint("full_attention_interval")
	hasSSM := kv.Has("ssm.state_size")
	recurrent := func(i int) bool {
		if interval > 0 && hasSSM {
			return uint64(i+1)%interval != 0
		}
		return false
	}

	layout := vramLayout{
		weights:   deviceWeights(modelFileSize(modelPath, f), f),
		headCount: kv.HeadCountMax(),
	}
	recurrentLayers := 0
	for i := range n {
		var layer vramLayer
		if i < n-nextn && recurrent(i) {
			recurrentLayers++
		} else {
			k, v := keyLen, valueLen
			if window > 0 && i < len(swaPattern) && swaPattern[i] {
				layer.window = window
				k, v = keyLenSWA, valueLenSWA
			}
			layer.kvElementsPerToken = kvHeads[i] * (k + v)
		}
		if i >= n-nextn {
			layout.nextnLayers = append(layout.nextnLayers, layer)
		} else {
			layout.layers = append(layout.layers, layer)
		}
	}

	if recurrentLayers > 0 {
		inner := kv.Uint("ssm.inner_size")
		state := kv.Uint("ssm.state_size")
		groups := kv.Uint("ssm.group_count", 1)
		kernel := kv.Uint("ssm.conv_kernel")
		convDim := inner + 2*groups*state
		perLayer := 4 * (inner*state + max(kernel, 1)*convDim - convDim)
		layout.recurrentStateBytes = perLayer * uint64(recurrentLayers)
	}

	if len(projectors) > 0 {
		if size, err := mmprojMemoryRequirement(modelPath, f, projectors); err == nil {
			layout.projector = size
		} else if info, err := os.Stat(projectors[0]); err == nil && projectors[0] != modelPath {
			layout.projector = uint64(info.Size())
		}
	}

	return layout
}

// predictFitTarget mirrors the --fit-target llama-server runs with: the
// user's LLAMA_ARG_FIT_TARGET (MiB) or llama.cpp's 1024 MiB default, plus the
// projector pad extraEnvsForStart adds.
func predictFitTarget(projector uint64) uint64 {
	target := uint64(predictFitTargetDefault)
	if v, ok := os.LookupEnv(llamaArgFitTargetEnv); ok {
		first, _, _ := strings.Cut(v, ",")
		if mib, err := strconv.ParseUint(strings.TrimSpace(first), 10, 64); err == nil {
			target = mib << 20
		}
	}
	if projector > 0 {
		target += projector
	}
	return target
}

// mtpDraftActive mirrors the launcher: the built-in MTP head is used only
// without an external draft model and with draft_num_predict > 0.
func mtpDraftActive(f *gguf.Model, draftModelPath string, opts api.Options) bool {
	return draftModelPath == "" && opts.DraftNumPredict > 0 && hasMTPDraft(f)
}

// VRAMPredictConfig describes the launch being predicted.
type VRAMPredictConfig struct {
	NumCtx         int // total context across NumParallel sequences
	NumParallel    int
	Devices        int // GPUs the model is split across; 0 or 1 = single GPU
	FlashAttention ml.FlashAttentionType
	DraftModelPath string
}

// PredictServerVRAM estimates llama-server device memory without spawning
// it. opts.NumBatch is used as n_ubatch.
func PredictServerVRAM(modelPath string, f *gguf.Model, projectors []string, opts api.Options, cfg VRAMPredictConfig) VRAMPrediction {
	if f == nil {
		return VRAMPrediction{}
	}
	cacheType := predictKVCacheType(cfg.FlashAttention)
	fa := cfg.FlashAttention != ml.FlashAttentionDisabled
	layout := newVRAMLayout(modelPath, f, projectors)

	draftN := 0
	if mtpDraftActive(f, cfg.DraftModelPath, opts) {
		draftN = opts.DraftNumPredict
	}
	p := vramPredictParams{
		numCtx:      cfg.NumCtx,
		numParallel: cfg.NumParallel,
		numUBatch:   opts.NumBatch,
		draftN:      draftN,
		cacheType:   cacheType,
		flashAttn:   fa,
		devices:     cfg.Devices,
		fitTarget:   predictFitTarget(layout.projector),
	}
	pred := layout.predict(p)

	// An external draft model gets its own context with an f16 KV cache.
	if cfg.DraftModelPath != "" && opts.DraftNumPredict > 0 {
		if df, err := LoadModel(cfg.DraftModelPath, 1024); err == nil {
			dl := newVRAMLayout(cfg.DraftModelPath, df, nil)
			dp := p
			dp.cacheType, dp.draftN, dp.fitTarget = "f16", 0, 0
			d := dl.predict(dp)
			pred.Shared += d.Shared
			pred.PerDevice += d.PerDevice
		}
	}
	return pred
}
