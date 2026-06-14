package diffusiongemma

import (
	"context"
	"math/rand"

	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model/base"
)

// Compile-time check that the model is a diffusion model the runner can drive.
var _ base.DiffusionModel = (*Model)(nil)

// Reference defaults (llama.cpp PR #24427, diffusion-gemma-cli.cpp) applied when
// the checkpoint's generation_config omits a value. The reference hardcodes these
// at runtime; we prefer the model's parsed values and fall back to these.
const (
	defCanvasLength = 256
	defMaxSteps     = 48
	// Self-conditioning gather width. The reference uses the full softmax; we
	// approximate with a top-k probability-weighted embedding. High-probability
	// mass dominates, so 32 keeps converging positions intact while bounding the
	// per-step embedding gather.
	defSelfCondK       = 32
	defStabilityThresh = 1
	defSeed            = 1234
)

const (
	defTMin       float32 = 0.4
	defTMax       float32 = 0.8
	defEntropy    float32 = 0.1
	defConfidence float32 = 0.005
)

// ResolveDiffuse derives the runtime block-diffusion config from the model's
// parsed defaults, the requested predict length (number of canvases), an optional
// step override, and a seed.
func (m *Model) ResolveDiffuse(nPredict, steps int, seed int64) base.DiffuseConfig {
	canvas := int(m.Diffusion.CanvasLength)
	if canvas <= 0 {
		canvas = defCanvasLength
	}
	st := steps
	if st <= 0 {
		st = int(m.Diffusion.MaxDenoisingSteps)
	}
	if st <= 0 {
		st = defMaxSteps
	}
	tMin := m.Diffusion.TMin
	if tMin <= 0 {
		tMin = defTMin
	}
	tMax := m.Diffusion.TMax
	if tMax <= 0 {
		tMax = defTMax
	}
	eb := m.Diffusion.EntropyBound
	if eb <= 0 {
		eb = defEntropy
	}
	conf := m.Diffusion.ConfidenceThreshold
	if conf <= 0 {
		conf = defConfidence
	}
	stab := int(m.Diffusion.StabilityThreshold)
	if stab <= 0 {
		stab = defStabilityThresh
	}
	if seed == 0 {
		seed = defSeed
	}
	return base.DiffuseConfig{
		Canvas:              canvas,
		Steps:               st,
		PredictTokens:       nPredict,
		MaxCanvases:         numCanvases(nPredict, canvas),
		SelfCondK:           defSelfCondK,
		StabilityThreshold:  stab,
		TMin:                tMin,
		TMax:                tMax,
		EntropyBound:        eb,
		ConfidenceThreshold: conf,
		Seed:                seed,
	}
}

// Diffuse runs the block-diffusion denoising loop: prefill the prompt (causal),
// then for each canvas iterate the bidirectional self-conditioned denoiser to
// convergence, emit the canvas's greedy (argmax) tokens, and commit the canvas as
// a causal prefix. emit is called for each generated token id in order.
func (m *Model) Diffuse(ctx context.Context, prompt []int32, cfg base.DiffuseConfig, emit func(token int32) error) error {
	caches := m.NewCaches()
	vocab := int(m.VocabSize)
	rng := rand.New(rand.NewSource(cfg.Seed))
	var keyCounter uint64 // distinct MLX RNG key per denoising step

	// Causal prefill of the prompt — the read-only prefix every canvas attends to.
	m.prefillPrompt(prompt, caches)
	nPast := len(prompt)
	emitted := 0

	for block := 0; block < cfg.MaxCanvases; block++ {
		if err := ctx.Err(); err != nil {
			return err
		}

		canvas := randomCanvas(rng, cfg.Canvas, vocab)
		checkpoint := CheckpointPrefix(caches)

		var prevArgmax, argmaxCanvas []int32
		var selfCond *SelfCond
		for step := cfg.Steps; step >= 1; step-- {
			if err := ctx.Err(); err != nil {
				return err
			}

			temp := tempAt(step, cfg.Steps, cfg.TMin, cfg.TMax)
			key := mlx.RandomKey(uint64(cfg.Seed) + keyCounter)
			keyCounter++
			s := m.decodeCanvasSample(canvas, int32(nPast), selfCond, caches, temp, cfg.SelfCondK, key)
			checkpoint.Rollback(caches) // discard this step's canvas K/V
			argmaxCanvas = s.argmax

			if stableAndConfident(s.argmax, prevArgmax, s.entropy, cfg.ConfidenceThreshold, cfg.StabilityThreshold) {
				break
			}
			prevArgmax = s.argmax
			accept := entropyBoundAccept(s.entropy, cfg.EntropyBound)
			selfCond = &SelfCond{IDs: s.scIDs, Probs: s.scProbs, K: cfg.SelfCondK}
			canvas = renoise(s.sampled, accept, rng, vocab)
			mlx.Sweep()
			mlx.ClearCache() // release MLX's per-step buffers
		}

		// Emit the settled canvas's greedy (argmax) tokens.
		remaining := len(argmaxCanvas)
		if cfg.PredictTokens > 0 {
			remaining = min(remaining, cfg.PredictTokens-emitted)
		}
		stop := false
		for _, tok := range argmaxCanvas[:remaining] {
			if m.tok != nil && m.tok.IsEOS(tok) {
				stop = true
				break
			}
			if err := emit(tok); err != nil {
				return err
			}
			emitted++
		}
		if stop || (cfg.PredictTokens > 0 && emitted >= cfg.PredictTokens) {
			break
		}

		if block+1 < cfg.MaxCanvases {
			m.commitCanvas(argmaxCanvas, int32(nPast), caches)
			nPast += cfg.Canvas
		}
	}
	return nil
}

// computeSelfCond builds the decoder self-conditioning addend:
//
//	soft = (Σ_k probs · embed[ids]) · sqrt(n_embd)
//	sc   = sc_down( gelu(sc_gate(rms_norm(soft))) · sc_up(rms_norm(soft)) )
func (m *Model) computeSelfCond(sc *SelfCond, B, L int32) *mlx.Array {
	k := int32(sc.K)
	ids := mlx.FromValues(sc.IDs, int(B), int(L*k))
	emb := m.EmbedTokens.Forward(ids)             // [B, L*k, n_embd] (raw gather)
	emb = mlx.Reshape(emb, B, L, k, m.HiddenSize) // [B, L, k, n_embd]
	probs := mlx.FromValues(sc.Probs, int(B), int(L), int(k), 1)
	soft := mlx.Sum(mlx.Mul(emb, probs), 2, false) // [B, L, n_embd]
	soft = mlx.MulScalar(soft, m.EmbedScale)       // · sqrt(n_embd)

	scn := mlx.RMSNormFn(soft, m.SelfCond.PreNormScaled, m.RMSNormEps)
	gated := mlx.GeGLU(m.SelfCond.Gate.Forward(scn), m.SelfCond.Up.Forward(scn))
	return m.SelfCond.Down.Forward(gated)
}

func (m *Model) prefillPrompt(tokens []int32, caches []cache.Cache) {
	const chunk = 2048
	for off := 0; off < len(tokens); off += chunk {
		end := min(off+chunk, len(tokens))
		m.forward(&batch.Batch{
			InputIDs:     mlx.FromValues(tokens[off:end], 1, end-off),
			SeqOffsets:   []int32{int32(off)},
			SeqQueryLens: []int32{int32(end - off)},
		}, caches, nil)
	}
	evalCaches(caches)
}

func (m *Model) commitCanvas(tokens []int32, nPast int32, caches []cache.Cache) {
	m.forward(&batch.Batch{
		InputIDs:     mlx.FromValues(tokens, 1, len(tokens)),
		SeqOffsets:   []int32{nPast},
		SeqQueryLens: []int32{int32(len(tokens))},
	}, caches, nil)
	evalCaches(caches)
}

func evalCaches(caches []cache.Cache) {
	var state []*mlx.Array
	for _, c := range caches {
		if c != nil {
			state = append(state, c.State()...)
		}
	}
	if len(state) > 0 {
		mlx.Eval(state...)
	}
}
