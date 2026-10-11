package pockettts

import (
	"context"
	"fmt"
	"maps"
	"math"
	"math/rand/v2"
	"net/http"
	"slices"
	"strings"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/cache"
	"github.com/ollama/ollama/mlxrunner/model"
	"github.com/ollama/ollama/mlxrunner/nn"
	"github.com/ollama/ollama/mlxrunner/tokenizer"
)

func init() { model.Register("PocketTTS", newModel) }

// config is the upstream pocket_tts YAML config as JSON.
type config struct {
	FlowLM struct {
		Transformer struct {
			NumHeads  int32   `json:"num_heads"`
			NumLayers int32   `json:"num_layers"`
			MaxPeriod float32 `json:"max_period"`
		} `json:"transformer"`
		Flow struct {
			Depth int32  `json:"depth"`
			Type  string `json:"type"`
		} `json:"flow"`
	} `json:"flow_lm"`
	Mimi struct {
		SampleRate int     `json:"sample_rate"`
		FrameRate  float64 `json:"frame_rate"`
		Seanet     struct {
			DilationBase int `json:"dilation_base"`
		} `json:"seanet"`
		Transformer struct {
			NumHeads  int32   `json:"num_heads"`
			NumLayers int32   `json:"num_layers"`
			Context   int     `json:"context"`
			MaxPeriod float32 `json:"max_period"`
		} `json:"transformer"`
	} `json:"mimi"`
	DefaultTemperature *float32 `json:"default_temperature"`
	textOptions
}

const (
	sampleRate   = 24000
	eosThreshold = -4
	// EOS is ignored on the first frames, where some voices cross the threshold before speaking.
	minFramesBeforeEOS = 6
	// mimiBatch is how many frames the codec decodes per call after the first.
	mimiBatch = 8
)

type Model struct {
	Embed       *nn.Embedding
	InputLinear *nn.Linear
	BOS         *mlx.Array
	Flow        *Transformer
	OutNorm     *nn.LayerNorm
	EOS         *nn.Linear
	Head        *FlowHead
	EmbMean     *mlx.Array
	EmbStd      *mlx.Array
	Mimi        *Mimi
	Voices      map[string]*Voice

	cfg *config
	tok *unigram

	// sample maps a transformer state and noise to a latent and EOS logit.
	sample mlx.CompileFunc
}

// Voice is a speaker prompt already run through the flow LM: per-layer keys
// and values [1, heads, frames, head_dim].
type Voice struct {
	Keys   []*mlx.Array
	Values []*mlx.Array
}

func newModel(root *model.Root) (model.Model, error) {
	var cfg config
	if err := root.Manifest.ReadConfigJSON("config.json", &cfg); err != nil {
		return nil, fmt.Errorf("load config: %w", err)
	}
	if cfg.Mimi.SampleRate != sampleRate || cfg.Mimi.FrameRate != 12.5 {
		return nil, fmt.Errorf("unsupported mimi rates: %d Hz, %g frames/s", cfg.Mimi.SampleRate, cfg.Mimi.FrameRate)
	}
	if cfg.FlowLM.Transformer.NumLayers <= 0 || cfg.FlowLM.Transformer.NumHeads <= 0 ||
		cfg.FlowLM.Transformer.MaxPeriod <= 0 || cfg.FlowLM.Flow.Depth <= 0 ||
		cfg.Mimi.Transformer.NumLayers <= 0 || cfg.Mimi.Transformer.NumHeads <= 0 || cfg.Mimi.Transformer.Context <= 0 {
		return nil, fmt.Errorf("invalid pocket-tts transformer configuration")
	}
	if t := cfg.FlowLM.Flow.Type; t != "" && t != "lsd" {
		return nil, fmt.Errorf("unsupported flow type %q", t)
	}
	if cfg.Mimi.Transformer.MaxPeriod == 0 {
		cfg.Mimi.Transformer.MaxPeriod = 10000
	}
	if cfg.Mimi.Seanet.DilationBase == 0 {
		cfg.Mimi.Seanet.DilationBase = 2
	}
	if cfg.DefaultTemperature == nil {
		temperature := float32(0.3)
		cfg.DefaultTemperature = &temperature
	}
	if *cfg.DefaultTemperature < 0 || math.IsInf(float64(*cfg.DefaultTemperature), 0) {
		return nil, fmt.Errorf("invalid default_temperature")
	}
	data, err := root.Manifest.ReadConfig("tokenizer.json")
	if err != nil {
		return nil, fmt.Errorf("load tokenizer: %w", err)
	}
	tok, err := newUnigram(data)
	if err != nil {
		return nil, fmt.Errorf("parse tokenizer: %w", err)
	}
	return &Model{cfg: &cfg, tok: tok}, nil
}

func (m *Model) LoadWeights(tensors map[string]*mlx.Array) error {
	// The flow LM transformer is bandwidth bound and runs in bfloat16; the
	// flow head and the Mimi decoder stay in float32.
	for name, t := range tensors {
		dtype := mlx.DTypeFloat32
		if strings.HasPrefix(name, "flow_lm.transformer.") || strings.HasPrefix(name, "voices.") ||
			name == "flow_lm.conditioner.embed.weight" || name == "flow_lm.input_linear.weight" {
			dtype = mlx.DTypeBFloat16
		}
		if t.DType() != dtype {
			tensors[name] = t.AsType(dtype)
		}
	}
	f := m.cfg.FlowLM
	var err error
	if m.Flow, err = loadTransformer(tensors, "flow_lm.transformer", f.Transformer.NumLayers, f.Transformer.NumHeads, f.Transformer.MaxPeriod); err != nil {
		return err
	}
	if m.Head, err = loadFlowHead(tensors, "flow_lm.flow_net", f.Flow.Depth); err != nil {
		return err
	}
	if m.Mimi, err = loadMimi(tensors, m.cfg); err != nil {
		return err
	}
	if w := tensors["flow_lm.conditioner.embed.weight"]; w != nil {
		m.Embed = nn.NewEmbedding(w)
	}
	m.InputLinear = linear(tensors, "flow_lm.input_linear")
	m.OutNorm = layerNorm(tensors, "flow_lm.out_norm", 1e-5)
	m.EOS = linear(tensors, "flow_lm.out_eos")
	m.BOS, m.EmbMean, m.EmbStd = tensors["flow_lm.bos_emb"], tensors["flow_lm.emb_mean"], tensors["flow_lm.emb_std"]
	if m.Embed == nil || m.InputLinear == nil || m.OutNorm == nil || m.EOS == nil || m.BOS == nil || m.EmbMean == nil || m.EmbStd == nil {
		return fmt.Errorf("missing flow_lm weights")
	}
	m.BOS = mlx.Reshape(m.BOS, 1, 1, -1)
	m.sample = mlx.Compile("pockettts_sample", func(in ...*mlx.Array) []*mlx.Array {
		h := m.OutNorm.Forward(mlx.Reshape(in[0].AsType(mlx.DTypeFloat32), 1, -1))
		latent := m.Head.sample(h, in[1])
		return []*mlx.Array{mlx.ExpandDims(latent, 0), m.EOS.Forward(h)}
	})

	// Voices are stored as voices.<name> [layers, 2, 1, frames, heads, head_dim].
	m.Voices = map[string]*Voice{}
	for name, t := range tensors {
		voice, ok := strings.CutPrefix(name, "voices.")
		if !ok {
			continue
		}
		if t.NumDims() != 6 || t.Dim(0) != len(m.Flow.Layers) || t.Dim(1) != 2 || t.Dim(2) != 1 ||
			t.Dim(3) == 0 || t.Dim(4) != int(f.Transformer.NumHeads) || t.Dim(4)*t.Dim(5) != m.Embed.Weight.Dim(1) {
			return fmt.Errorf("voice %q has shape %v", voice, t.Dims())
		}
		T, H, D := int32(t.Dim(3)), int32(t.Dim(4)), int32(t.Dim(5))
		kv := func(l, i int) *mlx.Array {
			x := t.Slice(mlx.Slice(l), mlx.Slice(i), mlx.Slice(), mlx.Slice(), mlx.Slice(), mlx.Slice())
			return mlx.Transpose(mlx.Reshape(x, 1, T, H, D), 0, 2, 1, 3)
		}
		v := &Voice{}
		for l := range t.Dim(0) {
			v.Keys = append(v.Keys, kv(l, 0))
			v.Values = append(v.Values, kv(l, 1))
		}
		m.Voices[voice] = v
	}
	if len(m.Voices) == 0 {
		return fmt.Errorf("model has no voices")
	}
	return nil
}

// Speech models load through model.Model but never run the text pipeline.
func (m *Model) NewCaches() []cache.Cache { return nil }
func (m *Model) Forward(*batch.Batch, []cache.Cache) (*mlx.Array, *mlx.Array) {
	panic("pockettts: text forward is not supported")
}
func (m *Model) Unembed(x *mlx.Array) *mlx.Array { return x }
func (m *Model) Tokenizer() *tokenizer.Tokenizer { return nil }
func (m *Model) MaxContextLength() int           { return 0 }

func (m *Model) VoiceNames() []string { return slices.Sorted(maps.Keys(m.Voices)) }

func (m *Model) Speak(ctx context.Context, req llm.SpeechRequest, emit func([]float32) error) error {
	name := req.Voice
	if name == "" {
		name = m.VoiceNames()[0]
	}
	voice, ok := m.Voices[name]
	if !ok {
		return api.StatusError{StatusCode: http.StatusBadRequest, ErrorMessage: fmt.Sprintf("voice %q not found; available voices: %s", req.Voice, strings.Join(m.VoiceNames(), ", "))}
	}
	chunks, err := m.tok.chunks(req.Input, m.cfg.textOptions)
	if err != nil {
		return api.StatusError{StatusCode: http.StatusBadRequest, ErrorMessage: err.Error()}
	}
	rng := rand.New(rand.NewPCG(rand.Uint64(), rand.Uint64()))
	for _, chunk := range chunks {
		if err := m.speak(ctx, chunk, voice, rng, emit); err != nil {
			return err
		}
	}
	return nil
}

// speak generates one chunk from a fresh copy of the voice state.
func (m *Model) speak(ctx context.Context, text string, voice *Voice, rng *rand.Rand, emit func([]float32) error) error {
	text, framesAfterEOS, err := m.cfg.prepare(text)
	if err != nil {
		return err
	}
	framesAfterEOS += 2
	tokens := m.tok.encode(text)
	maxFrames := int(math.Ceil((float64(len(tokens))/3 + 2) * m.cfg.Mimi.FrameRate))

	caches := make([]*cache.KVCache, len(m.Flow.Layers))
	attn := make([]attention, len(caches))
	for i := range caches {
		caches[i] = cache.NewKVCache()
		attn[i] = flowCache{caches[i]}
		defer caches[i].Free()
	}
	mimi := m.Mimi.newState()
	defer mimi.close()
	held := mlx.NewScope()
	defer held.Close()

	prev := m.BOS.Clone()
	held.Attach(prev)
	offset := voice.Keys[0].Dim(2)
	mlx.Scoped(func() {
		for i, c := range caches {
			c.Update(nil, voice.Keys[i], voice.Values[i])
		}
		ids := mlx.FromValues(tokens, 1, len(tokens))
		m.Flow.forward(m.Embed.Forward(ids), offset, attn)
	})
	offset += len(tokens)

	std := math.Sqrt(float64(*m.cfg.DefaultTemperature))
	noise := make([]float32, m.BOS.Dim(2))
	var latents, logits []*mlx.Array
	decoded, eosStep, stop := 0, -1, maxFrames
	for step := 0; step < stop; step++ {
		if err := ctx.Err(); err != nil {
			return err
		}
		for i := range noise {
			noise[i] = float32(rng.NormFloat64() * std)
		}
		out := mlx.ScopedArrays(func() []*mlx.Array {
			h := m.Flow.forward(m.InputLinear.Forward(prev.AsType(mlx.DTypeBFloat16)), offset+step, attn)
			out := m.sample(h, mlx.FromValues(noise, 1, len(noise)))
			prev.Set(out[0])
			return out
		})
		held.Attach(out...)
		mlx.AsyncEval(out...)
		latents, logits = append(latents, out[0]), append(logits, out[1])

		// Decode the first frame alone for latency, then in batches. EOS
		// is read at the same sync points; frames past it are dropped.
		if decoded > 0 && len(latents) < mimiBatch && step+1 < stop {
			continue
		}
		var eos []float32
		mlx.Scoped(func() { eos = mlx.Concatenate(logits, 0).Floats() })
		for i, e := range eos {
			if f := decoded + i; eosStep < 0 && f >= minFramesBeforeEOS && e > eosThreshold {
				eosStep = f
				stop = min(stop, f+framesAfterEOS)
			}
		}
		n := min(len(latents), stop-decoded)
		var samples []float32
		mlx.Scoped(func() {
			z := mlx.Add(mlx.Mul(mlx.Concatenate(latents[:n], 1), m.EmbStd), m.EmbMean)
			samples = m.Mimi.decode(mimi, z).Floats()
		})
		held.Discard(append(latents, logits...)...)
		latents, logits = latents[:0], logits[:0]
		if decoded == 0 {
			fade := min(len(samples), sampleRate/200)
			for i := range fade {
				samples[i] *= float32(i) / float32(fade-1)
			}
		}
		decoded += n
		if err := emit(samples); err != nil {
			return err
		}
	}
	return nil
}

type flowCache struct{ *cache.KVCache }

func (c flowCache) attend(q, k, v *mlx.Array, scale float32) *mlx.Array {
	h := c.Update(nil, k, v)
	mode := ""
	if q.Dim(2) > 1 {
		mode = "causal"
	}
	return mlx.FastScaledDotProductAttention(q, h.K(), h.V(), scale, mode, nil)
}
