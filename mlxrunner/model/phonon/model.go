package phonon

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

func init() {
	model.Register("GradiumPhononTTSForConditionalGeneration", newModel)
}

// config follows the upstream Phonon config.json.
type config struct {
	FlowLM struct {
		DModel      int     `json:"d_model"`
		NumHeads    int32   `json:"num_heads"`
		NumLayers   int32   `json:"num_layers"`
		MaxPeriod   float32 `json:"max_period"`
		FlowDepth   int32   `json:"flow_depth"`
		GELU        string  `json:"gelu"`
		TimeRMSNorm string  `json:"time_rms_norm"`
	} `json:"flow_lm"`
	Mimi struct {
		SampleRate   int     `json:"sample_rate"`
		FrameRate    float64 `json:"frame_rate"`
		DilationBase int     `json:"dilation_base"`
		NumHeads     int32   `json:"transformer_num_heads"`
		NumLayers    int32   `json:"transformer_num_layers"`
		Context      int     `json:"transformer_context"`
		MaxPeriod    float32 `json:"transformer_max_period"`
		GELU         string  `json:"gelu"`
	} `json:"mimi"`
	Temperature  float32 `json:"temp"`
	DecodeSteps  int     `json:"lsd_decode_steps"`
	EOSThreshold float32 `json:"eos_threshold"`
	Conditioners []struct {
		Type string `json:"type"`
		Name string `json:"name"`
		LUT  struct {
			PossibleValues []string `json:"possible_values"`
		} `json:"lut"`
		Continuous struct {
			Dim int `json:"dim"`
		} `json:"continuous"`
	} `json:"conditioners"`
	Fuser struct {
		Sum     []string `json:"sum"`
		Prepend []string `json:"prepend"`
	} `json:"fuser"`
}

const (
	sampleRate = 24000
	// mimiBatch is how many frames the codec decodes per call after the first.
	mimiBatch = 8
)

type Model struct {
	Embed       *nn.Embedding
	InputLinear *nn.Linear
	BOS         *mlx.Array
	Conditions  *mlx.Array
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
	cfg := config{Temperature: 0.3}
	if err := root.Manifest.ReadConfigJSON("config.json", &cfg); err != nil {
		return nil, fmt.Errorf("load config: %w", err)
	}
	var generation struct {
		Temperature *float32 `json:"temperature"`
	}
	if err := root.Manifest.ReadConfigJSON("generation_config.json", &generation); err != nil {
		return nil, fmt.Errorf("load generation config: %w", err)
	}
	if generation.Temperature != nil {
		cfg.Temperature = *generation.Temperature
	}
	if err := cfg.validate(); err != nil {
		return nil, err
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

func (c *config) validate() error {
	if c.FlowLM.DModel != 512 || len(c.Conditioners) != 2 ||
		c.Conditioners[0].Type != "lut" || c.Conditioners[0].Name != "num_speakers" ||
		!slices.Equal(c.Conditioners[0].LUT.PossibleValues, []string{"0", "1", "2"}) ||
		c.Conditioners[1].Type != "continuous" || c.Conditioners[1].Name != "duration_delta" || c.Conditioners[1].Continuous.Dim != 128 ||
		!slices.Equal(c.Fuser.Sum, []string{"num_speakers", "duration_delta"}) ||
		!slices.Equal(c.Fuser.Prepend, []string{"transcript_in_segment", "speaker_wavs"}) {
		return fmt.Errorf("unsupported Gradium Phonon conditioning layout")
	}
	for _, value := range []string{c.FlowLM.GELU, c.Mimi.GELU} {
		if value != "erf" {
			return fmt.Errorf("unsupported phonon GELU %q", value)
		}
	}
	if value := c.FlowLM.TimeRMSNorm; value != "rms" {
		return fmt.Errorf("unsupported phonon time norm %q", value)
	}
	if c.Mimi.SampleRate != sampleRate || c.Mimi.FrameRate != 12.5 {
		return fmt.Errorf("unsupported mimi rates: %d Hz, %g frames/s", c.Mimi.SampleRate, c.Mimi.FrameRate)
	}
	if c.FlowLM.NumLayers <= 0 || c.FlowLM.NumHeads <= 0 || c.FlowLM.MaxPeriod <= 0 || c.FlowLM.FlowDepth <= 0 ||
		c.Mimi.NumLayers <= 0 || c.Mimi.NumHeads <= 0 || c.Mimi.Context <= 0 || c.Mimi.MaxPeriod <= 0 || c.Mimi.DilationBase <= 0 {
		return fmt.Errorf("invalid phonon transformer configuration")
	}
	if c.DecodeSteps != 1 {
		return fmt.Errorf("phonon requires one LSD decode step, got %d", c.DecodeSteps)
	}
	if c.Temperature < 0 || math.IsNaN(float64(c.Temperature)) || math.IsInf(float64(c.Temperature), 0) ||
		math.IsNaN(float64(c.EOSThreshold)) || math.IsInf(float64(c.EOSThreshold), 0) {
		return fmt.Errorf("invalid phonon sampling configuration")
	}
	return nil
}

func (m *Model) LoadWeights(tensors map[string]*mlx.Array) error {
	// The flow LM transformer is bandwidth bound and runs in bfloat16; the
	// flow head and the Mimi decoder stay in float32.
	for name, t := range tensors {
		if d := t.DType(); d != mlx.DTypeFloat32 && d != mlx.DTypeBFloat16 && d != mlx.DTypeFloat16 {
			return fmt.Errorf("phonon requires unquantized floating-point weights: %s", name)
		}
		dtype := mlx.DTypeFloat32
		if strings.HasPrefix(name, "flow_lm.transformer.") || name == "flow_lm.input_linear.weight" {
			dtype = mlx.DTypeBFloat16
		}
		if strings.HasPrefix(name, "flow_lm.condition_provider.conditioners.") &&
			name != "flow_lm.condition_provider.conditioners.speaker_wavs.output_proj.weight" {
			dtype = mlx.DTypeBFloat16
		}
		if t.DType() != dtype {
			tensors[name] = t.AsType(dtype)
		}
	}
	f := m.cfg.FlowLM
	var err error
	if m.Flow, err = loadTransformer(tensors, "flow_lm.transformer", f.NumLayers, f.NumHeads, f.MaxPeriod, f.GELU); err != nil {
		return err
	}
	if m.Head, err = loadFlowHead(tensors, "flow_lm.flow_net", f.FlowDepth, f.TimeRMSNorm); err != nil {
		return err
	}
	if m.Mimi, err = loadMimi(tensors, m.cfg); err != nil {
		return err
	}
	if w := tensors["flow_lm.condition_provider.conditioners.transcript_in_segment.embed.weight"]; w != nil {
		m.Embed = nn.NewEmbedding(w)
	}
	m.InputLinear = linear(tensors, "flow_lm.input_linear")
	m.OutNorm = layerNorm(tensors, "flow_lm.out_norm", 1e-5)
	m.EOS = linear(tensors, "flow_lm.out_eos")
	m.BOS, m.EmbMean, m.EmbStd = tensors["flow_lm.bos_emb"], tensors["flow_lm.emb_mean"], tensors["flow_lm.emb_std"]
	if m.Embed == nil || m.InputLinear == nil || m.OutNorm == nil || m.EOS == nil || m.BOS == nil || m.EmbMean == nil || m.EmbStd == nil {
		return fmt.Errorf("missing flow_lm weights")
	}
	m.Conditions, err = gradiumConditions(tensors)
	if err != nil {
		return err
	}
	if m.Conditions == nil || !slices.Equal(m.Conditions.Dims(), []int{1, 1, m.Embed.Weight.Dim(1)}) {
		return fmt.Errorf("missing or invalid speech conditioning")
	}
	m.BOS = mlx.Reshape(m.BOS, 1, 1, -1)
	m.sample = mlx.Compile("phonon_sample", func(in ...*mlx.Array) []*mlx.Array {
		h := m.OutNorm.Forward(mlx.Reshape(in[0].AsType(mlx.DTypeFloat32), 1, -1))
		latent := m.Head.sample(h, in[1])
		return []*mlx.Array{mlx.ExpandDims(latent, 0), m.EOS.Forward(h)}
	})

	// The packager projects each speaker prompt into the flow LM's embedding
	// space. Prefill it once at load time; requests start from these cached keys
	// and values, without sharing mutable generation state.
	m.Voices = map[string]*Voice{}
	for name, t := range tensors {
		voice, ok := strings.CutPrefix(name, "voices.")
		if !ok {
			continue
		}
		projection := tensors["flow_lm.condition_provider.conditioners.speaker_wavs.output_proj.weight"]
		if projection == nil || t.NumDims() != 3 || t.Dim(0) != 1 || t.Dim(1) != projection.Dim(1) || t.Dim(2) == 0 {
			return fmt.Errorf("voice %q has invalid raw speaker latents %v", voice, t.Dims())
		}
		t = nn.NewLinear(projection, nil).Forward(mlx.Transpose(t, 0, 2, 1)).AsType(mlx.DTypeBFloat16)
		if voice == "" || t.NumDims() != 3 || t.Dim(0) != 1 || t.Dim(1) == 0 || t.Dim(2) != m.Embed.Weight.Dim(1) {
			return fmt.Errorf("voice %q has shape %v", voice, t.Dims())
		}
		m.Voices[voice] = m.prefillVoice(t)
	}
	if len(m.Voices) == 0 {
		return fmt.Errorf("model has no voices")
	}
	return nil
}

func (m *Model) prefillVoice(prompt *mlx.Array) *Voice {
	caches := make([]cache.Attention, len(m.Flow.Layers))
	for i := range caches {
		caches[i] = cache.NewKVCache()
		defer caches[i].Free()
	}
	kv := mlx.ScopedEval(func() []*mlx.Array {
		m.Flow.forward(prompt, 0, caches)
		var kv []*mlx.Array
		for _, c := range caches {
			kv = append(kv, c.State()...)
		}
		return kv
	})
	v := &Voice{}
	for i := 0; i < len(kv); i += 2 {
		v.Keys = append(v.Keys, kv[i])
		v.Values = append(v.Values, kv[i+1])
	}
	return v
}

// Speech models load through model.Model but never run the text pipeline.
func (m *Model) NewCaches() []cache.Cache { return nil }

func (m *Model) Forward(*batch.Batch, []cache.Cache) (*mlx.Array, *mlx.Array) {
	panic("phonon: text forward is not supported")
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
	chunks, err := m.tok.chunks(req.Input)
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
	text, framesAfterEOS, err := prepareText(text)
	if err != nil {
		return err
	}
	tokens := m.tok.encode(text)
	maxFrames := int(math.Ceil((float64(len(tokens))/3 + 2) * m.cfg.Mimi.FrameRate))

	caches := make([]cache.Attention, len(m.Flow.Layers))
	for i := range caches {
		caches[i] = cache.NewKVCache()
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
		m.Flow.forward(m.Embed.Forward(ids), offset, caches)
	})
	offset += len(tokens)

	std := math.Sqrt(float64(m.cfg.Temperature))
	noise := make([]float32, m.BOS.Dim(2))
	for generated, stop := 0, maxFrames; generated < stop; {
		// Decode the first frame alone for latency, then in batches. EOS
		// is read at the same sync points; frames past it are dropped.
		n := min(mimiBatch, stop-generated)
		if generated == 0 {
			n = 1
		}
		var latents, logits []*mlx.Array
		for step := generated; step < generated+n; step++ {
			if err := ctx.Err(); err != nil {
				return err
			}
			for i := range noise {
				noise[i] = float32(rng.NormFloat64() * std)
			}
			out := mlx.ScopedArrays(func() []*mlx.Array {
				input := mlx.Add(m.InputLinear.Forward(prev.AsType(mlx.DTypeBFloat16)), m.Conditions)
				h := m.Flow.forward(input, offset+step, caches)
				out := m.sample(h, mlx.FromValues(noise, 1, len(noise)))
				prev.Set(out[0])
				return out
			})
			held.Attach(out...)
			mlx.AsyncEval(out...)
			latents, logits = append(latents, out[0]), append(logits, out[1])
		}
		var eos []float32
		mlx.Scoped(func() { eos = mlx.Concatenate(logits, 0).Floats() })
		for i, e := range eos {
			if e > m.cfg.EOSThreshold {
				stop = min(stop, generated+i+1+framesAfterEOS)
				break
			}
		}
		n = min(n, stop-generated)
		var samples []float32
		mlx.Scoped(func() {
			z := mlx.Add(mlx.Mul(mlx.Concatenate(latents[:n], 1), m.EmbStd), m.EmbMean)
			samples = m.Mimi.decode(mimi, z).Floats()
		})
		held.Discard(latents...)
		held.Discard(logits...)
		generated += n
		if err := emit(samples); err != nil {
			return err
		}
	}
	return nil
}
