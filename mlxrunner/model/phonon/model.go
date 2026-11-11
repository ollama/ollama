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
	"unicode"

	"golang.org/x/text/unicode/norm"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/logutil"
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
	m.sample = mlx.Compile("phonon_sample", m.sampleFrame)

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
	if m.Voices["default"] == nil {
		return fmt.Errorf("model has no default voice")
	}
	return nil
}

func (m *Model) prefillVoice(prompt *mlx.Array) *Voice {
	caches := make([]cache.Cache, len(m.Flow.Layers))
	for i := range caches {
		caches[i] = cache.NewKVCache()
		defer caches[i].Free()
	}
	kv := mlx.ScopedEval(func() []*mlx.Array {
		m.Flow.forward(prompt, 0, caches)
		// Copy out the prompt's frames so the cache's padded buffers are freed.
		var kv []*mlx.Array
		for _, c := range caches {
			for _, s := range c.State() {
				kv = append(kv, mlx.Contiguous(s, false))
			}
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

func (m *Model) Forward(*batch.Batch, []cache.Layer) (*mlx.Array, *mlx.Array) {
	panic("phonon: text forward is not supported")
}
func (m *Model) Unembed(x *mlx.Array) *mlx.Array { return x }
func (m *Model) Tokenizer() *tokenizer.Tokenizer { return nil }
func (m *Model) MaxContextLength() int           { return 0 }

func (m *Model) VoiceNames() []string { return slices.Sorted(maps.Keys(m.Voices)) }

// voice finds a voice by name in any case and with or without accents, as
// the publisher's runtime does. No name means the checkpoint's own voice,
// default.
func (m *Model) voice(name string) *Voice {
	if name == "" {
		name = "default"
	}
	if v, ok := m.Voices[name]; ok {
		return v
	}
	for n, v := range m.Voices {
		if foldVoiceName(n) == foldVoiceName(name) {
			return v
		}
	}
	return nil
}

func foldVoiceName(s string) string {
	var b strings.Builder
	for _, r := range norm.NFD.String(s) {
		if !unicode.Is(unicode.Mn, r) {
			b.WriteRune(unicode.ToLower(r))
		}
	}
	return b.String()
}

func (m *Model) NewSpeech(ctx context.Context, req llm.SpeechRequest) (model.SpeechState, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	voice := m.voice(req.Voice)
	if voice == nil {
		return nil, api.StatusError{StatusCode: http.StatusBadRequest, ErrorMessage: fmt.Sprintf("voice %q not found; available voices: %s", req.Voice, strings.Join(m.VoiceNames(), ", "))}
	}
	chunks, err := m.tok.chunks(req.Input)
	if err != nil {
		return nil, api.StatusError{StatusCode: http.StatusBadRequest, ErrorMessage: err.Error()}
	}
	return &speechState{voice: voice, chunks: chunks, rng: rand.New(rand.NewPCG(rand.Uint64(), rand.Uint64()))}, nil
}

type speechState struct {
	voice                                   *Voice
	chunks                                  []string
	rng                                     *rand.Rand
	caches                                  []cache.Cache
	mimi                                    *mimiState
	held                                    *mlx.Scope
	prev                                    *mlx.Array
	offset, generated, stop, framesAfterEOS int
	noise                                   []float32
}

func (s *speechState) startChunk(m *Model) error {
	text, framesAfterEOS, err := prepareText(s.chunks[0])
	if err != nil {
		return err
	}
	s.chunks = s.chunks[1:]
	s.framesAfterEOS = framesAfterEOS
	tokens := m.tok.encode(text)
	s.stop = int(math.Ceil((float64(len(tokens))/3 + 2) * m.cfg.Mimi.FrameRate))
	s.generated = 0
	s.caches = make([]cache.Cache, len(m.Flow.Layers))
	for i := range s.caches {
		s.caches[i] = cache.NewKVCache()
	}
	s.mimi, s.held = m.Mimi.newState(), mlx.NewScope()
	s.prev = m.BOS.Clone()
	s.held.Attach(s.prev)
	s.offset = s.voice.Keys[0].Dim(2)
	mlx.Scoped(func() {
		for i, c := range s.caches {
			c.(cache.Attention).Update(nil, s.voice.Keys[i], s.voice.Values[i])
		}
		ids := mlx.FromValues(tokens, 1, len(tokens))
		m.Flow.forward(m.Embed.Forward(ids), s.offset, s.caches)
	})
	s.offset += len(tokens)
	s.noise = make([]float32, m.BOS.Dim(2))
	return nil
}

func (m *Model) sampleFrame(in ...*mlx.Array) []*mlx.Array {
	h := m.OutNorm.Forward(in[0].AsType(mlx.DTypeFloat32).Reshape(in[0].Dim(0), -1))
	latent := m.Head.sample(h, in[1])
	return []*mlx.Array{latent.ExpandDims(1), m.EOS.Forward(h)}
}

// speechFrame shares the transformer and flow head across active requests.
func (m *Model) speechFrame(prev *mlx.Array, offsets []int32, owners [][]cache.Cache, noise *mlx.Array) ([]*mlx.Array, error) {
	logutil.Trace("mlx speech frame batch", "rows", len(owners))
	layers, err := cache.Batch(owners)
	if err != nil {
		return nil, err
	}
	b := &batch.Batch{InputIDs: mlx.Zeros(mlx.DTypeInt32, len(owners), 1), SeqOffsets: offsets, SeqQueryLens: make([]int32, len(owners))}
	for i := range b.SeqQueryLens {
		b.SeqQueryLens[i] = 1
	}
	out := mlx.ScopedArrays(func() []*mlx.Array {
		input := mlx.Add(m.InputLinear.Forward(prev.AsType(mlx.DTypeBFloat16)), m.Conditions)
		h := m.Flow.forwardBatch(input, b, layers)
		return m.sample(h, noise)
	})
	return out, nil
}

func (m *Model) DecodeSpeech(states []model.SpeechState) ([]model.SpeechOutput, error) {
	output := make([]model.SpeechOutput, len(states))
	counts := make([]int, len(states))
	maxFrames := 0
	for i, state := range states {
		s := state.(*speechState)
		if s.held == nil && len(s.chunks) > 0 {
			if err := s.startChunk(m); err != nil {
				output[i].Err = err
				continue
			}
		}
		if s.held == nil {
			output[i].Done = true
			continue
		}
		counts[i] = min(mimiBatch, s.stop-s.generated)
		if s.generated == 0 {
			counts[i] = 1
		}
		maxFrames = max(maxFrames, counts[i])
	}
	latents, logits := make([][]*mlx.Array, len(states)), make([][]*mlx.Array, len(states))
	std := math.Sqrt(float64(m.cfg.Temperature))
	// Preserve each stream's original first-frame and eight-frame EOS sync
	// points. A newly joined stream does not alter its peers' sampling history.
	for step := range maxFrames {
		var ids []int
		var inputs []*mlx.Array
		var owners [][]cache.Cache
		var offsets []int32
		var noise []float32
		for i, state := range states {
			if step >= counts[i] {
				continue
			}
			s := state.(*speechState)
			for j := range s.noise {
				s.noise[j] = float32(s.rng.NormFloat64() * std)
			}
			noise = append(noise, s.noise...)
			ids, inputs, owners = append(ids, i), append(inputs, s.prev), append(owners, s.caches)
			offsets = append(offsets, int32(s.offset+s.generated+step))
		}
		out, err := m.speechFrame(mlx.Concatenate(inputs, 0), offsets, owners, mlx.FromValues(noise, len(ids), len(noise)/len(ids)))
		if err != nil {
			return nil, err
		}
		for row, id := range ids {
			s := states[id].(*speechState)
			latent := out[0].Slice(mlx.Slice(row, row+1), mlx.Slice(), mlx.Slice())
			logit := out[1].Slice(mlx.Slice(row, row+1), mlx.Slice())
			s.prev.Set(latent)
			s.held.Attach(latent, logit)
			latents[id], logits[id] = append(latents[id], latent), append(logits[id], logit)
		}
		mlx.AsyncEval(out...)
	}
	for i, state := range states {
		if counts[i] == 0 {
			continue
		}
		s := state.(*speechState)
		var eos []float32
		mlx.Scoped(func() { eos = mlx.Concatenate(logits[i], 0).Floats() })
		for j, e := range eos {
			if e > m.cfg.EOSThreshold {
				s.stop = min(s.stop, s.generated+j+1+s.framesAfterEOS)
				break
			}
		}
		n := min(counts[i], s.stop-s.generated)
		mlx.Scoped(func() {
			z := mlx.Add(mlx.Mul(mlx.Concatenate(latents[i][:n], 1), m.EmbStd), m.EmbMean)
			output[i].Samples = m.Mimi.decode(s.mimi, z).Floats()
		})
		s.held.Discard(latents[i]...)
		s.held.Discard(logits[i]...)
		s.generated += n
		if s.generated == s.stop {
			s.closeChunk()
		}
	}
	return output, nil
}

func (s *speechState) closeChunk() {
	for _, c := range s.caches {
		c.Free()
	}
	s.caches = nil
	if s.mimi != nil {
		s.mimi.close()
		s.mimi = nil
	}
	if s.held != nil {
		s.held.Close()
		s.held, s.prev = nil, nil
	}
}

func (s *speechState) Close() { s.closeChunk() }
