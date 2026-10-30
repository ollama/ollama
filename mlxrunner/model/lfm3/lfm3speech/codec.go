package lfm3speech

import (
	"fmt"
	"math"
	"strconv"
	"strings"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/cache"
	"github.com/ollama/ollama/mlxrunner/nn"
)

// The codec's LFM3 settings are fixed by the reference rather than its
// config.json; shapes come from the weights.
const (
	codecRopeTheta = 1e6
	codecNormEps   = 1e-5
)

// Codec is the LiquidCodec decoder: FSQ code frames to 24 kHz audio through
// an LFM3 stack at six times the code rate and an inverse STFT. Array fields
// are exported so mlx.Collect keeps them past the load scope.
type Codec struct {
	cfg           *CodecConfig
	ProjectOut    *nn.Linear
	FromQuantizer *nn.Linear
	stack         *stackConfig
	Layers        []*Layer
	Norm          *nn.RMSNorm
	Lin           *nn.Linear
	// Basis is the windowed inverse real DFT, [2*bins, n_fft] float32: the
	// first bins rows weight real parts, the rest imaginary parts.
	Basis *mlx.Array
	// window is the synthesis window, for the overlap-add envelope.
	window []float32
}

// LoadCodec builds the decoder from the codec tensors stored under prefix
// (e.g. "" for the subfolder's own model.safetensors) and its config.json.
// Encoder tensors are ignored.
func LoadCodec(tensors map[string]*mlx.Array, prefix string, configJSON []byte) (*Codec, error) {
	cfg, err := ParseCodecConfig(configJSON)
	if err != nil {
		return nil, err
	}
	c := &Codec{cfg: cfg}
	// The decoder runs in float32: the reference's bfloat16 decode is only
	// ~36 dB from its float32 decode and its chunked output drifts from the
	// whole decode; float32 streams exactly for ~60 MB more weights.
	decoder := map[string]*mlx.Array{}
	for name, t := range tensors {
		if rest, ok := strings.CutPrefix(name, prefix); ok && (strings.HasPrefix(rest, "decoder.") ||
			strings.HasPrefix(rest, "quantizer.project_out.") || strings.HasPrefix(rest, "from_quantizer.")) {
			if strings.HasSuffix(name, "_scale") || strings.HasSuffix(name, "_qbias") {
				return nil, fmt.Errorf("%s is quantized; the LFM3 codec must stay unquantized", name)
			}
			decoder[name] = t.AsType(mlx.DTypeFloat32)
		}
	}
	tensors = decoder
	latent := len(cfg.FSQLevels) * cfg.NumCodebooks
	if c.ProjectOut, err = dense(tensors, prefix+"quantizer.project_out", cfg.QuantizerDim, latent, true); err != nil {
		return nil, err
	}
	if c.FromQuantizer, err = dense(tensors, prefix+"from_quantizer", cfg.EncoderDim, cfg.QuantizerDim, true); err != nil {
		return nil, err
	}
	hidden := cfg.EncoderDim
	p := prefix + "decoder.lfm."
	if c.Norm, err = rmsNorm(tensors, p+"embedding_norm.weight", hidden, codecNormEps); err != nil {
		return nil, err
	}
	bins := cfg.NFFT/2 + 1
	if c.Lin, err = dense(tensors, prefix+"decoder.lin", 2*bins, hidden, true); err != nil {
		return nil, err
	}
	w := tensors[prefix+"decoder.istft.window"]
	if w == nil || w.NumDims() != 1 || w.Dim(0) != cfg.WinLength {
		return nil, fmt.Errorf("missing or misshapen %sdecoder.istft.window", prefix)
	}
	mlx.Scoped(func() { c.window = w.AsType(mlx.DTypeFloat32).Floats() })
	if err := c.loadLayers(tensors, p, hidden); err != nil {
		return nil, err
	}
	c.Basis = mlx.FromValues(synthesisBasis(c.window), 2*bins, cfg.NFFT)
	return c, nil
}

func (c *Codec) loadLayers(tensors map[string]*mlx.Array, p string, hidden int) error {
	count := 0
	for name := range tensors {
		if rest, ok := strings.CutPrefix(name, p+"layers."); ok {
			index, _, _ := strings.Cut(rest, ".")
			if i, err := strconv.Atoi(index); err == nil {
				count = max(count, i+1)
			}
		}
	}
	if count == 0 {
		return fmt.Errorf("missing %slayers", p)
	}
	cfg := &stackConfig{Hidden: int32(hidden), Window: c.cfg.Window, NormEps: codecNormEps, RopeTheta: codecRopeTheta}
	for i := range count {
		lp := fmt.Sprintf("%slayers.%d.", p, i)
		l := &Layer{MLP: &MLP{}}
		for _, n := range []struct {
			dst  **nn.RMSNorm
			name string
		}{
			{&l.OperatorNorm, "operator_norm"},
			{&l.PostOperatorNorm, "post_operator_norm"},
			{&l.FFNNorm, "ffn_norm"},
			{&l.PostFFNNorm, "post_ffn_norm"},
		} {
			norm, err := rmsNorm(tensors, lp+n.name+".weight", hidden, codecNormEps)
			if err != nil {
				return err
			}
			*n.dst = norm
		}
		w1 := tensors[lp+"feed_forward.w1.weight"]
		if w1 == nil || w1.NumDims() != 2 {
			return fmt.Errorf("missing %sfeed_forward.w1.weight", lp)
		}
		inter := w1.Dim(0)
		for _, lin := range []struct {
			dst     *nn.LinearLayer
			name    string
			out, in int
		}{
			{&l.MLP.W1, "feed_forward.w1", inter, hidden},
			{&l.MLP.W3, "feed_forward.w3", inter, hidden},
			{&l.MLP.W2, "feed_forward.w2", hidden, inter},
		} {
			layer, err := dense(tensors, lp+lin.name, lin.out, lin.in, false)
			if err != nil {
				return err
			}
			*lin.dst = layer
		}

		switch conv := tensors[lp+"conv.conv.weight"]; {
		case conv != nil:
			if conv.NumDims() != 3 || conv.Dim(0) != hidden || conv.Dim(1) != 1 {
				return fmt.Errorf("%sconv.conv.weight has shape %v", lp, conv.Dims())
			}
			if cfg.ConvKernel == 0 {
				cfg.ConvKernel = int32(conv.Dim(2))
			} else if int(cfg.ConvKernel) != conv.Dim(2) {
				return fmt.Errorf("codec conv kernels differ: %d and %d", cfg.ConvKernel, conv.Dim(2))
			}
			in, err := dense(tensors, lp+"conv.in_proj", 3*hidden, hidden, false)
			if err != nil {
				return err
			}
			out, err := dense(tensors, lp+"conv.out_proj", hidden, hidden, false)
			if err != nil {
				return err
			}
			l.Conv = &ShortConv{InProj: in, OutProj: out, Conv1D: nn.NewConv1d(mlx.Transpose(conv, 0, 2, 1), nil, 1, 0, 1, int32(hidden))}
		case tensors[lp+"self_attn.q_proj.weight"] != nil:
			qn := tensors[lp+"self_attn.q_norm.weight"]
			kw := tensors[lp+"self_attn.k_proj.weight"]
			if qn == nil || qn.NumDims() != 1 || kw == nil || kw.NumDims() != 2 {
				return fmt.Errorf("missing %sself_attn weights", lp)
			}
			headDim := qn.Dim(0)
			q := tensors[lp+"self_attn.q_proj.weight"].Dim(0)
			if headDim%2 != 0 || q%headDim != 0 || kw.Dim(0)%headDim != 0 {
				return fmt.Errorf("%sself_attn head width %d does not divide its projections", lp, headDim)
			}
			heads, kvHeads := int32(q/headDim), int32(kw.Dim(0)/headDim)
			if cfg.HeadDim == 0 {
				cfg.HeadDim, cfg.Heads, cfg.KVHeads = int32(headDim), heads, kvHeads
			} else if cfg.HeadDim != int32(headDim) || cfg.Heads != heads || cfg.KVHeads != kvHeads {
				return fmt.Errorf("codec attention layers differ in shape")
			}
			if heads%kvHeads != 0 {
				return fmt.Errorf("codec attention has %d heads for %d kv heads", heads, kvHeads)
			}
			a := &Attention{}
			for _, lin := range []struct {
				dst     *nn.LinearLayer
				name    string
				out, in int
			}{
				{&a.QProj, "q_proj", q, hidden},
				{&a.KProj, "k_proj", kw.Dim(0), hidden},
				{&a.VProj, "v_proj", kw.Dim(0), hidden},
				{&a.OProj, "o_proj", hidden, q},
				{&a.GateProj, "gate_proj", int(heads), hidden},
			} {
				layer, err := dense(tensors, lp+"self_attn."+lin.name, lin.out, lin.in, false)
				if err != nil {
					return err
				}
				*lin.dst = layer
			}
			var err error
			if a.QNorm, err = rmsNorm(tensors, lp+"self_attn.q_norm.weight", headDim, codecNormEps); err != nil {
				return err
			}
			if a.KNorm, err = rmsNorm(tensors, lp+"self_attn.k_norm.weight", headDim, codecNormEps); err != nil {
				return err
			}
			l.Attn = a
		default:
			return fmt.Errorf("%s is neither a conv nor an attention layer", lp)
		}
		c.Layers = append(c.Layers, l)
	}
	if cfg.HeadDim == 0 || cfg.ConvKernel == 0 {
		return fmt.Errorf("codec decoder needs both conv and attention layers")
	}
	cfg.Scale = float32(1 / math.Sqrt(float64(cfg.HeadDim)))
	c.stack = cfg
	return nil
}

// Config returns the codec configuration.
func (c *Codec) Config() *CodecConfig { return c.cfg }

// SampleRate is the output rate in Hz.
func (c *Codec) SampleRate() int { return c.cfg.SampleRate }

// fsq turns code frames into the quantizer latents [frames*codebooks*levels]:
// each code's mixed-radix digits, centered and scaled to [-1, 1].
func (c *Codec) fsq(frames [][]int32) ([]float32, error) {
	levels := c.cfg.FSQLevels
	size := int32(c.cfg.CodebookSize())
	out := make([]float32, 0, len(frames)*c.cfg.NumCodebooks*len(levels))
	for _, frame := range frames {
		if len(frame) != c.cfg.NumCodebooks {
			return nil, fmt.Errorf("frame has %d codes, want %d", len(frame), c.cfg.NumCodebooks)
		}
		for _, code := range frame {
			if code < 0 || code >= size {
				return nil, fmt.Errorf("codec code %d out of range [0, %d)", code, size)
			}
			for _, l := range levels {
				half := int32(l / 2)
				out = append(out, float32(code%int32(l)-half)/float32(half))
				code /= int32(l)
			}
		}
	}
	return out, nil
}

// frames runs codes through the network and the windowed inverse DFT,
// returning the STFT frames [frames*frame_stack, n_fft] on the host. With
// caches it continues a stream at STFT frame offset; without, it decodes
// the codes as a whole sequence.
func (c *Codec) frames(codes [][]int32, caches []cache.Cache, offset int) ([]float32, error) {
	latents, err := c.fsq(codes)
	if err != nil {
		return nil, err
	}
	var out []float32
	mlx.Scoped(func() {
		T := int32(len(codes))
		dtype := c.FromQuantizer.Weight.DType()
		x := mlx.FromValues(latents, 1, int(T), len(latents)/int(T)).AsType(dtype)
		x = c.FromQuantizer.Forward(c.ProjectOut.Forward(x))
		S, D := int32(c.cfg.FrameStack), c.stack.Hidden
		x = mlx.Reshape(mlx.BroadcastTo(mlx.Reshape(x, 1, T, 1, D), 1, T, S, D), 1, T*S, D)

		L := T * S
		b := &batch.Batch{
			InputIDs:     mlx.Zeros(mlx.DTypeInt32, 1, int(L)),
			SeqOffsets:   []int32{int32(offset)},
			SeqQueryLens: []int32{L},
		}
		positions := mlx.FromValues([]int32{int32(offset)}, 1)
		for i, l := range c.Layers {
			var lc cache.Cache
			if caches != nil {
				lc = caches[i]
			}
			x = l.forward(x, b, lc, positions, 1, L, c.stack)
		}
		spec := c.Lin.Forward(c.Norm.Forward(x, 0)).AsType(mlx.DTypeFloat32)
		bins := int32(c.cfg.NFFT/2 + 1)
		logMag := mlx.SliceStartStop(spec, []int32{0, 0, 0}, []int32{1, L, bins})
		phase := mlx.SliceStartStop(spec, []int32{0, 0, bins}, []int32{1, L, 2 * bins})
		mag := mlx.Exp(logMag)
		parts := mlx.Concatenate([]*mlx.Array{mlx.Mul(mag, mlx.Cos(phase)), mlx.Mul(mag, mlx.Sin(phase))}, 2)
		frames := mlx.Contiguous(mlx.Matmul(parts, c.Basis), false)
		eval := []*mlx.Array{frames}
		for _, lc := range caches {
			eval = append(eval, lc.State()...)
		}
		mlx.Eval(eval...)
		out = frames.Floats()
	})
	return out, nil
}

// Decode turns whole code frames into audio, 1920 samples per frame at 24
// kHz, as one offline pass.
func (c *Codec) Decode(codes [][]int32) ([]float32, error) {
	if len(codes) == 0 {
		return nil, nil
	}
	frames, err := c.frames(codes, nil, 0)
	if err != nil {
		return nil, err
	}
	o := newOverlapAdd(c.cfg, c.window)
	return append(o.add(frames), o.finish()...), nil
}

// Stream decodes a speech segment incrementally: Push code frames as they
// are sampled, then Flush. The concatenated output equals Decode of all
// frames. A Stream holds MLX state; Close it.
type Stream struct {
	codec  *Codec
	caches []cache.Cache
	ola    *overlapAdd
	offset int
	done   bool
}

// NewStream starts a stream with empty decoder state.
func (c *Codec) NewStream() *Stream {
	s := &Stream{codec: c, ola: newOverlapAdd(c.cfg, c.window)}
	for _, l := range c.Layers {
		if l.Conv != nil {
			s.caches = append(s.caches, cache.NewRecurrentCache(c.stack.ConvKernel-1, c.stack.Hidden, 1, 1, 1))
		} else {
			s.caches = append(s.caches, cache.NewRotatingKVCache(c.cfg.Window))
		}
	}
	return s
}

// Push decodes more frames and returns the samples they complete; samples
// that later frames still overlap are held back until the next Push or Flush.
func (s *Stream) Push(codes ...[]int32) ([]float32, error) {
	if s.done {
		return nil, fmt.Errorf("codec stream is finished")
	}
	if len(codes) == 0 {
		return nil, nil
	}
	frames, err := s.codec.frames(codes, s.caches, s.offset)
	if err != nil {
		return nil, err
	}
	s.offset += len(codes) * s.codec.cfg.FrameStack
	return s.ola.add(frames), nil
}

// Flush ends the stream and returns the held-back samples.
func (s *Stream) Flush() []float32 {
	if s.done {
		return nil
	}
	s.done = true
	return s.ola.finish()
}

// Close frees the stream's decoder state.
func (s *Stream) Close() {
	for _, c := range s.caches {
		c.Free()
	}
	s.caches = nil
	s.done = true
}

// overlapAdd accumulates windowed inverse-DFT frames at hop spacing,
// normalizes by the summed squared window, and trims (win-hop)/2 samples
// from both ends of the stream.
type overlapAdd struct {
	win, hop, trim int
	window         []float32
	frames         int       // STFT frames added so far
	tail           []float32 // unfinished sums at [hop*frames, hop*frames+win-hop)
	emitted        bool
}

func newOverlapAdd(cfg *CodecConfig, window []float32) *overlapAdd {
	return &overlapAdd{
		win:    cfg.WinLength,
		hop:    cfg.HopLength,
		trim:   (cfg.WinLength - cfg.HopLength) / 2,
		window: window,
		tail:   make([]float32, cfg.WinLength-cfg.HopLength),
	}
}

// add takes frames [n, win] and returns the samples no later frame reaches.
func (o *overlapAdd) add(frames []float32) []float32 {
	n := len(frames) / o.win
	if n == 0 {
		return nil
	}
	start := o.frames * o.hop
	buf := make([]float32, n*o.hop+o.win-o.hop)
	copy(buf, o.tail)
	for t := range n {
		row := frames[t*o.win : (t+1)*o.win]
		dst := buf[t*o.hop:]
		for i, v := range row {
			dst[i] += v
		}
	}
	o.frames += n
	done := n * o.hop
	copy(o.tail, buf[done:])
	out := buf[:done]
	for i := range out {
		out[i] /= o.envelope(start+i, -1)
	}
	if !o.emitted {
		o.emitted = true
		return out[o.trim:]
	}
	return out
}

// finish returns the tail minus the trailing trim.
func (o *overlapAdd) finish() []float32 {
	if o.frames == 0 {
		return nil
	}
	start := o.frames * o.hop
	out := make([]float32, len(o.tail)-o.trim)
	for i := range out {
		out[i] = o.tail[i] / o.envelope(start+i, o.frames)
	}
	return out
}

// envelope is the summed squared window at sample n over frames [0, total);
// total < 0 means every frame that can reach n exists.
func (o *overlapAdd) envelope(n, total int) float32 {
	var e float32
	for t := max(0, (n-o.win)/o.hop); t*o.hop <= n; t++ {
		if total >= 0 && t >= total {
			break
		}
		if i := n - t*o.hop; i < o.win {
			e += o.window[i] * o.window[i]
		}
	}
	return e
}

// synthesisBasis returns the windowed inverse real DFT as a [2*bins, n]
// matrix, norm="backward": real-part rows then imaginary-part rows. The
// imaginary parts of the DC and Nyquist bins are ignored, as irfft does.
func synthesisBasis(window []float32) []float32 {
	n := len(window)
	bins := n/2 + 1
	out := make([]float32, 2*bins*n)
	for k := range bins {
		scale := 2.0 / float64(n)
		if k == 0 || 2*k == n {
			scale = 1.0 / float64(n)
		}
		for j := range n {
			// Reduce k*j mod n first so the angle stays exact.
			angle := 2 * math.Pi * float64((k*j)%n) / float64(n)
			w := float64(window[j]) * scale
			out[k*n+j] = float32(w * math.Cos(angle))
			if k != 0 && 2*k != n {
				out[(bins+k)*n+j] = float32(-w * math.Sin(angle))
			}
		}
	}
	return out
}
