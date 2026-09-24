// Package gliner implements DeBERTa-v3 GLiNER span extraction on MLX.
package gliner

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"slices"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/model"
	"github.com/ollama/ollama/mlxrunner/tokenizer"
)

func init() { model.RegisterExtractor("GLiNER", newModel) }

type encoderConfig struct {
	ModelType            string   `json:"model_type"`
	HiddenSize           int      `json:"hidden_size"`
	EmbeddingSize        int      `json:"embedding_size"`
	IntermediateSize     int      `json:"intermediate_size"`
	NumHiddenLayers      int      `json:"num_hidden_layers"`
	NumAttentionHeads    int      `json:"num_attention_heads"`
	MaxPositions         int      `json:"max_position_embeddings"`
	MaxRelativePositions int      `json:"max_relative_positions"`
	PositionBuckets      int      `json:"position_buckets"`
	VocabSize            int      `json:"vocab_size"`
	LayerNormEps         float32  `json:"layer_norm_eps"`
	HiddenAct            string   `json:"hidden_act"`
	RelativeAttention    bool     `json:"relative_attention"`
	PositionBiasedInput  bool     `json:"position_biased_input"`
	ShareAttKey          bool     `json:"share_att_key"`
	NormRelEbd           string   `json:"norm_rel_ebd"`
	PosAttType           []string `json:"pos_att_type"`
	TypeVocabSize        int      `json:"type_vocab_size"`
	ConvKernelSize       int      `json:"conv_kernel_size"`
}

type Config struct {
	Encoder          encoderConfig `json:"encoder_config"`
	HiddenSize       int           `json:"hidden_size"`
	MaxWidth         int           `json:"max_width"`
	MaxLength        int           `json:"max_len"`
	SpanMode         string        `json:"span_mode"`
	SubtokenPooling  string        `json:"subtoken_pooling"`
	WordsSplitter    string        `json:"words_splitter_type"`
	HasRNN           bool          `json:"has_rnn"`
	FuseLayers       bool          `json:"fuse_layers"`
	EmbedEntToken    bool          `json:"embed_ent_token"`
	LabelsEncoder    string        `json:"labels_encoder"`
	PostFusionSchema string        `json:"post_fusion_schema"`
	ClassTokenIndex  int32         `json:"class_token_index"`
	EntToken         string        `json:"ent_token"`
	SepToken         string        `json:"sep_token"`
}

type Model struct {
	Weights                map[string]*mlx.Array
	cfg                    Config
	tok                    *tokenizer.Unigram
	cls, sep, ent, textSep int32
}

func newModel(root *model.Root) (model.Extractor, error) {
	if root.QuantType() != "" {
		return nil, fmt.Errorf("GLiNER requires unquantized weights")
	}
	config, err := root.Manifest.ReadConfig("config.json")
	if err != nil {
		return nil, err
	}
	tok, err := root.Manifest.ReadConfig("tokenizer.json")
	if err != nil {
		return nil, err
	}
	return New(config, tok)
}

// New constructs a model from the self-contained export produced by
// scripts/export_gliner.py. Weight loading remains on the MLX thread.
func New(config, tok []byte) (*Model, error) {
	var cfg Config
	if err := json.Unmarshal(config, &cfg); err != nil {
		return nil, err
	}
	if err := cfg.validate(); err != nil {
		return nil, err
	}
	t, err := tokenizer.LoadUnigram(tok)
	if err != nil {
		return nil, fmt.Errorf("GLiNER tokenizer: %w", err)
	}
	m := &Model{cfg: cfg, tok: t}
	if t.VocabSize() != cfg.Encoder.VocabSize {
		return nil, fmt.Errorf("GLiNER tokenizer vocabulary does not match encoder")
	}
	for token, target := range map[string]*int32{"[CLS]": &m.cls, "[SEP]": &m.sep, cfg.EntToken: &m.ent, cfg.SepToken: &m.textSep} {
		id, ok := t.TokenID(token)
		if !ok || id < 0 || int(id) >= cfg.Encoder.VocabSize {
			return nil, fmt.Errorf("missing or invalid GLiNER token %q", token)
		}
		*target = id
	}
	if m.ent != cfg.ClassTokenIndex {
		return nil, fmt.Errorf("class_token_index does not match tokenizer")
	}
	return m, nil
}

func (c Config) validate() error {
	e := c.Encoder
	if c.SpanMode != "markerV0" || c.SubtokenPooling != "first" || c.WordsSplitter != "whitespace" || !c.EmbedEntToken || c.FuseLayers || c.LabelsEncoder != "" || c.PostFusionSchema != "" {
		return fmt.Errorf("unsupported GLiNER variant: requires uni-encoder markerV0 spans, first-subtoken pooling, whitespace splitting, and entity-token embeddings")
	}
	if c.HiddenSize <= 0 || c.HiddenSize > 4096 || c.HiddenSize%2 != 0 || c.MaxLength <= 0 || c.MaxLength > 512 || c.MaxWidth <= 0 || c.MaxWidth > 128 || c.EntToken == "" || c.SepToken == "" || c.EntToken == c.SepToken {
		return fmt.Errorf("invalid GLiNER dimensions or special tokens")
	}
	if e.ModelType != "deberta-v2" || !e.RelativeAttention || e.PositionBiasedInput || !e.ShareAttKey || e.NormRelEbd != "layer_norm" || e.HiddenAct != "gelu" || e.TypeVocabSize != 0 || e.ConvKernelSize != 0 || (e.EmbeddingSize != 0 && e.EmbeddingSize != e.HiddenSize) || len(e.PosAttType) != 2 || !slices.Contains(e.PosAttType, "p2c") || !slices.Contains(e.PosAttType, "c2p") {
		return fmt.Errorf("unsupported GLiNER encoder: requires DeBERTa-v3 with shared relative attention and no convolution or absolute-position embeddings")
	}
	if e.HiddenSize <= 0 || e.HiddenSize > 4096 || e.NumAttentionHeads <= 0 || e.HiddenSize%e.NumAttentionHeads != 0 || e.NumHiddenLayers <= 0 || e.NumHiddenLayers > 48 || e.IntermediateSize <= 0 || e.IntermediateSize > 32768 || e.MaxPositions < 4 || e.MaxPositions > 512 || e.PositionBuckets < 4 || e.PositionBuckets > e.MaxPositions || e.VocabSize <= 0 || e.LayerNormEps <= 0 {
		return fmt.Errorf("invalid GLiNER encoder dimensions")
	}
	if e.MaxRelativePositions > 0 && e.MaxRelativePositions <= e.PositionBuckets/2+1 {
		return fmt.Errorf("invalid max_relative_positions")
	}
	return nil
}

const encoderPrefix = "token_rep_layer.bert_layer.model."
const spanPrefix = "span_rep_layer.span_rep_layer."

func (m *Model) LoadWeights(weights map[string]*mlx.Array) error {
	c, e := m.cfg, m.cfg.Encoder
	expected := make(map[string][]int)
	weight := func(name string, shape ...int) { expected[name] = shape }
	linear := func(name string, in, out int) { weight(name+".weight", out, in); weight(name+".bias", out) }
	norm := func(name string, width int) { weight(name+".weight", width); weight(name+".bias", width) }
	weight(encoderPrefix+"embeddings.word_embeddings.weight", e.VocabSize, e.HiddenSize)
	norm(encoderPrefix+"embeddings.LayerNorm", e.HiddenSize)
	weight(encoderPrefix+"encoder.rel_embeddings.weight", 2*e.PositionBuckets, e.HiddenSize)
	norm(encoderPrefix+"encoder.LayerNorm", e.HiddenSize)
	for i := range e.NumHiddenLayers {
		p := fmt.Sprintf("%sencoder.layer.%d.", encoderPrefix, i)
		for _, proj := range []string{"query_proj", "key_proj", "value_proj"} {
			linear(p+"attention.self."+proj, e.HiddenSize, e.HiddenSize)
		}
		linear(p+"attention.output.dense", e.HiddenSize, e.HiddenSize)
		norm(p+"attention.output.LayerNorm", e.HiddenSize)
		linear(p+"intermediate.dense", e.HiddenSize, e.IntermediateSize)
		linear(p+"output.dense", e.IntermediateSize, e.HiddenSize)
		norm(p+"output.LayerNorm", e.HiddenSize)
	}
	if e.HiddenSize != c.HiddenSize {
		linear("token_rep_layer.projection", e.HiddenSize, c.HiddenSize)
	}
	if c.HasRNN {
		for _, suffix := range []string{"", "_reverse"} {
			weight("rnn.lstm.weight_ih_l0"+suffix, 2*c.HiddenSize, c.HiddenSize)
			weight("rnn.lstm.weight_hh_l0"+suffix, 2*c.HiddenSize, c.HiddenSize/2)
			weight("rnn.lstm.bias_ih_l0"+suffix, 2*c.HiddenSize)
			weight("rnn.lstm.bias_hh_l0"+suffix, 2*c.HiddenSize)
		}
	}
	for _, name := range []string{spanPrefix + "project_start", spanPrefix + "project_end", spanPrefix + "out_project", "prompt_rep_layer"} {
		in := c.HiddenSize
		if name == spanPrefix+"out_project" {
			in *= 2
		}
		linear(name+".0", in, 4*c.HiddenSize)
		linear(name+".3", 4*c.HiddenSize, c.HiddenSize)
	}
	for name, shape := range expected {
		a := weights[name]
		if a == nil {
			return fmt.Errorf("missing GLiNER tensor %s", name)
		}
		if !slices.Equal(a.Dims(), shape) {
			return fmt.Errorf("GLiNER tensor %s has shape %v, expected %v", name, a.Dims(), shape)
		}
		if a.DType() != mlx.DTypeFloat32 {
			return fmt.Errorf("GLiNER currently requires float32 weights: %s has dtype %v", name, a.DType())
		}
	}
	for name := range weights {
		if _, ok := expected[name]; !ok {
			return fmt.Errorf("unsupported GLiNER tensor %s", name)
		}
	}
	m.Weights = weights
	return nil
}

func (m *Model) MaxContextLength() int { return m.cfg.Encoder.MaxPositions }

func (m *Model) linear(x *mlx.Array, name string) *mlx.Array {
	return m.Weights[name+".bias"].Addmm(x, m.Weights[name+".weight"].Transpose(1, 0), 1, 1)
}

func (m *Model) norm(x *mlx.Array, name string) *mlx.Array {
	n := mlx.LayerNorm{Weight: m.Weights[name+".weight"], Bias: m.Weights[name+".bias"]}
	return n.Forward(x, m.cfg.Encoder.LayerNormEps)
}

func relu(x *mlx.Array) *mlx.Array { return mlx.Maximum(x, mlx.FromValue(float32(0))) }
func (m *Model) project(x *mlx.Array, name string) *mlx.Array {
	return m.linear(relu(m.linear(x, name+".0")), name+".3")
}

// encode returns projected subword embeddings. Each request is unpadded,
// so all tokens attend bidirectionally to all other tokens.
func (m *Model) encode(ctx context.Context, ids []int32) (*mlx.Array, error) {
	e := m.cfg.Encoder
	n, heads, dim := len(ids), e.NumAttentionHeads, e.HiddenSize/e.NumAttentionHeads
	h := m.Weights[encoderPrefix+"embeddings.word_embeddings.weight"].TakeAxis(mlx.FromValues(ids, n), 0)
	h = m.norm(h, encoderPrefix+"embeddings.LayerNorm")
	rel := m.norm(m.Weights[encoderPrefix+"encoder.rel_embeddings.weight"], encoderPrefix+"encoder.LayerNorm")
	maxPos := e.MaxRelativePositions
	if maxPos <= 0 {
		maxPos = e.MaxPositions
	}
	c2p, p2c := relativePositions(n, e.PositionBuckets, maxPos)
	ci, pi := mlx.FromValues(c2p, 1, n, n), mlx.FromValues(p2c, 1, n, n)
	scale := mlx.FromValue(float32(math.Sqrt(float64(dim * 3))))
	reshapeHeads := func(x *mlx.Array, length int) *mlx.Array { return x.Reshape(length, heads, dim).Transpose(1, 0, 2) }
	for i := range e.NumHiddenLayers {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		p := fmt.Sprintf("%sencoder.layer.%d.", encoderPrefix, i)
		q := reshapeHeads(m.linear(h, p+"attention.self.query_proj"), n)
		k := reshapeHeads(m.linear(h, p+"attention.self.key_proj"), n)
		v := reshapeHeads(m.linear(h, p+"attention.self.value_proj"), n)
		pq := reshapeHeads(m.linear(rel, p+"attention.self.query_proj"), 2*e.PositionBuckets)
		pk := reshapeHeads(m.linear(rel, p+"attention.self.key_proj"), 2*e.PositionBuckets)
		scores := q.Matmul(k.Transpose(0, 2, 1).Divide(scale))
		contentPosition := q.Matmul(pk.Transpose(0, 2, 1)).TakeAlongAxis(ci, 2).Divide(scale)
		positionContent := k.Matmul(pq.Transpose(0, 2, 1)).TakeAlongAxis(pi, 2).Transpose(0, 2, 1).Divide(scale)
		scores = scores.Add(contentPosition).Add(positionContent)
		att := mlx.SoftmaxAxis(scores, -1, true).Matmul(v).Transpose(1, 0, 2).Reshape(n, e.HiddenSize)
		att = m.norm(h.Add(m.linear(att, p+"attention.output.dense")), p+"attention.output.LayerNorm")
		h = m.norm(att.Add(m.linear(mlx.GELU(m.linear(att, p+"intermediate.dense")), p+"output.dense")), p+"output.LayerNorm")
	}
	if e.HiddenSize != m.cfg.HiddenSize {
		h = m.linear(h, "token_rep_layer.projection")
	}
	return h, nil
}

func relativePositions(n, buckets, maxPosition int) ([]int32, []int32) {
	a, b := make([]int32, n*n), make([]int32, n*n)
	mid := buckets / 2
	for i := range n {
		for j := range n {
			d := i - j
			abs := int(math.Abs(float64(d)))
			if abs > mid {
				v := int(math.Ceil(math.Log(float64(abs)/float64(mid))/math.Log(float64(maxPosition-1)/float64(mid))*float64(mid-1))) + mid
				if d < 0 {
					d = -v
				} else {
					d = v
				}
			}
			a[i*n+j] = int32(min(max(d+buckets, 0), 2*buckets-1))
			b[i*n+j] = int32(min(max(-d+buckets, 0), 2*buckets-1))
		}
	}
	return a, b
}

func (m *Model) recurrent(ctx context.Context, words *mlx.Array) (*mlx.Array, error) {
	var directions []*mlx.Array
	n, width := words.Dim(0), m.cfg.HiddenSize/2
	for direction, suffix := range []string{"", "_reverse"} {
		x := m.Weights["rnn.lstm.bias_ih_l0"+suffix].Addmm(words, m.Weights["rnn.lstm.weight_ih_l0"+suffix].Transpose(1, 0), 1, 1)
		h, c := mlx.Zeros(mlx.DTypeFloat32, 1, width), mlx.Zeros(mlx.DTypeFloat32, 1, width)
		rows := make([]*mlx.Array, n)
		for step := range n {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			i := step
			if direction == 1 {
				i = n - 1 - step
			}
			gates := m.Weights["rnn.lstm.bias_hh_l0"+suffix].Addmm(h, m.Weights["rnn.lstm.weight_hh_l0"+suffix].Transpose(1, 0), 1, 1).Add(x.Slice(mlx.Slice(i, i+1), mlx.Slice()))
			gate := func(g int) *mlx.Array { return gates.Slice(mlx.Slice(), mlx.Slice(g*width, (g+1)*width)) }
			c = gate(1).Sigmoid().Multiply(c).Add(gate(0).Sigmoid().Multiply(gate(2).Tanh()))
			h = gate(3).Sigmoid().Multiply(c.Tanh())
			rows[i] = h
		}
		directions = append(directions, rows[0].Concatenate(0, rows[1:]...))
	}
	return directions[0].Concatenate(1, directions[1]), nil
}

func badInput(format string, args ...any) error {
	return api.StatusError{StatusCode: http.StatusBadRequest, ErrorMessage: fmt.Sprintf(format, args...)}
}

func (m *Model) Extract(ctx context.Context, req api.ExtractRequest) (*api.ExtractResponse, error) {
	if err := req.Validate(); err != nil {
		return nil, badInput("%s", err)
	}
	p, err := m.prepare(req.Input, req.Labels)
	if err != nil {
		return nil, err
	}
	out := &api.ExtractResponse{Entities: []api.Entity{}, PromptEvalCount: len(p.ids)}
	if len(p.words) == 0 {
		return out, nil
	}
	probabilities := mlx.ScopedArrays(func() []*mlx.Array {
		var logits *mlx.Array
		logits, err = m.forward(ctx, p)
		if err != nil {
			return nil
		}
		return []*mlx.Array{logits.Sigmoid()}
	})
	if err != nil {
		return nil, err
	}
	probs := probabilities[0].Floats()
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	out.Entities = decode(req.Input, req.Labels, p.words, p.spans, probs, req.ScoreThreshold())
	return out, nil
}

func (m *Model) forward(ctx context.Context, p *prepared) (*mlx.Array, error) {
	h, err := m.encode(ctx, p.ids)
	if err != nil {
		return nil, err
	}
	words := h.TakeAxis(mlx.FromValues(p.wordIndices, len(p.wordIndices)), 0)
	if m.cfg.HasRNN {
		words, err = m.recurrent(ctx, words)
		if err != nil {
			return nil, err
		}
	}
	labels := m.project(h.TakeAxis(mlx.FromValues(p.labelIndices, len(p.labelIndices)), 0), "prompt_rep_layer")
	starts, ends := make([]int32, len(p.spans)), make([]int32, len(p.spans))
	for i, span := range p.spans {
		starts[i], ends[i] = int32(span[0]), int32(span[1])
	}
	start := m.project(words, spanPrefix+"project_start").TakeAxis(mlx.FromValues(starts, len(starts)), 0)
	end := m.project(words, spanPrefix+"project_end").TakeAxis(mlx.FromValues(ends, len(ends)), 0)
	spans := m.project(relu(start.Concatenate(1, end)), spanPrefix+"out_project")
	return spans.Matmul(labels.Transpose(1, 0)), nil
}
