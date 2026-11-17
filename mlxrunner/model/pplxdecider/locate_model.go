package pplxdecider

import (
	"fmt"
	"slices"
	"strings"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/model/qwen3_5"
	"github.com/ollama/ollama/mlxrunner/nn"
)

const locatePromptPrefix = "Question:\nLocate verbatim spans:\n"

type locateConfig struct {
	Format   string `json:"format"`
	Marker   string `json:"marker"`
	MarkerID int32  `json:"marker_id"`
	Prompt   string `json:"prompt"`
	LoRA     struct {
		Rank   int     `json:"rank"`
		Alpha  float64 `json:"alpha"`
		Layers int     `json:"layers"`
		Merged bool    `json:"merged"`
	} `json:"lora"`
	Head struct {
		Width     int `json:"width"`
		MaxSpan   int `json:"max_span"`
		Proposals int `json:"proposals"`
	} `json:"head"`
	Thresholds struct {
		Default float32 `json:"default"`
	} `json:"thresholds"`
}

func (m *Model) validateLocateConfig() error {
	cfg := m.locate
	if cfg.Format != "pplx-decider-locate-span-v1" || cfg.Marker != "<|locate|>" ||
		cfg.MarkerID < 0 || cfg.MarkerID >= m.Config.VocabSize ||
		cfg.Prompt != locatePromptPrefix+"{instructions}"+cfg.Marker ||
		cfg.LoRA.Rank != 64 || cfg.LoRA.Alpha != 64 || cfg.LoRA.Layers != 0 || cfg.LoRA.Merged ||
		cfg.Head.Width != 128 || cfg.Head.MaxSpan != 128 || cfg.Head.Proposals != 128 ||
		cfg.Thresholds.Default <= 0 || cfg.Thresholds.Default >= 1 {
		return fmt.Errorf("unsupported Perplexity Locate configuration")
	}
	// Locate inserts the trained marker ID directly into its question tokens.
	// It must fit the embedding table without overlapping the tokenizer vocabulary.
	if int(cfg.MarkerID) < m.Tokenizer().VocabSize() {
		return fmt.Errorf("Perplexity Locate marker row %d overlaps the tokenizer vocabulary", cfg.MarkerID)
	}
	return nil
}

type locateLinear interface {
	setEnabled(bool)
}

type lowRankLinear struct {
	Base    nn.LinearLayer
	A       *mlx.Array
	B       *mlx.Array
	Enabled bool
}

func (l *lowRankLinear) setEnabled(enabled bool) { l.Enabled = enabled }
func (l *lowRankLinear) OutputDim() int32        { return l.Base.OutputDim() }

func (l *lowRankLinear) Forward(x *mlx.Array) *mlx.Array {
	base := l.Base.Forward(x)
	if !l.Enabled {
		return base
	}
	update := mlx.Matmul(mlx.Matmul(x, l.A.Transpose(1, 0)), l.B.Transpose(1, 0))
	return mlx.Add(base, update.AsType(base.DType()))
}

type packedLowRankLinear struct {
	Base    nn.LinearLayer
	First   *lowRankLinear
	Second  *lowRankLinear
	Enabled bool
}

func (l *packedLowRankLinear) setEnabled(enabled bool) { l.Enabled = enabled }
func (l *packedLowRankLinear) OutputDim() int32        { return l.Base.OutputDim() }

func (l *packedLowRankLinear) Forward(x *mlx.Array) *mlx.Array {
	base := l.Base.Forward(x)
	if !l.Enabled {
		return base
	}
	part := func(a, b *mlx.Array) *mlx.Array {
		return mlx.Matmul(mlx.Matmul(x, a.Transpose(1, 0)), b.Transpose(1, 0))
	}
	update := mlx.Concatenate([]*mlx.Array{
		part(l.First.A, l.First.B), part(l.Second.A, l.Second.B),
	}, -1)
	return mlx.Add(base, update.AsType(base.DType()))
}

type locateEmbedding struct {
	Base    nn.EmbeddingLayer
	Vector  *mlx.Array
	Marker  int32
	Enabled bool
}

func (e *locateEmbedding) AsLinear() nn.LinearLayer { return e.Base.AsLinear() }

func (e *locateEmbedding) Forward(ids *mlx.Array) *mlx.Array {
	base := e.Base.Forward(ids)
	if !e.Enabled {
		return base
	}
	marker := mlx.FromValues([]int32{e.Marker}, 1)
	selected := mlx.ExpandDims(ids.Equal(marker), -1)
	return mlx.Where(selected, e.Vector.AsType(base.DType()), base)
}

type spanHead struct {
	TS, TE, QS, QE, QP *nn.Linear
	Thresholds         *nn.Linear
	WidthBias          *mlx.Array
	StartBias          *mlx.Array
	EndBias            *mlx.Array
}

func (m *Model) loadLocateWeights(tensors map[string]*mlx.Array) error {
	consumed := map[string]bool{}
	get := func(name string, shape ...int) (*mlx.Array, error) {
		key := "locate." + name
		value := tensors[key]
		if value == nil || value.DType() != mlx.DTypeFloat32 || !slices.Equal(value.Dims(), shape) {
			return nil, fmt.Errorf("Perplexity Locate tensor %s must be FP32 %v", key, shape)
		}
		consumed[key] = true
		return value, nil
	}
	load := func(name string, base nn.LinearLayer) (*lowRankLinear, error) {
		if base == nil {
			return nil, fmt.Errorf("Perplexity Locate has no base projection %s", name)
		}
		prefix := "adapter.language_model." + name
		a := tensors["locate."+prefix+".lora_a"]
		b := tensors["locate."+prefix+".lora_b"]
		inputWidth := m.Config.HiddenSize
		switch {
		case strings.HasSuffix(name, ".linear_attn.out_proj"):
			inputWidth = m.Config.LinearNumValueHeads * m.Config.LinearValueHeadDim
		case strings.HasSuffix(name, ".self_attn.o_proj"):
			inputWidth = m.Config.NumAttentionHeads * m.Config.HeadDim
		case strings.HasSuffix(name, ".mlp.down_proj"):
			inputWidth = m.Config.IntermediateSize
		}
		if a == nil || b == nil || a.DType() != mlx.DTypeFloat32 || b.DType() != mlx.DTypeFloat32 ||
			a.NumDims() != 2 || b.NumDims() != 2 || a.Dim(0) != m.locate.LoRA.Rank || a.Dim(1) != int(inputWidth) ||
			b.Dim(1) != m.locate.LoRA.Rank || b.Dim(0) == 0 || int32(b.Dim(0)) > base.OutputDim() {
			return nil, fmt.Errorf("Perplexity Locate adapter %s has incompatible shape", name)
		}
		consumed["locate."+prefix+".lora_a"] = true
		consumed["locate."+prefix+".lora_b"] = true
		// The publisher evaluates these FP32 weights under BF16 autocast.
		return &lowRankLinear{Base: base, A: a.AsType(mlx.DTypeBFloat16), B: b.AsType(mlx.DTypeBFloat16)}, nil
	}
	wrap := func(base *nn.LinearLayer, name string) error {
		adapter, err := load(name, *base)
		if err != nil {
			return err
		}
		if adapter.B.Dim(0) != int(adapter.Base.OutputDim()) {
			return fmt.Errorf("Perplexity Locate adapter %s has wrong output width", name)
		}
		*base = adapter
		m.locateLoRA = append(m.locateLoRA, adapter)
		return nil
	}
	wrapPair := func(base *nn.LinearLayer, first, second string) error {
		a, err := load(first, *base)
		if err != nil {
			return err
		}
		b, err := load(second, *base)
		if err != nil {
			return err
		}
		if a.B.Dim(0)+b.B.Dim(0) != int((*base).OutputDim()) {
			return fmt.Errorf("Perplexity Locate packed adapter %s/%s has wrong width", first, second)
		}
		adapter := &packedLowRankLinear{Base: *base, First: a, Second: b}
		*base = adapter
		m.locateLoRA = append(m.locateLoRA, adapter)
		return nil
	}
	for i, layer := range m.Layers {
		prefix := fmt.Sprintf("layers.%d.", i)
		if layer.IsLinear {
			if err := wrapPair(&layer.Linear.InProjQKVZ, prefix+"linear_attn.in_proj_qkv", prefix+"linear_attn.in_proj_z"); err != nil {
				return err
			}
			if err := wrapPair(&layer.Linear.InProjBA, prefix+"linear_attn.in_proj_b", prefix+"linear_attn.in_proj_a"); err != nil {
				return err
			}
			if err := wrap(&layer.Linear.OutProj, prefix+"linear_attn.out_proj"); err != nil {
				return err
			}
		} else {
			for _, item := range []struct {
				base *nn.LinearLayer
				name string
			}{
				{&layer.FullAttn.QProj, "q_proj"},
				{&layer.FullAttn.KProj, "k_proj"},
				{&layer.FullAttn.VProj, "v_proj"},
				{&layer.FullAttn.OProj, "o_proj"},
			} {
				if err := wrap(item.base, prefix+"self_attn."+item.name); err != nil {
					return err
				}
			}
		}
		mlp, ok := layer.MLP.(*qwen3_5.DenseMLP)
		if !ok {
			return fmt.Errorf("Perplexity Locate requires dense layer %d", i)
		}
		for _, item := range []struct {
			base *nn.LinearLayer
			name string
		}{
			{&mlp.GateProj, "gate_proj"}, {&mlp.UpProj, "up_proj"}, {&mlp.DownProj, "down_proj"},
		} {
			if err := wrap(item.base, prefix+"mlp."+item.name); err != nil {
				return err
			}
		}
	}
	marker, err := get("adapter.language_model.embed_tokens.locate", int(m.Config.HiddenSize))
	if err != nil {
		return err
	}
	m.locateEmbed = &locateEmbedding{Base: m.EmbedTokens, Vector: marker, Marker: m.locate.MarkerID}
	m.EmbedTokens = m.locateEmbed
	loadHeadLinear := func(name string, out int) (*nn.Linear, error) {
		weight, err := get("head."+name+".weight", out, int(m.Config.HiddenSize))
		if err != nil {
			return nil, err
		}
		bias, err := get("head."+name+".bias", out)
		if err != nil {
			return nil, err
		}
		return nn.NewLinear(weight, bias), nil
	}
	head := &spanHead{}
	for _, item := range []struct {
		name string
		into **nn.Linear
	}{
		{"ts", &head.TS},
		{"te", &head.TE},
		{"qs", &head.QS},
		{"qe", &head.QE},
		{"qp", &head.QP},
		{"thresholds", &head.Thresholds},
	} {
		width := m.locate.Head.Width
		if item.name == "thresholds" {
			width = 3
		}
		linear, err := loadHeadLinear(item.name, width)
		if err != nil {
			return err
		}
		*item.into = linear
	}
	if head.WidthBias, err = get("head.width_bias.weight", 32, 1); err != nil {
		return err
	}
	if head.StartBias, err = get("head.start_bias"); err != nil {
		return err
	}
	if head.EndBias, err = get("head.end_bias"); err != nil {
		return err
	}
	for name := range tensors {
		if strings.HasPrefix(name, "locate.") && !consumed[name] {
			return fmt.Errorf("unexpected Perplexity Locate tensor %s", name)
		}
	}
	m.LocateHead = head
	return nil
}
