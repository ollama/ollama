package gemma

import (
	"cmp"
	"encoding/json"

	"github.com/ollama/ollama/types/model"
	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model/base"
)

type Model struct {
	Text   TextModel  `weight:"language_model"`
}

func (m *Model) NumLayers() int {
	return len(m.Text.Layers)
}

func (m Model) Cache() []cache.Cache {
	caches := make([]cache.Cache, m.NumLayers())
	for i := range caches {
		if (i+1)%m.Text.Options.SlidingWindowPattern == 0 {
			caches[i] = cache.NewKVCache()
		} else {
			caches[i] = cache.NewRotatingKVCache(m.Text.Options.SlidingWindow)
		}
	}
	return caches
}

func (m *Model) Forward(inputs *mlx.Array, cache []cache.Cache) *mlx.Array {
	return m.Text.Forward(inputs, cache)
}

func (m *Model) Unembed(x *mlx.Array) *mlx.Array {
	return m.Text.EmbedTokens.AsLinear().Forward(x)
}

func init() {
	base.Register("Gemma3ForConditionalGeneration", func(root *model.Root) (base.Model, error) {
		bts, err := root.ReadFile("config.json")
		if err != nil {
			return nil, err
		}

		var opts struct {
			Text TextOptions `json:"text_config"`
		}

		if err := json.Unmarshal(bts, &opts); err != nil {
			return nil, err
		}

		opts.Text.NumAttentionHeads = cmp.Or(opts.Text.NumAttentionHeads, 8)
		opts.Text.NumKeyValueHeads = cmp.Or(opts.Text.NumKeyValueHeads, 4)
		opts.Text.HeadDim = cmp.Or(opts.Text.HeadDim, 256)
		opts.Text.RMSNormEps = cmp.Or(opts.Text.RMSNormEps, 1e-6)
		opts.Text.SlidingWindowPattern = cmp.Or(opts.Text.SlidingWindowPattern, 6)

		// TODO: implement json.Unmarshaler
		opts.Text.RoPE = map[bool]mlx.RoPE{
			true:  {Dims: opts.Text.HeadDim, Traditional: false, Base: 1_000_000, Scale: 1. / 8.},
			false: {Dims: opts.Text.HeadDim, Traditional: false, Base: 10_000, Scale: 1},
		}

		return &Model{
			Text: TextModel{
				Layers:  make([]TextDecoderLayer, opts.Text.NumHiddenLayers),
				Options: opts.Text,
			},
		}, nil
	})
}

type RMSNorm struct {
	mlx.RMSNorm
}

func (m *RMSNorm) AfterLoad(*model.Root) error {
	m.Weight.Set(m.Weight.Add(mlx.FromValue(1)))
	return nil
}
