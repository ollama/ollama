package hy4_flash

import (
	"math"
	"strings"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/cache"
)

func TestTextForwardChunking(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		mlx.Scoped(func() {
			const width = 8
			cfg := textConfig{
				HiddenSize: width, Streams: 2, Layers: 2, RMSEps: 1e-5, HCEps: 1e-6, HCMagnitude: 2,
				AttentionTypes: []string{indexedAttention, slidingAttention},
				MLPTypes:       []string{"dense", "sparse"},
				Indexed:        attentionShape{heads: 2, plain: 2, rotary: 2, value: 2, theta: 10000},
				Sliding:        attentionShape{heads: 2, plain: 2, rotary: 2, value: 2, theta: 10000},
				QRank:          4, KVRank: 4, Window: 3, IndexRatio: 2, IndexTopK: 2, IndexHeads: 2, IndexDim: 4,
				Experts: 3, ExpertsPerTok: 2, LatentWidth: 4, ExpertWidth: 4, NormalizeRoute: true, RouteScale: 2, ExpertClamp: 10,
			}
			m := &Model{cfg: config{Text: cfg}, Layers: make([]decoderLayer, 2), tokenSlot: -1}
			data := map[string]*mlx.Array{}
			fill := func(name string, dims ...int) {
				count := 1
				for _, dim := range dims {
					count *= dim
				}
				values := make([]float32, count)
				for i := range values {
					values[i] = 0.2 + 0.15*float32(math.Sin(float64(i+len(data))*0.72))
				}
				a := mlx.FromValues(values, dims...)
				isFP32 := strings.Contains(name, "hc_") || strings.HasSuffix(name, "sink_param") ||
					strings.Contains(name, "key_value_projection") || strings.Contains(name, "key_gate_projection") ||
					strings.HasSuffix(name, "intra_group_ape") || strings.HasSuffix(name, "e_score_correction_bias")
				if !isFP32 {
					a = a.AsType(mlx.DTypeBFloat16)
				}
				data[name] = a
			}
			fill("model.embed_tokens.weight", 16, width)
			fill("lm_head.weight", 16, width)
			fill("model.norm.weight", width)
			fill("model.hc_head.hc_head_fn", 2, 16)
			fill("model.hc_head.hc_head_base", 2)
			fill("model.hc_head.hc_head_scale", 1)
			for _, prefix := range []string{"model.layers.0", "model.layers.1"} {
				for _, sub := range []string{".hc_attn_layer.hc_pre.hc_", ".hc_mlp_layer.hc_pre.hc_"} {
					fill(prefix+sub+"fn", 4, 16)
					fill(prefix+sub+"base", 4)
					fill(prefix+sub+"scale", 2)
				}
				fill(prefix+".input_layernorm.weight", width)
				fill(prefix+".post_attention_layernorm.weight", width)
				a := prefix + ".self_attn"
				fill(a+".q_a_proj.weight", 4, width)
				fill(a+".q_b_proj.weight", 8, 4)
				fill(a+".q_a_layernorm.weight", 4)
				fill(a+".kv_a_proj_with_mqa.weight", 6, width)
				fill(a+".kv_a_layernorm.weight", 4)
				fill(a+".kv_b_proj.weight", 8, 4)
				fill(a+".linear_gate.weight", 4, width)
				fill(a+".o_proj.weight", width, 4)
				fill(a+".post_proj_norm.weight", width)
				fill(a+".learnable_sink_param", 2)
			}
			index := "model.layers.0.self_attn.compressed_indexer"
			fill(index+".query_projection.weight", 8, 4)
			fill(index+".head_weight_projection.weight", 2, width)
			fill(index+".token_key_projection.weight", 4, width)
			fill(index+".token_key_norm.weight", 4)
			fill(index+".key_value_projection.weight", 4, 4)
			fill(index+".key_gate_projection.weight", 4, 4)
			fill(index+".key_norm.weight", 4)
			fill(index+".intra_group_ape", 2, 4)
			dense := "model.layers.0.mlp"
			fill(dense+".gate_proj.weight", 4, width)
			fill(dense+".up_proj.weight", 4, width)
			fill(dense+".down_proj.weight", width, 4)
			fill(dense+".post_fc2_norm.weight", width)
			moe := "model.layers.1.mlp"
			fill(moe+".experts.gate_up_proj", 3, 8, 4)
			fill(moe+".experts.down_proj", 3, 4, 4)
			fill(moe+".gate.weight", 3, width)
			fill(moe+".gate.e_score_correction_bias", 3)
			fill(moe+".routed_expert_down_proj.weight", 4, width)
			fill(moe+".routed_expert_up_proj.weight", width, 4)
			fill(moe+".shared_experts.gate_proj.weight", 4, width)
			fill(moe+".shared_experts.up_proj.weight", 4, width)
			fill(moe+".shared_experts.down_proj.weight", width, 4)
			fill(moe+".combine_output_norm.weight", width)
			if err := m.LoadWeights(data); err != nil {
				t.Fatal(err)
			}
			if got := len(mlx.Collect(m)); got < len(data)-2 {
				t.Fatalf("runner collected only %d arrays from %d checkpoint tensors", got, len(data))
			}

			run := func(ids []int32, start int, slots []cache.Layer) *mlx.Array {
				b := &batch.Batch{InputIDs: mlx.FromValues(ids, 1, len(ids)), SeqOffsets: []int32{int32(start)}, SeqQueryLens: []int32{int32(len(ids))}}
				h, _ := m.Forward(b, slots)
				return m.Unembed(h)
			}
			ids := []int32{2, 1, 5, 3, 9, 1, 4}
			full := run(ids, 0, nil)
			slots := m.NewCaches()
			for _, slot := range slots {
				defer slot.Free()
			}
			layers := cache.Layers(slots)
			part1 := run(ids[:3], 0, layers)
			part2 := run(ids[3:6], 3, layers)
			part3 := run(ids[6:], 6, layers)
			chunked := mlx.Concatenate([]*mlx.Array{part1, part2, part3}, 1)
			want, got := full.Floats(), chunked.Floats()
			for i := range want {
				if delta := math.Abs(float64(want[i] - got[i])); math.IsNaN(delta) || delta > 0.015 {
					t.Fatalf("chunked logit %d (token %d) = %g, unchunked = %g", i, i/16, got[i], want[i])
				}
			}
		})
	})
}
