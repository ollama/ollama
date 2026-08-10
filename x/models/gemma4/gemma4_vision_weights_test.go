package gemma4

import (
	"fmt"
	"testing"

	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model"
)

func testVisionTensor(shape ...int) *mlx.Array {
	n := 1
	for _, d := range shape {
		n *= d
	}
	values := make([]float32, n)
	for i := range values {
		values[i] = float32(i%7+1) / 16
	}
	return mlx.FromValues(values, shape...)
}

func testVisionProjection(tensors map[string]*mlx.Array, textHidden, visionHidden int) {
	tensors["model.embed_vision.embedding_projection.weight"] = testVisionTensor(textHidden, visionHidden)
}

func TestGemma4VisionWeightSchemes(t *testing.T) {
	skipIfNoMLX(t)

	t.Run("12b vision_embedder", func(t *testing.T) {
		cfg := &VisionConfig{
			ModelType:             "gemma4_unified_vision",
			HiddenSize:            4,
			PositionEmbeddingSize: 4,
			PatchSize:             2,
			PoolingKernelSize:     1,
			ImageSeqLength:        1,
			RMSNormEps:            1e-6,
		}
		m := &Model{
			TextConfig:   &TextConfig{HiddenSize: 6},
			VisionConfig: cfg,
			VisionEncoder: &VisionEncoder{
				Cfg: cfg,
			},
		}
		const prefix = "model.vision_embedder."
		tensors := map[string]*mlx.Array{
			prefix + "patch_ln1.weight":   testVisionTensor(12),
			prefix + "patch_ln1.bias":     testVisionTensor(12),
			prefix + "patch_dense.weight": testVisionTensor(4, 12),
			prefix + "patch_ln2.weight":   testVisionTensor(4),
			prefix + "patch_ln2.bias":     testVisionTensor(4),
			prefix + "pos_embedding":      testVisionTensor(4, 2, 4),
			prefix + "pos_norm.weight":    testVisionTensor(4),
			prefix + "pos_norm.bias":      testVisionTensor(4),
		}
		testVisionProjection(tensors, 6, 4)
		linears := model.NewLinearFactory(tensors, 0, 0, "", nil)
		if err := m.loadVisionWeights(tensors, linears); err != nil {
			t.Fatal(err)
		}
		if m.VisionEncoder.UnifiedPatchEmbedder == nil || m.VisionEncoder.PatchEmbedder != nil {
			t.Fatal("unified vision embedder was not selected")
		}
		assertVisionForwardShape(t, m, 6)
	})

	t.Run("26b 31b vision_tower", func(t *testing.T) {
		cfg := &VisionConfig{
			ModelType:             "gemma4_vision",
			HiddenSize:            4,
			IntermediateSize:      8,
			NumHiddenLayers:       1,
			NumAttentionHeads:     1,
			NumKeyValueHeads:      1,
			HeadDim:               4,
			PatchSize:             2,
			PositionEmbeddingSize: 4,
			PoolingKernelSize:     1,
			ImageSeqLength:        1,
			RMSNormEps:            1e-6,
			RopeTheta:             100,
			Scale:                 1,
		}
		m := &Model{
			TextConfig:   &TextConfig{HiddenSize: 6},
			VisionConfig: cfg,
			VisionEncoder: &VisionEncoder{
				Cfg: cfg,
			},
		}
		const tower = "model.vision_tower."
		tensors := map[string]*mlx.Array{
			tower + "patch_embedder.input_proj.weight":        testVisionTensor(4, 12),
			tower + "patch_embedder.position_embedding_table": testVisionTensor(2, 4, 4),
		}
		layer := tower + "encoder.layers.0"
		for _, name := range []string{
			"input_layernorm.weight",
			"post_attention_layernorm.weight",
			"pre_feedforward_layernorm.weight",
			"post_feedforward_layernorm.weight",
			"self_attn.q_norm.weight",
			"self_attn.k_norm.weight",
		} {
			tensors[layer+"."+name] = testVisionTensor(4)
		}
		for _, name := range []string{"q_proj", "k_proj", "v_proj", "o_proj"} {
			tensors[fmt.Sprintf("%s.self_attn.%s.linear.weight", layer, name)] = testVisionTensor(4, 4)
		}
		for _, name := range []string{"gate_proj", "up_proj"} {
			tensors[fmt.Sprintf("%s.mlp.%s.linear.weight", layer, name)] = testVisionTensor(8, 4)
		}
		tensors[layer+".mlp.down_proj.linear.weight"] = testVisionTensor(4, 8)
		testVisionProjection(tensors, 6, 4)

		linears := model.NewLinearFactory(tensors, 0, 0, "", nil)
		if err := m.loadVisionWeights(tensors, linears); err != nil {
			t.Fatal(err)
		}
		precomputeVisionScaledWeights(m.VisionEncoder)
		if m.VisionEncoder.PatchEmbedder == nil || len(m.VisionEncoder.Layers) != 1 {
			t.Fatal("transformer vision tower was not selected")
		}
		assertVisionForwardShape(t, m, 6)
	})
}

func assertVisionForwardShape(t *testing.T, m *Model, hidden int) {
	t.Helper()
	pixels := testVisionTensor(1, 3, 2, 2)
	out := m.ForwardVision(pixels)
	mlx.Eval(out)
	if got := out.Dims(); len(got) != 3 || got[0] != 1 || got[1] != 1 || got[2] != hidden {
		t.Fatalf("vision output shape = %v, want [1 1 %d]", got, hidden)
	}
}
