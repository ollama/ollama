package deepseekocr

import (
	"math"
	"slices"

	"github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/kvcache"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
	"github.com/ollama/ollama/model"
	"github.com/ollama/ollama/model/input"
)

type Model struct {
	model.Base
	model.TextProcessor

	Sam    *samModel    `gguf:"s"`
	Vision *visionModel `gguf:"v"`
	Text   *textModel

	ImageNewline ml.Tensor `gguf:"mm.image_newline"`
	//nolint:misspell // this misspelling is upstream. fixing it breaks the model
	ViewSeperator ml.Tensor `gguf:"mm.view_seperator"`

	Projector *nn.Linear `gguf:"mm.layers"`
}

func (m *Model) EncodeMultimodal(ctx ml.Context, bts []byte) ([]input.Multimodal, error) {
	patches, original, crop, err := ProcessImage(ctx, bts)
	if err != nil {
		return nil, err
	}

	var outputs []ml.Tensor
	if true { // TODO: local features if sum(patches) != 0
		samOutputs := m.Sam.Forward(ctx, patches)
		visionOutputs := m.Vision.Forward(ctx, patches, samOutputs)

		samOutputs = samOutputs.Reshape(ctx, -1, samOutputs.Dim(2), samOutputs.Dim(3)).Permute(ctx, 1, 0, 2, 3)
		visionOutputs = visionOutputs.Slice(ctx, 1, 1, visionOutputs.Dim(1), 1)
		localOutputs := visionOutputs.Concat(ctx, samOutputs, 0)
		localOutputs = m.Projector.Forward(ctx, localOutputs)

		hw := int(math.Sqrt(float64(localOutputs.Dim(1))))
		localOutputs = localOutputs.Reshape(ctx, -1, hw, crop[0], crop[1])
		localOutputs = localOutputs.Permute(ctx, 0, 2, 1, 3)
		localOutputs = localOutputs.Contiguous(ctx, -1, crop[0]*hw, crop[1]*hw)
		localOutputs = localOutputs.Concat(ctx, m.ImageNewline.Repeat(ctx, 2, localOutputs.Dim(2)), 1)
		localOutputs = localOutputs.Reshape(ctx, localOutputs.Dim(0), -1)

		outputs = append(outputs, localOutputs)
	}

	samOutputs := m.Sam.Forward(ctx, original)
	visionOutputs := m.Vision.Forward(ctx, original, samOutputs)

	samOutputs = samOutputs.Reshape(ctx, -1, samOutputs.Dim(2), samOutputs.Dim(3)).Permute(ctx, 1, 0, 2, 3)
	visionOutputs = visionOutputs.Slice(ctx, 1, 1, visionOutputs.Dim(1), 1)
	globalOutputs := visionOutputs.Concat(ctx, samOutputs, 0)
	globalOutputs = m.Projector.Forward(ctx, globalOutputs)

	hw := int(math.Sqrt(float64(globalOutputs.Dim(1))))
	globalOutputs = globalOutputs.Reshape(ctx, -1, hw, hw)
	globalOutputs = globalOutputs.Concat(ctx, m.ImageNewline.Repeat(ctx, 2, globalOutputs.Dim(2)), 1)
	globalOutputs = globalOutputs.Reshape(ctx, globalOutputs.Dim(0), -1)

	outputs = append(outputs, globalOutputs, m.ViewSeperator)
	return []input.Multimodal{
		{Tensor: outputs[0].Stack(ctx, 1, outputs[1:]...)},
	}, nil
}

func (m *Model) PostTokenize(inputs []*input.Input) ([]*input.Input, error) {
	outputs := make([]*input.Input, 0, len(inputs))
	for i := range inputs {
		if inputs[i].Multimodal == nil {
			outputs = append(outputs, inputs[i])
			continue
		}

		t := inputs[i].Multimodal[0].Tensor
		outputs = append(outputs, &input.Input{
			Token:          128815,
			Multimodal:     inputs[i].Multimodal,
			MultimodalHash: inputs[i].MultimodalHash,
			SameBatch:      t.Dim(1) - 1,
		})

		outputs = slices.Grow(outputs, t.Dim(1)-1)
		outputs = append(outputs, slices.Repeat([]*input.Input{{Token: 128815}}, t.Dim(1)-1)...)
	}
	return outputs, nil
}

func (m *Model) Forward(ctx ml.Context, batch input.Batch) (ml.Tensor, error) {
	inputsEmbeds := m.Text.TokenEmbedding.Forward(ctx, batch.Inputs).Duplicate(ctx)
	positions := ctx.Input().FromInts(batch.Positions, len(batch.Positions))

	for _, mm := range batch.Multimodal {
		t := mm.Multimodal[0].Tensor
		ctx.Forward(t.Copy(ctx, inputsEmbeds.View(ctx, mm.Index*inputsEmbeds.Stride(1), t.Dim(0)*t.Dim(1))))
	}

	hiddenStates := inputsEmbeds
	for i, block := range m.Text.Blocks {
		if m.Cache != nil {
			m.Cache.SetLayer(i)
		}

		var outputs ml.Tensor
		if i == len(m.Text.Blocks)-1 {
			outputs = batch.Outputs
		}

		hiddenStates = block.Forward(ctx, hiddenStates, positions, outputs, m.Cache, m.Text.Options)
	}

	hiddenStates = m.Text.OutputNorm.Forward(ctx, hiddenStates, m.Text.Options.eps)
	return m.Text.Output.Forward(ctx, hiddenStates), nil
}

func init() {
	model.Register("deepseekocr", func(c fs.Config) (model.Model, error) {
		textBlocks := make([]textBlock, c.Uint("block_count"))
		leadingDenseBlockCount := int(c.Uint("leading_dense_block_count", 1))
		for i := range textBlocks {
			if i >= leadingDenseBlockCount {
				textBlocks[i].FeedForward = &textMoe{}
			} else {
				textBlocks[i].FeedForward = &textMLP{}
			}
		}

		m := Model{
			TextProcessor: model.NewBytePairEncoding(
				&model.Vocabulary{
					Values: c.Strings("tokenizer.ggml.tokens"),
					Types:  c.Ints("tokenizer.ggml.token_type"),
					Merges: c.Strings("tokenizer.ggml.merges"),
					AddBOS: c.Bool("tokenizer.ggml.add_bos_token", true),
					BOS:    []int32{int32(c.Uint("tokenizer.ggml.bos_token_id"))},
					AddEOS: c.Bool("tokenizer.ggml.add_eos_token", false),
					EOS: append(
						[]int32{int32(c.Uint("tokenizer.ggml.eos_token_id"))},
						c.Ints("tokenizer.ggml.eos_token_ids")...,
					),
				},
				// Split regex into multiple parts (according to DeepSeek3's regex)
				"\\p{N}{1,3}",
				`[一-龥぀-ゟ゠-ヿ]+`,
				"[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+",
			),
			Text: &textModel{
				Blocks: textBlocks,
				Options: textOptions{
					hiddenSize:     int(c.Uint("embedding_length")),
					numHeads:       int(c.Uint("attention.head_count")),
					numKVHeads:     int(c.Uint("attention.head_count_kv")),
					numExperts:     int(c.Uint("expert_count")),
					numExpertsUsed: int(c.Uint("expert_used_count")),
					ropeBase:       c.Float("rope.freq_base", 10_000),
					ropeScale:      c.Float("rope.scaling.factor", 1.0),
					eps:            c.Float("attention.layer_norm_rms_epsilon", 1e-6),
				},
			},
			Vision: &visionModel{
				Blocks: make([]visionBlock, c.Uint("vision.block_count")),
				Options: visionOptions{
					hiddenSize: int(c.Uint("vision.embedding_length")),
					numHeads:   int(c.Uint("vision.head_count")),
					imageSize:  int(c.Uint("vision.image_size", 224)),
					patchSize:  int(c.Uint("vision.patch_size", 14)),
					eps:        c.Float("vision.attention.layer_norm_epsilon", 1e-5),
				},
			},
			Sam: &samModel{
				Blocks: make([]samBlock, c.Uint("sam.block_count")),
				Options: samOptions{
					hiddenSize:            int(c.Uint("sam.embedding_length")),
					numHeads:              int(c.Uint("sam.head_count")),
					eps:                   c.Float("sam.attention.layer_norm_epsilon", 1e-6),
					globalAttentionLayers: c.Ints("sam.global_attention_indexes"),
				},
			},
		}

		m.Cache = kvcache.NewCausalCache(m.Text.Shift)
		return &m, nil
	})
}
