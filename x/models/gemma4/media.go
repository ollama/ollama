package gemma4

import (
	"encoding/json"
	"fmt"
	"log/slog"

	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model"
	"github.com/ollama/ollama/x/mlxrunner/model/base"
	"github.com/ollama/ollama/x/models/nn"
)

// multimodalConfig carries the top-level config.json fields the text_config
// unwrap in parseTextConfig discards.
type multimodalConfig struct {
	VisionConfig             *VisionConfig `json:"vision_config"`
	ImageTokenID             int32         `json:"image_token_id"`
	BOITokenID               int32         `json:"boi_token_id"`
	EOITokenID               int32         `json:"eoi_token_id"`
	VisionSoftTokensPerImage int32         `json:"vision_soft_tokens_per_image"`

	AudioConfig  *AudioConfig `json:"audio_config"`
	AudioTokenID int32        `json:"audio_token_id"`
	BOATokenID   int32        `json:"boa_token_id"`
	EOATokenID   int32        `json:"eoa_token_id"`
	// Some checkpoints carry the end-of-audio token only under this key.
	EOATokenIndex int32 `json:"eoa_token_index"`
}

func parseMultimodalConfig(configData []byte) (multimodalConfig, error) {
	var mm multimodalConfig
	if err := json.Unmarshal(configData, &mm); err != nil {
		return multimodalConfig{}, fmt.Errorf("parse multimodal config: %w", err)
	}

	switch v := mm.VisionConfig; {
	case v != nil && v.ModelType == "gemma4_vision":
		if v.HeadDim == 0 && v.NumAttentionHeads > 0 {
			v.HeadDim = v.HiddenSize / v.NumAttentionHeads
		}
		if v.PatchSize == 0 {
			v.PatchSize = 16
		}
		if v.PoolingKernelSize == 0 {
			v.PoolingKernelSize = 3
		}
		if v.DefaultOutputLen == 0 {
			v.DefaultOutputLen = 280
		}
		if v.RMSNormEps == 0 {
			v.RMSNormEps = 1e-6
		}
		v.RopeTheta = 100
		if v.RopeParameters != nil && v.RopeParameters.RopeTheta > 0 {
			v.RopeTheta = v.RopeParameters.RopeTheta
		}
	case v != nil && v.unified():
		if v.PatchSize == 0 {
			v.PatchSize = 16
		}
		if v.PoolingKernelSize == 0 {
			v.PoolingKernelSize = 3
		}
		if v.NumSoftTokens == 0 {
			v.NumSoftTokens = 280
		}
		if v.RMSNormEps == 0 {
			v.RMSNormEps = 1e-6
		}
	default:
		mm.VisionConfig = nil
	}

	switch a := mm.AudioConfig; {
	case a != nil && a.ModelType == "gemma4_audio":
		applyAudioDefaults(a)
	case a != nil && a.unified():
		if a.SamplesPerToken == 0 {
			a.SamplesPerToken = 640
		}
		if a.SamplesPerToken < 0 {
			return multimodalConfig{}, fmt.Errorf("invalid audio_samples_per_token %d", a.SamplesPerToken)
		}
		if a.RMSNormEps == 0 {
			a.RMSNormEps = 1e-6
		}
	default:
		mm.AudioConfig = nil
	}
	if mm.EOATokenID == 0 {
		mm.EOATokenID = mm.EOATokenIndex
	}
	return mm, nil
}

// MultimodalEmbedder projects pooled features into the text embedding
// space: a scale-less RMSNorm then a linear.
type MultimodalEmbedder struct {
	Projection nn.LinearLayer
}

// ClippableLinear clamps a linear's input and output to checkpoint-provided
// bounds; the loader guarantees all four are present.
type ClippableLinear struct {
	Inner                                    nn.LinearLayer
	InputMin, InputMax, OutputMin, OutputMax *mlx.Array
}

func (c *ClippableLinear) Forward(x *mlx.Array) *mlx.Array {
	x = mlx.Clip(x, c.InputMin, c.InputMax)
	x = c.Inner.Forward(x)
	return mlx.Clip(x, c.OutputMin, c.OutputMax)
}

func (c *ClippableLinear) OutputDim() int32 { return c.Inner.OutputDim() }

// makeClippableLinear builds one tower projection: the weight sits under a
// .linear suffix, with optional clamp scalars beside it.
func makeClippableLinear(linears model.LinearFactory, tensors map[string]*mlx.Array, name string) (nn.LinearLayer, error) {
	inner := linears.Make(name + ".linear")
	if inner == nil {
		inner = linears.Make(name)
	}
	if inner == nil {
		return nil, fmt.Errorf("missing weight: %s", name)
	}

	c := &ClippableLinear{
		Inner:     inner,
		InputMin:  tensors[name+".input_min"],
		InputMax:  tensors[name+".input_max"],
		OutputMin: tensors[name+".output_min"],
		OutputMax: tensors[name+".output_max"],
	}
	if c.InputMin == nil && c.InputMax == nil && c.OutputMin == nil && c.OutputMax == nil {
		return inner, nil
	}
	if c.InputMin == nil || c.InputMax == nil || c.OutputMin == nil || c.OutputMax == nil {
		return nil, fmt.Errorf("weight %s has a partial clamp set", name)
	}
	return c, nil
}

// PrepareMedia implements base.MediaModel: splice each media segment's
// placeholder expansion — the begin token, the soft-token run(s), the end
// token — into the stream, with all preprocessing on the CPU. An audio
// segment yields one item per chunk, all inside one boa/eoa pair; the runs
// attend causally, so chunked prefill may split them.
func (m *Model) PrepareMedia(segments []base.Segment) (*base.PreparedRequest, error) {
	prepared := &base.PreparedRequest{}
	for s, seg := range segments {
		switch seg.Kind {
		case "":
			prepared.Tokens = append(prepared.Tokens, seg.Tokens...)
		case "image":
			if !m.visionLoaded() {
				return nil, fmt.Errorf("this model does not support image input")
			}

			pixels, positions, geom, err := m.preprocessImage(seg.Data)
			if err != nil {
				return nil, err
			}

			start := len(prepared.Tokens)
			prepared.Tokens = append(prepared.Tokens, m.MM.BOITokenID)
			for range geom.NumSoftTokens {
				prepared.Tokens = append(prepared.Tokens, m.MM.ImageTokenID)
			}
			prepared.Tokens = append(prepared.Tokens, m.MM.EOITokenID)

			n := int(geom.PatchesH * geom.PatchesW)
			prepared.Items = append(prepared.Items, base.PreparedItem{
				Range:     [2]int{start, len(prepared.Tokens)},
				Source:    s,
				MediaData: pixels,
				Dims:      []int{n, len(pixels) / n},
				Opaque:    preparedImage{positions: positions, geom: geom},
			})
		case "audio":
			if !m.audioLoaded() {
				return nil, fmt.Errorf("this model does not support audio input")
			}

			// A chunk's MediaData rows are mel frames for the conformer
			// and one raw waveform frame per soft token for the unified
			// embedder.
			var chunks []audioChunk
			if m.Audio.unified() {
				frames, tokens, err := processUnifiedAudio(seg.Data, int(m.Audio.SamplesPerToken))
				if err != nil {
					return nil, err
				}
				chunks = []audioChunk{{data: frames, frames: tokens, numTokens: tokens}}
			} else {
				var err error
				if chunks, err = processAudio(seg.Data); err != nil {
					return nil, err
				}
			}

			prepared.Tokens = append(prepared.Tokens, m.MM.BOATokenID)
			for _, chunk := range chunks {
				start := len(prepared.Tokens)
				for range chunk.numTokens {
					prepared.Tokens = append(prepared.Tokens, m.MM.AudioTokenID)
				}
				prepared.Items = append(prepared.Items, base.PreparedItem{
					Range:     [2]int{start, len(prepared.Tokens)},
					Source:    s,
					MediaData: chunk.data,
					Dims:      []int{chunk.frames, len(chunk.data) / chunk.frames},
					Opaque:    preparedAudio{numTokens: int32(chunk.numTokens)},
					Causal:    true,
				})
			}
			prepared.Tokens = append(prepared.Tokens, m.MM.EOATokenID)
		default:
			return nil, fmt.Errorf("gemma4 does not support %s input", seg.Kind)
		}
	}
	return prepared, nil
}

// EncodeMedia implements base.MediaModel: run the matching encoder over one
// whole item, returning the lazy [soft tokens, hidden] features.
func (m *Model) EncodeMedia(item *base.PreparedItem, data *mlx.Array) *mlx.Array {
	switch p := item.Opaque.(type) {
	case preparedAudio:
		if m.AudioTower != nil {
			return m.encodeAudio(data)
		}
		return m.encodeUnifiedAudio(data)
	case preparedImage:
		if m.UnifiedEmbedder != nil {
			return m.encodeUnifiedImage(data, p.positions, p.geom)
		}
		return m.encodeImage(data, p.positions, p.geom)
	}
	panic("gemma4: unknown media item")
}

// softRun returns a media item's feature-bearing token range: an image
// expansion is boi + soft*N + eoi, so the run starts one past the splice;
// an audio item's range is exactly its soft run.
func softRun(item batch.MediaItem) (start, end int) {
	switch p := item.Opaque.(type) {
	case preparedAudio:
		return item.Pos, item.Pos + int(p.numTokens)
	case preparedImage:
		return item.Pos + 1, item.Pos + 1 + int(p.geom.NumSoftTokens)
	}
	panic("gemma4: unknown media item")
}

// buildMasks returns the per-layer-type masks. Both start as unmaterialized
// causal masks; when the checkpoint enables bidirectional vision attention,
// the sliding mask relaxes each image run intersecting the chunk while the
// full-attention mask stays causal.
func (m *Model) buildMasks(b *batch.Batch) (sliding, full nn.AttentionMask) {
	sliding, full = nn.CausalMask(), nn.CausalMask()
	if !m.BidirectionalVisionAttention || len(b.Media) == 0 {
		return
	}

	for _, item := range b.Media {
		// Only image runs are bidirectional; audio attends causally.
		if _, ok := item.Opaque.(preparedImage); !ok {
			continue
		}
		start, end := softRun(item)
		off := int(b.SeqOffsets[item.Seq])
		if end <= off || start >= off+int(b.SeqQueryLens[item.Seq]) {
			continue
		}
		sliding = sliding.Relax(item.Seq, start, end, start, end)
	}
	return sliding, full
}

// scatterMedia overwrites the soft-token rows this chunk covers with the
// item's projected features.
func (m *Model) scatterMedia(h *mlx.Array, b *batch.Batch) *mlx.Array {
	for _, item := range b.Media {
		if item.Features == nil {
			continue
		}
		start, end := softRun(item)
		off := int(b.SeqOffsets[item.Seq])
		qLo := max(start, off)
		qHi := min(end, off+int(b.SeqQueryLens[item.Seq]))
		if qHi <= qLo {
			continue
		}

		feat := item.Features.Slice(mlx.Slice(qLo-start, qHi-start), mlx.Slice())
		feat = mlx.Reshape(feat.AsType(h.DType()), 1, int32(qHi-qLo), m.HiddenSize)
		h = h.SliceUpdate(feat, mlx.Slice(item.Seq, item.Seq+1), mlx.Slice(qLo-off, qHi-off), mlx.Slice())
	}
	return h
}

// validateMediaTokens warns when the tokenizer's special tokens disagree
// with the config's media token IDs; the config values are authoritative.
func (m *Model) validateMediaTokens() {
	type mediaToken struct {
		name string
		id   int32
	}
	var tokens []mediaToken
	if m.Vision != nil {
		tokens = append(tokens,
			mediaToken{"<|image|>", m.MM.ImageTokenID},
			mediaToken{"<|image>", m.MM.BOITokenID},
			mediaToken{"<image|>", m.MM.EOITokenID})
	}
	if m.Audio != nil {
		tokens = append(tokens,
			mediaToken{"<|audio|>", m.MM.AudioTokenID},
			mediaToken{"<|audio>", m.MM.BOATokenID},
			mediaToken{"<audio|>", m.MM.EOATokenID})
	}
	for _, tok := range tokens {
		if id, ok := m.tok.GetSpecialToken(tok.name); ok && id != tok.id {
			slog.Warn("media token mismatch", "token", tok.name, "config", tok.id, "tokenizer", id)
		}
	}
}
