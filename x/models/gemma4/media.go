package gemma4

import (
	"bytes"
	"encoding/json"
	"fmt"
	"image"
	"math"

	"golang.org/x/image/draw"

	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model/base"
	"github.com/ollama/ollama/x/models/nn"
)

const (
	gemma4MinImageTokens     = 40
	gemma4DefaultImageTokens = 280
)

type preparedImage struct {
	height    int
	width     int
	numTokens int
}

func parseImageTokenLimit(data []byte, fallback int) int {
	if fallback <= 0 {
		fallback = gemma4DefaultImageTokens
	}
	var cfg struct {
		ImageProcessor struct {
			MaxSoftTokens int `json:"max_soft_tokens"`
			ImageSeqLen   int `json:"image_seq_length"`
		} `json:"image_processor"`
		ImageSeqLen int `json:"image_seq_length"`
	}
	if json.Unmarshal(data, &cfg) != nil {
		return fallback
	}
	for _, n := range []int{cfg.ImageProcessor.MaxSoftTokens, cfg.ImageProcessor.ImageSeqLen, cfg.ImageSeqLen} {
		if n > 0 {
			return n
		}
	}
	return fallback
}

// PrepareMedia implements base.MediaModel: decode and resize each image on the
// CPU, then splice image_start + soft-token placeholders + image_end.
func (m *Model) PrepareMedia(segments []base.Segment) (*base.PreparedRequest, error) {
	prepared := &base.PreparedRequest{}
	for s, seg := range segments {
		if seg.Data == nil {
			prepared.Tokens = append(prepared.Tokens, seg.Tokens...)
			continue
		}
		if m.VisionEncoder == nil || m.MultimodalEmbed == nil {
			return nil, fmt.Errorf("this model does not support %s input", seg.Kind)
		}
		if seg.Kind != "image" {
			return nil, fmt.Errorf("gemma4 does not support %s input", seg.Kind)
		}

		pixels, geom, err := m.preprocessImage(seg.Data, m.imageTokenLimit)
		if err != nil {
			return nil, fmt.Errorf("preprocess image: %w", err)
		}

		start := len(prepared.Tokens)
		prepared.Tokens = append(prepared.Tokens, m.imageStartTokenID)
		for range geom.numTokens {
			prepared.Tokens = append(prepared.Tokens, m.imageTokenID)
		}
		prepared.Tokens = append(prepared.Tokens, m.imageEndTokenID)

		prepared.Items = append(prepared.Items, base.PreparedItem{
			Range:     [2]int{start, len(prepared.Tokens)},
			Source:    s,
			MediaData: pixels,
			Dims:      []int{1, 3, geom.height, geom.width},
			Opaque:    geom,
		})
	}
	return prepared, nil
}

// EncodeMedia runs the selected Gemma 4 vision path and returns lazy
// [numTokens, hidden] features. Evaluation is pulled by the text forward.
func (m *Model) EncodeMedia(_ *base.PreparedItem, data *mlx.Array) *mlx.Array {
	return mlx.Squeeze(m.ForwardVision(data), 0)
}

func gemma4SoftRun(item batch.MediaItem) (start, end int) {
	geom := item.Opaque.(preparedImage)
	return item.Pos + 1, item.Pos + 1 + geom.numTokens
}

func (m *Model) scatterMedia(h *mlx.Array, b *batch.Batch) *mlx.Array {
	for _, item := range b.Media {
		if item.Features == nil {
			continue
		}
		start, end := gemma4SoftRun(item)
		off := int(b.SeqOffsets[item.Seq])
		qLo := max(start, off)
		qHi := min(end, off+int(b.SeqQueryLens[item.Seq]))
		if qHi <= qLo {
			continue
		}

		features := item.Features.Slice(mlx.Slice(qLo-start, qHi-start), mlx.Slice())
		features = mlx.Reshape(features.AsType(h.DType()), 1, int32(qHi-qLo), m.HiddenSize)
		h = h.SliceUpdate(features,
			mlx.Slice(item.Seq, item.Seq+1),
			mlx.Slice(qLo-off, qHi-off),
			mlx.Slice(),
		)
	}
	return h
}

// PLE has a separate token table. Vision rows use zero there while their
// hidden-state contribution comes from the projected image features.
func maskMediaTokens(tokens *mlx.Array, b *batch.Batch) *mlx.Array {
	masked := tokens
	for _, item := range b.Media {
		start, end := gemma4SoftRun(item)
		off := int(b.SeqOffsets[item.Seq])
		qLo := max(start, off)
		qHi := min(end, off+int(b.SeqQueryLens[item.Seq]))
		if qHi <= qLo {
			continue
		}
		zeros := mlx.Zeros(mlx.DTypeInt32, 1, qHi-qLo)
		masked = masked.SliceUpdate(zeros,
			mlx.Slice(item.Seq, item.Seq+1),
			mlx.Slice(qLo-off, qHi-off),
		)
	}
	return masked
}

func mediaAttentionMask(b *batch.Batch) nn.AttentionMask {
	mask := nn.CausalMask()
	for _, item := range b.Media {
		start, end := gemma4SoftRun(item)
		mask = mask.Relax(item.Seq, start, end, start, end)
	}
	return mask
}

func (m *Model) preprocessImage(data []byte, maxTokens int) ([]float32, preparedImage, error) {
	if m.VisionConfig == nil {
		return nil, preparedImage{}, fmt.Errorf("model has no vision config")
	}
	src, _, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		return nil, preparedImage{}, fmt.Errorf("decode: %w", err)
	}

	bounds := src.Bounds()
	if bounds.Dx() <= 0 || bounds.Dy() <= 0 {
		return nil, preparedImage{}, fmt.Errorf("invalid image dimensions %dx%d", bounds.Dx(), bounds.Dy())
	}

	cfg := m.VisionConfig
	if maxTokens <= 0 {
		maxTokens = max(gemma4DefaultImageTokens, int(cfg.ImageSeqLength))
	}
	patchSize := int(cfg.PatchSize)
	alignment := int(cfg.PoolingKernelSize) * patchSize
	pixelsPerToken := alignment * alignment
	minTokens := min(maxTokens, max(gemma4MinImageTokens, int(cfg.ImageSeqLength)))
	targetH, targetW := gemma4ImageSize(
		bounds.Dy(),
		bounds.Dx(),
		alignment,
		minTokens*pixelsPerToken,
		maxTokens*pixelsPerToken,
	)

	resized := image.NewRGBA(image.Rect(0, 0, targetW, targetH))
	draw.CatmullRom.Scale(resized, resized.Bounds(), src, bounds, draw.Src, nil)

	pixels := make([]float32, 3*targetH*targetW)
	plane := targetH * targetW
	for y := range targetH {
		for x := range targetW {
			srcAt := y*resized.Stride + x*4
			dstAt := y*targetW + x
			pixels[dstAt] = float32(resized.Pix[srcAt]) / 255
			pixels[plane+dstAt] = float32(resized.Pix[srcAt+1]) / 255
			pixels[2*plane+dstAt] = float32(resized.Pix[srcAt+2]) / 255
		}
	}

	return pixels, preparedImage{
		height:    targetH,
		width:     targetW,
		numTokens: m.VisionTokenCount(targetH, targetW),
	}, nil
}

func gemma4ImageSize(height, width, alignment, minPixels, maxPixels int) (int, int) {
	roundTo := func(value float64) int {
		return int(math.Round(value/float64(alignment))) * alignment
	}

	targetH := max(alignment, roundTo(float64(height)))
	targetW := max(alignment, roundTo(float64(width)))
	switch {
	case float64(targetH)*float64(targetW) > float64(maxPixels):
		targetH, targetW = gemma4FitImageSize(height, width, alignment, maxPixels)
	case float64(targetH)*float64(targetW) < float64(minPixels):
		targetH, targetW = gemma4FitImageSize(height, width, alignment, minPixels)
	}
	return targetH, targetW
}

func gemma4FitImageSize(height, width, alignment, pixelBudget int) (int, int) {
	scale := math.Sqrt(float64(pixelBudget) / (float64(height) * float64(width)))
	targetH := int(math.Floor(scale*float64(height)/float64(alignment))) * alignment
	targetW := int(math.Floor(scale*float64(width)/float64(alignment))) * alignment

	maxSide := pixelBudget / (alignment * alignment) * alignment
	switch {
	case targetH == 0 && targetW == 0:
		return alignment, alignment
	case targetH == 0:
		targetH = alignment
		targetW = min(int(math.Floor(float64(width)/float64(height)))*alignment, maxSide)
	case targetW == 0:
		targetW = alignment
		targetH = min(int(math.Floor(float64(height)/float64(width)))*alignment, maxSide)
	}
	return targetH, targetW
}
