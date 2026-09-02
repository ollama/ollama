package nemotron_h

import (
	"bytes"
	"fmt"
	"image"
	"math"

	"golang.org/x/image/draw"

	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model/base"
)

// preparedImage is the model-private state travelling with a prepared item.
type preparedImage struct {
	height int
	width  int
}

// PrepareMedia implements base.MediaModel: splice each image segment's
// placeholder expansion — image start, the feature-token run, image end —
// into the stream, decoding and resizing on the CPU.
func (m *Model) PrepareMedia(segments []base.Segment) (*base.PreparedRequest, error) {
	prepared := &base.PreparedRequest{}
	for s, seg := range segments {
		if seg.Data == nil {
			prepared.Tokens = append(prepared.Tokens, seg.Tokens...)
			continue
		}
		if seg.Kind != "image" {
			return nil, fmt.Errorf("nemotron_h does not support %s input", seg.Kind)
		}
		if m.VisionEncoder == nil || m.Projector == nil || m.VisionConfig == nil {
			if m.visionErr != nil {
				return nil, fmt.Errorf("nemotron_h vision is unavailable: %w", m.visionErr)
			}
			return nil, fmt.Errorf("this model does not support %s input", seg.Kind)
		}

		pixels, height, width, err := m.preprocessImage(seg.Data, nemotronImagePatchBudget(m.VisionConfig))
		if err != nil {
			return nil, err
		}

		start := len(prepared.Tokens)
		prepared.Tokens = append(prepared.Tokens, m.imageStartTokenID)
		for range m.visionTokenCount(height, width) {
			prepared.Tokens = append(prepared.Tokens, m.imageTokenID)
		}
		prepared.Tokens = append(prepared.Tokens, m.imageEndTokenID)

		prepared.Items = append(prepared.Items, base.PreparedItem{
			Range:     [2]int{start, len(prepared.Tokens)},
			Source:    s,
			MediaData: pixels,
			Dims:      []int{1, 3, height, width},
			Opaque:    preparedImage{height: height, width: width},
			// Image rows use the text stack's causal mask, so chunked prefill
			// may split the expansion after the vision features are encoded.
			Causal: true,
		})
	}
	return prepared, nil
}

// EncodeMedia implements base.MediaModel: run the RADIO tower and projector
// over one image, returning the lazy [1, tokens, hidden] features.
func (m *Model) EncodeMedia(_ *base.PreparedItem, data *mlx.Array) *mlx.Array {
	return mlx.Squeeze(m.forwardVision(data.AsType(m.VisionEncoder.Position.DType())), 0)
}

// scatterMedia overwrites each expansion's feature rows with the encoded
// features, for whatever part of the expansion this forward covers. The
// expansion is imageStart + features + imageEnd, so the run starts one past
// the splice. delta shifts each row's offset to the sequence position of
// batch column 0, which the MTP draft needs: its slots hold look-ahead
// tokens rather than the tokens at their own offset.
func (m *Model) scatterMedia(h *mlx.Array, b *batch.Batch, delta int) *mlx.Array {
	for _, item := range b.Media {
		if item.Features == nil {
			continue
		}
		img := item.Opaque.(preparedImage)
		start := item.Pos + 1
		end := start + m.visionTokenCount(img.height, img.width)

		off := int(b.SeqOffsets[item.Seq]) + delta
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

func (m *Model) preprocessImage(data []byte, maxPatches int) (pixels []float32, height, width int, err error) {
	cfg := m.VisionConfig
	if cfg == nil {
		return nil, 0, 0, fmt.Errorf("model has no vision config")
	}

	src, _, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		return nil, 0, 0, fmt.Errorf("decode image: %w", err)
	}
	bounds := src.Bounds()
	srcW, srcH := bounds.Dx(), bounds.Dy()
	if srcW <= 0 || srcH <= 0 {
		return nil, 0, 0, fmt.Errorf("invalid image dimensions %dx%d", srcW, srcH)
	}

	patchH, patchW := nemotronImagePatchGrid(srcH, srcW, maxPatches, cfg)
	targetH := patchH * int(cfg.PatchSize)
	targetW := patchW * int(cfg.PatchSize)
	dst := image.NewRGBA(image.Rect(0, 0, targetW, targetH))
	draw.CatmullRom.Scale(dst, dst.Bounds(), src, bounds, draw.Src, nil)

	pixels = make([]float32, 3*targetH*targetW)
	plane := targetH * targetW
	for y := range targetH {
		for x := range targetW {
			offset := y*dst.Stride + x*4
			i := y*targetW + x
			pixels[i] = (float32(dst.Pix[offset])/255 - cfg.Mean[0]) / cfg.Std[0]
			pixels[plane+i] = (float32(dst.Pix[offset+1])/255 - cfg.Mean[1]) / cfg.Std[1]
			pixels[2*plane+i] = (float32(dst.Pix[offset+2])/255 - cfg.Mean[2]) / cfg.Std[2]
		}
	}

	return pixels, targetH, targetW, nil
}

func nemotronImagePatchBudget(cfg *VisionConfig) int {
	return max(cfg.MinNumPatches, min(cfg.MaxNumPatches, (cfg.MaxModelLen-nemotronVisionReservedTokens)*int(cfg.DownsampleFactor)*int(cfg.DownsampleFactor)))
}

func nemotronImagePatchGrid(srcH, srcW, maxPatches int, cfg *VisionConfig) (int, int) {
	patchSize := float64(cfg.PatchSize)
	closestH := max(1, int(math.RoundToEven(float64(srcH)/patchSize+0.5)))
	closestW := max(1, int(math.RoundToEven(float64(srcW)/patchSize+0.5)))
	sourcePatches := max(1, closestH*closestW)
	budget := max(cfg.MinNumPatches, min(maxPatches, cfg.MaxNumPatches))

	scale := math.Min(math.Sqrt(float64(budget)/float64(sourcePatches)), 1)
	targetH := max(1, int(math.Floor(scale*float64(closestH))))
	targetW := max(1, int(math.Floor(scale*float64(closestW))))
	if budget > cfg.MinNumPatches && targetH*targetW < cfg.MinNumPatches {
		scaleUp := math.Sqrt(float64(cfg.MinNumPatches) / float64(targetH*targetW))
		targetH = max(1, int(math.Ceil(scaleUp*float64(targetH))))
		targetW = max(1, int(math.Ceil(scaleUp*float64(targetW))))
	}

	divisor := int(cfg.DownsampleFactor)
	if rem := targetH % divisor; rem != 0 {
		inc := divisor - rem
		if (targetH+inc)*targetW <= budget {
			targetH += inc
		} else {
			targetH = max(divisor, targetH-rem)
		}
	}
	if rem := targetW % divisor; rem != 0 {
		inc := divisor - rem
		if targetH*(targetW+inc) <= budget {
			targetW += inc
		} else {
			targetW = max(divisor, targetW-rem)
		}
	}

	return targetH, targetW
}
