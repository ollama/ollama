package hy4_flash

import (
	"bytes"
	"fmt"
	"image"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/model"
	"golang.org/x/image/draw"
)

type imageGrid struct{ rows, cols int }

func (g imageGrid) tokenCount(merge int) int { return (g.rows / merge) * (g.cols/merge + 1) }

type positionLayout struct {
	positions [][3]int32
	delta     int32
}

func (m *Model) PrepareMedia(segments []model.Segment) (*model.PreparedRequest, error) {
	prepared := &model.PreparedRequest{}
	layout := &positionLayout{}
	var pos int32
	appendText := func(tokens ...int32) {
		prepared.Tokens = append(prepared.Tokens, tokens...)
		for range tokens {
			layout.positions = append(layout.positions, [3]int32{pos, pos, pos})
			pos++
		}
	}
	for i, seg := range segments {
		if seg.Data == nil {
			appendText(seg.Tokens...)
			continue
		}
		if seg.Kind != "image" || m.Vision == nil {
			return nil, fmt.Errorf("Hy4 Flash does not support %s here", seg.Kind)
		}
		pixels, dims, grid, err := m.prepareImage(seg.Data)
		if err != nil {
			return nil, err
		}
		start := len(prepared.Tokens)
		appendText(m.cfg.ImageStartTokenID)
		rows, cols := grid.rows/m.cfg.Vision.Merge, grid.cols/m.cfg.Vision.Merge+1
		for h := range rows {
			for w := range cols {
				prepared.Tokens = append(prepared.Tokens, m.cfg.ImageTokenID)
				layout.positions = append(layout.positions, [3]int32{pos, pos + int32(h), pos + int32(w)})
			}
		}
		pos += int32(max(rows, cols))
		appendText(m.cfg.ImageEndTokenID)
		prepared.Items = append(prepared.Items, model.PreparedItem{
			Range: [2]int{start, len(prepared.Tokens)}, Source: i, MediaData: pixels, Dims: dims, Opaque: grid, Causal: true,
		})
	}
	if len(prepared.Items) > 0 {
		layout.delta = pos - int32(len(prepared.Tokens))
		prepared.Layout = layout
	}
	return prepared, nil
}

func (m *Model) prepareImage(raw []byte) ([]float32, []int, imageGrid, error) {
	img, _, err := image.Decode(bytes.NewReader(raw))
	if err != nil {
		return nil, nil, imageGrid{}, fmt.Errorf("decode Hy4 image: %w", err)
	}
	bounds := img.Bounds()
	cfg, proc := m.cfg.Vision, m.processor.Image
	h, w, err := model.FitImagePixels(int32(bounds.Dy()), int32(bounds.Dx()), int32(cfg.Patch*cfg.Merge), proc.Size.Min, proc.Size.Max)
	if err != nil {
		return nil, nil, imageGrid{}, err
	}
	resized := image.NewRGBA(image.Rect(0, 0, int(w), int(h)))
	draw.CatmullRom.Scale(resized, resized.Bounds(), model.DropImageAlpha(img, bounds), bounds, draw.Src, nil)
	pixels := make([]float32, int(h)*int(w)*3)
	for y := range int(h) {
		for x := range int(w) {
			ip := resized.PixOffset(x, y)
			op := 3 * (y*int(w) + x)
			for channel := range 3 {
				pixels[op+channel] = (float32(resized.Pix[ip+channel])*proc.Scale - proc.Mean[channel]) / proc.Std[channel]
			}
		}
	}
	return pixels, []int{1, int(h), int(w), 3}, imageGrid{int(h) / cfg.Patch, int(w) / cfg.Patch}, nil
}

func (m *Model) scatterImageFeatures(x *mlx.Array, b *batch.Batch) *mlx.Array {
	for _, item := range b.Media {
		if item.Features == nil {
			continue
		}
		imageStart := item.Pos + 1
		imageEnd := imageStart + item.Opaque.(imageGrid).tokenCount(m.cfg.Vision.Merge)
		base := int(b.SeqOffsets[item.Seq])
		lo, hi := max(imageStart, base), min(imageEnd, base+int(b.SeqQueryLens[item.Seq]))
		if lo >= hi {
			continue
		}
		features := item.Features.Slice(mlx.Slice(lo-imageStart, hi-imageStart), mlx.Slice()).AsType(x.DType())
		x = x.SliceUpdate(features.Reshape(1, hi-lo, x.Dim(2)), mlx.Slice(item.Seq, item.Seq+1), mlx.Slice(lo-base, hi-base), mlx.Slice())
	}
	return x
}
