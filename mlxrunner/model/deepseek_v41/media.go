package deepseek_v41

import (
	"bytes"
	"fmt"
	"image"
	"math"

	"golang.org/x/image/draw"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/model"
)

var _ model.MediaModel = (*Model)(nil)

type imageGeometry struct {
	patchRows int
	patchCols int
	textRows  int
	textCols  int
}

// Admit high-resolution photos while bounding a common RGBA decode to about
// 200 MiB before the token-budget resize.
const maxSourceImagePixels = 50_000_000

func (g imageGeometry) tokenCount() int { return g.textRows*(g.textCols+1) + 2 }

func imageSize(width, height int, v visionConfig) (int, int, imageGeometry, error) {
	if width <= 0 || height <= 0 || v.PatchSize <= 0 || v.DownsampleRatio <= 0 {
		return 0, 0, imageGeometry{}, fmt.Errorf("invalid image dimensions %dx%d", width, height)
	}
	w, h := float64(width), float64(height)
	if float64(width)*float64(height) < float64(v.MinPixels) {
		scale := math.Sqrt(float64(v.MinPixels) / (w * h))
		w, h = math.Floor(w*scale), math.Floor(h*scale)
	}
	p := v.PatchSize
	targetW, targetH := int(math.Ceil(w/float64(p)))*p, int(math.Ceil(h/float64(p)))*p
	grid := func(tw, th int) imageGeometry {
		rows, cols := th/p, tw/p
		return imageGeometry{rows, cols, (rows + v.DownsampleRatio - 1) / v.DownsampleRatio, (cols + v.DownsampleRatio - 1) / v.DownsampleRatio}
	}
	g := grid(targetW, targetH)
	if g.tokenCount() > v.MaxImageTokens {
		cell := p * v.DownsampleRatio
		aspect := h / w
		colLimit := math.Sqrt(float64(v.MaxImageTokens-2)/aspect+0.25) - 0.5
		rowLimit := colLimit * aspect
		if colLimit < 1 {
			targetW, targetH = cell, ((v.MaxImageTokens-2)/2)*cell
		} else if rowLimit < 1 {
			targetW, targetH = (v.MaxImageTokens-3)*cell, cell
		} else {
			factor := math.Min(math.Floor(colLimit)*float64(cell)/w, math.Floor(rowLimit)*float64(cell)/h)
			targetW = int(math.Floor(w*factor/float64(p))) * p
			targetH = int(math.Floor(h*factor/float64(p))) * p
		}
		g = grid(targetW, targetH)
	}
	if targetW <= 0 || targetH <= 0 || g.tokenCount() > v.MaxImageTokens {
		return 0, 0, imageGeometry{}, fmt.Errorf("image cannot fit %d-token budget", v.MaxImageTokens)
	}
	return targetW, targetH, g, nil
}

func (m *Model) preprocessImage(data []byte) ([]float32, imageGeometry, error) {
	config, _, err := image.DecodeConfig(bytes.NewReader(data))
	if err != nil {
		return nil, imageGeometry{}, fmt.Errorf("read DeepSeek image dimensions: %w", err)
	}
	if config.Width <= 0 || config.Height <= 0 || int64(config.Width)*int64(config.Height) > maxSourceImagePixels {
		return nil, imageGeometry{}, fmt.Errorf("DeepSeek image exceeds %d source pixels", maxSourceImagePixels)
	}
	tw, th, g, err := imageSize(config.Width, config.Height, m.Config.Vision)
	if err != nil {
		return nil, imageGeometry{}, err
	}
	source, _, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		return nil, imageGeometry{}, fmt.Errorf("decode DeepSeek image: %w", err)
	}
	b := source.Bounds()
	// Match the publisher's centered fit with gray letterboxing. A single
	// resize preserves the source aspect ratio before patch extraction.
	scale := math.Min(float64(tw)/float64(b.Dx()), float64(th)/float64(b.Dy()))
	rw := min(tw, max(1, int(math.Round(float64(b.Dx())*scale))))
	rh := min(th, max(1, int(math.Round(float64(b.Dy())*scale))))
	canvas := image.NewNRGBA(image.Rect(0, 0, tw, th))
	for i := 0; i < len(canvas.Pix); i += 4 {
		canvas.Pix[i], canvas.Pix[i+1], canvas.Pix[i+2], canvas.Pix[i+3] = 127, 127, 127, 255
	}
	left, top := (tw-rw)/2, (th-rh)/2
	draw.CatmullRom.Scale(canvas, image.Rect(left, top, left+rw, top+rh), source, b, draw.Src, nil)
	p := m.Config.Vision.PatchSize
	patches := make([]float32, g.patchRows*g.patchCols*3*p*p)
	for row := range g.patchRows {
		for col := range g.patchCols {
			base := (row*g.patchCols + col) * 3 * p * p
			for channel := range 3 {
				for y := range p {
					for x := range p {
						pixel := (row*p+y)*canvas.Stride + (col*p+x)*4 + channel
						patches[base+channel*p*p+y*p+x] = 2*float32(canvas.Pix[pixel])/255 - 1
					}
				}
			}
		}
	}
	return patches, g, nil
}

func (m *Model) PrepareMedia(segments []model.Segment) (*model.PreparedRequest, error) {
	result := &model.PreparedRequest{}
	for source, segment := range segments {
		if segment.Data == nil {
			result.Tokens = append(result.Tokens, segment.Tokens...)
			continue
		}
		if segment.Kind != "image" || m.VisionTower == nil {
			return nil, fmt.Errorf("DeepSeek V4.1 does not support %s input", segment.Kind)
		}
		patches, g, err := m.preprocessImage(segment.Data)
		if err != nil {
			return nil, err
		}
		start := len(result.Tokens)
		for range g.tokenCount() {
			result.Tokens = append(result.Tokens, int32(m.Config.ImageTokenID))
		}
		result.Items = append(result.Items, model.PreparedItem{
			Range: [2]int{start, len(result.Tokens)}, Source: source,
			MediaData: patches, Dims: []int{g.patchRows * g.patchCols, 3 * m.Config.Vision.PatchSize * m.Config.Vision.PatchSize},
			Opaque: g,
		})
	}
	return result, nil
}

func (m *Model) EncodeMedia(item *model.PreparedItem, data *mlx.Array) *mlx.Array {
	g := item.Opaque.(imageGeometry)
	return m.VisionTower.forward(data, g)
}

func imageMask(b *batch.Batch) *mlx.Array {
	if len(b.Media) == 0 {
		return nil
	}
	return mlx.FromValues(imagePositions(b), b.InputIDs.Dim(0), b.InputIDs.Dim(1))
}

func imagePositions(b *batch.Batch) []bool {
	length := b.InputIDs.Dim(1)
	mask := make([]bool, b.InputIDs.Dim(0)*length)
	for _, item := range b.Media {
		if item.Seq >= b.InputIDs.Dim(0) {
			continue
		}
		start := max(0, item.Pos-int(b.SeqOffsets[item.Seq]))
		end := min(length, item.Pos+item.Opaque.(imageGeometry).tokenCount()-int(b.SeqOffsets[item.Seq]))
		for i := start; i < end; i++ {
			mask[item.Seq*length+i] = true
		}
	}
	return mask
}

func (m *Model) scatterImages(hidden *mlx.Array, b *batch.Batch) *mlx.Array {
	width := int32(m.Config.HiddenSize)
	for _, item := range b.Media {
		if item.Features == nil {
			continue
		}
		g := item.Opaque.(imageGeometry)
		pieces := make([]*mlx.Array, 0, 2*g.textRows+2)
		pieces = append(pieces, mlx.Reshape(m.VisionTower.start, 1, width))
		for row := range g.textRows {
			pieces = append(pieces, item.Features.Slice(mlx.Slice(row*g.textCols, (row+1)*g.textCols), mlx.Slice()))
			pieces = append(pieces, mlx.Reshape(m.VisionTower.newline, 1, width))
		}
		pieces = append(pieces, mlx.Reshape(m.VisionTower.end, 1, width))
		features := mlx.Concatenate(pieces, 0)
		offset := int(b.SeqOffsets[item.Seq])
		start := max(item.Pos, offset)
		end := min(item.Pos+g.tokenCount(), offset+int(b.SeqQueryLens[item.Seq]))
		if end <= start {
			continue
		}
		part := features.Slice(mlx.Slice(start-item.Pos, end-item.Pos), mlx.Slice())
		part = mlx.Reshape(part.AsType(hidden.DType()), 1, int32(end-start), width)
		hidden = hidden.SliceUpdate(part, mlx.Slice(item.Seq, item.Seq+1), mlx.Slice(start-offset, end-offset), mlx.Slice())
	}
	return hidden
}
