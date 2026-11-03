package lfm3

import (
	"bytes"
	"fmt"
	"image"
	"math"

	"golang.org/x/image/draw"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/cache"
	"github.com/ollama/ollama/mlxrunner/model"
)

const (
	visionStartTokenID  = 125009
	visionEndTokenID    = 125010
	maxImageAspectRatio = 200
)

var _ model.MediaModel = (*VLModel)(nil)

type preparedImage struct {
	Height int32
	Width  int32
	Tokens int32
}

type preparedAudio struct {
	Tokens int32
	// Lead counts span markers before the first audio token.
	Lead int32
}

// mediaSpan locates an item's soft tokens relative to its first placeholder.
func mediaSpan(opaque any) (lead, tokens int) {
	switch item := opaque.(type) {
	case preparedImage:
		return 1, int(item.Tokens)
	case preparedAudio:
		return int(item.Lead), int(item.Tokens)
	}
	return 0, 0
}

func resizeImageGrid(height, width, factor, minPixels, maxPixels int32) (int32, int32, error) {
	if height <= 0 || width <= 0 || factor <= 0 || minPixels <= 0 || maxPixels < minPixels {
		return 0, 0, fmt.Errorf("invalid image size or processor limits")
	}
	if float64(max(height, width))/float64(min(height, width)) > maxImageAspectRatio {
		return 0, 0, fmt.Errorf("image aspect ratio exceeds %d", maxImageAspectRatio)
	}
	f := float64(factor)
	h := math.RoundToEven(float64(height)/f) * f
	w := math.RoundToEven(float64(width)/f) * f
	area := h * w
	switch {
	case area > float64(maxPixels):
		scale := math.Sqrt(float64(height) * float64(width) / float64(maxPixels))
		h = math.Max(f, math.Floor(float64(height)/scale/f)*f)
		w = math.Max(f, math.Floor(float64(width)/scale/f)*f)
	case area < float64(minPixels):
		scale := math.Sqrt(float64(minPixels) / float64(height) / float64(width))
		h = math.Ceil(float64(height)*scale/f) * f
		w = math.Ceil(float64(width)*scale/f) * f
	}
	return int32(h), int32(w), nil
}

type resampleTap struct {
	index  int
	weight float64
}

func cubicWeight(x float64) float64 {
	x = math.Abs(x)
	switch {
	case x < 1:
		return (1.5*x-2.5)*x*x + 1
	case x < 2:
		return ((-0.5*x+2.5)*x-4)*x + 2
	default:
		return 0
	}
}

func bicubicPlan(source, target int) [][]resampleTap {
	plan := make([][]resampleTap, target)
	scale := float64(source) / float64(target)
	filterScale := math.Max(scale, 1)
	for output := range target {
		center := (float64(output) + 0.5) * scale
		first := max(0, int(math.Floor(center-2*filterScale-0.5)))
		last := min(source, int(math.Ceil(center+2*filterScale-0.5))+1)
		var total float64
		for input := first; input < last; input++ {
			weight := cubicWeight((float64(input) + 0.5 - center) / filterScale)
			if weight == 0 {
				continue
			}
			plan[output] = append(plan[output], resampleTap{index: input, weight: weight})
			total += weight
		}
		for i := range plan[output] {
			plan[output][i].weight /= total
		}
	}
	return plan
}

func roundedByte(v float64) uint8 {
	return uint8(max(0, min(255, int(math.Floor(v+0.5)))))
}

func resizeBicubic(source *image.NRGBA, width, height int) *image.NRGBA {
	sourceW, sourceH := source.Bounds().Dx(), source.Bounds().Dy()
	horizontal := image.NewNRGBA(image.Rect(0, 0, width, sourceH))
	for x, taps := range bicubicPlan(sourceW, width) {
		for y := range sourceH {
			for channel := range 3 {
				var value float64
				for _, tap := range taps {
					value += float64(source.Pix[source.PixOffset(tap.index, y)+channel]) * tap.weight
				}
				horizontal.Pix[horizontal.PixOffset(x, y)+channel] = roundedByte(value)
			}
			horizontal.Pix[horizontal.PixOffset(x, y)+3] = 255
		}
	}
	output := image.NewNRGBA(image.Rect(0, 0, width, height))
	for y, taps := range bicubicPlan(sourceH, height) {
		for x := range width {
			for channel := range 3 {
				var value float64
				for _, tap := range taps {
					value += float64(horizontal.Pix[horizontal.PixOffset(x, tap.index)+channel]) * tap.weight
				}
				output.Pix[output.PixOffset(x, y)+channel] = roundedByte(value)
			}
			output.Pix[output.PixOffset(x, y)+3] = 255
		}
	}
	return output
}

func (v *VisionModel) preprocessImage(data []byte) ([]float32, preparedImage, error) {
	source, _, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		return nil, preparedImage{}, fmt.Errorf("decode LFM3 image: %w", err)
	}
	bounds := source.Bounds()
	patch := v.Config.PatchSize
	merge := v.Config.SpatialMergeSize
	temporal := v.Config.TemporalPatchSize
	height, width, err := resizeImageGrid(int32(bounds.Dy()), int32(bounds.Dx()), patch*merge, v.MinPixels, v.MaxPixels)
	if err != nil {
		return nil, preparedImage{}, err
	}
	// The processor consumes RGB. Preserve color channels when an input has alpha.
	rgb := image.NewNRGBA(bounds)
	draw.Draw(rgb, bounds, source, bounds.Min, draw.Src)
	for i := 3; i < len(rgb.Pix); i += 4 {
		rgb.Pix[i] = 0xff
	}
	resized := resizeBicubic(rgb, int(width), int(height))
	gridH, gridW := height/patch, width/patch
	channels := int32(3) * temporal * patch * patch
	pixels := make([]float32, gridH*gridW*channels)
	row := int32(0)
	for blockH := int32(0); blockH < gridH; blockH += merge {
		for blockW := int32(0); blockW < gridW; blockW += merge {
			for innerH := range merge {
				for innerW := range merge {
					patchH, patchW := blockH+innerH, blockW+innerW
					for channel := range int32(3) {
						for time := range temporal {
							for y := range patch {
								for x := range patch {
									pixel := resized.PixOffset(int(patchW*patch+x), int(patchH*patch+y))
									index := row*channels + (((channel*temporal+time)*patch+y)*patch + x)
									pixels[index] = float32(resized.Pix[pixel+int(channel)])/127.5 - 1
								}
							}
						}
					}
					row++
				}
			}
		}
	}
	return pixels, preparedImage{Height: gridH, Width: gridW, Tokens: gridH * gridW / (merge * merge)}, nil
}

func (m *VLModel) PrepareMedia(segments []model.Segment) (*model.PreparedRequest, error) {
	request := &model.PreparedRequest{}
	for source, segment := range segments {
		if segment.Data == nil && segment.Kind == "" {
			request.Tokens = append(request.Tokens, segment.Tokens...)
			continue
		}
		if segment.Kind == "audio" && m.Audio != nil {
			item, tokens, err := m.Audio.prepare(segment.Data)
			if err != nil {
				return nil, err
			}
			item.Range = [2]int{len(request.Tokens), len(request.Tokens) + len(tokens)}
			item.Source = source
			request.Tokens = append(request.Tokens, tokens...)
			request.Items = append(request.Items, item)
			continue
		}
		if segment.Kind != "image" || m.Vision == nil {
			return nil, fmt.Errorf("this LFM3 model does not support %s input", segment.Kind)
		}
		pixels, geometry, err := m.Vision.preprocessImage(segment.Data)
		if err != nil {
			return nil, err
		}
		start := len(request.Tokens)
		request.Tokens = append(request.Tokens, visionStartTokenID)
		for range geometry.Tokens {
			request.Tokens = append(request.Tokens, m.Vision.ImageTokenID)
		}
		request.Tokens = append(request.Tokens, visionEndTokenID)
		request.Items = append(request.Items, model.PreparedItem{
			Range:     [2]int{start, len(request.Tokens)},
			Source:    source,
			MediaData: pixels,
			Dims:      []int{int(geometry.Height * geometry.Width), len(pixels) / int(geometry.Height*geometry.Width)},
			Opaque:    geometry,
			Causal:    true,
		})
	}
	return request, nil
}

func (m *VLModel) EncodeMedia(item *model.PreparedItem, data *mlx.Array) *mlx.Array {
	if _, ok := item.Opaque.(preparedAudio); ok {
		return m.Audio.Enc.Forward(data)
	}
	geometry := item.Opaque.(preparedImage)
	return m.Vision.Forward(data, geometry.Height, geometry.Width)
}

func (m *VLModel) Forward(b *batch.Batch, caches []cache.Cache) (hidden, auxHidden *mlx.Array) {
	h := m.EmbedTokens.Forward(b.InputIDs)
	for _, item := range b.Media {
		if item.Features == nil {
			continue
		}
		lead, tokens := mediaSpan(item.Opaque)
		softStart := item.Pos + lead
		softEnd := softStart + tokens
		offset := int(b.SeqOffsets[item.Seq])
		start := max(softStart, offset)
		end := min(softEnd, offset+int(b.SeqQueryLens[item.Seq]))
		if end <= start {
			continue
		}
		features := item.Features.Slice(mlx.Slice(start-softStart, end-softStart), mlx.Slice())
		features = mlx.Reshape(features.AsType(h.DType()), 1, int32(end-start), m.HiddenSize)
		h = h.SliceUpdate(features, mlx.Slice(item.Seq, item.Seq+1), mlx.Slice(start-offset, end-offset), mlx.Slice())
	}
	return m.forwardHidden(h, b, caches)
}
