package qwen3_5

import (
	"bytes"
	"fmt"
	"image"
	"math"

	"golang.org/x/image/draw"

	"github.com/ollama/ollama/mlxrunner/model"
)

// Processor bounds from the family's preprocessor config: total pixels are
// clamped to [shortest_edge, longest_edge] with both sides rounded to
// multiples of patch_size * spatial_merge_size.
const (
	visionMinPixels = 65536
	visionMaxPixels = 16777216
)

// preparedImage is the model-private state for one image: the pre-merge
// patch grid and the CPU-computed lookups the encoder consumes, all in the
// block-major merge order the pixel rows use.
type preparedImage struct {
	gridH, gridW int32
	// ropePos is (h, w) per patch for the tower's 2D rotary tables.
	ropePos []int32
	// posIdx/posWt are the four bilinear corners per patch into the learned
	// position table, with their interpolation weights.
	posIdx []int32
	posWt  []float32
}

func (p preparedImage) numPatches() int32 { return p.gridH * p.gridW }

func (p preparedImage) numTokens(merge int32) int32 {
	return p.gridH * p.gridW / (merge * merge)
}

func smartResize(height, width, factor, maxPixels int32) (int32, int32, error) {
	return model.FitImagePixels(height, width, factor, visionMinPixels, maxPixels)
}

// preprocessImage decodes and prepares one image: aspect-preserving resize,
// (x/255-0.5)/0.5 normalization, and patchification into the tower's
// block-major row layout.
func (m *Model) preprocessImage(data []byte) (pixels []float32, prep preparedImage, err error) {
	img, _, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		return nil, preparedImage{}, fmt.Errorf("decode image: %w", err)
	}

	v := m.Vision
	patch, merge, temporal := v.PatchSize, v.SpatialMergeSize, v.TemporalPatchSize
	bounds := img.Bounds()
	img = model.DropImageAlpha(img, bounds)
	maxPixels := m.VisionMaxPixels
	if maxPixels == 0 {
		maxPixels = visionMaxPixels
	}
	targetH, targetW, err := smartResize(int32(bounds.Dy()), int32(bounds.Dx()), patch*merge, maxPixels)
	if err != nil {
		return nil, preparedImage{}, err
	}

	resized := image.NewRGBA(image.Rect(0, 0, int(targetW), int(targetH)))
	draw.CatmullRom.Scale(resized, resized.Bounds(), img, bounds, draw.Src, nil)
	gridH, gridW := targetH/patch, targetW/patch
	prep = preparedImage{gridH: gridH, gridW: gridW}
	pixels = patchifyImage(resized, gridH, gridW, patch, merge, temporal)
	prep.ropePos, prep.posIdx, prep.posWt = visionPatchLookups(gridH, gridW, merge, v.NumPositionEmbeddings)
	return pixels, prep, nil
}

// patchifyImage converts the resized image to the tower's row layout:
// block-major over merge blocks, each row [channel][temporal duplicate]
// [pixel row][pixel column].
func patchifyImage(resized *image.RGBA, gridH, gridW, patch, merge, temporal int32) []float32 {
	n := int(gridH * gridW)
	rowLen := int(3 * temporal * patch * patch)
	pixels := make([]float32, n*rowLen)

	p, t := int(patch), int(temporal)
	patchArea := p * p
	row := 0
	for bh := range gridH / merge {
		for bw := range gridW / merge {
			for mh := range merge {
				for mw := range merge {
					gh, gw := bh*merge+mh, bw*merge+mw
					out := pixels[row*rowLen:]
					baseX, baseY := int(gw)*p, int(gh)*p
					for py := range p {
						px0 := resized.PixOffset(baseX, baseY+py)
						for px := range p {
							pix := resized.Pix[px0+px*4:]
							o := py*p + px
							for c := range 3 {
								val := float32(pix[c])/127.5 - 1
								for ti := range t {
									out[(c*t+ti)*patchArea+o] = val
								}
							}
						}
					}
					row++
				}
			}
		}
	}
	return pixels
}

// visionPatchLookups computes, in the block-major token order, each patch's
// (h, w) rotary position and its four bilinear corners into the learned
// side*side position grid with their weights.
func visionPatchLookups(gridH, gridW, merge, numPosEmbeddings int32) (ropePos, posIdx []int32, posWt []float32) {
	side := int32(math.Sqrt(float64(numPosEmbeddings)))
	n := int(gridH * gridW)
	ropePos = make([]int32, 0, 2*n)
	posIdx = make([]int32, 0, 4*n)
	posWt = make([]float32, 0, 4*n)

	// linspace(0, side-1, g) per axis over the pre-merge grid.
	coord := func(i, g int32) float64 {
		if g == 1 {
			return 0
		}
		return float64(i) * float64(side-1) / float64(g-1)
	}

	for bh := range gridH / merge {
		for bw := range gridW / merge {
			for mh := range merge {
				for mw := range merge {
					gh, gw := bh*merge+mh, bw*merge+mw
					ropePos = append(ropePos, gh, gw)

					hg, wg := coord(gh, gridH), coord(gw, gridW)
					hf, wf := int32(hg), int32(wg)
					hc, wc := min(hf+1, side-1), min(wf+1, side-1)
					hFrac, wFrac := float32(hg-float64(hf)), float32(wg-float64(wf))
					posIdx = append(posIdx, hf*side+wf, hf*side+wc, hc*side+wf, hc*side+wc)
					posWt = append(posWt,
						(1-hFrac)*(1-wFrac), (1-hFrac)*wFrac, hFrac*(1-wFrac), hFrac*wFrac)
				}
			}
		}
	}
	return ropePos, posIdx, posWt
}
