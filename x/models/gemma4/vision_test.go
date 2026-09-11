package gemma4

import (
	"image"
	"math"
	"slices"
	"testing"

	"github.com/ollama/ollama/x/internal/mlxtest"

	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/models/nn"
)

func TestVisionTargetSize(t *testing.T) {
	cases := []struct {
		name           string
		h, w, budget   int32
		wantH, wantW   int32
		wantSoftTokens int32
	}{
		{"landscape", 768, 1024, 280, 672, 912, 266},
		{"square", 896, 896, 280, 768, 768, 256},
		{"square small budget", 896, 896, 70, 384, 384, 64},
		{"square large budget", 896, 896, 1120, 1584, 1584, 1089},
		{"extreme aspect clamps", 10, 10000, 280, 48, 13440, 280},
		{"tiny upscales", 20, 20, 280, 768, 768, 256},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			gotH, gotW, err := visionTargetSize(c.h, c.w, 16, 3, c.budget*9)
			if err != nil {
				t.Fatal(err)
			}
			if gotH != c.wantH || gotW != c.wantW {
				t.Fatalf("target = %dx%d, want %dx%d", gotW, gotH, c.wantW, c.wantH)
			}
			soft := (gotH / 16) * (gotW / 16) / 9
			if soft != c.wantSoftTokens {
				t.Fatalf("soft tokens = %d, want %d", soft, c.wantSoftTokens)
			}
		})
	}
}

// TestEncodeImagePooling checks the encode chain end to end on a towerless
// config (identity projections, zero position table, no encoder layers):
// the output must equal the reference computation — normalize, 3x3 grid
// mean, sqrt(hidden) scale, RMS norm — in the reference soft-token order.
func TestEncodeImagePooling(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		const (
			grid    = 6
			pool    = 3
			patchD  = 12 // patch 2x2 x RGB, doubling as hidden size
			patches = grid * grid
		)

		identity := make([]float32, patchD*patchD)
		for i := range patchD {
			identity[i*patchD+i] = 1
		}

		v := &VisionConfig{
			HiddenSize:        patchD,
			HeadDim:           4,
			PoolingKernelSize: pool,
			PatchSize:         2,
			RMSNormEps:        1e-6,
			RopeTheta:         100,
		}
		m := &Model{
			Vision: v,
			VisionTower: &VisionTower{
				PatchEmbedder: &PatchEmbedder{
					InputProj:              nn.NewLinear(mlx.FromValues(identity, patchD, patchD), nil),
					PositionEmbeddingTable: mlx.Zeros(mlx.DTypeFloat32, 2, 8, patchD),
				},
			},
			EmbedVision: &MultimodalEmbedder{Projection: nn.NewLinear(mlx.FromValues(identity, patchD, patchD), nil)},
		}

		pixels := make([]float32, patches*patchD)
		positions := make([]int32, 2*patches)
		for p := range patches {
			positions[2*p] = int32(p % grid)
			positions[2*p+1] = int32(p / grid)
			for d := range patchD {
				pixels[p*patchD+d] = float32((p*31+d*7)%97) / 97
			}
		}

		geom := ImageGeometry{PatchesW: grid, PatchesH: grid, NumSoftTokens: patches / (pool * pool)}
		out := m.encodeImage(mlx.FromValues(pixels, patches, patchD), positions, geom)
		out = out.AsType(mlx.DTypeFloat32)
		mlx.Eval(out)
		got := out.Floats()

		scale := float32(math.Sqrt(patchD))
		for by := range grid / pool {
			for bx := range grid / pool {
				var block [patchD]float32
				for iy := range pool {
					for ix := range pool {
						p := (by*pool+iy)*grid + (bx*pool + ix)
						for d := range patchD {
							block[d] += 2 * (pixels[p*patchD+d] - 0.5)
						}
					}
				}
				var sumsq float64
				for d := range patchD {
					block[d] = block[d] / (pool * pool) * scale
					sumsq += float64(block[d]) * float64(block[d])
				}
				rms := float32(math.Sqrt(sumsq/patchD + 1e-6))

				b := by*(grid/pool) + bx
				for d := range patchD {
					want := block[d] / rms
					if diff := math.Abs(float64(got[b*patchD+d] - want)); diff > 0.03 {
						t.Fatalf("soft token %d dim %d = %v, want %v", b, d, got[b*patchD+d], want)
					}
				}
			}
		}
	})
}

// TestPatchifyMerged checks the unified layout against an independent index
// computation: one raster patch of pool*patchSize pixels per soft token,
// (pixel row, pixel column, RGB) across the whole merged patch.
func TestPatchifyMerged(t *testing.T) {
	const patch, pool = 2, 2
	const merged = patch * pool
	const w, h = 8, 4 // 2x1 model patches of 4px

	img := image.NewRGBA(image.Rect(0, 0, w, h))
	pixel := func(x, y, c int) uint8 { return uint8(x*16 + y*3 + c) }
	for y := range h {
		for x := range w {
			o := img.PixOffset(x, y)
			for c := range 3 {
				img.Pix[o+c] = pixel(x, y, c)
			}
			img.Pix[o+3] = 255
		}
	}

	pixels, positions, geom := patchify(img, w, h, merged, 1)
	if geom.PatchesW != 2 || geom.PatchesH != 1 || geom.NumSoftTokens != 2 {
		t.Fatalf("geometry = %+v", geom)
	}
	if want := []int32{0, 0, 1, 0}; !slices.Equal(positions, want) {
		t.Fatalf("positions = %v, want %v", positions, want)
	}

	patchLen := merged * merged * 3
	for i, got := range pixels {
		p := i / patchLen
		e := i % patchLen
		px, py, c := e/3%merged, e/3/merged, e%3
		x := p*merged + px // single model-patch row, so no gy term
		y := py
		if want := float32(pixel(x, y, c)) / 255; got != want {
			t.Fatalf("pixels[%d] = %v, want %v (x=%d y=%d c=%d)", i, got, want, x, y, c)
		}
	}
}

// TestEncodeUnifiedImage checks the encode chain end to end on an identity-sized
// config against a reference computation of LN, position add, LN, and RMS
// norm.
func TestEncodeUnifiedImage(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		const d = 12 // merged patch dim (1px teacher patches, 2x2 merge, RGB), doubling as embed dim
		identity := make([]float32, d*d)
		for i := range d {
			identity[i*d+i] = 1
		}
		ones := make([]float32, d)
		for i := range ones {
			ones[i] = 1
		}
		unitLN := func() *nn.LayerNorm {
			return &nn.LayerNorm{Weight: mlx.FromValues(ones, d), Bias: mlx.Zeros(mlx.DTypeFloat32, d), Eps: 1e-5}
		}

		posTable := make([]float32, 2*2*d)
		for r := range 2 {
			for a := range 2 {
				for i := range d {
					posTable[(r*2+a)*d+i] = float32(r+1) * float32(a+1) / 8
				}
			}
		}

		m := &Model{
			Vision: &VisionConfig{ModelType: "gemma4_unified_vision", RMSNormEps: 1e-6},
			UnifiedEmbedder: &UnifiedVisionEmbedder{
				PatchLN1:     unitLN(),
				PatchDense:   nn.NewLinear(mlx.FromValues(identity, d, d), nil),
				PatchLN2:     unitLN(),
				PosEmbedding: mlx.FromValues(posTable, 2, 2, d),
				PosNorm:      unitLN(),
			},
			EmbedVision: &MultimodalEmbedder{Projection: nn.NewLinear(mlx.FromValues(identity, d, d), nil)},
		}

		const n = 4
		patches := make([]float32, n*d)
		for i := range patches {
			patches[i] = float32((i*13)%29) / 29
		}
		positions := []int32{0, 0, 1, 0, 0, 1, 1, 1}

		out := m.encodeUnifiedImage(mlx.FromValues(patches, n, d), positions, ImageGeometry{PatchesW: 2, PatchesH: 2, NumSoftTokens: n})
		out = out.AsType(mlx.DTypeFloat32)
		mlx.Eval(out)
		got := out.Floats()

		ln := func(v []float64) []float64 {
			var mean, varSum float64
			for _, x := range v {
				mean += x
			}
			mean /= float64(len(v))
			for _, x := range v {
				varSum += (x - mean) * (x - mean)
			}
			varSum /= float64(len(v))
			outV := make([]float64, len(v))
			for i, x := range v {
				outV[i] = (x - mean) / math.Sqrt(varSum+1e-5)
			}
			return outV
		}

		for p := range n {
			v := make([]float64, d)
			for i := range d {
				v[i] = float64(patches[p*d+i])
			}
			v = ln(ln(v)) // LN1, identity dense, LN2
			x, y := int(positions[2*p]), int(positions[2*p+1])
			for i := range d {
				v[i] += float64(posTable[(x*2+0)*d+i]) + float64(posTable[(y*2+1)*d+i])
			}
			v = ln(v)
			var sumsq float64
			for _, x := range v {
				sumsq += x * x
			}
			rms := math.Sqrt(sumsq/d + 1e-6)
			for i := range d {
				want := v[i] / rms
				if diff := math.Abs(float64(got[p*d+i]) - want); diff > 0.03 {
					t.Fatalf("patch %d dim %d = %v, want %v", p, i, got[p*d+i], want)
				}
			}
		}
	})
}
