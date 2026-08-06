package gemma4

import (
	"bytes"
	"image"
	"image/png"
	"math"
	"slices"
	"testing"

	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model/base"
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

func TestParseMultimodalConfig(t *testing.T) {
	visionJSON := []byte(`{
		"image_token_id": 258880, "boi_token_id": 255999, "eoi_token_id": 258882,
		"vision_soft_tokens_per_image": 280,
		"vision_config": {"model_type": "gemma4_vision", "hidden_size": 1152,
			"num_hidden_layers": 27, "num_attention_heads": 16, "head_dim": 72,
			"patch_size": 16, "pooling_kernel_size": 3, "default_output_length": 280,
			"standardize": true, "rope_parameters": {"rope_theta": 100.0}},
		"text_config": {"hidden_size": 2816, "use_bidirectional_attention": "vision"}}`)

	mm, err := parseMultimodalConfig(visionJSON)
	if err != nil {
		t.Fatal(err)
	}
	if mm.VisionConfig == nil {
		t.Fatal("vision config not retained")
	}
	if mm.ImageTokenID != 258880 || mm.BOITokenID != 255999 || mm.EOITokenID != 258882 {
		t.Fatalf("token ids = %d/%d/%d", mm.ImageTokenID, mm.BOITokenID, mm.EOITokenID)
	}
	if mm.VisionConfig.RopeTheta != 100 || !mm.VisionConfig.Standardize {
		t.Fatalf("vision config = %+v", mm.VisionConfig)
	}

	cfg, err := parseTextConfig(visionJSON)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.UseBidirectionalAttention != "vision" {
		t.Fatalf("use_bidirectional_attention = %q", cfg.UseBidirectionalAttention)
	}

	// The 12B unified family uses the encoder-free embedder.
	unified, err := parseMultimodalConfig([]byte(`{
		"image_token_id": 258880,
		"vision_config": {"model_type": "gemma4_unified_vision", "mm_embed_dim": 3840,
			"model_patch_size": 48, "num_soft_tokens": 280, "patch_size": 16,
			"pooling_kernel_size": 3, "rms_norm_eps": 1e-6}}`))
	if err != nil {
		t.Fatal(err)
	}
	uv := unified.VisionConfig
	if uv == nil || !uv.unified() {
		t.Fatal("unified vision config not retained")
	}
	if uv.ModelPatchSize != 48 || uv.NumSoftTokens != 280 || uv.MMEmbedDim != 3840 {
		t.Fatalf("unified config = %+v", uv)
	}

	// An unknown vision architecture loads text-only.
	unknown, err := parseMultimodalConfig([]byte(`{
		"vision_config": {"model_type": "gemma5_vision"}}`))
	if err != nil {
		t.Fatal(err)
	}
	if unknown.VisionConfig != nil {
		t.Fatal("unknown vision config should be ignored")
	}

	textOnly, err := parseMultimodalConfig([]byte(`{"text_config": {"hidden_size": 640}}`))
	if err != nil {
		t.Fatal(err)
	}
	if textOnly.VisionConfig != nil {
		t.Fatal("text-only checkpoint grew a vision config")
	}

	if err := validateSoftTokenBudget(280); err != nil {
		t.Fatal(err)
	}
	if err := validateSoftTokenBudget(300); err == nil {
		t.Fatal("budget 300 accepted")
	}
}

func visionTestModel() *Model {
	return &Model{
		TextConfig: &TextConfig{UseBidirectionalAttention: "vision"},
		MM:         multimodalConfig{ImageTokenID: 258880, BOITokenID: 255999, EOITokenID: 258882},
	}
}

func mediaItemAt(pos int, softTokens int32) batch.MediaItem {
	return batch.MediaItem{
		Pos:    pos,
		Opaque: preparedImage{geom: ImageGeometry{NumSoftTokens: softTokens}},
	}
}

func TestBuildMasks(t *testing.T) {
	m := visionTestModel()

	// Soft run [3, 7): boi at 2, eoi at 7.
	item := mediaItemAt(2, 4)
	chunk := func(off, qLen int32) *batch.Batch {
		return &batch.Batch{
			SeqOffsets:   []int32{off},
			SeqQueryLens: []int32{qLen},
			Media:        []batch.MediaItem{item},
		}
	}

	sliding, full := m.buildMasks(&batch.Batch{SeqOffsets: []int32{0}, SeqQueryLens: []int32{8}})
	if !sliding.IsCausal() || !full.IsCausal() {
		t.Fatal("text-only batch left the causal fast path")
	}

	sliding, full = m.buildMasks(chunk(0, 8))
	if sliding.IsCausal() {
		t.Fatal("intersecting run did not relax the sliding mask")
	}
	if !full.IsCausal() {
		t.Fatal("full-attention mask relaxed under \"vision\" semantics")
	}

	// A decode step past the run keeps both masks on the fast path.
	sliding, full = m.buildMasks(chunk(20, 1))
	if !sliding.IsCausal() || !full.IsCausal() {
		t.Fatal("non-intersecting item left the causal fast path")
	}

	m.UseBidirectionalAttention = "all"
	sliding, full = m.buildMasks(chunk(0, 8))
	if sliding.IsCausal() || full.IsCausal() {
		t.Fatal("\"all\" semantics should relax both masks")
	}

	m.UseBidirectionalAttention = ""
	sliding, full = m.buildMasks(chunk(0, 8))
	if !sliding.IsCausal() || !full.IsCausal() {
		t.Fatal("causal-only config relaxed a mask")
	}
}

func TestScatterMedia(t *testing.T) {
	useMLXTestThread(t)

	m := visionTestModel()
	m.HiddenSize = 4

	features := mlx.FromValues([]float32{
		1, 1, 1, 1,
		2, 2, 2, 2,
		3, 3, 3, 3,
		4, 4, 4, 4,
	}, 4, 4)
	item := mediaItemAt(1, 4) // soft run [2, 6)
	item.Features = features

	// Split evaluation: first chunk covers [0, 4) — soft rows 2, 3 — and the
	// resumed chunk [4, 8) covers rows 4, 5 with feature rows sliced by
	// overlap.
	rowVal := func(h *mlx.Array, row int) float32 {
		vals := h.Slice(mlx.Slice(0, 1), mlx.Slice(row, row+1), mlx.Slice()).AsType(mlx.DTypeFloat32)
		mlx.Eval(vals)
		return vals.Floats()[0]
	}

	first := m.scatterMedia(mlx.Zeros(mlx.DTypeFloat32, 1, 4, 4), &batch.Batch{
		SeqOffsets:   []int32{0},
		SeqQueryLens: []int32{4},
		Media:        []batch.MediaItem{item},
	})
	for row, want := range []float32{0, 0, 1, 2} {
		if got := rowVal(first, row); got != want {
			t.Fatalf("first chunk row %d = %v, want %v", row, got, want)
		}
	}

	resumed := m.scatterMedia(mlx.Zeros(mlx.DTypeFloat32, 1, 4, 4), &batch.Batch{
		SeqOffsets:   []int32{4},
		SeqQueryLens: []int32{4},
		Media:        []batch.MediaItem{item},
	})
	for row, want := range []float32{3, 4, 0, 0} {
		if got := rowVal(resumed, row); got != want {
			t.Fatalf("resumed chunk row %d = %v, want %v", row, got, want)
		}
	}

	// Featureless items (decode) scatter nothing.
	item.Features = nil
	decode := m.scatterMedia(mlx.Zeros(mlx.DTypeFloat32, 1, 1, 4), &batch.Batch{
		SeqOffsets:   []int32{7},
		SeqQueryLens: []int32{1},
		Media:        []batch.MediaItem{item},
	})
	if got := rowVal(decode, 0); got != 0 {
		t.Fatalf("featureless scatter wrote %v", got)
	}
}

// TestEncodeImagePooling checks the encode chain end to end on a towerless
// config (identity projections, zero position table, no encoder layers):
// the output must equal the reference computation — normalize, 3x3 grid
// mean, sqrt(hidden) scale, RMS norm — in the reference soft-token order.
func TestEncodeImagePooling(t *testing.T) {
	useMLXTestThread(t)

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

// TestEncodeUnified checks the encode chain end to end on an identity-sized
// config against a reference computation of LN, position add, LN, and RMS
// norm.
func TestEncodeUnified(t *testing.T) {
	useMLXTestThread(t)

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

	out := m.encodeUnified(mlx.FromValues(patches, n, d), positions, ImageGeometry{PatchesW: 2, PatchesH: 2, NumSoftTokens: n})
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
}

func TestPrepareMediaExpansion(t *testing.T) {
	m := visionTestModel()
	m.Vision = &VisionConfig{
		ModelType:         "gemma4_vision",
		HiddenSize:        12,
		PatchSize:         16,
		PoolingKernelSize: 3,
		DefaultOutputLen:  70,
		RMSNormEps:        1e-6,
	}
	m.VisionTower = &VisionTower{}

	var buf bytes.Buffer
	if err := png.Encode(&buf, image.NewRGBA(image.Rect(0, 0, 96, 96))); err != nil {
		t.Fatal(err)
	}

	prepared, err := m.PrepareMedia([]base.Segment{
		{Tokens: []int32{9}},
		{Kind: "image", Data: buf.Bytes()},
	})
	if err != nil {
		t.Fatal(err)
	}

	// 96x96 at budget 70 resizes to 384x384: 24x24 patches, 64 soft tokens,
	// spliced after the one-token text run.
	wantSoft := 64
	if len(prepared.Tokens) != 1+wantSoft+2 || prepared.Tokens[0] != 9 {
		t.Fatalf("stream length = %d, want %d", len(prepared.Tokens), 1+wantSoft+2)
	}
	if len(prepared.Items) != 1 {
		t.Fatalf("items = %d, want 1", len(prepared.Items))
	}
	item := prepared.Items[0]
	if item.Range != [2]int{1, 1 + wantSoft + 2} || item.Source != 1 {
		t.Fatalf("item range = %v source = %d", item.Range, item.Source)
	}
	exp := prepared.Tokens[item.Range[0]:item.Range[1]]
	if exp[0] != m.MM.BOITokenID || exp[len(exp)-1] != m.MM.EOITokenID {
		t.Fatal("expansion not wrapped in boi/eoi")
	}
	for _, tok := range exp[1 : len(exp)-1] {
		if tok != m.MM.ImageTokenID {
			t.Fatalf("soft token = %d", tok)
		}
	}
	if item.Dims[0] != 24*24 || item.Dims[1] != 16*16*3 {
		t.Fatalf("dims = %v", item.Dims)
	}
	if len(item.MediaData) != item.Dims[0]*item.Dims[1] {
		t.Fatalf("media data length = %d", len(item.MediaData))
	}
	p := item.Opaque.(preparedImage)
	if p.geom.NumSoftTokens != int32(wantSoft) || len(p.positions) != 2*24*24 {
		t.Fatalf("geometry = %+v, positions = %d", p.geom, len(p.positions))
	}

	if _, err := m.PrepareMedia([]base.Segment{{Kind: "audio", Data: []byte{1}}}); err == nil {
		t.Fatal("audio accepted")
	}
	m.VisionTower = nil
	if _, err := m.PrepareMedia([]base.Segment{{Kind: "image", Data: buf.Bytes()}}); err == nil {
		t.Fatal("towerless model accepted an image")
	}
}
