package glimmer

import (
	"fmt"
	"math"

	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model"
	"github.com/ollama/ollama/x/models/nn"
)

type VisionEncoder struct {
	Conv1Linear            nn.LinearLayer
	PositionalEmbeddingVLM *mlx.Array
	LNPre                  *nn.LayerNorm
	Layers                 []*VisionBlock
	LNPost                 *nn.LayerNorm
}

type VisionBlock struct {
	LN1       *nn.LayerNorm
	Attention *VisionAttention
	LN2       *nn.LayerNorm
	MLP       *VisionMLP
}

type VisionAttention struct {
	QProj           nn.LinearLayer
	KProj           nn.LinearLayer
	VProj           nn.LinearLayer
	OProj           nn.LinearLayer
	InterleavedRoPE bool
}

type VisionMLP struct {
	FC   nn.LinearLayer
	Proj nn.LinearLayer
}

type VisionAdapter struct {
	FC   nn.LinearLayer
	Proj nn.LinearLayer
}

func (m *Model) loadVisionWeights(tensors map[string]*mlx.Array, linears model.LinearFactory) error {
	officialHF := tensors["model.vision_tower.patch_embedder.position_embedding_table.weight"] != nil
	encoderPrefix := "model.vision_encoder."
	convPath := encoderPrefix + "conv1_linear"
	positionPath := encoderPrefix + "positional_embedding_vlm"
	if officialHF {
		encoderPrefix = "model.vision_tower."
		convPath = encoderPrefix + "patch_embedder.patch_embedding"
		positionPath = encoderPrefix + "patch_embedder.position_embedding_table.weight"
	}
	encoder := &VisionEncoder{
		Conv1Linear:            linears.Make(convPath),
		PositionalEmbeddingVLM: tensors[positionPath],
		LNPre: &nn.LayerNorm{
			Weight: tensors[encoderPrefix+"ln_pre.weight"],
			Bias:   tensors[encoderPrefix+"ln_pre.bias"],
			Eps:    1e-5,
		},
		Layers: make([]*VisionBlock, m.VisionLayers),
		LNPost: &nn.LayerNorm{
			Weight: tensors[encoderPrefix+"ln_post.weight"],
			Bias:   tensors[encoderPrefix+"ln_post.bias"],
			Eps:    1e-5,
		},
	}
	if encoder.Conv1Linear == nil || encoder.PositionalEmbeddingVLM == nil ||
		encoder.LNPre.Weight == nil || encoder.LNPre.Bias == nil ||
		encoder.LNPost.Weight == nil || encoder.LNPost.Bias == nil {
		return fmt.Errorf("missing vision encoder input or normalization weights")
	}

	for i := range m.VisionLayers {
		prefix := fmt.Sprintf("%stransformer.%d.", encoderPrefix, i)
		ln1, ln2 := "ln_1", "ln_2"
		attnQ, attnK, attnV, attnO := "attn.q_proj", "attn.k_proj", "attn.v_proj", "attn.o_proj"
		mlpFC, mlpProj := "mlp.c_fc", "mlp.c_proj"
		if officialHF {
			prefix = fmt.Sprintf("%slayers.%d.", encoderPrefix, i)
			ln1, ln2 = "norm1", "norm2"
			attnQ, attnK, attnV, attnO = "attn.q_proj", "attn.k_proj", "attn.v_proj", "attn.proj"
			mlpFC, mlpProj = "mlp.fc1", "mlp.fc2"
		}
		block := &VisionBlock{
			LN1: &nn.LayerNorm{
				Weight: tensors[prefix+ln1+".weight"],
				Bias:   tensors[prefix+ln1+".bias"],
				Eps:    1e-5,
			},
			Attention: &VisionAttention{
				QProj:           linears.Make(prefix + attnQ),
				KProj:           linears.Make(prefix + attnK),
				VProj:           linears.Make(prefix + attnV),
				OProj:           linears.Make(prefix + attnO),
				InterleavedRoPE: m.VisionRoPEInterleaved,
			},
			LN2: &nn.LayerNorm{
				Weight: tensors[prefix+ln2+".weight"],
				Bias:   tensors[prefix+ln2+".bias"],
				Eps:    1e-5,
			},
			MLP: &VisionMLP{
				FC:   linears.Make(prefix + mlpFC),
				Proj: linears.Make(prefix + mlpProj),
			},
		}
		if block.LN1.Weight == nil || block.LN1.Bias == nil ||
			block.LN2.Weight == nil || block.LN2.Bias == nil ||
			block.Attention.QProj == nil || block.Attention.KProj == nil ||
			block.Attention.VProj == nil || block.Attention.OProj == nil ||
			block.MLP.FC == nil || block.MLP.Proj == nil {
			return fmt.Errorf("vision layer %d: missing weights", i)
		}
		encoder.Layers[i] = block
	}

	adapterFC, adapterProj := "model.vision_adapter.c_fc", "model.vision_adapter.c_proj"
	if officialHF {
		adapterFC, adapterProj = "model.vision_adapter.fc1", "model.vision_adapter.fc2"
	}
	adapter := &VisionAdapter{FC: linears.Make(adapterFC), Proj: linears.Make(adapterProj)}
	projection := linears.Make("model.vision_projection")
	if adapter.FC == nil || adapter.Proj == nil || projection == nil {
		return fmt.Errorf("missing vision adapter or projection weights")
	}

	m.VisionEncoder = encoder
	m.VisionAdapter = adapter
	m.VisionProjection = projection
	return nil
}

// encodeVision runs the vision tower over one uploaded image, returning the
// lazy [outputTokens, hidden] features. Graph construction only, per the
// base.MediaModel contract — the consuming forward's evaluation pulls the
// encoder, and its intermediates are transient within that single eval.
func (m *Model) encodeVision(patches *mlx.Array, gridH, gridW int) *mlx.Array {
	encoder := m.VisionEncoder
	inputCount := gridH * gridW
	permutation, sparseLens := sparseVisionPermutation(
		gridH,
		gridW,
		int(m.VisionPosEmbeddingGridH),
		int(m.VisionPosEmbeddingGridW),
	)
	positional := interpolateVisionPositions(
		encoder.PositionalEmbeddingVLM,
		gridH,
		gridW,
		int(m.VisionPosEmbeddingGridH),
		int(m.VisionPosEmbeddingGridW),
	)
	cos, sin := visionRoPE(gridH, gridW, int(m.VisionLatentDim/m.VisionHeads))

	x := encoder.Conv1Linear.Forward(patches.AsType(encoder.PositionalEmbeddingVLM.DType()))
	x = mlx.Add(x, positional.ExpandDims(0))
	x = encoder.LNPre.Forward(x)

	var sparseMask *mlx.Array
	if m.VisionSparseAttentionFactor > 1 {
		indices := mlx.FromValues(permutation, len(permutation))
		x = mlx.Take(x, indices, 1)
		cos = mlx.Take(cos, indices, 0)
		sin = mlx.Take(sin, indices, 0)
		sparseMask = visionBlockDiagonalMask(sparseLens, x.DType())
	}

	for i, layer := range encoder.Layers {
		// Full-attention layers attend across the whole single image: no mask.
		mask := sparseMask
		if m.VisionLayerTypes[i] == "full_attention" {
			mask = nil
		}
		x = layer.Forward(x, cos, sin, mask, int(m.VisionHeads))
	}

	if m.VisionSparseAttentionFactor > 1 {
		restore := make([]int32, inputCount)
		for reordered, original := range permutation {
			restore[original] = int32(reordered)
		}
		x = mlx.Take(x, mlx.FromValues(restore, len(restore)), 1)
	}

	x = encoder.LNPost.Forward(x)
	features := pixelShuffleVision(x.Squeeze(0), gridH, gridW, int(m.VisionDownsampleFactor))
	features = m.VisionAdapter.Forward(features)
	features = m.VisionProjection.Forward(features)
	if m.NormalizeVisionEmbeddings {
		features = mlx.RMSNormFn(features, nil, m.RMSNormEps)
	}
	return features
}

func (b *VisionBlock) Forward(x, cos, sin, mask *mlx.Array, heads int) *mlx.Array {
	h := b.LN1.Forward(x)
	h = b.Attention.Forward(h, cos, sin, mask, heads)
	x = mlx.Add(x, h)
	h = b.LN2.Forward(x)
	return mlx.Add(x, b.MLP.Forward(h))
}

func (a *VisionAttention) Forward(x, cos, sin, mask *mlx.Array, heads int) *mlx.Array {
	shape := x.Dims()
	batchSize, sequence, width := int32(shape[0]), int32(shape[1]), int32(shape[2])
	headDim := width / int32(heads)

	q := mlx.Reshape(a.QProj.Forward(x), batchSize, sequence, int32(heads), headDim)
	k := mlx.Reshape(a.KProj.Forward(x), batchSize, sequence, int32(heads), headDim)
	v := mlx.Reshape(a.VProj.Forward(x), batchSize, sequence, int32(heads), headDim)
	q = applyVisionRoPE(q, cos, sin, a.InterleavedRoPE)
	k = applyVisionRoPE(k, cos, sin, a.InterleavedRoPE)
	q = mlx.Transpose(q, 0, 2, 1, 3)
	k = mlx.Transpose(k, 0, 2, 1, 3)
	v = mlx.Transpose(v, 0, 2, 1, 3)

	out := mlx.FastScaledDotProductAttention(q, k, v, float32(1/math.Sqrt(float64(headDim))), "", mask)
	out = mlx.Transpose(out, 0, 2, 1, 3)
	return a.OProj.Forward(mlx.Reshape(out, batchSize, sequence, width))
}

func (m *VisionMLP) Forward(x *mlx.Array) *mlx.Array {
	return m.Proj.Forward(mlx.GELU(m.FC.Forward(x)))
}

func (a *VisionAdapter) Forward(x *mlx.Array) *mlx.Array {
	return mlx.GELU(a.Proj.Forward(mlx.GELU(a.FC.Forward(x))))
}

func sparseVisionPermutation(gridH, gridW, windowH, windowW int) ([]int32, []int) {
	if windowH <= 0 || windowW <= 0 {
		permutation := make([]int32, gridH*gridW)
		for i := range permutation {
			permutation[i] = int32(i)
		}
		return permutation, []int{len(permutation)}
	}

	var permutation []int32
	var lengths []int
	for top := 0; top < gridH; top += windowH {
		for left := 0; left < gridW; left += windowW {
			before := len(permutation)
			for y := top; y < min(top+windowH, gridH); y++ {
				for x := left; x < min(left+windowW, gridW); x++ {
					permutation = append(permutation, int32(y*gridW+x))
				}
			}
			lengths = append(lengths, len(permutation)-before)
		}
	}
	return permutation, lengths
}

func visionBlockDiagonalMask(lengths []int, dtype mlx.DType) *mlx.Array {
	if len(lengths) <= 1 {
		return nil
	}
	total := 0
	for _, length := range lengths {
		total += length
	}
	if total == 0 {
		return nil
	}
	// Expand a per-block matrix by each patch's block index, so the
	// quadratic tensor is built on the device.
	blocks := make([]float32, len(lengths)*len(lengths))
	for i := range blocks {
		blocks[i] = float32(math.Inf(-1))
	}
	ids := make([]int32, total)
	offset := 0
	for b, length := range lengths {
		blocks[b*len(lengths)+b] = 0
		for i := offset; i < offset+length; i++ {
			ids[i] = int32(b)
		}
		offset += length
	}
	index := mlx.FromValues(ids, total)
	m := mlx.Take(mlx.FromValues(blocks, len(lengths), len(lengths)), index, 0)
	m = mlx.Take(m, index, 1)
	return mlx.Reshape(m, 1, 1, int32(total), int32(total)).AsType(dtype)
}

func interpolateVisionPositions(table *mlx.Array, gridH, gridW, tableH, tableW int) *mlx.Array {
	count := gridH * gridW
	indices := [4][]int32{
		make([]int32, count),
		make([]int32, count),
		make([]int32, count),
		make([]int32, count),
	}
	weights := [4][]float32{
		make([]float32, count),
		make([]float32, count),
		make([]float32, count),
		make([]float32, count),
	}

	at := 0
	// This order and coordinate assignment intentionally match the reference's
	// meshgrid(ys, xs, indexing="xy") call.
	for x := range gridW {
		for y := range gridH {
			sourceX := (float64(y)+0.5)*float64(tableW)/float64(gridH) - 0.5
			sourceY := (float64(x)+0.5)*float64(tableH)/float64(gridW) - 0.5
			x0, y0 := int(math.Floor(sourceX)), int(math.Floor(sourceY))
			dx, dy := float32(sourceX-float64(x0)), float32(sourceY-float64(y0))
			neighbors := [4]struct {
				x, y   int
				weight float32
			}{
				{x0, y0, (1 - dx) * (1 - dy)},
				{x0 + 1, y0, dx * (1 - dy)},
				{x0, y0 + 1, (1 - dx) * dy},
				{x0 + 1, y0 + 1, dx * dy},
			}
			for i, neighbor := range neighbors {
				if neighbor.x >= 0 && neighbor.x < tableW && neighbor.y >= 0 && neighbor.y < tableH {
					indices[i][at] = int32(neighbor.y*tableW + neighbor.x)
					weights[i][at] = neighbor.weight
				}
			}
			at++
		}
	}

	var result *mlx.Array
	for i := range indices {
		selected := mlx.Take(table, mlx.FromValues(indices[i], len(indices[i])), 0)
		weight := mlx.FromValues(weights[i], len(weights[i]), 1).AsType(table.DType())
		term := mlx.Mul(selected, weight)
		if result == nil {
			result = term
		} else {
			result = mlx.Add(result, term)
		}
	}
	return result
}

func visionRoPE(gridH, gridW, headDim int) (*mlx.Array, *mlx.Array) {
	half := headDim / 2
	quarter := half / 2
	cosValues := make([]float32, gridH*gridW*half)
	sinValues := make([]float32, len(cosValues))
	for y := range gridH {
		for x := range gridW {
			base := (y*gridW + x) * half
			for i := range quarter {
				frequency := math.Pow(10000, -2*float64(i)/float64(half))
				widthAngle := float64(x+1) * frequency
				heightAngle := float64(y+1) * frequency
				cosValues[base+i] = float32(math.Cos(widthAngle))
				sinValues[base+i] = float32(math.Sin(widthAngle))
				cosValues[base+quarter+i] = float32(math.Cos(heightAngle))
				sinValues[base+quarter+i] = float32(math.Sin(heightAngle))
			}
		}
	}
	return mlx.FromValues(cosValues, gridH*gridW, half), mlx.FromValues(sinValues, gridH*gridW, half)
}

func applyVisionRoPE(x, cos, sin *mlx.Array, interleaved bool) *mlx.Array {
	shape := x.Dims()
	batchSize, sequence, heads, headDim := int32(shape[0]), int32(shape[1]), int32(shape[2]), int32(shape[3])
	dtype := x.DType()
	if !interleaved {
		x = x.AsType(mlx.DTypeFloat32)
		cos = mlx.Concatenate([]*mlx.Array{cos, cos}, 1).AsType(mlx.DTypeFloat32).ExpandDims(0).ExpandDims(2)
		sin = mlx.Concatenate([]*mlx.Array{sin, sin}, 1).AsType(mlx.DTypeFloat32).ExpandDims(0).ExpandDims(2)
		half := int(headDim / 2)
		first := x.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(), mlx.Slice(0, half))
		second := x.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(), mlx.Slice(half, int(headDim)))
		rotated := mlx.Concatenate([]*mlx.Array{mlx.Neg(second), first}, 3)
		return mlx.Add(mlx.Mul(x, cos), mlx.Mul(rotated, sin)).AsType(dtype)
	}

	pairs := mlx.Reshape(x.AsType(mlx.DTypeFloat32), batchSize, sequence, heads, headDim/2, 2)
	real := pairs.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(), mlx.Slice(), mlx.Slice(0)).Squeeze(4)
	imag := pairs.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(), mlx.Slice(), mlx.Slice(1)).Squeeze(4)
	cos = cos.AsType(mlx.DTypeFloat32).ExpandDims(0).ExpandDims(2)
	sin = sin.AsType(mlx.DTypeFloat32).ExpandDims(0).ExpandDims(2)
	outReal := mlx.Sub(mlx.Mul(real, cos), mlx.Mul(imag, sin))
	outImag := mlx.Add(mlx.Mul(real, sin), mlx.Mul(imag, cos))
	return mlx.Reshape(mlx.Stack([]*mlx.Array{outReal, outImag}, 4), batchSize, sequence, heads, headDim).AsType(dtype)
}

func pixelShuffleVision(x *mlx.Array, gridH, gridW, factor int) *mlx.Array {
	outputH, outputW := gridH/factor, gridW/factor
	permutation := make([]int32, 0, gridH*gridW)
	for y := range outputH {
		for x := range outputW {
			for innerY := range factor {
				for innerX := range factor {
					permutation = append(permutation, int32((y*factor+innerY)*gridW+x*factor+innerX))
				}
			}
		}
	}
	x = mlx.Take(x, mlx.FromValues(permutation, len(permutation)), 0)
	width := int32(x.Dim(1))
	x = mlx.Reshape(x, int32(outputH*outputW), int32(factor*factor), width)
	x = mlx.Transpose(x, 0, 2, 1)
	return mlx.Reshape(x, int32(outputH*outputW), width*int32(factor*factor))
}
