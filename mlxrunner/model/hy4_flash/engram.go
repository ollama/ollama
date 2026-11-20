package hy4_flash

import (
	"encoding/binary"
	"fmt"
	"math"
	"slices"
	"strconv"

	"github.com/ollama/ollama/fs/safetensors"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/cache"
	"github.com/ollama/ollama/mlxrunner/model"
	"github.com/ollama/ollama/mlxrunner/nn"
)

// Engram uses the FP32 accumulator's epsilon, independently of the text RMSNorm.
const engramRMSEpsilon = 1.1920929e-7

type engram struct {
	table       *model.HostMapping
	offset      int64
	tokenMap    []int32
	multipliers []int64
	moduli      []int
	headOffsets []int
	HeadNorm    *mlx.Array
	Value       nn.LinearLayer
	Gamma       *mlx.Array
	Streams     []engramStream
}

type engramStream struct {
	Key                nn.LinearLayer
	KeyNorm, QueryNorm *nn.RMSNorm
	Scale              *mlx.Array
}

func openEngram(path, name string, moduli [][]int, multipliers []int64, c *textConfig) (*engram, error) {
	if len(moduli) != c.EngramOrder-1 || len(multipliers) != c.EngramOrder {
		return nil, fmt.Errorf("%s: invalid Engram hash metadata", name)
	}
	for _, multiplier := range multipliers {
		if multiplier <= 0 || multiplier%2 == 0 {
			return nil, fmt.Errorf("%s: invalid Engram hash multiplier", name)
		}
	}
	headModuli := make([]int, 0, (c.EngramOrder-1)*c.EngramHeads)
	for _, order := range moduli {
		if len(order) != c.EngramHeads {
			return nil, fmt.Errorf("%s: invalid Engram head count", name)
		}
		headModuli = append(headModuli, order...)
	}
	rows := 0
	offsets := make([]int, len(headModuli))
	for i, modulus := range headModuli {
		if modulus <= 0 || modulus > math.MaxInt32-rows {
			return nil, fmt.Errorf("%s: invalid Engram modulus %d", name, modulus)
		}
		offsets[i] = rows
		rows += modulus
	}
	data, err := safetensors.OpenForExtraction(path)
	if err != nil {
		return nil, err
	}
	defer data.Close()
	tensor, err := data.GetTensor(name)
	if err != nil {
		return nil, err
	}
	width := c.EngramWidth / c.EngramHeads
	if tensor.Dtype != "BF16" || !slices.Equal(tensor.Shape, []int32{int32(rows), int32(width)}) {
		return nil, fmt.Errorf("%s: Engram needs a BF16 table with shape [%d, %d], got %s %v", name, rows, width, tensor.Dtype, tensor.Shape)
	}
	mapped, err := model.OpenHostMapping(path)
	if err != nil {
		return nil, err
	}
	return &engram{table: mapped, offset: tensor.Offset(), multipliers: multipliers, moduli: headModuli, headOffsets: offsets}, nil
}

func engramTableName(layer int) string {
	return fmt.Sprintf("model.layers.%d.engram_layer.multi_head_embedding.embedding.weight", layer-1)
}

func (m *Model) loadEngrams(root *model.Root) error {
	if len(m.cfg.Text.EngramLayers) == 0 {
		return nil
	}
	var metadata struct {
		Multipliers map[string][]int64 `json:"hash_multipliers"`
	}
	if err := root.Manifest.ReadConfigJSON("engram_metadata.json", &metadata); err != nil {
		return fmt.Errorf("load Hy4 Flash Engram metadata: %w", err)
	}
	blobs := make(map[string]string)
	for _, layer := range root.Manifest.TensorLayers() {
		blobs[layer.Name] = layer.Digest
	}
	for _, id := range m.cfg.Text.EngramLayers {
		if id <= 0 || id > len(m.Layers) || m.Layers[id-1].Engram != nil {
			return fmt.Errorf("invalid Engram layer %d", id)
		}
		name := engramTableName(id)
		digest, ok := blobs[name]
		if !ok {
			return fmt.Errorf("missing Engram table %s", name)
		}
		path, err := manifest.BlobsPath(digest)
		if err != nil {
			return err
		}
		key := strconv.Itoa(id)
		e, err := openEngram(path, name, m.cfg.Text.EngramModuli[key], metadata.Multipliers[key], &m.cfg.Text)
		if err != nil {
			return err
		}
		m.Layers[id-1].Engram = e
	}
	return nil
}

func (e *engram) loadWeights(w weights, prefix string, c *textConfig) error {
	tokens := w.tensor(prefix + ".lookup_table")
	e.Value = w.linear(prefix + ".value_proj")
	e.Gamma = w.tensor(prefix + ".unified_pre_gamma")
	norm := func(name string) *nn.RMSNorm {
		return nn.NewRMSNorm(w.tensor(name).AsType(mlx.DTypeFloat32), engramRMSEpsilon)
	}
	headWeights := make([]*mlx.Array, len(e.moduli))
	for i := range headWeights {
		headWeights[i] = w.tensor(fmt.Sprintf("%s.lookup_per_head_norms.%d.weight", prefix, i)).AsType(mlx.DTypeFloat32)
	}
	e.Streams = make([]engramStream, c.Streams)
	for i := range e.Streams {
		suffix := "." + strconv.Itoa(i)
		e.Streams[i] = engramStream{
			Key:       w.linear(prefix + ".key_projs" + suffix),
			KeyNorm:   norm(prefix + ".norm_key" + suffix + ".weight"),
			QueryNorm: norm(prefix + ".norm_query" + suffix + ".weight"),
			Scale:     w.tensor(prefix + ".unified_per_hc_scales" + suffix),
		}
	}
	if *w.err != nil {
		return *w.err
	}
	e.HeadNorm = mlx.Stack(headWeights, 0)
	e.tokenMap = tokens.AsType(mlx.DTypeInt32).Ints()
	if int(c.EngramPadID) >= len(e.tokenMap) {
		return fmt.Errorf("%s: Engram pad token outside token map", prefix)
	}
	maxToken := math.MaxInt64 / slices.Max(e.multipliers)
	for _, token := range e.tokenMap {
		if token < 0 || int64(token) > maxToken {
			return fmt.Errorf("%s: Engram token map is incompatible with the hash multipliers", prefix)
		}
	}
	return nil
}

func engramTokens(b *batch.Batch, slot cache.Layer, order int) [][]int32 {
	if slot == nil {
		tokens := b.InputIDs.AsType(mlx.DTypeInt32).Ints()
		length := b.InputIDs.Dim(1)
		result := make([][]int32, b.InputIDs.Dim(0))
		for row := range result {
			result[row] = tokens[row*length : row*length+int(b.SeqQueryLens[row])]
		}
		return result
	}
	ids := b.InputIDs.Reshape(b.InputIDs.Dim(0), 1, b.InputIDs.Dim(1), 1)
	placeholder := mlx.Zeros(mlx.DTypeInt32, ids.Dim(0), 1, ids.Dim(2), 0)
	history := slot.(cache.Attention).Update(b, ids, placeholder)
	rows := history.Rows()
	if rows == nil {
		rows = []*nn.KVHistory{history}
	}
	result := make([][]int32, len(rows))
	for row, h := range rows {
		keys := h.K()
		start := max(0, keys.Dim(2)-int(b.SeqQueryLens[row])-order+1)
		result[row] = mlx.Contiguous(keys.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(start, mlx.End), mlx.Slice()), false).AsType(mlx.DTypeInt32).Ints()
	}
	return result
}

func (e *engram) gather(tokens [][]int32, b *batch.Batch, cfg *config) (*mlx.Array, *mlx.Array, error) {
	c := &cfg.Text
	B, L := b.InputIDs.Dim(0), b.InputIDs.Dim(1)
	width := c.EngramWidth / c.EngramHeads
	heads := len(e.moduli)
	values := make([]float32, B*L*heads*width)
	mask := make([]bool, B*L)
	for row, history := range tokens {
		first := len(history) - int(b.SeqQueryLens[row])
		for i := first; i < len(history); i++ {
			token := history[i]
			pos := row*L + i - first
			if token == c.EngramPadID || token == cfg.ImageTokenID || token == cfg.VideoTokenID {
				continue
			}
			mask[pos] = true
			var mixed int64
			for shift, multiplier := range e.multipliers {
				id := c.EngramPadID
				if i >= shift && history[i-shift] >= 0 && int(history[i-shift]) < len(e.tokenMap) {
					id = history[i-shift]
				}
				mixed ^= int64(e.tokenMap[id]) * multiplier
				if shift == 0 {
					continue
				}
				for h := (shift - 1) * c.EngramHeads; h < shift*c.EngramHeads; h++ {
					rowID := e.headOffsets[h] + int(mixed%int64(e.moduli[h]))
					offset := e.offset + int64(rowID)*int64(width*2)
					raw, err := e.table.Bytes(offset, offset+int64(width*2))
					if err != nil {
						return nil, nil, err
					}
					out := values[(pos*heads+h)*width:]
					for j := range width {
						out[j] = math.Float32frombits(uint32(binary.LittleEndian.Uint16(raw[2*j:])) << 16)
					}
				}
			}
		}
	}
	return mlx.FromValues(values, B, L, heads, width), mlx.FromValues(mask, B, L, 1, 1), nil
}

func (e *engram) forward(x *mlx.Array, tokens [][]int32, b *batch.Batch, cfg *config) *mlx.Array {
	embeddings, mask, err := e.gather(tokens, b, cfg)
	if err != nil {
		panic(fmt.Errorf("read Hy4 Flash Engram: %w", err))
	}
	c := &cfg.Text
	embeddings = embeddings.AsType(x.DType())
	if c.EngramFP8 {
		scale := mlx.DivScalar(mlx.Maximum(embeddings.Abs().MaxAxis(-1, true), mlx.FromValue(float32(1e-12)).AsType(x.DType())), 448)
		scaled := mlx.Div(embeddings, scale)
		limit := mlx.FromValue(float32(448)).AsType(x.DType())
		rounded := mlx.Mul(mlx.FromFP8(mlx.ToFP8(mlx.Clip(scaled, limit.Negative(), limit)), x.DType()), scale)
		embeddings = mlx.Add(embeddings, mlx.Sub(rounded, embeddings))
	}
	// RMSNorm each hash vector independently in one dispatch, then apply
	// its learned weight in FP32 before rounding back to the input dtype.
	normalized := mlx.RMSNormFn(embeddings.AsType(mlx.DTypeFloat32), nil, engramRMSEpsilon)
	memory := mlx.Mul(normalized, e.HeadNorm).AsType(x.DType()).Reshape(x.Dim(0), x.Dim(1), -1)
	signal := mlx.Mul(mlx.RMSNormFn(e.Value.Forward(memory), nil, engramRMSEpsilon), e.Gamma)
	output := make([]*mlx.Array, len(e.Streams))
	invDim := float32(1 / math.Sqrt(float64(c.HiddenSize)))
	for i, stream := range e.Streams {
		query := mlx.Squeeze(x.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(i, i+1), mlx.Slice()), 2)
		query = engramNorm(stream.QueryNorm, query)
		key := engramNorm(stream.KeyNorm, stream.Key.Forward(memory))
		dot := mlx.Sum(mlx.Mul(query, key), -1, true).AsType(mlx.DTypeFloat32)
		dot = mlx.MulScalar(dot, invDim).AsType(x.DType())
		magnitude := mlx.Maximum(dot.Abs(), mlx.FromValue(float32(1e-6)).AsType(dot.DType())).Sqrt()
		gate := mlx.Mul(magnitude, dot.Sign()).AsType(mlx.DTypeFloat32).Sigmoid().AsType(x.DType())
		output[i] = mlx.Add(mlx.Mul(signal, gate), mlx.Mul(signal, stream.Scale))
	}
	result := mlx.Stack(output, 2).AsType(x.DType())
	return mlx.Where(mask, result, mlx.FromValue(float32(0)).AsType(result.DType()))
}

func engramNorm(n *nn.RMSNorm, x *mlx.Array) *mlx.Array {
	return n.Forward(x.AsType(mlx.DTypeFloat32), 0).AsType(x.DType())
}
