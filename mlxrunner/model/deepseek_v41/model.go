package deepseek_v41

import (
	"fmt"
	"strings"

	"github.com/ollama/ollama/fs/safetensors"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/cache"
	"github.com/ollama/ollama/mlxrunner/model"
	"github.com/ollama/ollama/mlxrunner/nn"
	"github.com/ollama/ollama/mlxrunner/tokenizer"
)

func init() {
	model.Register("DeepseekV41ForCausalLM", func(root *model.Root) (model.Model, error) {
		return NewModel(root)
	})
}

var (
	_ model.Model               = (*Model)(nil)
	_ model.SkippedTensorsModel = (*Model)(nil)
)

// Model holds the DeepSeek text and vision weights plus host-backed engram
// tables. The optional MTP draft branch is not loaded yet.
type Model struct {
	Config      *Config
	Embedding   *nn.Embedding
	Head        *mlx.Array
	FinalNorm   *nn.RMSNorm
	Layers      []*transformerLayer
	VisionTower *visionTower
	// Collect visits exported fields only. Keep the tower's private weights
	// here so the load scope does not free them after initialization.
	VisionWeights   []*mlx.Array
	EngramTables    map[int]*hostEngramTable
	EngramLayout    *engramLayout
	TokenMap        []int32
	HashMultipliers [][]int64
	Skipped         []string
	quant           map[string]*model.TensorQuantInfo
	tok             *tokenizer.Tokenizer
}

func NewModel(root *model.Root) (*Model, error) {
	configJSON, err := root.Manifest.ReadConfig("config.json")
	if err != nil {
		return nil, err
	}
	cfg, err := parseConfig(configJSON)
	if err != nil {
		return nil, err
	}
	tokenizerJSON, err := root.Manifest.ReadConfig("tokenizer.json")
	if err != nil {
		return nil, err
	}
	metadataJSON, err := root.Manifest.ReadConfig("engram_metadata.json")
	if err != nil {
		return nil, fmt.Errorf("DeepSeek engram metadata is required: %w", err)
	}
	tokenMap, multipliers, err := parseEngramMetadata(metadataJSON, tokenizerJSON, &cfg)
	if err != nil {
		return nil, err
	}
	layout, err := newEngramLayout(&cfg)
	if err != nil {
		return nil, err
	}
	tokenizerConfig := &tokenizer.TokenizerConfig{ConfigJSON: configJSON}
	if data, err := root.Manifest.ReadConfig("tokenizer_config.json"); err == nil {
		tokenizerConfig.TokenizerConfigJSON = data
	}
	tok, err := tokenizer.LoadFromBytesWithConfig(tokenizerJSON, tokenizerConfig)
	if err != nil {
		return nil, fmt.Errorf("load DeepSeek tokenizer: %w", err)
	}
	m := &Model{
		Config: &cfg, Layers: make([]*transformerLayer, cfg.NumHiddenLayers),
		EngramLayout: layout, TokenMap: tokenMap, HashMultipliers: multipliers, quant: root.AllTensorQuant(), tok: tok,
	}
	loaded := false
	defer func() {
		if !loaded {
			_ = m.Close()
		}
	}()
	m.EngramTables, m.Skipped, err = openHostEngramTables(root, &cfg)
	if err != nil {
		return nil, err
	}
	for _, layer := range root.Manifest.TensorLayers() {
		if !strings.HasPrefix(layer.Name, "mtp.") {
			continue
		}
		path, err := manifest.BlobsPath(layer.Digest)
		if err != nil {
			return nil, err
		}
		extractor, err := safetensors.OpenForExtraction(path)
		if err != nil {
			return nil, err
		}
		m.Skipped = append(m.Skipped, extractor.ListTensors()...)
		if err := extractor.Close(); err != nil {
			return nil, err
		}
	}
	loaded = true
	return m, nil
}

func (m *Model) SkippedTensorNames() []string { return m.Skipped }

// HostBackedTensorNames reports the engram tensors the loader keeps off GPU.
func HostBackedTensorNames(root *model.Root) ([]string, error) {
	configJSON, err := root.Manifest.ReadConfig("config.json")
	if err != nil {
		return nil, err
	}
	cfg, err := parseConfig(configJSON)
	if err != nil {
		return nil, err
	}
	if _, err := root.Manifest.ReadConfig("engram_metadata.json"); err != nil {
		return nil, err
	}
	tables, names, err := openHostEngramTables(root, &cfg)
	if err != nil {
		return nil, err
	}
	defer closeHostEngramTables(tables)
	return names, nil
}

func openHostEngramTables(root *model.Root, cfg *Config) (_ map[int]*hostEngramTable, _ []string, err error) {
	layers := make(map[string]manifest.Layer)
	for _, layer := range root.Manifest.TensorLayers() {
		layers[layer.Name] = layer
	}
	tables := make(map[int]*hostEngramTable, len(cfg.EngramLayerIDs))
	defer func() {
		if err != nil {
			closeHostEngramTables(tables)
		}
	}()
	var names []string
	for _, id := range cfg.EngramLayerIDs {
		name := fmt.Sprintf("layers.%d.engram.embed.weight", id)
		layer, ok := layers[name]
		if !ok {
			return nil, nil, fmt.Errorf("missing host-backed engram tensor %s", name)
		}
		path, pathErr := manifest.BlobsPath(layer.Digest)
		if pathErr != nil {
			return nil, nil, pathErr
		}
		table, openErr := openHostEngramTable(path, name, root.TensorQuant(name))
		if openErr != nil {
			return nil, nil, openErr
		}
		tables[id] = table
		names = append(names, name, name+".scale")
	}
	return tables, names, nil
}

func closeHostEngramTables(tables map[int]*hostEngramTable) {
	for _, table := range tables {
		_ = table.close()
	}
}

// Close releases the host-backed engram files. It is safe to call again.
func (m *Model) Close() error {
	var first error
	for _, table := range m.EngramTables {
		if err := table.close(); err != nil && first == nil {
			first = err
		}
	}
	return first
}

func (m *Model) LoadWeights(tensors map[string]*mlx.Array) error {
	var err error
	embedding, err := requireTensor(tensors, "embed.weight")
	if err != nil {
		return err
	}
	m.Embedding = nn.NewEmbedding(embedding)
	m.Head, err = requireTensor(tensors, "head.weight")
	if err != nil {
		return err
	}
	// Unembed consumes FP32 weights. Keep the conversion across decode steps.
	m.Head = m.Head.AsType(mlx.DTypeFloat32)
	norm, err := requireTensor(tensors, "norm.weight")
	if err != nil {
		return err
	}
	m.FinalNorm = nn.NewRMSNorm(norm, float32(m.Config.RMSNormEps))
	for i := range m.Layers {
		m.Layers[i], err = loadTransformerLayer(tensors, m.quant, m.Config, i, m.EngramTables[i])
		if err != nil {
			return fmt.Errorf("load DeepSeek layer %d: %w", i, err)
		}
	}
	m.VisionTower, err = loadVision(tensors, m.Config)
	if err != nil {
		return fmt.Errorf("load DeepSeek vision: %w", err)
	}
	if m.VisionTower != nil {
		for name, weight := range tensors {
			if strings.HasPrefix(name, "vision.") || strings.HasPrefix(name, "aligner.") || strings.HasPrefix(name, "image_") {
				m.VisionWeights = append(m.VisionWeights, weight)
			}
		}
	}
	return nil
}

func (m *Model) NewCaches() []cache.Cache {
	states := make([]cache.Cache, len(m.Layers)+1)
	for i := range states {
		states[i] = cache.NewKVCache()
	}
	return states
}

func (m *Model) Forward(b *batch.Batch, states []cache.Layer) (*mlx.Array, *mlx.Array) {
	if len(states) != 0 && len(states) < len(m.Layers)+1 {
		panic(fmt.Sprintf("DeepSeek V4.1 has %d cache slots, need %d", len(states), len(m.Layers)+1))
	}
	var tokenState cache.Layer
	if len(states) > 0 {
		tokenState = states[len(m.Layers)]
	}
	rows := m.hashCurrentTokens(b, tokenState)
	imageMask := imageMask(b)
	embedded := m.Embedding.Forward(b.InputIDs)
	if imageMask != nil {
		embedded = m.scatterImages(embedded, b)
	}
	batchSize, length, width := int32(embedded.Dim(0)), int32(embedded.Dim(1)), int32(embedded.Dim(2))
	streams := mlx.BroadcastTo(mlx.ExpandDims(embedded, 2), batchSize, length, int32(m.Config.HCMult), width)
	preMixValues := make([]float32, m.Config.HCMult)
	preMixValues[0] = 1
	preMix := mlx.FromValues(preMixValues, 1, 1, m.Config.HCMult)
	positionValues := make([]int32, batchSize*length)
	for row := range int(batchSize) {
		for token := range int(length) {
			positionValues[row*int(length)+token] = b.SeqOffsets[row] + int32(token)
		}
	}
	positions := mlx.FromValues(positionValues, int(batchSize), int(length))
	sparse := make([]*sparseState, batchSize)
	for row := range sparse {
		sparse[row] = &sparseState{}
	}
	for i, layer := range m.Layers {
		var state cache.Layer
		if len(states) > 0 {
			state = states[i]
		}
		streams, preMix = layer.forward(streams, preMix, positions, b, state, rows[i], sparse, imageMask)
	}
	hidden := m.FinalNorm.Forward(collapseConnection(streams, preMix), 0)
	return hidden, hidden
}

func (m *Model) Unembed(hidden *mlx.Array) *mlx.Array {
	return mlx.Matmul(hidden.AsType(mlx.DTypeFloat32), mlx.Transpose(m.Head.AsType(mlx.DTypeFloat32), 1, 0))
}

func (m *Model) Tokenizer() *tokenizer.Tokenizer { return m.tok }

// Large-context candidate selection is implemented, but the full model has
// only been validated within this provisional limit.
func (m *Model) MaxContextLength() int { return min(m.Config.MaxPositionEmbeddings, 32768) }

// makeProjection reads the runner's normalized _scale companion and rejects
// any quantization other than the published exact MXFP8 layout.
func makeProjection(tensors map[string]*mlx.Array, quant map[string]*model.TensorQuantInfo, name string) (nn.LinearLayer, error) {
	weightName := name + ".weight"
	weight := tensors[weightName]
	if weight == nil {
		return nil, fmt.Errorf("missing DeepSeek projection %s", weightName)
	}
	scales := tensors[weightName+"_scale"]
	if scales == nil {
		if weight.DType() != mlx.DTypeBFloat16 && weight.DType() != mlx.DTypeFloat32 {
			return nil, fmt.Errorf("DeepSeek projection %s lacks MXFP8 scales", weightName)
		}
		return nn.NewLinear(weight, nil), nil
	}
	if weight.DType() != mlx.DTypeUint32 || scales.DType() != mlx.DTypeUint8 {
		return nil, fmt.Errorf("DeepSeek projection %s needs U32 weights and U8 scales", weightName)
	}
	group, bits, mode := model.ResolveLinearQuantParams(32, 8, "mxfp8", quant, weightName, weight, scales)
	if group != 32 || bits != 8 || mode != "mxfp8" {
		return nil, fmt.Errorf("DeepSeek projection %s uses unsupported quantization %s/%d/%d", weightName, mode, bits, group)
	}
	return &nn.QuantizedLinear{Weight: weight, Scales: scales, GroupSize: 32, Bits: 8, Mode: "mxfp8"}, nil
}
