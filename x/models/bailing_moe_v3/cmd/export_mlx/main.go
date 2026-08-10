package main

import (
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"

	"github.com/ollama/ollama/x/mlxrunner"
	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	bailing "github.com/ollama/ollama/x/models/bailing_moe_v3"
	"github.com/ollama/ollama/x/models/nn"
)

type metadata struct {
	Model     string           `json:"model"`
	TokenIDs  []int32          `json:"token_ids"`
	Arrays    map[string][]int `json:"arrays"`
	RouterIDs map[string][]int `json:"router_ids,omitempty"`
}

func parseTokenIDs(value string) ([]int32, error) {
	fields := strings.Split(value, ",")
	ids := make([]int32, 0, len(fields))
	for _, field := range fields {
		field = strings.TrimSpace(field)
		if field == "" {
			continue
		}
		id, err := strconv.ParseInt(field, 10, 32)
		if err != nil {
			return nil, fmt.Errorf("invalid token ID %q: %w", field, err)
		}
		ids = append(ids, int32(id))
	}
	if len(ids) == 0 {
		return nil, fmt.Errorf("at least one token ID is required")
	}
	return ids, nil
}

func dumpArray(dir, name string, value *mlx.Array) ([]int, error) {
	value = value.AsType(mlx.DTypeFloat32)
	mlx.Eval(value)
	if value.DType() != mlx.DTypeFloat32 {
		return nil, fmt.Errorf("%s: cast produced %s", name, value.DType())
	}
	f, err := os.Create(filepath.Join(dir, name+".f32"))
	if err != nil {
		return nil, err
	}
	if err := binary.Write(f, binary.LittleEndian, value.Floats()); err != nil {
		f.Close()
		return nil, err
	}
	if err := f.Close(); err != nil {
		return nil, err
	}
	return value.Dims(), nil
}

func retainOnly(value *mlx.Array) {
	mlx.Eval(value)
	mlx.Pin(value)
	mlx.Sweep()
	mlx.Unpin(value)
}

func interleavedToHalf(x *mlx.Array) *mlx.Array {
	dims := x.Dims()
	dims32 := make([]int32, len(dims))
	for i, dim := range dims {
		dims32[i] = int32(dim)
	}
	d := dims[len(dims)-1]
	paired := append(append([]int32(nil), dims32[:len(dims32)-1]...), int32(d/2), 2)
	x = mlx.Reshape(x, paired...)
	axes := make([]int, len(paired))
	for i := range axes {
		axes[i] = i
	}
	axes[len(axes)-2], axes[len(axes)-1] = axes[len(axes)-1], axes[len(axes)-2]
	return mlx.Contiguous(mlx.Reshape(mlx.Transpose(x, axes...), dims32...), false)
}

func traceMLA(dir string, meta *metadata, attention *bailing.MLAAttention, x *mlx.Array, b *batch.Batch, positions *mlx.Array, cfg *bailing.Config) error {
	dump := func(name string, value *mlx.Array) error {
		shape, err := dumpArray(dir, "layer_03_mla_"+name, value)
		if err == nil {
			meta.Arrays["layer_03_mla_"+name] = shape
		}
		return err
	}
	B, L := int32(1), int32(x.Dim(1))
	q := attention.QBProj.Forward(attention.QALayerNorm.Forward(attention.QAProj.Forward(x), cfg.RMSNormEps))
	q = mlx.Transpose(mlx.Reshape(q, B, L, cfg.NumAttentionHeads, cfg.QKHeadDim), 0, 2, 1, 3)
	qNope := mlx.SliceStartStop(q, []int32{0, 0, 0, 0}, []int32{B, cfg.NumAttentionHeads, L, cfg.QKNopeHeadDim})
	qPE := mlx.SliceStartStop(q, []int32{0, 0, 0, cfg.QKNopeHeadDim}, []int32{B, cfg.NumAttentionHeads, L, cfg.QKHeadDim})
	compressed := attention.KVAProjWithMQA.Forward(x)
	kvCompressed := mlx.SliceStartStop(compressed, []int32{0, 0, 0}, []int32{B, L, cfg.KVLoraRank})
	kPE := mlx.SliceStartStop(compressed, []int32{0, 0, cfg.KVLoraRank}, []int32{B, L, cfg.KVLoraRank + cfg.QKRopeHeadDim})
	kPE = mlx.Transpose(mlx.Reshape(kPE, B, L, 1, cfg.QKRopeHeadDim), 0, 2, 1, 3)
	kvLatent := attention.KVALayerNorm.Forward(kvCompressed, cfg.RMSNormEps)
	kvExpanded := mlx.Transpose(mlx.Reshape(
		attention.KVBProj.Forward(kvLatent), B, L, cfg.NumAttentionHeads, cfg.QKNopeHeadDim+cfg.VHeadDim,
	), 0, 2, 1, 3)
	kNope := mlx.SliceStartStop(kvExpanded, []int32{0, 0, 0, 0}, []int32{B, cfg.NumAttentionHeads, L, cfg.QKNopeHeadDim})
	values := mlx.SliceStartStop(kvExpanded, []int32{0, 0, 0, cfg.QKNopeHeadDim}, []int32{B, cfg.NumAttentionHeads, L, cfg.QKNopeHeadDim + cfg.VHeadDim})
	qPE = mlx.RoPEWithBase(interleavedToHalf(qPE), int(cfg.QKRopeHeadDim), false, cfg.RopeTheta, 1, positions)
	kPE = mlx.RoPEWithBase(interleavedToHalf(kPE), int(cfg.QKRopeHeadDim), false, cfg.RopeTheta, 1, positions)
	kPE = mlx.Tile(kPE, []int32{1, cfg.NumAttentionHeads, 1, 1})
	queries := mlx.Concatenate([]*mlx.Array{qNope, qPE}, 3)
	keys := mlx.Concatenate([]*mlx.Array{kNope, kPE}, 3)
	core := nn.ScaledDotProductAttention(b, queries, cfg.Scale, nn.WithKV(keys, values, b.SeqQueryLens), nn.WithMask(nn.CausalMask()))
	gate := mlx.ExpandDims(mlx.Transpose(mlx.Sigmoid(attention.GateProj.Forward(x).AsType(mlx.DTypeFloat32)), 0, 2, 1), -1)
	gated := mlx.Mul(core.AsType(mlx.DTypeFloat32), gate).AsType(x.DType())
	denseInput := mlx.Reshape(mlx.Transpose(gated, 0, 2, 1, 3), B, L, cfg.NumAttentionHeads*cfg.VHeadDim)
	dense := attention.OutProj.Forward(denseInput)
	for name, value := range map[string]*mlx.Array{
		"q": q, "q_nope": qNope, "q_rope": qPE,
		"compressed": compressed, "kv_latent": kvLatent, "kv_expanded": kvExpanded,
		"k_nope": kNope, "k_rope": kPE, "values": values,
		"queries": queries, "keys": keys, "core": core, "gate": gate,
		"gated": gated, "dense_input": denseInput, "dense": dense,
	} {
		if err := dump(name, value); err != nil {
			return err
		}
	}
	return nil
}

func run() error {
	modelName := flag.String("model", "opd-200", "Ollama model name")
	outputDir := flag.String("output", "", "directory for float32 trace files")
	tokenIDsValue := flag.String("token-ids", "", "comma-separated input token IDs")
	flag.Parse()
	if *outputDir == "" {
		return fmt.Errorf("-output is required")
	}
	tokenIDs, err := parseTokenIDs(*tokenIDsValue)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(*outputDir, 0o755); err != nil {
		return err
	}
	if err := mlx.CheckInit(); err != nil {
		return fmt.Errorf("initialize MLX: %w", err)
	}

	runner := &mlxrunner.Runner{}
	if err := runner.Load(*modelName); err != nil {
		return fmt.Errorf("load model: %w", err)
	}
	m, ok := runner.Model.(*bailing.Model)
	if !ok {
		return fmt.Errorf("model %q has type %T, not BailingMoeV3", *modelName, runner.Model)
	}

	length := int32(len(tokenIDs))
	b := &batch.Batch{
		InputIDs:     mlx.FromValues(tokenIDs, 1, len(tokenIDs)),
		SeqOffsets:   []int32{0},
		SeqQueryLens: []int32{length},
	}
	positions := mlx.FromValues([]int32{0}, 1)
	mlx.Pin(b.InputIDs, positions)
	defer mlx.Unpin(b.InputIDs, positions)
	meta := metadata{
		Model: *modelName, TokenIDs: tokenIDs,
		Arrays: map[string][]int{}, RouterIDs: map[string][]int{},
	}

	hidden := m.EmbedTokens.Forward(b.InputIDs)
	meta.Arrays["embedding"], err = dumpArray(*outputDir, "embedding", hidden)
	if err != nil {
		return err
	}
	retainOnly(hidden)

	for i, layer := range m.Layers {
		normalized := layer.InputNorm.Forward(hidden, m.RMSNormEps)
		attention := layer.Attention.Forward(normalized, b, nil, positions, 1, length, m.Config)
		postAttention := mlx.Add(hidden, attention)
		postNormalized := layer.PostAttentionNorm.Forward(postAttention, m.RMSNormEps)
		if i == 0 {
			for name, value := range map[string]*mlx.Array{
				"layer_00_input_norm": normalized,
				"layer_00_attention":  attention,
			} {
				meta.Arrays[name], err = dumpArray(*outputDir, name, value)
				if err != nil {
					return err
				}
			}
		}
		if i == 3 {
			for name, value := range map[string]*mlx.Array{
				"layer_03_input_norm": normalized,
				"layer_03_attention":  attention,
				"layer_03_post_attn":  postAttention,
				"layer_03_post_norm":  postNormalized,
			} {
				meta.Arrays[name], err = dumpArray(*outputDir, name, value)
				if err != nil {
					return err
				}
			}
			if moe, ok := layer.MLP.(*bailing.SparseMoE); ok {
				indices, weights := moe.Router.Forward(postNormalized, m.Config)
				mlx.Eval(indices, weights)
				meta.RouterIDs["layer_03"] = indices.Ints()
				meta.Arrays["layer_03_router_weights"], err = dumpArray(*outputDir, "layer_03_router_weights", weights)
				if err != nil {
					return err
				}
			}
			if mla, ok := layer.Attention.(*bailing.MLAAttention); ok {
				if err := traceMLA(*outputDir, &meta, mla, normalized, b, positions, m.Config); err != nil {
					return err
				}
			}
		}
		mlp := layer.MLP.Forward(postNormalized, m.Config)
		hidden = mlx.Add(postAttention, mlp)
		if i == 3 {
			meta.Arrays["layer_03_mlp"], err = dumpArray(*outputDir, "layer_03_mlp", mlp)
			if err != nil {
				return err
			}
		}
		name := fmt.Sprintf("layer_%02d", i)
		meta.Arrays[name], err = dumpArray(*outputDir, name, hidden)
		if err != nil {
			return err
		}
		retainOnly(hidden)
	}

	hidden = m.Norm.Forward(hidden, m.RMSNormEps)
	meta.Arrays["final_norm"], err = dumpArray(*outputDir, "final_norm", hidden)
	if err != nil {
		return err
	}
	logits := m.Unembed(hidden)
	logits = mlx.SliceStartStop(logits,
		[]int32{0, length - 1, 0},
		[]int32{1, length, m.VocabSize})
	logits = mlx.Squeeze(logits, 1)
	meta.Arrays["logits"], err = dumpArray(*outputDir, "logits", logits)
	if err != nil {
		return err
	}

	data, err := json.MarshalIndent(meta, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(*outputDir, "metadata.json"), append(data, '\n'), 0o644)
}

func main() {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
