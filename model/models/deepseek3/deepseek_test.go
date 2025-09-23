package deepseek3

import (
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"log/slog"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/logutil"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model"
	"github.com/ollama/ollama/model/input"
	typemodel "github.com/ollama/ollama/types/model"
)

var args struct {
	model,
	prompt string
	layers int
}

func TestMain(m *testing.M) {
	flag.StringVar(&args.model, "model", "", "path to model")
	flag.StringVar(&args.prompt, "prompt", "The capital of France is", "model prompt")
	flag.IntVar(&args.layers, "layers", math.MaxInt, "num of gpu layers")
	flag.Parse()

	slog.SetDefault(logutil.NewLogger(os.Stderr, envconfig.LogLevel()))

	os.Exit(m.Run())
}

func blob(tb testing.TB, model string) string {
	tb.Helper()

	models := envconfig.Models()
	manifest, err := os.Open(filepath.Join(models, "manifests", typemodel.ParseName(model).Filepath()))
	if err != nil {
		tb.Fatal(err)
	}
	defer manifest.Close()

	var m struct {
		Layers []struct {
			MediaType string `json:"mediaType"`
			Digest    string `json:"digest"`
		} `json:"layers"`
	}

	if err := json.NewDecoder(manifest).Decode(&m); err != nil {
		tb.Fatal(err)
	}

	for _, layer := range m.Layers {
		if layer.MediaType == "application/vnd.ollama.image.model" {
			tb.Log("using model blob", layer.Digest)
			return filepath.Join(models, "blobs", strings.ReplaceAll(layer.Digest, ":", "-"))
		}
	}

	return ""
}

func loadFloatsFromBinary(filename string) ([]float32, error) {
	f, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	fi, err := f.Stat()
	if err != nil {
		return nil, err
	}
	if fi.Size()%4 != 0 {
		return nil, fmt.Errorf("file size %d not multiple of 4", fi.Size())
	}

	n := int(fi.Size() / 4)
	floats := make([]float32, n)
	if err := binary.Read(f, binary.LittleEndian, floats); err != nil {
		return nil, err
	}
	return floats, nil
}

func TestForward(t *testing.T) {
	m, err := model.New(blob(t, args.model), ml.BackendParams{AllocMemory: true})
	if err != nil {
		t.Fatal(err)
	}

	if err := m.Backend().Load(t.Context(), func(float32) {}); err != nil {
		t.Fatal(err)
	}

	t.Logf("%+v", m.(*Transformer).TransformerBlocks[0].Attention.QA)

	attentionBlock := m.(*Transformer).TransformerBlocks[0].Attention
	ctx := m.Backend().NewContext()
	filePath := "/Users/graceguo/Downloads/hidden_states.bin"

	hsFloats, err := loadFloatsFromBinary(filePath)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("hs len=%d, expected=%d", len(hsFloats), 7168*4*1)
	t.Logf("DEBUG: hsFloats: %v", hsFloats[:10])
	hiddenStates := ctx.Input().FromFloatSlice(hsFloats, 7168, 4, 1)
	t.Logf("DEBUG: hiddenStates.shape: %v", hiddenStates.Shape())
	positionIndices := []int32{0, 1, 2, 3}
	positions := ctx.Input().FromIntSlice(positionIndices, 4)

	qLoraRankVal := 1536
	options := &Options{
		kvLoraRank:         512,
		qkNopeHeadDim:      128,
		qkRopeHeadDim:      64,
		kqNopeHeadDim:      128,      // key part dimension (256 - 128 = 128)
		qkHeadDim:          128 + 64, // qk_nope_head_dim + qk_rope_head_dim
		qLoraRank:          &qLoraRankVal,
		attnImplementation: "sdpa",
		vHeadDim:           128,
		hiddenSize:         7168,
		numHeads:           128,
		numKVHeads:         128,
		keyLength:          128,
		valueLength:        128,
		eps:       1e-06,
		ropeBase:  10000,
		ropeScale: 40,

		yarn_log_multiplier:   0.1,
		originalContextLength: 4096,
	}
	result := attentionBlock.Forward(ctx, hiddenStates, positions, nil, options)
	result = result.Contiguous(ctx)
	ctx.Forward(result).Compute(result)

	t.Logf("shape=%v dtype=%v", result.Shape(), result.DType())

	// filePath = "/Users/graceguo/workspace/ollama/model/models/deepseek3/hello5.bin"
	// print("DEBUG: filePath: %v\n", filePath)
	// err = os.WriteFile(filePath, result.Bytes(), 0644)
	// if err != nil {
	// 	t.Fatal(err)
	// }

	t.Logf("Forward pass completed, result shape: %v", result.Shape())
}

func TestTopKIndicesComplex(t *testing.T) {
	m, err := model.New(blob(t, args.model), ml.BackendParams{AllocMemory: true})
	if err != nil {
		t.Fatal(err)
	}
	if err := m.Backend().Load(t.Context(), func(float32) {}); err != nil {
		t.Fatal(err)
	}
	mlp := m.(*Transformer).TransformerBlocks[3].MLP
	ctx := m.Backend().NewContext()

	filePath := "/Users/graceguo/Downloads/hidden_states.bin"

	hsFloats, err := loadFloatsFromBinary(filePath)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("hs len=%d, expected=%d", len(hsFloats), 7168*4*1)
	t.Logf("DEBUG: hsFloats: %v", hsFloats[:10])
	hiddenStates := ctx.Input().FromFloatSlice(hsFloats, 7168, 4, 1)
	t.Logf("DEBUG: hiddenStates.shape: %v", hiddenStates.Shape())

	options := &Options{
		numExperts:          256,
		numExpertsUsed:      8,
		normTopKProb:        true,
		routedScalingFactor: 2.5,
	}

	result := mlp.Forward(ctx, hiddenStates, options)
	result = result.Contiguous(ctx)
	ctx.Forward(result).Compute(result)

	t.Logf("shape=%v dtype=%v", result.Shape(), result.DType())

	// filePath = "/Users/graceguo/workspace/ollama/model/models/deepseek3/post_moe.bin"
	// print("DEBUG: filePath: %v\n", filePath)
	// err = os.WriteFile(filePath, result.Bytes(), 0644)
	// if err != nil {
	// 	t.Fatal(err)
	// }

	t.Logf("Forward pass completed, result shape: %v", result.Shape())
	t.Logf("Result shape: %v", result.Shape())
}

func TestFullForward(t *testing.T) {
	m, err := model.New(blob(t, args.model), ml.BackendParams{AllocMemory: true})
	if err != nil {
		t.Fatal(err)
	}
	if err := m.Backend().Load(t.Context(), func(float32) {}); err != nil {
		t.Fatal(err)
	}

	ctx := m.Backend().NewContext()

	prompt := args.prompt
	if prompt == "" {
		prompt = "Hello world! How's it going? 123 一二三"
	}

	tp := m.(model.TextProcessor)
	tokens, err := tp.Encode(prompt, true)
	if err != nil {
		t.Fatal(err)
	}

	t.Logf("tokens: %q", tokens)

	decoded, err := tp.Decode(tokens)
	if err != nil { t.Fatal(err) }
	t.Logf("decoded: %q", decoded)

	inputsTensor := ctx.Input().FromIntSlice(tokens, len(tokens))
	positions := make([]int32, len(tokens))
	sequences := make([]int, len(tokens))
	for i := range tokens {
		positions[i] = int32(i)
		sequences[i] = 0
	}
	outputs := ctx.Input().FromIntSlice([]int32{int32(len(tokens) - 1)}, 1)

	batch := input.Batch{
		Inputs:    inputsTensor,
		Positions: positions,
		Sequences: sequences,
		Outputs:   outputs,
	}
	if cache := m.Config().Cache; cache != nil {
		cache.Init(m.Backend(), ml.DTypeF16, 1, 4096, len(tokens))
	}

	result, err := model.Forward(ctx, m, batch)
	if err != nil {
		t.Fatal(err)
	}

	result = result.Contiguous(ctx)
	ctx.Forward(result).Compute(result)

	t.Logf("Forward pass completed, result shape: %v", result.Shape())
}

func TestTokenization(t *testing.T) {
	m, err := model.New(blob(t, args.model), ml.BackendParams{AllocMemory: true})
	if err != nil {
		t.Fatal(err)
	}
	if err := m.Backend().Load(t.Context(), func(float32) {}); err != nil {
		t.Fatal(err)
	}

	prompt := args.prompt
	if prompt == "" {
		prompt = "hello"
	}

	tp := m.(model.TextProcessor)
	tokens, err := tp.Encode(prompt, true)
	if err != nil {
		t.Fatal(err)
	}

	t.Logf("tokens: %v", tokens)
}
