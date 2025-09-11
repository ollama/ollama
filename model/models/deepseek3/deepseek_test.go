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
	typemodel "github.com/ollama/ollama/types/model"
)

// import

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

	// Load hidden states from binary file
	filePath := "/Users/graceguo/Downloads/hidden_states.bin"

	hsFloats, err := loadFloatsFromBinary(filePath)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("hs len=%d, expected=%d", len(hsFloats), 7168*4*1)
	// [1, 4, 7168]
	t.Logf("DEBUG: hsFloats: %v", hsFloats[:10])
	hiddenStates := ctx.Input().FromFloatSlice(hsFloats, 7168, 4, 1)
	t.Logf("DEBUG: hiddenStates.shape: %v", hiddenStates.Shape())
	// t.Logf("DEBUG: hsFloats: %v", hsFloats)
	// t.Logf("DEBUG: hiddenStates: %v", hiddenStates)

	// Create position indices for sequence length 4 (not frequency data!)
	// RoPE expects position indices like [0, 1, 2, 3] for a sequence of length 4
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
		// originalContextLength: 128000,
		eps:       1e-06,
		ropeBase:  1000000,
		ropeScale: 1,
	}

	// cache := m.(*Transformer).Cache
	// cache := m.(*Transformer).Cache

	// cache.Init(m.Backend(), ml.DTypeF16, 1, 4096, 512)
	// print("DEBUG: completed init\n")

	// N := 128
	// positionIndices2 := make([]int32, N)
	// for i := 0; i < N; i++ {
	// 	positionIndices2[i] = int32(i)
	// }
	// // positions2 := ctx.Input().FromIntSlice(positionIndices2, 128)

	// sequences := make([]int, len(positionIndices2)) // []int{0,0,0,0}
	// batch := input.Batch{
	// 	Positions: positionIndices2, // []int32{0,1,2,3}
	// 	Sequences: sequences,
	// }
	// if err := cache.StartForward(ctx, batch, false); err != nil {
	// 	t.Fatal(err)
	// }

	// print("DEBUG: completed start forward\n")
	// result := attentionBlock.Forward(ctx, hiddenStates, positions, cache, options)
	result := attentionBlock.Forward(ctx, hiddenStates, positions, nil, options)
	// bf 16 to f32?
	// result = result.Cast(ctx, ml.DTypeF32)
	result = result.Contiguous(ctx)
	ctx.Forward(result).Compute(result)

	t.Logf("shape=%v dtype=%v", result.Shape(), result.DType())

	filePath = "/Users/graceguo/workspace/ollama/model/models/deepseek3/attn_outputFinal.bin"
	print("DEBUG: filePath: %v\n", filePath)
	err = os.WriteFile(filePath, result.Bytes(), 0644)
	if err != nil {
		t.Fatal(err)
	}

	t.Logf("Forward pass completed, result shape: %v", result.Shape())
}
