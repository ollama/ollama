package gptoss

import (
	"encoding/binary"
	"encoding/json"
	"flag"
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

// loadFloatsFromBinary reads float32 values from a binary file
func loadFloatsFromBinary(filename string) ([]float32, error) {
	f, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var floats []float32
	if err := binary.Read(f, binary.LittleEndian, &floats); err != nil {
		return nil, err
	}

	return floats, nil
}

func TestOssForward(t *testing.T) {
	m, err := model.New(blob(t, args.model), ml.BackendParams{AllocMemory: true})
	if err != nil {
		t.Fatal(err)
	}

	if err := m.Backend().Load(t.Context(), func(float32) {}); err != nil {
		t.Fatal(err)
	}

	// t.Logf("%+v", m.(*Transformer).TransformerBlocks[0].Attention.QA)

	attentionBlock := m.(*Transformer).TransformerBlocks[0].Attention
	ctx := m.Backend().NewContext()

	// Load hidden states from binary file
	filePath := "/Users/graceguo/workspace/transformers/src/transformers/models/deepseek_v3/hidden_states.bin"

	hsFloats, err := loadFloatsFromBinary(filePath)
	if err != nil {
		t.Fatal(err)
	}
	// [1, 4, 7168]
	hiddenStates := ctx.Input().FromFloatSlice(hsFloats, 2880, 4, 1)
	t.Logf("DEBUG: hiddenStates: %v\n", hiddenStates.Shape())

	positionIndices := []int32{0, 1, 2, 3}
	positions := ctx.Input().FromIntSlice(positionIndices, 4)
	t.Logf("DEBUG: positions: %v\n", positions.Shape())

	// DEBUG: options: &{2880 64 8 64 64 128 4 131072 1e-05 150000 1}

	options := &Options{
		hiddenSize:            2880,
		numHeads:              64,
		numKVHeads:            8,
		keyLength:             64, // head_dim
		valueLength:           64, // head_dim
		numExperts:            128,
		numExpertsUsed:        4,
		originalContextLength: 131072,
		eps:                   1e-05,
		ropeBase:              150000,
		ropeScale:             1,
	}

	result := attentionBlock.Forward(ctx, hiddenStates, positions, nil, options)
	t.Logf("Forward pass completed, result shape: %v", result.Shape())
}
