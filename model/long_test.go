package model_test

import (
	"encoding/json"
	"flag"
	"log/slog"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/logutil"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model"
	"github.com/ollama/ollama/model/input"
	_ "github.com/ollama/ollama/model/models"
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
			return filepath.Join(models, "blobs", strings.ReplaceAll(layer.Digest, ":", "-"))
		}
	}

	return ""
}

func testForward(tb testing.TB, m model.Model, ids, positions []int32) ([]int32, []int32) {
	ctx := m.Backend().NewContext()
	defer ctx.Close()

	batch := input.Batch{
		Inputs:    ctx.Input().FromIntSlice(ids, len(ids)),
		Positions: positions,
		Sequences: slices.Repeat([]int{0}, len(ids)),
		Outputs:   []int32{positions[len(positions)-1]},
	}
	if cache := m.Config().Cache; cache != nil {
		if err := cache.StartForward(ctx, batch, false); err != nil {
			tb.Fatal(err)
		}
	}

	tt, err := m.Forward(ctx, batch)
	if err != nil {
		tb.Fatal(err)
	}

	ctx.Forward(tt).Compute(tt)

	f32s := tt.Floats()

	ids = []int32{int32(slices.Index(f32s, slices.Max(f32s)))}
	positions = []int32{positions[len(positions)-1] + 1}
	return ids, positions
}

func BenchmarkTextGeneration(b *testing.B) {
	b.Setenv("OLLAMA_LIBRARY_PATH", filepath.Join("..", "build", "lib", "ollama"))

	m, err := model.New(blob(b, args.model), ml.BackendParams{NumGPULayers: args.layers})
	if err != nil {
		b.Fatal(err)
	}

	if err := m.Backend().Load(b.Context(), func(float32) {}); err != nil {
		b.Fatal(err)
	}

	maxBatch := 64
	if cache := m.Config().Cache; cache != nil {
		cache.Init(m.Backend(), ml.DTypeF16, 1, 4<<10, maxBatch)
		defer cache.Close()
	}

	processor, ok := m.(model.TextProcessor)
	if !ok {
		b.Fatal("not a text processor")
	}

	ids, err := processor.Encode(args.prompt, false)
	if err != nil {
		b.Fatal(err)
	}

	// TODO: handle inputs larger than the batch size
	if len(ids) > maxBatch {
		b.Fatal("inputs larger than batch size")
	}

	positions := make([]int32, len(ids))
	for i := range ids {
		positions[i] = int32(i)
	}

	prefillCount := len(ids)
	now := time.Now()
	ids, positions = testForward(b, m, ids, positions)
	prefillDuration := time.Since(now)

	for b.Loop() {
		ids, positions = testForward(b, m, ids, positions)
	}

	b.ReportMetric(float64(prefillCount)/float64(prefillDuration.Seconds()), "prefilltokens/s")
	b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "tokens/s")
	// suppress ns/op
	b.ReportMetric(0, "ns/op")
}
