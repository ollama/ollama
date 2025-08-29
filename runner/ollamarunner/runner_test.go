package ollamarunner

import (
	"encoding/json"
	"flag"
	"log/slog"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/discover"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/logutil"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model"
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
			tb.Log("using model blob", layer.Digest)
			return filepath.Join(models, "blobs", strings.ReplaceAll(layer.Digest, ":", "-"))
		}
	}

	return ""
}

func BenchmarkRunner(b *testing.B) {
	libraryPath, ok := os.LookupEnv("OLLAMA_LIBRARY_PATH")
	if !ok {
		libraryPath = filepath.Join("..", "..", "build", "lib", "ollama")
	}

	b.Setenv("OLLAMA_LIBRARY_PATH", libraryPath)
	if runtime.GOOS == "windows" {
		b.Setenv("PATH", strings.Join(append(filepath.SplitList(os.Getenv("PATH")), libraryPath), string(filepath.ListSeparator)))
	}

	var s Server
	s.modelPath = blob(b, args.model)
	s.batchSize = 512
	s.ready.Add(1)

	model, err := model.New(s.modelPath, ml.BackendParams{})
	if err != nil {
		b.Fatal(err)
	}

	layers := args.layers
	if layers < 0 || layers > int(model.Backend().Config().Uint("block_count")+1) {
		layers = int(model.Backend().Config().Uint("block_count") + 1)
	}

	gpus := discover.GetGPUInfo()
	if err := s.allocModel(s.modelPath, ml.BackendParams{
		AllocMemory: true,
		NumThreads:  1,
		GPULayers: ml.GPULayersList{
			ml.GPULayers{
				ID: gpus[0].ID,
				Layers: slices.Collect(func(yield func(int) bool) {
					for i := range layers {
						if !yield(i) {
							return
						}
					}
				}),
			},
		},
		FlashAttention: envconfig.FlashAttention(),
	}, nil, 1, "f16", int(envconfig.ContextLength()), false); err != nil {
		b.Fatal(err)
	}

	s.loadModel()

	seq, err := s.NewSequence(args.prompt, nil, NewSequenceParams{})
	if err != nil {
		b.Fatal(err)
	}

	seq.cache, seq.inputs, err = s.cache.LoadCacheSlot(seq.inputs)
	if err != nil {
		b.Fatal(err)
	}

	s.seqs = []*Sequence{seq}

	go func() {
		for s := range seq.responses {
			slog.Debug("response", "text", s)
		}
	}()

	// process prompt
	if err := s.processBatch(); err != nil {
		b.Fatal(err)
	}

	for b.Loop() {
		if err := s.processBatch(); err != nil {
			b.Fatal(err)
		}
	}

	b.ReportMetric(float64(seq.numPromptInputs)/float64(seq.startGenerationTime.Sub(seq.startProcessingTime).Seconds()), "prefilltokens/s")
	b.ReportMetric(float64(seq.numPredicted)/float64(time.Since(seq.startGenerationTime).Seconds()), "tokens/s")
	// suppress ns/op
	b.ReportMetric(0, "ns/op")
}
