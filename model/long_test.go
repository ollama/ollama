package model_test

import (
	"errors"
	"flag"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/ollama/ollama/model"
	_ "github.com/ollama/ollama/model/llama"
	"github.com/ollama/ollama/sample"
)

var args struct {
	model  string
	tokens int
	debug  bool
}

func TestMain(m *testing.M) {
	flag.StringVar(&args.model, "model", "", "path to model")
	flag.IntVar(&args.tokens, "tokens", 10, "number of tokens to generate")
	flag.BoolVar(&args.debug, "debug", false, "enable debug logging")
	flag.Parse()

	level := slog.LevelInfo
	if args.debug {
		level = slog.LevelDebug
	}

	slog.SetDefault(slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{
		Level:     level,
		AddSource: true,
		ReplaceAttr: func(_ []string, attr slog.Attr) slog.Attr {
			if attr.Key == slog.SourceKey {
				source := attr.Value.Any().(*slog.Source)
				source.File = filepath.Base(source.File)
			}

			return attr
		},
	})))

	os.Exit(m.Run())
}

func TestLong(t *testing.T) {
	if testing.Short() || args.model == "" || len(flag.Args()) < 1 {
		t.Skip("usage: go test -v -run TestLong -model path/to/model ...")
	}

	m, err := model.New(args.model)
	if err != nil {
		t.Fatal(err)
	}

	inputIDs, err := m.(model.TextProcessor).Encode(strings.Join(flag.Args(), " "))
	if err != nil {
		t.Fatal(err)
	}

	for range args.tokens {
		logit, err := model.Forward(m, model.WithInputIDs(inputIDs))
		if err != nil {
			t.Fatal(err)
		}

		f32s := logit.Floats()
		f64s := make([]float64, len(f32s))
		for i, f32 := range f32s {
			f64s[i] = float64(f32)
		}

		// do sampling
		f64s, err = sample.Sample(f64s, sample.Greedy())
		if err != nil {
			t.Fatal(err)
		}

		var outputIDs []int32
		for _, f64 := range f64s {
			if !m.(model.TextProcessor).Is(uint32(f64), model.SpecialEOS) {
				outputIDs = append(outputIDs, int32(f64))
			}
		}

		if len(outputIDs) == 0 {
			break
		}

		if _, err := m.(model.TextProcessor).Decode(outputIDs); errors.Is(err, io.EOF) {
			break
		} else if err != nil {
			t.Fatal(err)
		}

		inputIDs = append(inputIDs, outputIDs...)
	}
}
