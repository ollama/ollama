package main

import (
	"errors"
	"flag"
	"fmt"
	"image"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"strings"

	"github.com/ollama/ollama/cache"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model"
	_ "github.com/ollama/ollama/model/llama"
	_ "github.com/ollama/ollama/model/mllama"
	"github.com/ollama/ollama/sample"
)

var args struct {
	n     int
	debug bool
	image string
	cache bool
}

func temp() error {
	flag.IntVar(&args.n, "n", 10, "number of samples")
	flag.BoolVar(&args.debug, "debug", false, "enable debug logging")
	flag.StringVar(&args.image, "image", "", "path to image file")
	flag.BoolVar(&args.cache, "cache", false, "enable KV cache")

	flag.Parse()

	var prompt string
	if n := len(flag.Args()); n == 1 {
		bts, err := io.ReadAll(os.Stdin)
		if err != nil {
			return err
		}

		prompt = string(bts)
	} else if n > 1 {
		prompt = strings.Join(flag.Args()[1:], " ")
	} else {
		return fmt.Errorf("usage: %s path/to/file <prompt\n", filepath.Base(os.Args[0]))
	}

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

	m, err := model.New(flag.Arg(0))
	if err != nil {
		return err
	}

	inputIDs, err := m.(model.TextProcessor).Encode(prompt)
	if err != nil {
		return err
	}

	var opts []model.OptionsFunc
	if args.cache {
		opts = append(opts, model.WithCache(&cache.Simple{
			Capacity: 2048,
			DType:    ml.DTypeF32,
		}))
	}

	if args.image != "" {
		if err := func() error {
			f, err := os.Open(args.image)
			if err != nil {
				return err
			}
			defer f.Close()

			img, _, err := image.Decode(f)
			if err != nil {
				return err
			}

			opts = append(opts, model.WithImage(img))
			return nil
		}(); err != nil {
			return err
		}
	}

	var offset int
	for range args.n {
		logit, err := model.Forward(m, append(opts, model.WithInputIDs(inputIDs), model.WithOffset(offset))...)
		if err != nil {
			return err
		}

		f32s := logit.Floats()
		f64s := make([]float64, len(f32s))
		for i, f32 := range f32s {
			f64s[i] = float64(f32)
		}

		// do sampling
		f64s, err = sample.Sample(f64s, sample.Greedy())
		if err != nil {
			return err
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

		s, err := m.(model.TextProcessor).Decode(outputIDs)
		if errors.Is(err, io.EOF) {
			break
		} else if err != nil {
			return err
		}

		fmt.Print(s)

		inputIDs = append(inputIDs, outputIDs...)
		if args.cache {
			offset = len(inputIDs) - 1
		}
	}

	return nil
}

func main() {
	if err := temp(); err != nil {
		fmt.Println("err", err)
		os.Exit(1)
	}
}
