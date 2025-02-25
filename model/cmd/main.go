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
	"time"

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
	// start := time.Now()
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

	// Schema for a list of friends with their info
	// Maps to JSON like:
	// {
	// 	"name": "string",
	// 	"age": integer,
	// 	"is_available": boolean
	// }
	schema := &sample.Schema{
		Name: "root",
		Type: "object",
		Properties: []*sample.Schema{
			{Name: "name", Type: "string"},
			{Name: "age", Type: "integer"},
			{Name: "is_available", Type: "boolean"},
		},
	}

	// fmt.Println("schema", schema)
	// schema = nil
	jsonTransform, err := sample.NewJSONSampler(m.(model.TextProcessor), schema)
	if err != nil {
		return err
	}

	transforms := []sample.Transform{
		jsonTransform,
	}

	var offset int
	var stringBuffer string
	// var ttft time.Duration
	var totalSamplingTime time.Duration
	count := 0
	for range args.n {
		logits, err := model.Forward(m, append(opts, model.WithInputIDs(inputIDs), model.WithOffset(offset))...)
		if err != nil {
			return err
		}

		samplingStart := time.Now()
		sampler := sample.Greedy()
		sampledIdx, err := sampler.Sample(logits.Floats(), transforms...)
		if err != nil {
			return err
		}

		samplingTime := time.Since(samplingStart)
		totalSamplingTime += samplingTime

		// fmt.Println("sampling time", samplingTime)
		// fmt.Printf("Sample time: %vms\n", finishTime.Sub(sampleTime).Milliseconds())

		var outputIDs []int32

		if !m.(model.TextProcessor).Is(uint32(sampledIdx), model.SpecialEOS) {
			outputIDs = append(outputIDs, int32(sampledIdx))
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

		// if ttft == 0 {
		// 	ttft = time.Since(start)
		// fmt.Printf("Time to first token: %vms\n", ttft.Milliseconds())
		// }

		// fmt.Printf("--- token: %q\n", s)
		// fmt.Printf("--- outputIDs: %v\n", outputIDs)
		stringBuffer += s
		count++
		fmt.Println("--- stringBuffer", stringBuffer)

		outputIDs, err = jsonTransform.UpdateState(outputIDs)
		if err != nil {
			return err
		}

		// can do fun shifting stuff here if needed
		inputIDs = append(inputIDs, outputIDs...)
		if args.cache {
			offset = len(inputIDs) - 1
		}
	}
	fmt.Println("\n------ Output: ------")
	fmt.Println(stringBuffer)
	fmt.Println("--------------------")
	fmt.Println("sample average time", totalSamplingTime/time.Duration(count))
	return nil
}

func main() {
	if err := temp(); err != nil {
		fmt.Println("err", err)
		os.Exit(1)
	}
}
