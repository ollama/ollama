package model_test

import (
	"cmp"
	"encoding/binary"
	"encoding/json"
	"errors"
	"flag"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"testing"

	gocmp "github.com/google/go-cmp/cmp"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/logutil"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model"
	"github.com/ollama/ollama/model/input"
	_ "github.com/ollama/ollama/model/models"
	"github.com/ollama/ollama/tokenizer"
	typesmodel "github.com/ollama/ollama/types/model"
)

var args struct {
	model, prompt, image, output string
	layers, threads              int
	vv                           bool
}

func setup(tb testing.TB) model.Model {
	tb.Helper()
	if args.model == "" {
		tb.Skip("model argument is required")
	}

	path := args.model
	if _, err := os.Stat(path); errors.Is(err, os.ErrNotExist) {
		name := typesmodel.ParseName(path)
		if !name.IsValid() {
			tb.Fatalf("model path %q does not exist and is not a valid model name", path)
		}

		models := envconfig.Models()

		f, err := os.Open(filepath.Join(models, "manifests", name.Filepath()))
		if err != nil {
			tb.Fatal(err)
		}
		defer f.Close()

		var manifest struct {
			Layers []struct {
				MediaType string `json:"mediaType"`
				Digest    string `json:"digest"`
			} `json:"layers"`
		}

		if err := json.NewDecoder(f).Decode(&manifest); err != nil {
			tb.Fatal(err)
		}

		for _, layer := range manifest.Layers {
			if layer.MediaType == "application/vnd.ollama.image.model" {
				path = filepath.Join(models, "blobs", strings.ReplaceAll(layer.Digest, ":", "-"))
				break
			}
		}
	} else if err != nil {
		tb.Fatal(err)
	}

	return setupFile(tb, path)
}

func setupFile(tb testing.TB, path string) model.Model {
	tb.Log("using file", path)

	if s := os.Getenv("OLLAMA_LIBRARY_PATH"); s != "" {
		abspath, err := filepath.Abs(s)
		if err != nil {
			tb.Fatal(err)
		}

		tb.Setenv("PATH", strings.Join([]string{abspath, os.Getenv("PATH")}, string(os.PathListSeparator)))
	}

	m, err := model.New(path, ml.BackendParams{
		AllocMemory: true,
		NumThreads:  args.threads,
		GPULayers: []ml.GPULayers{
			{
				DeviceID: ml.DeviceID{
					ID:      cmp.Or(os.Getenv("OLLAMA_DEVICE_ID"), "0"),
					Library: cmp.Or(os.Getenv("OLLAMA_DEVICE_LIBRARY"), "Metal"),
				},
				Layers: func() (layers []int) {
					layers = make([]int, args.layers)
					for i := range layers {
						layers[i] = i
					}
					return layers
				}(),
			},
		},
	})
	if err != nil {
		tb.Fatal(err)
	}

	if err := m.Backend().Load(tb.Context(), func(float32) {}); err != nil {
		tb.Fatal(err)
	}

	return m
}

func dump(tb testing.TB, ctx ml.Context, tt ml.Tensor) {
	tt = tt.Contiguous(ctx)
	ctx.Forward(tt).Compute(tt)

	tb.Log("shape", tt.Shape(), "dtype", tt.DType(), "nbytes", len(tt.Bytes()))
	if args.vv {
		tb.Log(ml.Dump(ctx, tt))
	}

	if args.output != "" {
		f, err := os.Create(args.output)
		if err != nil {
			tb.Fatal(err)
		}
		defer f.Close()

		if err := binary.Write(f, binary.LittleEndian, tt.Bytes()); err != nil {
			tb.Fatal(err)
		}
	}
}

func TestMain(m *testing.M) {
	flag.StringVar(&args.model, "model", "", "path to model")
	flag.StringVar(&args.prompt, "prompt", "The capital of France is", "model prompt")
	flag.StringVar(&args.image, "image", "", "model image")
	flag.StringVar(&args.output, "output", "", "output file")

	flag.IntVar(&args.layers, "layers", 99, "num of gpu layers")
	flag.IntVar(&args.threads, "threads", runtime.NumCPU(), "num of threads")

	flag.BoolVar(&args.vv, "vv", false, "enable verbose logging")
	flag.Parse()

	slog.SetDefault(logutil.NewLogger(os.Stderr, envconfig.LogLevel()))
	os.Exit(m.Run())
}

func TestTokenizer(t *testing.T) {
	m := setup(t)

	processor, ok := m.(tokenizer.Tokenizer)
	if !ok {
		t.Fatal("not a text processor")
	}

	cases := map[string][]int32{
		"kingfisher": {7037, 506, 13761},
		"prospect":   {4892, 25594},
	}

	for wantText, wantIds := range cases {
		t.Run(wantText, func(t *testing.T) {
			encoded, err := processor.Encode(wantText, false)
			if err != nil {
				t.Fatal(err)
			}

			t.Log("encoded", "ids", encoded)
			if diff := gocmp.Diff(wantIds, encoded); diff != "" {
				t.Error("-want +got:", diff)
			}

			// if len(wantIds) < 10 {
			// 	for _, wantId := range wantIds {
			// 		decoded, err := processor.Decode([]int32{wantId})
			// 		if err != nil {
			// 			t.Fatal(err)
			// 		}
			//
			// 		t.Log("decoded", "id", wantId, "text", decoded)
			// 	}
			// }

			decoded, err := processor.Decode(wantIds)
			if err != nil {
				t.Fatal(err)
			}

			if diff := gocmp.Diff(wantText, decoded); diff != "" {
				t.Error("-want +got:", diff)
			}
		})
	}
}

func TestDecodeText(t *testing.T) {
	m := setup(t)

	maxBatch := 512
	if cache := m.Config().Cache; cache != nil {
		cache.Init(m.Backend(), ml.DTypeF16, 1, 32<<10, maxBatch)
		defer cache.Close()
	}

	processor, ok := m.(tokenizer.Tokenizer)
	if !ok {
		t.Fatal("not a text processor")
	}

	// prompt := "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nthe capital of france is<|im_end|>\n<|im_start|>assistant\n"
	// prompt := `<｜begin▁of▁sentence｜>hello<image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image>hello`
	// prompt := "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nthe capital of france is<|im_end|>\n<|im_start|>assistant\n"
	// t.Log("prompt", args.prompt)
	// prompt := args.prompt
	prompt := "[gMASK]<sop><|user|>\nThe capital of France is<|assistant|>\n"
	ids, err := processor.Encode(prompt, true)
	if err != nil {
		t.Fatal(err)
	}

	positions := make([]int32, len(ids))
	for i := range ids {
		positions[i] = int32(i)
	}

	for i := 0; i < len(ids); i += maxBatch {
		t.Run("batch/"+strconv.Itoa(i), func(t *testing.T) {
			j := min(len(ids), i+maxBatch)

			slog.Info("", "ids", ids[i:j], "positions", positions[i:j])

			ctx := m.Backend().NewContext()
			defer ctx.Close()

			batch := input.Batch{
				Inputs:    ctx.Input().FromInts(ids[i:j], j-i),
				Positions: positions[i:j],
				Sequences: slices.Repeat([]int{0}, j-i),
				// Outputs:   ctx.Input().FromInts([]int32{positions[j-1]}, 1),
			}

			tt, err := model.Forward(ctx, m, batch)
			if err != nil {
				t.Fatal(err)
			}

			dump(t, ctx, tt)
		})
	}
}

func TestEncodeImage(t *testing.T) {
	m := setup(t)

	processor, ok := m.(model.MultimodalProcessor)
	if !ok {
		t.Fatal("model is not a MultimodalProcessor")
	}

	if args.image == "" {
		t.Fatal("image argument is required for multimodal models")
	}

	f, err := os.Open(args.image)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	bts, err := io.ReadAll(f)
	if err != nil {
		t.Fatal(err)
	}

	ctx := m.Backend().NewContext()
	defer ctx.Close()

	mm, err := processor.EncodeMultimodal(ctx, bts)
	if err != nil {
		t.Fatal(err)
	}

	if len(mm) < 1 {
		t.Fatal("no multimodal tensors returned")
	}

	dump(t, ctx, mm[0].Tensor)
}

func TestMultimodal(t *testing.T) {
	m := setup(t)

	maxBatch := 2048
	if cache := m.Config().Cache; cache != nil {
		cache.Init(m.Backend(), ml.DTypeF16, 1, 32<<10, maxBatch)
		defer cache.Close()
	}

	ctx := m.Backend().NewContext()
	defer ctx.Close()

	textProcessor, ok := m.(tokenizer.Tokenizer)
	if !ok {
		t.Fatal("not a text processor")
	}

	prompt := "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|image_pad|>transcribe the text in this image<|im_end|>\n<|im_start|>assistant\n"
	ids, err := textProcessor.Encode(prompt, true)
	if err != nil {
		t.Fatal(err)
	}

	imageProcessor, ok := m.(model.MultimodalProcessor)
	if !ok {
		t.Fatal("model is not a MultimodalProcessor")
	}

	f, err := os.Open(args.image)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	bts, err := io.ReadAll(f)
	if err != nil {
		t.Fatal(err)
	}

	mm, err := imageProcessor.EncodeMultimodal(ctx, bts)
	if err != nil {
		t.Fatal(err)
	}

	imagePad, err := textProcessor.Encode("<|image_pad|>", false)
	if err != nil {
		t.Fatal(err)
	} else if len(imagePad) != 1 {
		t.Fatalf("expected image pad to be a single token, got %d tokens", len(imagePad))
	}

	inputs := make([]*input.Input, len(ids))
	for i, id := range ids {
		inputs[i] = &input.Input{Token: id}
		if id == imagePad[0] {
			inputs[i].Multimodal = mm
		}
	}

	t.Log("inputs", len(inputs))
	t.Log("inputs", func() (s []int) {
		s = make([]int, len(inputs))
		for i, input := range inputs {
			s[i] = int(input.Token)
		}
		return s
	}())

	inputs, err = imageProcessor.PostTokenize(inputs)
	if err != nil {
		t.Fatal(err)
	}

	t.Log("post tokenize inputs", len(inputs))

	tt, err := model.Forward(ctx, m, input.Batch{
		Inputs: ctx.Input().FromInts(slices.Collect(func(yield func(int32) bool) {
			for _, input := range inputs {
				if !yield(input.Token) {
					return
				}
			}
		}), len(inputs)),
		Positions: slices.Collect(func(yield func(int32) bool) {
			for i := range inputs {
				if !yield(int32(i)) {
					return
				}
			}
		}),
		Sequences: slices.Repeat([]int{0}, len(inputs)),
		Multimodal: slices.Collect(func(yield func(input.MultimodalIndex) bool) {
			for i := range inputs {
				if mm := inputs[i].Multimodal; mm != nil {
					if !yield(input.MultimodalIndex{Index: i, Multimodal: mm}) {
						return
					}
				}
			}
		}),
	})
	if err != nil {
		t.Fatal(err)
	}

	dump(t, ctx, tt)
}
