package convert

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"io/fs"
	"log/slog"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"golang.org/x/exp/maps"

	"github.com/ollama/ollama/fs/ggml"
)

type tensorData struct {
	Offsets []int  `json:"data_offsets"`
	Type    string `json:"dtype"`
	Shape   []int  `json:"shape"`
}

func convertFull(t *testing.T, fsys fs.FS) (*os.File, ggml.KV, ggml.Tensors) {
	t.Helper()

	f, err := os.CreateTemp(t.TempDir(), "f16")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	if err := ConvertModel(fsys, f); err != nil {
		t.Fatal(err)
	}

	r, err := os.Open(f.Name())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { r.Close() })

	m, _, err := ggml.Decode(r, math.MaxInt)
	if err != nil {
		t.Fatal(err)
	}

	if _, err := r.Seek(0, io.SeekStart); err != nil {
		t.Fatal(err)
	}

	return r, m.KV(), m.Tensors()
}

func generateResultsJSON(t *testing.T, f *os.File, kv ggml.KV, tensors ggml.Tensors) map[string]string {
	actual := make(map[string]string)
	for k, v := range kv {
		if s, ok := v.(json.Marshaler); !ok {
			actual[k] = fmt.Sprintf("%v", v)
		} else {
			bts, err := json.Marshal(s)
			if err != nil {
				t.Fatal(err)
			}

			actual[k] = fmt.Sprintf("%x", sha256.Sum256(bts))
		}
	}

	for _, tensor := range tensors.Items() {
		sha256sum := sha256.New()
		sr := io.NewSectionReader(f, int64(tensors.Offset+tensor.Offset), int64(tensor.Size()))
		if _, err := io.Copy(sha256sum, sr); err != nil {
			t.Fatal(err)
		}

		actual[tensor.Name] = hex.EncodeToString(sha256sum.Sum(nil))
	}

	return actual
}

func TestMain(m *testing.M) {
	var level slog.Level
	flag.TextVar(&level, "level", slog.LevelInfo, "log level")
	flag.Parse()
	slog.SetLogLoggerLevel(level)
	os.Exit(m.Run())
}

func TestConvertModel(t *testing.T) {
	cases := []string{
		"Meta-Llama-3-8B-Instruct",
		"Meta-Llama-3.1-8B-Instruct",
		"Mistral-7B-Instruct-v0.2",
		"Mixtral-8x7B-Instruct-v0.1",
		"gemma-2b-it",
		"gemma-2-2b-it",
		// microsoft/Phi-3-mini-128-instruct@d548c233192db00165d842bf8edff054bb3212f8
		"Phi-3-mini-128k-instruct",
		"all-MiniLM-L6-v2",
		"gemma-2-9b-it",
		"Qwen2.5-0.5B-Instruct",
		"c4ai-command-r-v01",
	}

	for i := range cases {
		tt := cases[i]
		t.Run(tt, func(t *testing.T) {
			t.Parallel()

			p := filepath.Join("testdata", tt)
			if testing.Short() {
				t.Skip("skipping in short mode")
			} else if _, err := os.Stat(p); err != nil {
				t.Skipf("%s not found", p)
			}

			f, kv, tensors := convertFull(t, os.DirFS(p))
			actual := generateResultsJSON(t, f, kv, tensors)

			expectFile, err := os.Open(filepath.Join("testdata", fmt.Sprintf("%s.json", tt)))
			if err != nil {
				t.Fatal(err)
			}

			var expect map[string]string
			if err := json.NewDecoder(expectFile).Decode(&expect); err != nil {
				t.Fatal(err)
			}

			keys := maps.Keys(expect)
			slices.Sort(keys)
			for _, k := range keys {
				if v, ok := actual[k]; !ok {
					t.Errorf("missing %s", k)
				} else if v != expect[k] {
					t.Errorf("unexpected %s: want %s, got %s", k, expect[k], v)
				}
			}
		})
	}
}

func TestConvertInvalidTensorNames(t *testing.T) {
	f, err := os.CreateTemp(t.TempDir(), "testmodel")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	tempDir := t.TempDir()

	td := map[string]*tensorData{}
	offset := 4096

	td["model.layers.0.self_attn.q_proj.weight"] = &tensorData{
		Offsets: []int{0, offset},
		Type:    "F32",
		Shape:   []int{4096, 4096},
	}
	td["blk.0.attn_q.weight"] = &tensorData{
		Offsets: []int{offset, offset * 2},
		Type:    "F32",
		Shape:   []int{4096, 4096},
	}
	generateSafetensorTestData(t, tempDir, td)

	err = ConvertModel(os.DirFS(tempDir), f)
	if err == nil || !strings.HasPrefix(err.Error(), "duplicate tensor name") {
		t.Errorf("expected error but didn't get one")
	}
}

func TestConvertInvalidDatatype(t *testing.T) {
	f, err := os.CreateTemp(t.TempDir(), "testmodel")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	tempDir := t.TempDir()

	td := map[string]*tensorData{}
	offset := 4096 * 14336

	td["model.layers.0.mlp.down_proj.weight"] = &tensorData{
		Offsets: []int{0, offset},
		Type:    "I8",
		Shape:   []int{4096, 14336},
	}
	td["model.layers.0.mlp.down_proj.weight_format"] = &tensorData{
		Offsets: []int{offset, offset},
		Type:    "U8",
		Shape:   []int{},
	}
	generateSafetensorTestData(t, tempDir, td)

	err = ConvertModel(os.DirFS(tempDir), f)
	if err == nil || err.Error() != "unsupported safetensors model" {
		t.Errorf("expected error but didn't get one")
	}
}

func generateSafetensorTestData(t *testing.T, tempDir string, tensorData map[string]*tensorData) {
	data, err := json.Marshal(tensorData)
	if err != nil {
		t.Fatal(err)
	}

	var buf bytes.Buffer

	l := int64(len(data))
	err = binary.Write(&buf, binary.LittleEndian, l)
	if err != nil {
		t.Fatal(err)
	}

	_, err = buf.Write(data)
	if err != nil {
		t.Fatal(err)
	}

	fdata, err := os.Create(filepath.Join(tempDir, "model-00001-of-00001.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	defer fdata.Close()

	_, err = fdata.Write(buf.Bytes())
	if err != nil {
		t.Fatal(err)
	}

	configData := `
{
  "architectures": [
    "LlamaForCausalLM"
  ]
}
`

	f, err := os.Create(filepath.Join(tempDir, "config.json"))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	_, err = f.WriteString(configData)
	if err != nil {
		t.Fatal(err)
	}

	tokenizerData := `
{
}
`

	f, err = os.Create(filepath.Join(tempDir, "tokenizer.json"))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	_, err = f.WriteString(tokenizerData)
	if err != nil {
		t.Fatal(err)
	}
}

func TestConvertAdapter(t *testing.T) {
	type AdapterCase struct {
		Name     string
		BaseKV   map[string]any
		Expected map[string]string
	}

	cases := []AdapterCase{
		{
			Name: "discollama",
			BaseKV: map[string]any{
				"general.architecture":          "llama",
				"llama.attention.head_count":    uint32(32),
				"llama.attention.head_count_kv": uint32(8),
			},
			Expected: map[string]string{
				"general.architecture":          "llama",
				"general.file_type":             "1",
				"general.parameter_count":       "106496",
				"general.type":                  "adapter",
				"general.version":               "v0.2",
				"adapter.lora.alpha":            "16",
				"adapter.type":                  "lora",
				"llama.attention.head_count":    "32",
				"llama.attention.head_count_kv": "8",
				"blk.31.attn_q.weight.lora_a":   "0eb3318b02cd313429bcc7621b539fdbb10240fea190c56c9e5f93fcd37a4e50",
				"blk.31.attn_q.weight.lora_b":   "0eb3318b02cd313429bcc7621b539fdbb10240fea190c56c9e5f93fcd37a4e50",
				"blk.31.attn_v.weight.lora_a":   "0eb3318b02cd313429bcc7621b539fdbb10240fea190c56c9e5f93fcd37a4e50",
				"blk.31.attn_v.weight.lora_b":   "071dcafe89df065d6e1c935ecb8fdf6479b3c202eb912e7da938597673ff5857",
			},
		},
	}

	for _, c := range cases {
		t.Run(c.Name, func(t *testing.T) {
			t.Parallel()

			f, err := os.CreateTemp(t.TempDir(), "f16")
			if err != nil {
				t.Fatal(err)
			}
			defer f.Close()

			tempDir := t.TempDir()
			generateLoraTestData(t, tempDir)

			if err = ConvertAdapter(os.DirFS(tempDir), f, c.BaseKV); err != nil {
				t.Fatal(err)
			}

			r, err := os.Open(f.Name())
			if err != nil {
				t.Fatal(err)
			}
			defer r.Close()

			m, _, err := ggml.Decode(r, math.MaxInt)
			if err != nil {
				t.Fatal(err)
			}

			if _, err := r.Seek(0, io.SeekStart); err != nil {
				t.Fatal(err)
			}

			actual := generateResultsJSON(t, r, m.KV(), m.Tensors())

			keys := maps.Keys(c.Expected)
			slices.Sort(keys)
			for _, k := range keys {
				if v, ok := actual[k]; !ok {
					t.Errorf("missing %s", k)
				} else if v != c.Expected[k] {
					t.Errorf("unexpected %s: want %s, got %s", k, c.Expected[k], v)
				}
			}
		})
	}
}

func generateLoraTestData(t *testing.T, tempDir string) {
	offset := 4096 * 8 * 4

	td := map[string]*tensorData{"__metadata__": nil}
	td["model.layers.31.self_attn.q_proj.lora_a"] = &tensorData{
		Offsets: []int{0, offset},
		Type:    "F32",
		Shape:   []int{4096, 8},
	}
	td["model.layers.31.self_attn.q_proj.lora_b"] = &tensorData{
		Offsets: []int{offset, offset * 2},
		Type:    "F32",
		Shape:   []int{8, 4096},
	}
	td["model.layers.31.self_attn.v_proj.lora_a"] = &tensorData{
		Offsets: []int{offset * 2, offset * 3},
		Type:    "F32",
		Shape:   []int{4096, 8},
	}
	td["model.layers.31.self_attn.v_proj.lora_b"] = &tensorData{
		Offsets: []int{offset * 3, offset*3 + 8*1024*4},
		Type:    "F32",
		Shape:   []int{8, 1024},
	}

	data, err := json.Marshal(td)
	if err != nil {
		t.Fatal(err)
	}

	var buf bytes.Buffer

	l := int64(len(data))
	err = binary.Write(&buf, binary.LittleEndian, l)
	if err != nil {
		t.Fatal(err)
	}

	_, err = buf.Write(data)
	if err != nil {
		t.Fatal(err)
	}

	// write some data for the tensors

	ones := make([]float32, 4096*8)
	for i := range ones {
		ones[i] = float32(1)
	}

	for range 3 {
		err = binary.Write(&buf, binary.LittleEndian, ones)
		if err != nil {
			t.Fatal(err)
		}
	}

	ones = make([]float32, 1024*8)
	for i := range ones {
		ones[i] = float32(1)
	}

	err = binary.Write(&buf, binary.LittleEndian, ones)
	if err != nil {
		t.Fatal(err)
	}

	fdata, err := os.Create(filepath.Join(tempDir, "adapters.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	defer fdata.Close()

	_, err = fdata.Write(buf.Bytes())
	if err != nil {
		t.Fatal(err)
	}

	configData := `
{
    "adapter_path": "adapters-test",
    "batch_size": 8,
    "config": "config-tiny.json",
    "data": "../discollama-completion",
    "grad_checkpoint": null,
    "iters": 1000,
    "learning_rate": 1e-05,
    "lora_layers": 1,
    "lora_parameters": {
        "rank": 8,
        "alpha": 16,
        "dropout": 0.0,
        "scale": 2.0
    },
    "lr_schedule": null,
    "max_seq_length": 2048,
    "model": "/Users/pdevine/git/Meta-Llama-3-8B-Instruct",
    "resume_adapter_file": null,
    "save_every": 100,
    "seed": 0,
    "steps_per_eval": 200,
    "steps_per_report": 10,
    "test": false,
    "test_batches": 500,
    "train": true,
    "use_dora": false,
    "val_batches": 25
}
`
	f, err := os.Create(filepath.Join(tempDir, "adapter_config.json"))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	_, err = f.WriteString(configData)
	if err != nil {
		t.Fatal(err)
	}
}
