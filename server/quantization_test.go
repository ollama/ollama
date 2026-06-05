package server

import (
	"bytes"
	"io"
	"os"
	"path/filepath"
	"slices"
	"testing"

	fsggml "github.com/ollama/ollama/fs/ggml"
)

func TestLlamaQuantizeArgs(t *testing.T) {
	tests := []struct {
		name     string
		arch     string
		fileType fsggml.FileType
		typeName string
		want     []string
	}{
		{
			name:     "default",
			arch:     "llama",
			fileType: fsggml.FileTypeQ4_K_M,
			want:     []string{"--allow-requantize", "in.gguf", "out.gguf", "Q4_K_M"},
		},
		{
			name:     "qwen35moe k quant keeps mtp projection q8",
			arch:     "qwen35moe",
			fileType: fsggml.FileTypeQ4_K_M,
			want: []string{
				"--allow-requantize",
				"--tensor-type", `^blk\.[0-9]+\.nextn\.eh_proj\.weight$=q8_0`,
				"in.gguf", "out.gguf", "Q4_K_M",
			},
		},
		{
			name:     "qwen35 k quant keeps mtp projection q8",
			arch:     "qwen35",
			fileType: fsggml.FileTypeQ4_K_S,
			want: []string{
				"--allow-requantize",
				"--tensor-type", `^blk\.[0-9]+\.nextn\.eh_proj\.weight$=q8_0`,
				"in.gguf", "out.gguf", "Q4_K_S",
			},
		},
		{
			name:     "qwen35moe f16 keeps mtp projection unquantized",
			arch:     "qwen35moe",
			fileType: fsggml.FileTypeF16,
			want:     []string{"--allow-requantize", "in.gguf", "out.gguf", "F16"},
		},
		{
			name:     "qwen35moe bf16 keeps mtp projection unquantized",
			arch:     "qwen35moe",
			fileType: fsggml.FileTypeBF16,
			want:     []string{"--allow-requantize", "in.gguf", "out.gguf", "BF16"},
		},
		{
			name:     "qwen35moe q8 already satisfies mtp projection floor",
			arch:     "qwen35moe",
			fileType: fsggml.FileTypeQ8_0,
			want:     []string{"--allow-requantize", "in.gguf", "out.gguf", "Q8_0"},
		},
		{
			name:     "gemma3n k quant keeps per layer token embedding f16",
			arch:     "gemma3n",
			fileType: fsggml.FileTypeQ4_K_M,
			want: []string{
				"--allow-requantize",
				"--tensor-type", `^per_layer_token_embd\.weight$=f16`,
				"in.gguf", "out.gguf", "Q4_K_M",
			},
		},
		{
			name:     "deepseek2 k quant keeps mla tensors q8",
			arch:     "deepseek2",
			fileType: fsggml.FileTypeQ4_K_M,
			want: []string{
				"--allow-requantize",
				"--tensor-type", `attn_k_b\.weight$=q8_0`,
				"--tensor-type", `attn_q_a\.weight$=q8_0`,
				"--tensor-type", `attn_q_b\.weight$=q8_0`,
				"--tensor-type", `attn_v_b\.weight$=q8_0`,
				"--tensor-type", `attn_kv_a_mqa\.weight$=q8_0`,
				"in.gguf", "out.gguf", "Q4_K_M",
			},
		},
		{
			name:     "gemma3n q8 does not add k quant override",
			arch:     "gemma3n",
			fileType: fsggml.FileTypeQ8_0,
			want:     []string{"--allow-requantize", "in.gguf", "out.gguf", "Q8_0"},
		},
		{
			name:     "glmocr k quant keeps input and output embeddings f16",
			arch:     "glmocr",
			fileType: fsggml.FileTypeQ4_K_M,
			want: []string{
				"--allow-requantize",
				"--tensor-type", `^token_embd\.weight$=f16`,
				"--tensor-type", `^output\.weight$=f16`,
				"in.gguf", "out.gguf", "Q4_K_M",
			},
		},
		{
			name:     "glm4 k quant keeps glm-ocr split text embeddings f16",
			arch:     "glm4",
			fileType: fsggml.FileTypeQ4_K_M,
			want: []string{
				"--allow-requantize",
				"--tensor-type", `^token_embd\.weight$=f16`,
				"--tensor-type", `^output\.weight$=f16`,
				"in.gguf", "out.gguf", "Q4_K_M",
			},
		},
		{
			name:     "copy does not add quantization overrides",
			arch:     "gemma3n",
			fileType: fsggml.FileTypeQ4_K_M,
			typeName: "COPY",
			want:     []string{"--allow-requantize", "in.gguf", "out.gguf", "COPY"},
		},
		{
			name:     "qwen35moe copy does not add mtp projection override",
			arch:     "qwen35moe",
			fileType: fsggml.FileTypeQ4_K_M,
			typeName: "COPY",
			want:     []string{"--allow-requantize", "in.gguf", "out.gguf", "COPY"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			typeName := tt.fileType.String()
			if tt.typeName != "" {
				typeName = tt.typeName
			}
			got := llamaQuantizeArgs(tt.arch, tt.fileType, "in.gguf", "out.gguf", typeName)
			if !slices.Equal(got, tt.want) {
				t.Fatalf("llamaQuantizeArgs = %#v, want %#v", got, tt.want)
			}
		})
	}
}

func TestDisableLlamaCppCompat(t *testing.T) {
	got := disableLlamaCppCompat([]string{"A=1", llamaCppCompatEnv + "=1", "B=2"})
	want := []string{"A=1", "B=2", llamaCppCompatEnv + "=0"}
	if !slices.Equal(got, want) {
		t.Fatalf("disableLlamaCppCompat = %#v, want %#v", got, want)
	}
}

func TestLlamaQuantizeEnv(t *testing.T) {
	env := []string{"A=1", llamaCppCompatEnv + "=0", "B=2"}
	tests := []struct {
		name         string
		enableCompat bool
		want         []string
	}{
		{
			name:         "clean GGUF validation disables compat",
			enableCompat: false,
			want:         []string{"A=1", "B=2", llamaCppCompatEnv + "=0"},
		},
		{
			name:         "embedded compatibility quantization allows compat",
			enableCompat: true,
			want:         []string{"A=1", "B=2"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := llamaQuantizeEnv(env, tt.enableCompat); !slices.Equal(got, tt.want) {
				t.Fatalf("llamaQuantizeEnv = %#v, want %#v", got, tt.want)
			}
		})
	}
}

func TestRestoreEmbeddedCompatibilityTensorsReplacesExistingCopies(t *testing.T) {
	dir := t.TempDir()

	origFile, orig := writeQuantizationTestGGUF(t, dir, "orig.gguf", fsggml.KV{
		"general.architecture": "qwen35",
	}, []*fsggml.Tensor{
		testTensor("blk.0.weight", fsggml.TensorTypeF32, []uint64{1}, []byte{1, 2, 3, 4}),
		testTensor("v.pos_embed.weight", fsggml.TensorTypeF16, []uint64{2}, []byte{5, 6, 7, 8}),
	})
	defer origFile.Close()

	outFile, _ := writeQuantizationTestGGUF(t, dir, "out.gguf", fsggml.KV{
		"general.architecture": "qwen35",
		"general.file_type":    fsggml.FileTypeQ4_K_M,
	}, []*fsggml.Tensor{
		testTensor("blk.0.weight", fsggml.TensorTypeF32, []uint64{1}, []byte{9, 10, 11, 12}),
		testTensor("v.pos_embed.weight", fsggml.TensorTypeF32, []uint64{1}, []byte{13, 14, 15, 16}),
	})
	defer outFile.Close()

	if err := restoreEmbeddedCompatibilityTensors(origFile, outFile, orig, fsggml.FileTypeQ4_K_M); err != nil {
		t.Fatal(err)
	}
	if _, err := outFile.Seek(0, io.SeekStart); err != nil {
		t.Fatal(err)
	}
	got, err := fsggml.Decode(outFile, -1)
	if err != nil {
		t.Fatal(err)
	}

	tensors := map[string]*fsggml.Tensor{}
	for _, tensor := range got.Tensors().Items() {
		tensors[tensor.Name] = tensor
	}
	if got := tensors["blk.0.weight"].Kind; got != uint32(fsggml.TensorTypeF32) {
		t.Fatalf("blk.0.weight kind = %v, want F32", got)
	}
	if got := tensors["v.pos_embed.weight"].Kind; got != uint32(fsggml.TensorTypeF16) {
		t.Fatalf("v.pos_embed.weight kind = %v, want F16 from original", got)
	}
	if got := tensors["v.pos_embed.weight"].Shape; !slices.Equal(got, []uint64{2}) {
		t.Fatalf("v.pos_embed.weight shape = %v, want original shape [2]", got)
	}
}

func writeQuantizationTestGGUF(t *testing.T, dir, name string, kv fsggml.KV, tensors []*fsggml.Tensor) (*os.File, *fsggml.GGML) {
	t.Helper()

	file, err := os.Create(filepath.Join(dir, name))
	if err != nil {
		t.Fatal(err)
	}
	if err := fsggml.WriteGGUF(file, kv, tensors); err != nil {
		t.Fatal(err)
	}
	if _, err := file.Seek(0, io.SeekStart); err != nil {
		t.Fatal(err)
	}
	ggml, err := fsggml.Decode(file, -1)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := file.Seek(0, io.SeekStart); err != nil {
		t.Fatal(err)
	}
	return file, ggml
}

func testTensor(name string, kind fsggml.TensorType, shape []uint64, data []byte) *fsggml.Tensor {
	return &fsggml.Tensor{
		Name:     name,
		Kind:     uint32(kind),
		Shape:    shape,
		WriterTo: bytes.NewReader(data),
	}
}
