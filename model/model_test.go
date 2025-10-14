package model

import (
	"errors"
	"reflect"
	"slices"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/ollama/ollama/fs"
	fsggml "github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/backend/ggml"
	"github.com/ollama/ollama/ml/nn"
)

func TestParseTags(t *testing.T) {
	cases := []struct {
		value string
		want  Tag
	}{
		{
			value: "output",
			want: Tag{
				name: "output",
			},
		},
		{
			value: "output,alt:token_embd",
			want: Tag{
				name: "output",
				alternatives: []string{
					"token_embd",
				},
			},
		},
	}

	for _, tt := range cases {
		t.Run(tt.value, func(t *testing.T) {
			got := parseTag(tt.value)
			if diff := cmp.Diff(tt.want, got, cmp.AllowUnexported((Tag{}))); diff != "" {
				t.Errorf("ParseTags() returned unexpected values (-want +got):\n%s", diff)
			}
		})
	}
}

type fakeBackend struct {
	*ggml.Backend
	names []string
}

type fakeTensor struct {
	*ggml.Tensor
	Name string
}

func (m *fakeBackend) Get(name string) ml.Tensor {
	if slices.Contains(m.names, name) {
		return &fakeTensor{Name: name}
	}

	return nil
}

func TestPopulateFields(t *testing.T) {
	type fakeLayer struct {
		Query  *nn.Linear `gguf:"attn_q"`
		Key    *nn.Linear `gguf:"attn_k"`
		Value  *nn.Linear `gguf:"attn_v"`
		Output *nn.Linear `gguf:"attn_o"`
	}

	type fakeModel struct {
		Input      *nn.Embedding `gguf:"input"`
		OutputNorm *nn.RMSNorm   `gguf:"output_norm"`
		Output     *nn.Linear    `gguf:"output"`
		Layers     [2]fakeLayer  `gguf:"blk"`
	}

	var m fakeModel
	v := reflect.ValueOf(&m)
	v.Elem().Set(populateFields(Base{b: &fakeBackend{
		names: []string{
			"input.weight",
			"blk.0.attn_q.weight",
			"blk.0.attn_k.weight",
			"blk.0.attn_v.weight",
			"blk.1.attn_q.weight",
			"blk.1.attn_k.weight",
			"blk.1.attn_v.weight",
			"output_norm.weight",
			"output.weight",
		},
	}}, v.Elem()))

	if diff := cmp.Diff(fakeModel{
		Input:      &nn.Embedding{Weight: &fakeTensor{Name: "input.weight"}},
		OutputNorm: &nn.RMSNorm{Weight: &fakeTensor{Name: "output_norm.weight"}},
		Output:     &nn.Linear{Weight: &fakeTensor{Name: "output.weight"}},
		Layers: [2]fakeLayer{
			{
				Query: &nn.Linear{Weight: &fakeTensor{Name: "blk.0.attn_q.weight"}},
				Key:   &nn.Linear{Weight: &fakeTensor{Name: "blk.0.attn_k.weight"}},
				Value: &nn.Linear{Weight: &fakeTensor{Name: "blk.0.attn_v.weight"}},
			},
			{
				Query: &nn.Linear{Weight: &fakeTensor{Name: "blk.1.attn_q.weight"}},
				Key:   &nn.Linear{Weight: &fakeTensor{Name: "blk.1.attn_k.weight"}},
				Value: &nn.Linear{Weight: &fakeTensor{Name: "blk.1.attn_v.weight"}},
			},
		},
	}, m); diff != "" {
		t.Errorf("populateFields() set incorrect values (-want +got):\n%s", diff)
	}
}

func TestPopulateFieldsAlternateName(t *testing.T) {
	type nested struct {
		Weight *nn.Linear `gguf:"a,alt:b"`
	}

	type fakeModel struct {
		Input  *nn.Embedding `gguf:"input"`
		Output *nn.Linear    `gguf:"output,alt:input"`
		Nested *nested       `gguf:"nested"`
		Tensor ml.Tensor     `gguf:"leaf,alt:tensor"`
	}

	var m fakeModel
	v := reflect.ValueOf(&m)
	v.Elem().Set(populateFields(Base{b: &fakeBackend{
		names: []string{
			"input.weight",
			"nested.b.weight",
			"leaf",
		},
	}}, v.Elem()))

	if diff := cmp.Diff(fakeModel{
		Input:  &nn.Embedding{Weight: &fakeTensor{Name: "input.weight"}},
		Output: &nn.Linear{Weight: &fakeTensor{Name: "input.weight"}},
		Nested: &nested{
			Weight: &nn.Linear{Weight: &fakeTensor{Name: "nested.b.weight"}},
		},
		Tensor: &fakeTensor{Name: "leaf"},
	}, m); diff != "" {
		t.Errorf("populateFields() set incorrect values (-want +got):\n%s", diff)
	}
}

func TestPopulateFieldsPrefixSuffixName(t *testing.T) {
	type fakeBlock struct {
		A  *nn.Linear `gguf:"a"`
		B  *nn.Linear `gguf:",pre:b_"`
		C  *nn.Linear `gguf:",suf:_c"`
		XY *nn.Linear `gguf:",pre:x_,suf:_y"`
	}

	type fakeModel struct {
		Blocks []fakeBlock `gguf:"blk"`
	}

	m := fakeModel{
		Blocks: make([]fakeBlock, 2),
	}
	v := reflect.ValueOf(&m)
	v.Elem().Set(populateFields(Base{b: &fakeBackend{
		names: []string{
			"blk.0.a.weight",
			"blk.0.b_weight",
			"blk.0.b_bias",
			"blk.0.weight_c",
			"blk.0.x_weight_y",
			"blk.1.a.weight",
			"blk.1.b_weight",
			"blk.1.b_bias",
			"blk.1.weight_c",
			"blk.1.x_weight_y",
		},
	}}, v.Elem()))

	if diff := cmp.Diff(fakeModel{
		Blocks: []fakeBlock{
			{
				A:  &nn.Linear{Weight: &fakeTensor{Name: "blk.0.a.weight"}},
				B:  &nn.Linear{Weight: &fakeTensor{Name: "blk.0.b_weight"}, Bias: &fakeTensor{Name: "blk.0.b_bias"}},
				C:  &nn.Linear{Weight: &fakeTensor{Name: "blk.0.weight_c"}},
				XY: &nn.Linear{Weight: &fakeTensor{Name: "blk.0.x_weight_y"}},
			},
			{
				A:  &nn.Linear{Weight: &fakeTensor{Name: "blk.1.a.weight"}},
				B:  &nn.Linear{Weight: &fakeTensor{Name: "blk.1.b_weight"}, Bias: &fakeTensor{Name: "blk.1.b_bias"}},
				C:  &nn.Linear{Weight: &fakeTensor{Name: "blk.1.weight_c"}},
				XY: &nn.Linear{Weight: &fakeTensor{Name: "blk.1.x_weight_y"}},
			},
		},
	}, m); diff != "" {
		t.Errorf("populateFields() set incorrect values (-want +got):\n%s", diff)
	}
}

func TestModelForArch(t *testing.T) {
	type fakeModel struct {
		Model
	}

	type fakeEmbeddingModel struct {
		Model
	}

	models["model"] = func(c fs.Config) (Model, error) { return fakeModel{}, nil }
	models["model_embed"] = func(c fs.Config) (Model, error) { return fakeEmbeddingModel{}, nil }

	cases := []struct {
		name   string
		config fs.Config
		want   any
		err    error
	}{
		{
			name: "model",
			config: fsggml.KV{
				"general.architecture": "model",
			},
			want: fakeModel{},
		},
		{
			name: "embedding",
			config: fsggml.KV{
				"general.architecture": "model",
				"model.pooling_type":   uint32(1),
			},
			want: fakeEmbeddingModel{},
		},
		{
			name: "unsupported",
			config: fsggml.KV{
				"general.architecture": "unsupported",
			},
			err: ErrUnsupportedModel,
		},
	}

	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			got, err := modelForArch(tt.config)
			if !errors.Is(err, tt.err) {
				t.Fatal(err)
			}

			if diff := cmp.Diff(tt.want, got); diff != "" {
				t.Errorf("modelForArch() returned unexpected values (-want +got):\n%s", diff)
			}
		})
	}
}
