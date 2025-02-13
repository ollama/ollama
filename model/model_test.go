package model

import (
	"reflect"
	"slices"
	"testing"

	"github.com/google/go-cmp/cmp"
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
				Name: "output",
			},
		},
		{
			value: "output,alt:token_embd",
			want: Tag{
				Name: "output",
				Alternate: []string{
					"token_embd",
				},
			},
		},
	}

	for _, tt := range cases {
		t.Run(tt.value, func(t *testing.T) {
			got := ParseTags(tt.value)
			if diff := cmp.Diff(tt.want, got); diff != "" {
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
	type fakeModel struct {
		Input  *nn.Embedding `gguf:"input"`
		Output *nn.Linear    `gguf:"output,alt:input"`
	}

	m := fakeModel{}
	v := reflect.ValueOf(&m)
	v.Elem().Set(populateFields(Base{b: &fakeBackend{
		names: []string{
			"input.weight",
		},
	}}, v.Elem()))

	if diff := cmp.Diff(fakeModel{
		Input:  &nn.Embedding{Weight: &fakeTensor{Name: "input.weight"}},
		Output: &nn.Linear{Weight: &fakeTensor{Name: "input.weight"}},
	}, m); diff != "" {
		t.Errorf("populateFields() set incorrect values (-want +got):\n%s", diff)
	}
}
