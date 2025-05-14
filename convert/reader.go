package convert

import (
	"errors"
	"io"
	"io/fs"
	"strings"
)

type Tensor interface {
	Name() string
	Shape() []uint64
	Kind() uint32
	SetRepacker(Repacker)
	WriteTo(io.Writer) (int64, error)
	Clone() Tensor
}

type tensorBase struct {
	name     string
	shape    []uint64
	repacker Repacker
}

func (t tensorBase) Name() string {
	return t.name
}

func (t tensorBase) Shape() []uint64 {
	return t.shape
}

const (
	tensorKindF32 uint32 = iota
	tensorKindF16
)

func (t tensorBase) Kind() uint32 {
	if strings.HasSuffix(t.name, ".ffn_gate_inp.weight") ||
		t.name == "token_types.weight" ||
		t.name == "v.positional_embedding_vlm" ||
		t.name == "v.tile_position_embd.weight" ||
		t.name == "v.pre_tile_position_embd.weight" ||
		t.name == "v.post_tile_position_embd.weight" {
		// these tensors are always F32
		return 0
	}

	switch len(t.shape) {
	case 0:
		panic("invalid tensor shape")
	case 1:
		return tensorKindF32
	default:
		return tensorKindF16
	}
}

func (t *tensorBase) SetRepacker(fn Repacker) {
	t.repacker = fn
}

type Repacker func(string, []float32, []uint64) ([]float32, error)

func parseTensors(fsys fs.FS, replacer *strings.Replacer) ([]Tensor, error) {
	patterns := []struct {
		Pattern string
		Func    func(fs.FS, *strings.Replacer, ...string) ([]Tensor, error)
	}{
		{"*.safetensors", parseSafetensors},
		{"pytorch_model-*-of-*.bin", parseTorch},
		{"pytorch_model.bin", parseTorch},
		{"consolidated.*.pth", parseTorch},
	}

	for _, pattern := range patterns {
		matches, err := fs.Glob(fsys, pattern.Pattern)
		if err != nil {
			return nil, err
		}

		if len(matches) > 0 {
			return pattern.Func(fsys, replacer, matches...)
		}
	}

	return nil, errors.New("unknown tensor format")
}
