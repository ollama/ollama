package convert

import (
	"errors"
	"io"
	"path/filepath"
	"strings"
)

type Tensor interface {
	Name() string
	Shape() []uint64
	Kind() uint32
	SetRepacker(repacker)
	WriteTo(io.Writer) (int64, error)
}

type tensorBase struct {
	name  string
	shape []uint64
	repacker
}

func (t tensorBase) Name() string {
	return t.name
}

func (t tensorBase) Shape() []uint64 {
	return t.shape
}

func (t tensorBase) Kind() uint32 {
	if strings.HasSuffix(t.name, ".block_sparse_moe.gate.weight") {
		return 0
	}

	switch len(t.shape) {
	case 0:
		panic("invalid tensor shape")
	case 1:
		return 0
	default:
		return 1
	}
}

func (t *tensorBase) SetRepacker(fn repacker) {
	t.repacker = fn
}

type repacker func(string, []float32, []uint64) ([]float32, error)

func parseTensors(d string) ([]Tensor, error) {
	patterns := map[string]func(...string) ([]Tensor, error){
		"model-*-of-*.safetensors": parseSafetensors,
		"model.safetensors":        parseSafetensors,
		"pytorch_model-*-of-*.bin": parseTorch,
		"pytorch_model.bin":        parseTorch,
		"consolidated.*.pth":       parseTorch,
	}

	for pattern, parseFn := range patterns {
		matches, err := filepath.Glob(filepath.Join(d, pattern))
		if err != nil {
			return nil, err
		}

		if len(matches) > 0 {
			return parseFn(matches...)
		}
	}

	return nil, errors.New("unknown tensor format")
}
