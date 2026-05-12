package pooling

import (
	"github.com/ollama/ollama/ml"
)

type Type uint32

const (
	TypeNone Type = iota
	TypeMean
	TypeCLS
	TypeLast
)

func (t Type) String() string {
	switch t {
	case TypeMean:
		return "Mean"
	case TypeCLS:
		return "CLS"
	case TypeLast:
		return "Last"
	default:
		return "Unknown"
	}
}

func (t Type) Forward(ctx ml.Context, hiddenStates ml.Tensor) ml.Tensor {
	switch t {
	case TypeMean:
		hiddenStates = hiddenStates.Permute(ctx, 1, 0, 2, 3).Contiguous(ctx).Mean(ctx)
		return hiddenStates.Permute(ctx, 1, 0, 2, 3).Contiguous(ctx)
	case TypeCLS:
		return hiddenStates.Slice(ctx, 1, 0, 1, 1)
	case TypeLast:
		return hiddenStates.Slice(ctx, 1, hiddenStates.Dim(1)-1, hiddenStates.Dim(1), 1)
	default:
		panic("unknown pooling type")
	}
}
