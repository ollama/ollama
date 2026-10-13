package pooling

import (
	"github.com/ollama/ollama/x/ml"
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
	// case TypeMean:
	// 	hiddenStates = hiddenStates.Transpose(ctx, 1, 0, 2, 3).Contiguous(ctx, false).Mean(ctx)
	// 	return hiddenStates.Transpose(ctx, 1, 0, 2, 3).Contiguous(ctx, false)
	// case TypeCLS:
	// 	return hiddenStates.Slice(ctx, 1, 0, 1, 1)
	// case TypeLast:
	// 	return hiddenStates.Slice(ctx, 1, hiddenStates.Dim(1)-1, hiddenStates.Dim(1), 1)
	default:
		panic("unknown pooling type")
	}
}
