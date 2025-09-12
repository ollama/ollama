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
	TypeRank

	TypeUnknown     = 0xFFFFFFFE
	TypeUnspecified = 0xFFFFFFFF
)

func Pooling(ctx ml.Context, hiddenStates ml.Tensor, poolingType Type) ml.Tensor {
	switch poolingType {
	case TypeNone:
		return hiddenStates
	case TypeMean:
		hiddenStates = hiddenStates.Permute(ctx, 1, 0, 2, 3).Contiguous(ctx).Mean(ctx)
		return hiddenStates.Permute(ctx, 1, 0, 2, 3).Contiguous(ctx)
	case TypeCLS:
		return hiddenStates.View(ctx, 0, hiddenStates.Dim(0))
	case TypeLast:
		panic("not implemented")
	case TypeRank:
		panic("not implemented")
	default:
		panic("not implemented")
	}
}
