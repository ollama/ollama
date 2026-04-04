package pooling

import (
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
)

type Type uint32

const (
	TypeNone Type = iota
	TypeMean
	TypeCLS
	TypeLast
	TypeRank
)

func (t Type) String() string {
	switch t {
	case TypeMean:
		return "Mean"
	case TypeCLS:
		return "CLS"
	case TypeLast:
		return "Last"
	case TypeRank:
		return "Rank"
	default:
		return "Unknown"
	}
}

func (t Type) Forward(ctx ml.Context,
	hiddenStates ml.Tensor,
	cls *nn.Linear,
	clsOut *nn.Linear,
) ml.Tensor {
	switch t {
	case TypeMean:
		hiddenStates = hiddenStates.Permute(ctx, 1, 0, 2, 3).Contiguous(ctx).Mean(ctx)
		return hiddenStates.Permute(ctx, 1, 0, 2, 3).Contiguous(ctx)
	case TypeCLS:
		return hiddenStates.Slice(ctx, 1, 0, 1, 1)
	case TypeLast:
		return hiddenStates.Slice(ctx, 1, hiddenStates.Dim(1)-1, hiddenStates.Dim(1), 1)
	case TypeRank:
		hiddenStates = hiddenStates.Slice(ctx, 1, hiddenStates.Dim(1)-1, hiddenStates.Dim(1), 1)
		if cls != nil && cls.Weight != nil {
			hiddenStates = cls.Forward(ctx, hiddenStates)
			hiddenStates = hiddenStates.Tanh(ctx)
		}
		if clsOut != nil && clsOut.Weight != nil {
			hiddenStates = clsOut.Forward(ctx, hiddenStates)
		}
		return hiddenStates
	default:
		panic("unknown pooling type")
	}
}
