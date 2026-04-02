package mlx

// RecurrentConv1d performs a depthwise causal 1D convolution with pool-based
// state management.
//
// x: [1, totalTokens, C] packed input (sum of SeqLens = totalTokens)
// weight: [C, K] depthwise conv weight (K = kernel size)
// statePool: [poolSize, convTail, C] conv state pool
// history: PageTable maps batch positions to pool rows, SeqLens gives
//
//	per-sequence token counts
//
// For the common aligned case (pool rows == [0..B-1], uniform SeqLens),
// reshapes to [B, T, C] and runs a single batched conv. Otherwise splits
// by SeqLens and runs per-sequence.
//
// Returns (output [1, totalTokens, C], nextState [B, convTail, C]).
func RecurrentConv1d(x, weight, statePool *Array, history KVHistory, stepSeqLens []int) (output, nextState *Array) {
	B := len(stepSeqLens)
	indices := history.PageTable.Ints()
	convTail := statePool.Dim(1)

	// Check alignment: pool rows contiguous [0..B-1] and all step SeqLens equal
	aligned := statePool.Dim(0) >= B
	uniformL := 0
	if aligned && B > 0 {
		uniformL = stepSeqLens[0]
		for i := range B {
			if indices[i] != i || stepSeqLens[i] != uniformL {
				aligned = false
				break
			}
		}
	}

	if aligned && B > 0 {
		batchState := statePool
		if statePool.Dim(0) > B {
			batchState = SliceStartStop(statePool,
				[]int32{0, 0, 0},
				[]int32{int32(B), int32(convTail), int32(statePool.Dim(2))})
		}
		// Reshape packed [1, B*L, C] to [B, L, C]
		bx := Reshape(x, int32(B), int32(uniformL), int32(x.Dim(2)))
		out, ns := depthwiseConv1dWithState(bx, weight, batchState, uniformL, convTail)
		// Reshape output back to packed [1, B*L, C]
		out = Reshape(out, 1, int32(B*uniformL), int32(x.Dim(2)))
		return out, ns
	}

	// TODO: use masked kernels to avoid per-sequence splitting (see GatedDelta).
	outs := make([]*Array, B)
	nstates := make([]*Array, B)
	offset := 0
	for i := range B {
		seqLen := stepSeqLens[i]
		idx := int32(indices[i])
		o := int32(offset)
		xi := SliceStartStop(x, []int32{0, o, 0}, []int32{1, o + int32(seqLen), int32(x.Dim(2))})
		si := SliceStartStop(statePool, []int32{idx, 0, 0}, []int32{idx + 1, int32(convTail), int32(statePool.Dim(2))})
		outs[i], nstates[i] = depthwiseConv1dWithState(xi, weight, si, seqLen, convTail)
		offset += seqLen
	}
	return Concatenate(outs, 1), Concatenate(nstates, 0)
}

// depthwiseConv1dWithState runs depthwise causal conv1d on [B, L, C] input
// with [B, convTail, C] state prepended. Returns (output [B, L, C], nextState [B, convTail, C]).
func depthwiseConv1dWithState(x, weight, state *Array, L, convTail int) (output, nextState *Array) {
	B := int32(x.Dim(0))
	C := int32(weight.Dim(0))
	K := int32(weight.Dim(1))

	convInput := state.Concatenate(1, x)

	var out *Array
	for i := int32(0); i < K; i++ {
		seg := SliceStartStop(convInput,
			[]int32{0, i, 0},
			[]int32{B, i + int32(L), C})
		wi := SliceStartStop(weight,
			[]int32{0, i},
			[]int32{C, i + 1})
		wi = Reshape(wi, 1, 1, C)
		term := Mul(seg, wi)
		if out == nil {
			out = term
		} else {
			out = Add(out, term)
		}
	}

	// Next state is the tail of convInput
	total := int32(convInput.Dim(1))
	ns := SliceStartStop(convInput,
		[]int32{0, total - int32(convTail), 0},
		[]int32{B, total, int32(convInput.Dim(2))})

	return out, ns
}
