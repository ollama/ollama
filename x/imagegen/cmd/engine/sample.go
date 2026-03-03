//go:build mlx

package main

import "github.com/ollama/ollama/x/imagegen/mlx"

// sampleTopK samples from top-k logits using global random state
func sampleTopK(scaledLogits *mlx.Array, k int) *mlx.Array {
	neg := mlx.Neg(scaledLogits)
	indices := mlx.Argpartition(neg, k-1, -1)
	topKIdx := mlx.Slice(indices, []int32{0}, []int32{int32(k)})
	values := mlx.TakeAlongAxis(scaledLogits, topKIdx, -1)
	sampled := mlx.RandomCategorical(values, -1, 1)
	return mlx.Take(topKIdx, sampled, -1)
}

// sampleTopP samples using nucleus sampling with global random state
func sampleTopP(scaledLogits *mlx.Array, p float32, vocabSize int32) *mlx.Array {
	sorted := mlx.Argsort(mlx.Neg(scaledLogits), -1)
	sortedLogits := mlx.TakeAlongAxis(scaledLogits, sorted, -1)
	probs := mlx.Softmax(sortedLogits, -1)
	cumProbs := mlx.Cumsum(probs, -1)
	mask := mlx.LessScalar(cumProbs, p)
	negInf := mlx.FullDtype(float32(-1e9), scaledLogits.Dtype(), vocabSize)
	masked := mlx.Where(mask, sortedLogits, negInf)
	sampled := mlx.RandomCategorical(masked, -1, 1)
	return mlx.Take(sorted, sampled, -1)
}

// sample samples from logits at the last position
func sample(logits *mlx.Array, temp float32, topK int, topP float32, vocab int32) *mlx.Array {
	// Get last position logits: [1, L, vocab] -> [vocab]
	shape := logits.Shape()
	seqLen := shape[1]
	lastLogits := mlx.Slice(logits, []int32{0, seqLen - 1, 0}, []int32{1, seqLen, vocab})
	lastLogits = mlx.Reshape(lastLogits, vocab)

	if temp == 0 {
		return mlx.Argmax(lastLogits, -1, false)
	}
	scaled := mlx.DivScalar(lastLogits, temp)
	if topK > 0 && topK < int(vocab) {
		return sampleTopK(scaled, topK)
	}
	if topP > 0 && topP < 1.0 {
		return sampleTopP(scaled, topP, vocab)
	}
	return mlx.RandomCategorical(scaled, -1, 1)
}
