package mlxrunner

import (
	"math"
	"testing"

	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

func TestApplyTokenMask(t *testing.T) {
	skipIfNoMLX(t)
	const (
		bitsPerMaskWord       = 32
		firstTokenID          = 0                   // Least-significant bit of the first mask word.
		interiorTokenID       = 7                   // Last bit of the first mask word's low byte.
		lastIDInFirstMaskWord = bitsPerMaskWord - 1 // Sign bit of the int32-backed first mask word.
		lastVocabID           = 40                  // Final valid ID in a partially used second mask word.
		vocabSize             = lastVocabID + 1
	)
	allowedIDs := []int{firstTokenID, interiorTokenID, lastIDInFirstMaskWord, lastVocabID}
	word0 := uint32(1)<<firstTokenID |
		uint32(1)<<interiorTokenID |
		uint32(1)<<lastIDInFirstMaskWord
	packed := []int32{
		int32(word0),
		int32(uint32(1) << (lastVocabID - bitsPerMaskWord)),
	}
	e := &grammarEngine{}
	e.initMask(vocabSize)
	logits := mlx.Zeros(mlx.DTypeFloat32, 1, vocabSize)
	masked := e.apply(logits, mlx.FromValues(packed, 1, len(packed)))
	mlx.Eval(masked)
	got := masked.Floats()
	for id := range vocabSize {
		allowed := false
		for _, a := range allowedIDs {
			if id == a {
				allowed = true
			}
		}
		if allowed && got[id] != 0 {
			t.Fatalf("allowed token %d masked to %v", id, got[id])
		}
		if !allowed && !math.IsInf(float64(got[id]), -1) {
			t.Fatalf("disallowed token %d = %v, want -Inf", id, got[id])
		}
	}
}
