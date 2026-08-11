package mlxrunner

import (
	"slices"
	"testing"
)

func TestUnpackTokenMask(t *testing.T) {
	const (
		bitsPerMaskWord       = 32
		firstTokenID          = 0                   // Least-significant bit of the first mask word.
		interiorTokenID       = 7                   // Arbitrary non-boundary bit in the first mask word.
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
	got := make([]bool, vocabSize)
	unpackTokenMaskInto(got, packed)
	want := make([]bool, vocabSize)
	for _, id := range allowedIDs {
		want[id] = true
	}
	if !slices.Equal(got, want) {
		t.Fatalf("unpackTokenMask = %v, want %v", got, want)
	}
}
