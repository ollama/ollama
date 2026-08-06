package mlxrunner

import (
	"slices"
	"testing"
)

func TestEffectiveKeyTokens(t *testing.T) {
	tokens := []int32{10, 20, 500, 500, 500, 30}
	items := []mediaItem{{pos: 2, length: 3, fold: foldValue([]byte("img"), []int{1})}}

	eff := effectiveKeyTokens(tokens, items)
	want := []uint32{10, 20, items[0].fold, items[0].fold, items[0].fold, 30}
	if !slices.Equal(eff, want) {
		t.Fatalf("got %v, want %v", eff, want)
	}

	// Text-only streams never alias a media stream: folds carry bit 31.
	for _, e := range effectiveKeyTokens(tokens, nil) {
		if e&(1<<31) != 0 {
			t.Fatalf("token key %d has bit 31 set", e)
		}
	}
}

// Two prompts that differ only in their image diverge at the expansion's
// first key — one position earlier under bigram packing — and prompts with
// the same image share keys through the whole expansion.
func TestKeyFoldDivergence(t *testing.T) {
	prompt := func(fold uint32) []uint32 {
		tokens := []int32{1, 2, 900, 900, 900, 3, 4}
		return effectiveKeyTokens(tokens, []mediaItem{{pos: 2, length: 3, fold: fold}})
	}
	imgA := foldValue([]byte("a"), []int{1})
	imgB := foldValue([]byte("b"), []int{1})
	if imgA != foldValue([]byte("a"), []int{1}) {
		t.Fatal("fold not deterministic")
	}
	if imgA == foldValue([]byte("a"), []int{2}) {
		t.Fatal("different dims produced the same fold under identical bytes")
	}

	for _, lookahead := range []int{0, 1} {
		pc := &prefixCache{draftLookahead: lookahead}
		keysA := pc.key(prompt(imgA))
		keysB := pc.key(prompt(imgB))
		keysA2 := pc.key(prompt(imgA))

		if !slices.Equal(keysA, keysA2) {
			t.Fatalf("lookahead %d: same image produced different keys", lookahead)
		}

		// Bigram packing pulls the divergence one position early: the key
		// before the expansion packs (token, fold). Keys re-converge in value
		// after the expansion (shared trailing text), which is fine — the trie
		// paths forked at the first difference.
		divergeAt, convergeAt := 2-lookahead, 5
		for i := range keysA {
			same := keysA[i] == keysB[i]
			if i < divergeAt && !same {
				t.Fatalf("lookahead %d: keys diverge at %d, before the expansion", lookahead, i)
			}
			if i >= divergeAt && i < convergeAt && same {
				t.Fatalf("lookahead %d: keys agree at %d, inside the expansion", lookahead, i)
			}
		}
	}
}
