package mlxrunner

import (
	"context"
	"errors"
	"slices"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
	"github.com/ollama/ollama/mlxrunner/model"
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

func TestExtendChunk(t *testing.T) {
	m := &requestMedia{
		items: []mediaItem{
			{pos: 10, length: 4, item: &model.PreparedItem{}},
			{pos: 40, length: 8, item: &model.PreparedItem{Causal: true}},
			{pos: 96, length: 4, item: &model.PreparedItem{}},
		},
		inputLen: 100,
	}

	cases := []struct{ pos, n, want int }{
		{0, 10, 10}, // ends at the expansion start: not inside
		{0, 12, 10}, // expansion starts inside: cut so it begins the next chunk
		{0, 14, 14}, // ends at the expansion end: not inside
		{10, 2, 4},  // chunk starts at the expansion: extend to its end
		{12, 1, 2},  // resume mid-expansion: extend to its end
		{38, 6, 6},  // causal expansion: ending inside is legal
		{42, 4, 4},  // causal expansion at chunk start: no extension
		{90, 7, 6},  // trailing expansion starts inside: cut before it
		{96, 2, 3},  // trailing expansion at chunk start: clip one short of the prompt
	}
	for _, c := range cases {
		if got := m.extendChunk(c.pos, c.n); got != c.want {
			t.Errorf("extendChunk(%d, %d) = %d, want %d", c.pos, c.n, got, c.want)
		}
	}

	var nilMedia *requestMedia
	if got := nilMedia.extendChunk(0, 12); got != 12 {
		t.Errorf("nil extendChunk = %d, want 12", got)
	}
}

// encodeCountingModel counts EncodeMedia calls and returns a real array so
// the pin/release lifecycle runs against live handles.
type encodeCountingModel struct {
	stubMediaModel
	calls *int
	// nilData counts encodes that got no pixels; nil means none are expected.
	nilData *int
}

func (m encodeCountingModel) EncodeMedia(item *model.PreparedItem, data *mlx.Array) *mlx.Array {
	*m.calls++
	if data == nil && m.nilData != nil {
		*m.nilData++
	}
	return mlx.Zeros(mlx.DTypeFloat32, item.Range[1]-item.Range[0], 4)
}

func TestBatchMediaLifecycle(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		calls := 0
		prepared := &model.PreparedItem{
			Range:     [2]int{2, 6},
			MediaData: []float32{1, 2},
			Dims:      []int{2},
			Opaque:    7,
		}
		r := &Runner{Model: encodeCountingModel{calls: &calls}}
		request := Request{
			Tokens:     make([]int32, 8),
			MediaItems: []mediaItem{{pos: 2, length: 4, item: prepared}},
		}

		m := r.openMedia(request)
		if m == nil {
			t.Fatal("openMedia returned nil for a media request")
		}
		if m.manifest[0].Pos != 2 || m.manifest[0].Opaque != 7 {
			t.Fatalf("manifest = %+v", m.manifest[0])
		}

		ctx := context.Background()
		if items, _ := m.batchMedia(ctx, 0, 2); items[0].Features != nil || calls != 0 {
			t.Fatal("non-overlapping chunk encoded features")
		}
		if items, _ := m.batchMedia(ctx, 0, 4); items[0].Features == nil || calls != 1 {
			t.Fatalf("overlap did not encode once (calls=%d)", calls)
		}
		if items, _ := m.batchMedia(ctx, 4, 2); items[0].Features == nil || calls != 1 {
			t.Fatalf("second overlap re-encoded (calls=%d)", calls)
		}

		m.free(4)
		if m.manifest[0].Features == nil {
			t.Fatal("release dropped features before the expansion was evaluated")
		}
		m.free(6)
		if m.manifest[0].Features != nil {
			t.Fatal("release kept features past the expansion end")
		}
		m.close()

		if r.openMedia(Request{Tokens: make([]int32, 8)}) != nil {
			t.Fatal("openMedia returned non-nil for a text-only request")
		}
	})
}

// TestBatchMediaLazyPixels checks that a Pixels producer runs only when its
// item is first encoded, and that its error stops the chunk.
func TestBatchMediaLazyPixels(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		calls, produced := 0, 0
		lazy := &model.PreparedItem{
			Range: [2]int{2, 6},
			Dims:  []int{2},
			Pixels: func(context.Context) ([]float32, error) {
				produced++
				return []float32{1, 2}, nil
			},
		}
		broken := &model.PreparedItem{
			Range:  [2]int{6, 8},
			Dims:   []int{2},
			Pixels: func(context.Context) ([]float32, error) { return nil, errors.New("decode failed") },
		}
		// A producer with nothing to upload: the item encodes from shared
		// model state.
		shared := &model.PreparedItem{
			Range:  [2]int{8, 9},
			Dims:   []int{0, 1},
			Pixels: func(context.Context) ([]float32, error) { return nil, nil },
		}
		nilData := 0
		r := &Runner{Model: encodeCountingModel{calls: &calls, nilData: &nilData}}
		m := r.openMedia(Request{
			Tokens:     make([]int32, 10),
			MediaItems: []mediaItem{{pos: 2, length: 4, item: lazy}, {pos: 6, length: 2, item: broken}, {pos: 8, length: 1, item: shared}},
		})
		defer m.close()

		ctx := context.Background()
		if _, err := m.batchMedia(ctx, 0, 2); err != nil || produced != 0 {
			t.Fatalf("pixels produced before the item was reached (produced=%d, err=%v)", produced, err)
		}
		if items, err := m.batchMedia(ctx, 0, 5); err != nil || items[0].Features == nil || produced != 1 || calls != 1 {
			t.Fatalf("first overlap: produced=%d calls=%d err=%v", produced, calls, err)
		}
		if _, err := m.batchMedia(ctx, 5, 1); err != nil || produced != 1 {
			t.Fatalf("pixels produced again (produced=%d, err=%v)", produced, err)
		}
		if _, err := m.batchMedia(ctx, 5, 3); err == nil || err.Error() != "decode failed" {
			t.Fatalf("producer error = %v, want decode failed", err)
		}
		if items, err := m.batchMedia(ctx, 8, 1); err != nil || items[2].Features == nil || nilData != 1 {
			t.Fatalf("shared item: features=%v nilData=%d err=%v", items != nil && items[2].Features != nil, nilData, err)
		}
	})
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
