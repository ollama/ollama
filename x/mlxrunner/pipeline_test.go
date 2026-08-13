package mlxrunner

import (
	"fmt"
	"slices"
	"strings"
	"testing"

	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model/base"
	"github.com/ollama/ollama/x/tokenizer"
)

// textOnlyModel satisfies base.Model but not base.MediaModel.
type textOnlyModel struct{}

func (textOnlyModel) LoadWeights(map[string]*mlx.Array) error { return nil }
func (textOnlyModel) NewCaches() []cache.Cache                { return nil }
func (textOnlyModel) Forward(*batch.Batch, []cache.Cache) (*mlx.Array, *mlx.Array) {
	return nil, nil
}
func (textOnlyModel) Unembed(x *mlx.Array) *mlx.Array { return x }
func (textOnlyModel) Tokenizer() *tokenizer.Tokenizer { return nil }
func (textOnlyModel) MaxContextLength() int           { return 0 }

func TestPrepareRejectsMediaWithoutSupport(t *testing.T) {
	r := &Runner{Model: textOnlyModel{}}
	req := &Request{
		CompletionRequest: CompletionRequest{
			Prompt: "[img-0] what is this?",
			Media:  []llm.MediaData{{ID: 0, Kind: llm.MediaKindImage, Data: []byte{1}}},
		},
	}

	err := r.Prepare(req)
	if err == nil {
		t.Fatal("expected error for media on a text-only model")
	}
	if !strings.Contains(err.Error(), "does not support image input") {
		t.Fatalf("unexpected error: %v", err)
	}
}

// stubMediaModel expands every media segment to the same placeholder
// tokens. Expansion token values are deliberately outside the test
// tokenizer's vocabulary: they are spliced, never encoded.
type stubMediaModel struct {
	textOnlyModel
	expansion []int32
	layout    any
}

func (m stubMediaModel) PrepareMedia(segments []base.Segment) (*base.PreparedRequest, error) {
	prepared := &base.PreparedRequest{Layout: m.layout}
	for s, seg := range segments {
		if seg.Data == nil {
			prepared.Tokens = append(prepared.Tokens, seg.Tokens...)
			continue
		}
		if seg.Kind != string(llm.MediaKindImage) {
			return nil, fmt.Errorf("stub does not support %s input", seg.Kind)
		}
		if len(m.expansion) == 0 {
			continue
		}
		start := len(prepared.Tokens)
		prepared.Tokens = append(prepared.Tokens, m.expansion...)
		prepared.Items = append(prepared.Items, base.PreparedItem{
			Range:     [2]int{start, len(prepared.Tokens)},
			Source:    s,
			MediaData: []float32{float32(len(seg.Data))},
			Dims:      []int{1},
			Opaque:    len(seg.Data),
		})
	}
	return prepared, nil
}

func (stubMediaModel) EncodeMedia(*base.PreparedItem, *mlx.Array) *mlx.Array { return nil }

func mediaTestRunner(t *testing.T) *Runner {
	t.Helper()
	return &Runner{
		Model:         stubMediaModel{expansion: []int32{70, 500, 500, 71}, layout: "L"},
		Tokenizer:     newTestTokenizer(t, []int32{7}),
		contextLength: 4096,
	}
}

func TestPrepareExpandsMediaTags(t *testing.T) {
	r := mediaTestRunner(t)
	req := &Request{
		CompletionRequest: CompletionRequest{
			Prompt: "01[img-0]23",
			Media:  []llm.MediaData{{ID: 0, Kind: llm.MediaKindImage, Data: []byte("img")}},
		},
	}
	if err := r.Prepare(req); err != nil {
		t.Fatal(err)
	}

	want := []int32{0, 1, 70, 500, 500, 71, 2, 3}
	if !slices.Equal(req.Tokens, want) {
		t.Fatalf("tokens = %v, want %v", req.Tokens, want)
	}
	if len(req.MediaItems) != 1 {
		t.Fatalf("items = %d, want 1", len(req.MediaItems))
	}
	item := req.MediaItems[0]
	if item.pos != 2 || item.length != 4 {
		t.Fatalf("item at [%d,+%d), want [2,+4)", item.pos, item.length)
	}
	if item.fold&(1<<31) == 0 {
		t.Fatal("item fold missing bit 31")
	}
	if item.item == nil || item.item.Opaque != 3 {
		t.Fatal("prepared item not carried")
	}
	if req.Layout != "L" {
		t.Fatalf("layout = %v, want the model's", req.Layout)
	}
}

// rawMediaModel returns a fixed PreparedRequest, letting tests exercise the
// runner's validation of model-authored items.
type rawMediaModel struct {
	textOnlyModel
	prepared base.PreparedRequest
}

func (m rawMediaModel) PrepareMedia([]base.Segment) (*base.PreparedRequest, error) {
	p := m.prepared
	return &p, nil
}
func (rawMediaModel) EncodeMedia(*base.PreparedItem, *mlx.Array) *mlx.Array { return nil }

func TestPrepareValidatesAuthoredItems(t *testing.T) {
	media := []llm.MediaData{{ID: 0, Kind: llm.MediaKindImage, Data: []byte("img")}}
	item := func(lo, hi, src int) base.PreparedItem {
		return base.PreparedItem{Range: [2]int{lo, hi}, Source: src, Dims: []int{1}}
	}
	// Prompt "0[img-0]1" produces segments 0=text, 1=media, 2=text.
	cases := []struct {
		name  string
		items []base.PreparedItem
		want  string // "" means the items must be accepted
	}{
		{"per-tile items", []base.PreparedItem{item(1, 3, 1), item(3, 4, 1)}, ""},
		{"overlap", []base.PreparedItem{item(1, 3, 1), item(2, 4, 1)}, "invalid range"},
		{"out of bounds", []base.PreparedItem{item(2, 9, 1)}, "invalid range"},
		{"empty range", []base.PreparedItem{item(2, 2, 1)}, "invalid range"},
		{"text source", []base.PreparedItem{item(1, 3, 0)}, "non-media segment"},
	}
	for _, c := range cases {
		r := mediaTestRunner(t)
		r.Model = rawMediaModel{prepared: base.PreparedRequest{Tokens: []int32{0, 5, 5, 5, 1}, Items: c.items}}
		err := r.Prepare(&Request{CompletionRequest: CompletionRequest{Prompt: "0[img-0]1", Media: media}})
		if c.want == "" {
			if err != nil {
				t.Fatalf("%s: unexpected error %v", c.name, err)
			}
			continue
		}
		if err == nil || !strings.Contains(err.Error(), c.want) {
			t.Fatalf("%s: err = %v, want %q", c.name, err, c.want)
		}
	}
}

func TestPrepareMediaErrors(t *testing.T) {
	media := []llm.MediaData{{ID: 0, Kind: llm.MediaKindImage, Data: []byte("img")}}

	r := mediaTestRunner(t)
	err := r.Prepare(&Request{CompletionRequest: CompletionRequest{Prompt: "0[img-3]1", Media: media}})
	if err == nil || !strings.Contains(err.Error(), "invalid image index: 3") {
		t.Fatalf("missing-ID error = %v", err)
	}

	// Unreferenced media is ignored with a warning; the prompt still works.
	req := &Request{CompletionRequest: CompletionRequest{Prompt: "01", Media: media}}
	if err := r.Prepare(req); err != nil {
		t.Fatal(err)
	}
	if len(req.MediaItems) != 0 {
		t.Fatalf("items = %d, want 0", len(req.MediaItems))
	}

	// Duplicate references are allowed; each occurrence is its own item with
	// the same fold.
	req = &Request{CompletionRequest: CompletionRequest{Prompt: "[img-0]0[img-0]", Media: media}}
	if err := r.Prepare(req); err != nil {
		t.Fatal(err)
	}
	if len(req.MediaItems) != 2 || req.MediaItems[0].fold != req.MediaItems[1].fold {
		t.Fatalf("duplicate items = %+v", req.MediaItems)
	}

	// A zero-length expansion cannot carry identity into the trie keys.
	r.Model = stubMediaModel{}
	err = r.Prepare(&Request{CompletionRequest: CompletionRequest{Prompt: "[img-0]", Media: media}})
	if err == nil || !strings.Contains(err.Error(), "no tokens") {
		t.Fatalf("zero-expansion error = %v", err)
	}
}

func TestGenerationBudget(t *testing.T) {
	tests := []struct {
		name          string
		numPredict    int
		numCtx        int
		contextLength int
		promptLen     int
		want          int
	}{
		{
			// A checkpoint's max_position_embeddings is not a usable budget:
			// the request window is.
			name:          "open ended bounded by context window",
			numPredict:    -1,
			numCtx:        4096,
			contextLength: 131072,
			promptLen:     12,
			want:          40960,
		},
		{
			name:          "open ended bounded by model context when window is larger",
			numPredict:    -1,
			numCtx:        131072,
			contextLength: 8192,
			promptLen:     192,
			want:          8000,
		},
		{
			name:          "open ended without a context window uses the model context",
			numPredict:    -1,
			numCtx:        0,
			contextLength: 4096,
			promptLen:     96,
			want:          4000,
		},
		{
			name:          "explicit budget preserved",
			numPredict:    128,
			numCtx:        4096,
			contextLength: 131072,
			promptLen:     12,
			want:          128,
		},
		{
			name:          "explicit budget capped by model context",
			numPredict:    9000,
			numCtx:        131072,
			contextLength: 8192,
			promptLen:     192,
			want:          8000,
		},
		{
			name:          "zero budget still generates within the model context",
			numPredict:    0,
			numCtx:        4096,
			contextLength: 131072,
			promptLen:     12,
			want:          131060,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := generationBudget(tt.numPredict, tt.numCtx, tt.contextLength, tt.promptLen); got != tt.want {
				t.Fatalf("generationBudget(%d, %d, %d, %d) = %d, want %d",
					tt.numPredict, tt.numCtx, tt.contextLength, tt.promptLen, got, tt.want)
			}
		})
	}
}
