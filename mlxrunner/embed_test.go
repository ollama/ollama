package mlxrunner

import (
	"context"
	"errors"
	"fmt"
	"math"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/cache"
	"github.com/ollama/ollama/mlxrunner/model"
	"github.com/ollama/ollama/mlxrunner/tokenizer"
)

// stubModel is a generative-only model.Model (no EmbeddingDim).
type stubModel struct{}

func (s *stubModel) LoadWeights(map[string]*mlx.Array) error { return nil }
func (s *stubModel) NewCaches() []cache.Cache                { return nil }
func (s *stubModel) Forward(*batch.Batch, []cache.Layer) (*mlx.Array, *mlx.Array) {
	return nil, nil
}
func (s *stubModel) Unembed(*mlx.Array) *mlx.Array   { return nil }
func (s *stubModel) Tokenizer() *tokenizer.Tokenizer { return nil }
func (s *stubModel) MaxContextLength() int           { return 8 }

// stubEmbeddingModel adds EmbeddingDim so the handler's gate passes.
type stubEmbeddingModel struct{ stubModel }

func (s *stubEmbeddingModel) EmbeddingDim() int { return 768 }

func TestHandleEmbedGate(t *testing.T) {
	// Generative-only model: expect 501.
	r := &Runner{Model: &stubModel{}}
	req := httptest.NewRequest("POST", "/v1/embeddings", strings.NewReader(`{"content":"hi"}`))
	w := httptest.NewRecorder()
	r.handleEmbed(w, req)
	if w.Code != http.StatusNotImplemented {
		t.Fatalf("gate: got %d, want 501; body=%q", w.Code, w.Body.String())
	}

	// Invalid wire input must be rejected before scheduling a forward.
	for _, body := range []string{`{`, `{"content":""}`} {
		r := &Runner{Model: &stubEmbeddingModel{}}
		req := httptest.NewRequest("POST", "/v1/embeddings", strings.NewReader(body))
		w := httptest.NewRecorder()
		r.handleEmbed(w, req)
		if w.Code != http.StatusBadRequest {
			t.Fatalf("body %q: got %d, want 400; resp=%q", body, w.Code, w.Body.String())
		}
	}
}

func TestHandleEmbedCancellation(t *testing.T) {
	// Cancellation before the runner consumes: the handler must not panic or
	// hang when the request context is already done.
	r := &Runner{
		Model: &stubEmbeddingModel{},
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	req := httptest.NewRequest("POST", "/v1/embeddings", strings.NewReader(`{"content":"hi"}`)).WithContext(ctx)
	w := httptest.NewRecorder()
	done := make(chan struct{})
	go func() { r.handleEmbed(w, req); close(done) }()
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("handleEmbed hung on cancelled context")
	}
}

// TestEmbedMediaWireDecode covers the base64 boundary in handleEmbed: the
// model is never reached on decode failure; a valid entry reaches capability validation.
func TestEmbedMediaWireDecode(t *testing.T) {
	t.Run("bad base64 rejected with index", func(t *testing.T) {
		r := &Runner{Model: &stubEmbeddingModel{}}
		body := `{"content":"hi","media":["aGVsbG8=","!!!not base64!!!","d29ybGQ="]}`
		req := httptest.NewRequest("POST", "/v1/embeddings", strings.NewReader(body))
		w := httptest.NewRecorder()
		r.handleEmbed(w, req)
		if w.Code != http.StatusBadRequest {
			t.Fatalf("got %d, want 400; body=%q", w.Code, w.Body.String())
		}
		if !strings.Contains(w.Body.String(), "media[1]") {
			t.Errorf("400 body should name media[1], got %q", w.Body.String())
		}
	})

	t.Run("valid base64 reaches runner then 501s", func(t *testing.T) {
		r := &Runner{Model: &stubEmbeddingModel{}}

		body := `{"content":"hi","media":["aGVsbG8="]}`
		req := httptest.NewRequest("POST", "/v1/embeddings", strings.NewReader(body))
		w := httptest.NewRecorder()
		r.handleEmbed(w, req)
		if w.Code != http.StatusNotImplemented {
			t.Fatalf("got %d, want 501; body=%q", w.Code, w.Body.String())
		}
		if !strings.Contains(w.Body.String(), "media embedding not supported") {
			t.Errorf("501 body should say media is not supported, got %q", w.Body.String())
		}
	})
}

// miniEmbedTokenizer builds the smallest BPE tokenizer runEmbed can drive:
// byte-level, no merges, lowercase letters + space, BOS=0 EOS=1.
func miniEmbedTokenizer(t *testing.T) *tokenizer.Tokenizer {
	t.Helper()
	var sb strings.Builder
	sb.WriteString(`{"model": {"type": "BPE", "vocab": {`)
	id := 0
	add := func(s string) {
		if id > 0 {
			sb.WriteString(", ")
		}
		fmt.Fprintf(&sb, "%q: %d", s, id)
		id++
	}
	add("<bos>")
	add("<eos>")
	for c := 'a'; c <= 'z'; c++ {
		add(string(c))
	}
	add(" ")
	add("<img_placeholder>") // id 29: reserved ID so tests can distinguish placeholder rows
	sb.WriteString(`}, "merges": []}}`)
	tok, err := tokenizer.LoadFromBytesWithConfig([]byte(sb.String()), &tokenizer.TokenizerConfig{
		GenerationConfigJSON: []byte(`{"bos_token_id": 0, "eos_token_id": 1}`),
	})
	if err != nil {
		t.Fatal(err)
	}
	return tok
}

// stubEmbedMediaModel is a stubEmbeddingModel that also implements the embedMedia
// surface runEmbed dispatchers on. Placeholder expansion: each image segment
// becomes one token per entry of placeholderTokens. The Forward capture lets
// tests assert the runner spliced media at the right positions.
type stubEmbedMediaModel struct {
	stubEmbeddingModel

	lastBatch *batch.Batch
	prepared  *model.PreparedRequest
	// runEmbed scopes the forward: batch arrays are freed at scope exit,
	// so the stub materializes its captures as CPU data instead.
	capturedTokens []int32
	capturedMedia  []batch.MediaItem
}

const (
	stubPlaceholderToken = 29 // <img_placeholder> in miniEmbedTokenizer's vocab
	stubPlaceholdersN    = 2
)

func (s *stubEmbedMediaModel) SupportsImages() bool { return true }
func (s *stubEmbedMediaModel) SupportsAudio() bool  { return false }

func (s *stubEmbedMediaModel) PrepareMedia(segments []model.Segment) (*model.PreparedRequest, error) {
	prepared := &model.PreparedRequest{}
	for i, seg := range segments {
		switch seg.Kind {
		case "":
			prepared.Tokens = append(prepared.Tokens, seg.Tokens...)
		case "image":
			start := len(prepared.Tokens)
			for range stubPlaceholdersN {
				prepared.Tokens = append(prepared.Tokens, stubPlaceholderToken)
			}
			prepared.Items = append(prepared.Items, model.PreparedItem{
				Range:     [2]int{start, start + stubPlaceholdersN},
				Source:    i,
				MediaData: []float32{0}, Dims: []int{1},
			})
		default:
			return nil, fmt.Errorf("unsupported segment kind %q", seg.Kind)
		}
	}
	s.prepared = prepared
	return prepared, nil
}

func (s *stubEmbedMediaModel) EncodeMedia(item *model.PreparedItem, _ *mlx.Array) *mlx.Array {
	rows := item.Range[1] - item.Range[0]
	vals := make([]float32, rows*s.EmbeddingDim())
	for j := range vals {
		vals[j] = float32((j*13)%31) / 31
	}
	return mlx.FromValues(vals, rows, s.EmbeddingDim())
}

func (s *stubEmbedMediaModel) Forward(b *batch.Batch, _ []cache.Layer) (*mlx.Array, *mlx.Array) {
	s.lastBatch = b
	// Scope-safe capture: runEmbed frees the batch's arrays at scope exit,
	// so copy what the assertions need as CPU data now.
	ids := b.InputIDs.AsType(mlx.DTypeInt32)
	mlx.Eval(ids)
	s.capturedTokens = append(s.capturedTokens[:0], ids.Ints()...)
	for _, item := range b.Media {
		cp := item
		cp.Features = nil // shapes are captured separately below
		s.capturedMedia = append(s.capturedMedia, cp)
	}
	L := b.InputIDs.Dim(1)
	D := s.EmbeddingDim()
	vals := make([]float32, L*D)
	for i := range vals {
		vals[i] = float32((i*7)%29) / 29
	}
	return mlx.FromValues(vals, 1, L, D), nil
}

// pngSniff passable blob: only the 8-byte PNG magic is checked runner-side.
var stubPNG = []byte{0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n', 0, 0, 0, 0}

// TestEmbedMediaRunEmbedExercises covers the media branch of runEmbed end to
// end with a stub embedMedia model: a marker-less caller text plus image blob
// must reach Forward with BOS + caller tokens + one placeholder run + EOS, at
// the position the runner computed. Placeholder expansion per blob is now
// unconditional; caller text is never scanned for markers.
func TestEmbedMediaRunEmbedExercises(t *testing.T) {
	tok := miniEmbedTokenizer(t)

	t.Run("marker-less text plus one image blob", func(t *testing.T) {
		mlxtest.Run(t, func(t *mlxtest.T) {
			m := &stubEmbedMediaModel{}
			r := newScoreTestRunner(m)
			r.Tokenizer = tok
			req := embeddingRequest{
				Content: "ab",
				Media:   [][]byte{stubPNG},
			}
			resp, err := r.runEmbed(context.Background(), req)
			if err != nil {
				t.Fatalf("runEmbed: %v", err)
			}
			if len(resp.Embedding) != m.EmbeddingDim() {
				t.Fatalf("embedding len %d, want %d", len(resp.Embedding), m.EmbeddingDim())
			}

			if m.prepared == nil || len(m.prepared.Items) != 1 {
				t.Fatalf("PrepareMedia items = %+v, want exactly one", m.prepared)
			}
			if m.lastBatch == nil {
				t.Fatal("model Forward was never called")
			}
			if len(m.capturedMedia) != 1 {
				t.Fatalf("Forward batch Media len %d, want 1", len(m.capturedMedia))
			}
			item := m.capturedMedia[0]
			// Construct the expected stream from caller text, not from the
			// prepared object that runEmbed has already rebased in place.
			text := tok.Encode(req.Content, false)
			wantPos := 1 + len(text)
			rng := m.prepared.Items[0].Range
			if item.Pos != wantPos || rng != [2]int{wantPos, wantPos + stubPlaceholdersN} {
				t.Fatalf("media pos %d range %v, want [%d,%d)", item.Pos, rng, wantPos, wantPos+stubPlaceholdersN)
			}
			wantTokens := append([]int32{tok.BOS()}, text...)
			for range stubPlaceholdersN {
				wantTokens = append(wantTokens, stubPlaceholderToken)
			}
			wantTokens = append(wantTokens, tok.EOS())
			if !slices.Equal(m.capturedTokens, wantTokens) {
				t.Fatalf("forward tokens %v, want %v", m.capturedTokens, wantTokens)
			}
			// Feature rows must equal the placeholder-run length. runEmbed
			// frees the feature arrays at scope exit, so the stub captured
			// only CPU data; assert the run length it was sliced from.
			if got := rng[1] - rng[0]; got != stubPlaceholdersN {
				t.Errorf("placeholder run %d, want %d", got, stubPlaceholdersN)
			}
		})
	})
}

// TestRunEmbedMediaNotSupported: a runner model that implements
// EmbeddingDim but not embedMedia must 501 a media request (the wire-level
// twin of this gate is already covered in TestEmbedMediaWireDecode).
func TestEmbedMediaRunEmbedNotSupported(t *testing.T) {
	r := &Runner{Model: &stubEmbeddingModel{}, Tokenizer: miniEmbedTokenizer(t)}
	req := embeddingRequest{
		Content: "a<boi><eoi>b",
		Media:   [][]byte{stubPNG},
	}
	_, err := r.runEmbed(t.Context(), req)
	var status api.StatusError
	if !errors.As(err, &status) || status.StatusCode != http.StatusNotImplemented {
		t.Fatalf("error = %v, want 501", err)
	}
	if !strings.Contains(status.ErrorMessage, "media embedding not supported") {
		t.Errorf("501 body %q, want 'media embedding not supported'", status.ErrorMessage)
	}
}

// TestMeanPoolAndNormalize checks the ST module chain's math directly: mean
// over the first seqLen token rows (pads excluded), then L2.
func TestMeanPoolAndNormalize(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		// rows = [[1,0],[3,0],[99,99]] L=3 D=2, seqLen=2 → mean = [2,0], L2 → [1,0]
		rows := mlx.FromValues([]float32{1, 0, 3, 0, 99, 99}, 3, 2)
		got := meanPoolAndNormalize(rows, 2)
		if len(got) != 2 {
			t.Fatalf("len %d, want 2", len(got))
		}
		if got[0] < 0.9999 || got[1] > 1e-4 {
			t.Errorf("got %v, want ~[1, 0]", got)
		}
	})
}

// stubLongEmbedModel raises MaxContextLength so the token-count gate passes
// and the allocation pre-flight is what fires. Forward returns a minimal
// real hidden state so the unguarded path completes (and the test's
// assertion fires) when the gate is absent.
type stubLongEmbedModel struct{ stubEmbeddingModel }

func (s *stubLongEmbedModel) MaxContextLength() int { return 1 << 24 }

func (s *stubLongEmbedModel) Forward(b *batch.Batch, _ []cache.Layer) (*mlx.Array, *mlx.Array) {
	L := b.InputIDs.Dim(1)
	h := mlx.Zeros(mlx.DTypeFloat32, len(b.SeqOffsets), L, 1)
	return h, h
}

// TestRunEmbedRejectsOversizedAllocation: a token count whose predicted
// allocations exceed the capacity budget must 413 BEFORE any forward work —
// the failure mode that used to panic at metal::malloc.
func TestRunEmbedRejectsOversizedAllocation(t *testing.T) {
	tok := miniEmbedTokenizer(t)
	mlxtest.Run(t, func(t *mlxtest.T) {
		r := newScoreTestRunner(&stubLongEmbedModel{})
		r.Tokenizer = tok

		// Token count guaranteeing predicted > budget: solve
		// L*L*4 > budget for L (mask dominates).
		budget := EmbedCapacityBudget()
		L := int(math.Sqrt(float64(budget)/4)) + 64
		content := strings.Repeat("ab ", L/2+1) // ~L tokens for this vocab

		req := embeddingRequest{
			Content: content,
		}
		_, err := r.runEmbed(context.Background(), req)
		var status api.StatusError
		if !errors.As(err, &status) || status.StatusCode != http.StatusRequestEntityTooLarge {
			t.Fatalf("error = %v, want 413", err)
		}
		if !strings.Contains(status.ErrorMessage, "would allocate") {
			t.Errorf("message should name the allocation: %q", status.ErrorMessage)
		}

		// Rejection must leave the slot usable.
		if _, err := r.runEmbed(context.Background(), embeddingRequest{Content: "hello"}); err != nil {
			t.Fatalf("runEmbed after reject: %v", err)
		}
	})
}

// TestPredictedEmbedAllocationBytes pins the dominant-allocation formula:
// attention mask B*L*L + hidden output [1,L,D] + token row, all float32.
func TestPredictedEmbedAllocationBytes(t *testing.T) {
	tests := []struct {
		name string
		tok  int
		hid  int32
		want int64 // exact value for the formula
	}{
		{"empty", 0, 512, 0},
		{"tiny", 8, 512, int64(8*8*4 + 8*512*4 + 8*4)},
		{"300tok mteb shape", 300, 512, int64(300*300*4 + 300*512*4 + 300*4)},
	}
	for _, tt := range tests {
		if got := PredictedEmbedAllocationBytes(tt.tok, tt.hid); got != tt.want {
			t.Errorf("%s: got %d, want %d", tt.name, got, tt.want)
		}
	}
	if got := PredictedEmbedAllocationBytes(-5, 512); got != 0 {
		t.Errorf("negative tokens: got %d, want 0", got)
	}
}

// TestEmbedCapacityBudget must always return a usable positive budget with a
// live MLX device: its recommended working set halved for transient headroom.
// Without a device the function falls back to a constant; CI boxes have no
// MLX dylib, so SkipIfUnavailable short-circuits before any cgo call.
func TestEmbedCapacityBudget(t *testing.T) {
	mlxtest.SkipIfUnavailable(t)

	b := EmbedCapacityBudget()
	if b <= 0 {
		t.Fatalf("budget %d, want > 0", b)
	}
	if limit, err := mlx.MaxRecommendedWorkingSetSize(); err == nil && limit > 0 {
		if want := int64(float64(limit) * embedBudgetFraction); b != want {
			t.Errorf("budget = %d, want working set %d * fraction = %d", b, limit, want)
		}
	}
}
