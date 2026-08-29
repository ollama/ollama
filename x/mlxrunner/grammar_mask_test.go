package mlxrunner

import (
	"errors"
	"io/fs"
	"math"
	"path/filepath"
	"slices"
	"testing"

	"github.com/ollama/ollama/x/internal/mlxtest"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/xgrammar"
)

func TestApplyTokenMask(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
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
		t.Cleanup(e.close)
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
	})
}

const (
	draftTestEOS   int32 = 32
	draftTestVocab int   = 40
)

func draftTestVocabulary() []string {
	pieces := []string{
		"{", "}", `"`, ":", ",", "a", "n", "s", "w", "e", "r", "o", "k", " ",
		"1", "2", "[", "]", "true", "false", "ok", "answer",
	}
	for len(pieces) < draftTestVocab {
		pieces = append(pieces, "")
	}
	pieces[draftTestEOS] = "<eos>"
	return pieces
}

func draftPieceIDs(t *mlxtest.T, pieces ...string) []int32 {
	t.Helper()
	vocab := draftTestVocabulary()
	ids := make([]int32, len(pieces))
	for i, piece := range pieces {
		id := slices.Index(vocab, piece)
		if id < 0 {
			t.Fatalf("test vocabulary does not contain %q", piece)
		}
		ids[i] = int32(id)
	}
	return ids
}

// testDraftGrammar builds a grammar over the small draft-test vocabulary and
// an engine sized to match, skipping when the native payloads are not built.
func testDraftGrammar(t *mlxtest.T, schema string) (*grammarEngine, *grammar) {
	t.Helper()
	path, err := mlx.LoadedLibraryPath()
	if err != nil {
		t.Skipf("native MLX payload is not built: %v", err)
	}
	compiler, err := xgrammar.New(filepath.Dir(path), draftTestVocabulary(), draftTestVocab, []int32{draftTestEOS}, 8, 128<<20)
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			t.Skipf("native xgrammar payload is not built: %v", err)
		}
		t.Fatal(err)
	}
	t.Cleanup(compiler.Close)
	m, err := compiler.Compile(xgrammar.JSONSchema, schema)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(m.Close)
	e := &grammarEngine{}
	e.initMask(draftTestVocab)
	t.Cleanup(e.close)
	return e, &grammar{m: m}
}

// requireSameMask asserts two matchers fill identical masks, proving their
// parser states are equivalent.
func requireSameMask(t *mlxtest.T, got, want *grammar, vocab int) {
	t.Helper()
	words := (vocab + 31) / 32
	gotRow, wantRow := make([]int32, words), make([]int32, words)
	gotConstrained, err := got.m.Fill(gotRow)
	if err != nil {
		t.Fatal(err)
	}
	wantConstrained, err := want.m.Fill(wantRow)
	if err != nil {
		t.Fatal(err)
	}
	if gotConstrained != wantConstrained || !slices.Equal(gotRow, wantRow) {
		t.Fatalf("mask %v %032b, want %v %032b", gotConstrained, gotRow, wantConstrained, wantRow)
	}
}

// acceptRun advances a grammar over committed tokens through the engine's
// accept, failing the test on any rejection.
func acceptRun(t *mlxtest.T, e *grammarEngine, g *grammar, ids []int32) {
	t.Helper()
	for _, id := range ids {
		if err := errors.Join(e.accept([]*grammar{g}, []int32{id})...); err != nil {
			t.Fatal(err)
		}
	}
}

// maskedRows evaluates mask's logits, returning per-row allowed flags.
func maskedRows(t *mlxtest.T, masked *mlx.Array, rows int) [][]bool {
	t.Helper()
	mlx.Eval(masked)
	vals := masked.Floats()
	out := make([][]bool, rows)
	for i := range rows {
		out[i] = make([]bool, draftTestVocab)
		for id := range draftTestVocab {
			out[i][id] = !math.IsInf(float64(vals[i*draftTestVocab+id]), -1)
		}
	}
	return out
}

const draftTestSchema = `{"type":"object","properties":{"answer":{"type":"string","enum":["ok"]}},"required":["answer"],"additionalProperties":false}`

func TestMaskDraftPositions(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		e, g := testDraftGrammar(t, draftTestSchema)
		_, shadow := testDraftGrammar(t, draftTestSchema)

		drafts := draftPieceIDs(t, "{", `"`, "answer")
		logits := mlx.Zeros(mlx.DTypeFloat32, 1, len(drafts)+1, draftTestVocab)
		masked, errs := e.mask([]*grammar{g}, logits, [][]int32{drafts})
		if errs != nil {
			t.Fatal(errs)
		}
		// The walk left no matcher state behind: the mask matches a fresh shadow's.
		requireSameMask(t, g, shadow, draftTestVocab)

		rows := maskedRows(t, masked, len(drafts)+1)
		open := draftPieceIDs(t, "{", "}")
		if !rows[0][open[0]] || rows[0][open[1]] {
			t.Fatal("row 0 does not constrain to the object opener")
		}
		for i, id := range drafts {
			if !rows[i][id] {
				t.Fatalf("row %d masks its own draft token %d", i, id)
			}
		}
		quote := draftPieceIDs(t, `"`)[0]
		if !rows[3][quote] {
			t.Fatal("bonus row masks the value-opening quote")
		}

		// Verification rejects the third draft; the committed run reaches the
		// grammar through accept, like every emitted token.
		final := draftPieceIDs(t, "answer")[0]
		run := append(drafts[:2:2], final)
		acceptRun(t, e, g, run)
		for _, id := range run {
			if err := shadow.m.Accept(id); err != nil {
				t.Fatalf("shadow accept %d: %v", id, err)
			}
		}
		requireSameMask(t, g, shadow, draftTestVocab)
	})
}

func TestMaskStopsAtRejectedDraft(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		e, g := testDraftGrammar(t, draftTestSchema)
		_, shadow := testDraftGrammar(t, draftTestSchema)

		// "}" is invalid after "{": the answer property is required.
		drafts := draftPieceIDs(t, "{", "}", `"`)
		logits := mlx.Zeros(mlx.DTypeFloat32, 1, len(drafts)+1, draftTestVocab)
		masked, errs := e.mask([]*grammar{g}, logits, [][]int32{drafts})
		if errs != nil {
			t.Fatal(errs)
		}
		requireSameMask(t, g, shadow, draftTestVocab)

		rows := maskedRows(t, masked, len(drafts)+1)
		if rows[1][drafts[1]] {
			t.Fatal("row 1 allows the invalid draft")
		}
		for i := 2; i < len(rows); i++ {
			for id, ok := range rows[i] {
				if !ok {
					t.Fatalf("unreached row %d masks token %d", i, id)
				}
			}
		}
	})
}

func TestMaskCrossesAnAcceptedStop(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		e, g := testDraftGrammar(t, `{"type":"object"}`)

		drafts := draftPieceIDs(t, "{", "}", "<eos>")
		logits := mlx.Zeros(mlx.DTypeFloat32, 1, len(drafts)+1, draftTestVocab)
		masked, errs := e.mask([]*grammar{g}, logits, [][]int32{drafts})
		if errs != nil {
			t.Fatal(errs)
		}
		rows := maskedRows(t, masked, len(drafts)+1)
		if !rows[2][draftTestEOS] {
			t.Fatal("row 2 masks EOS after the object completes")
		}
		// The walk accepts the stop and the restore un-terminates: the position
		// past it is unconstrained, and the matcher replays from the start.
		for id, ok := range rows[3] {
			if !ok {
				t.Fatalf("row past the stop masks token %d", id)
			}
		}
		if g.m.Terminated() {
			t.Fatal("matcher terminated after the restore")
		}
		acceptRun(t, e, g, drafts)
		if !g.m.Terminated() {
			t.Fatal("matcher not terminated after the committed EOS")
		}
		// A terminated grammar accepts nothing further and needs nothing
		// accepted: the row is skipped, not faulted.
		if err := errors.Join(e.accept([]*grammar{g}, []int32{drafts[0]})...); err != nil {
			t.Fatalf("accept after termination: %v", err)
		}
	})
}

func TestMaskLeavesDeadPositionsUnmasked(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		// After `{"` this schema requires a property name no vocabulary piece can
		// start, so that position's fill rejects every token.
		schema := `{"type":"object","properties":{"z":{"type":"string"}},"required":["z"],"additionalProperties":false}`
		e, g := testDraftGrammar(t, schema)
		_, shadow := testDraftGrammar(t, schema)

		drafts := draftPieceIDs(t, "{", `"`)
		logits := mlx.Zeros(mlx.DTypeFloat32, 1, len(drafts)+1, draftTestVocab)
		masked, errs := e.mask([]*grammar{g}, logits, [][]int32{drafts})
		if errs != nil {
			t.Fatal(errs)
		}
		requireSameMask(t, g, shadow, draftTestVocab)
		rows := maskedRows(t, masked, len(drafts)+1)
		if !rows[0][drafts[0]] || !rows[1][drafts[1]] {
			t.Fatal("leading rows mask their own drafts")
		}
		for id, ok := range rows[2] {
			if !ok {
				t.Fatalf("dead position masks token %d", id)
			}
		}
		// A run committed into the dead state fails at its accept.
		acceptRun(t, e, g, drafts)
		if err := errors.Join(e.accept([]*grammar{g}, draftPieceIDs(t, "a"))...); err == nil {
			t.Fatal("accept into a dead grammar state unexpectedly succeeded")
		}

		// At position 0 the dead state is already committed: the row fails.
		e0, g0 := testDraftGrammar(t, `{"enum":[3]}`)
		logits0 := mlx.Zeros(mlx.DTypeFloat32, 1, 2, draftTestVocab)
		if _, errs := e0.mask([]*grammar{g0}, logits0, [][]int32{draftPieceIDs(t, "1")}); errs == nil || errs[0] == nil {
			t.Fatal("dead position 0 did not fail the row")
		}
	})
}

func TestMaskBatchesRows(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		e, g0 := testDraftGrammar(t, draftTestSchema)
		_, shadow0 := testDraftGrammar(t, draftTestSchema)
		_, g1 := testDraftGrammar(t, `{"type":"object"}`)
		_, shadow1 := testDraftGrammar(t, `{"type":"object"}`)

		// Ragged rows: row 0 advances three drafts, row 1 one; row 1's tail
		// positions pad with all-ones.
		drafts := [][]int32{draftPieceIDs(t, "{", `"`, "answer"), draftPieceIDs(t, "{")}
		logits := mlx.Zeros(mlx.DTypeFloat32, 2, 4, draftTestVocab)
		masked, errs := e.mask([]*grammar{g0, g1}, logits, drafts)
		if errs != nil {
			t.Fatal(errs)
		}
		requireSameMask(t, g0, shadow0, draftTestVocab)
		requireSameMask(t, g1, shadow1, draftTestVocab)

		rows := maskedRows(t, masked, 8)
		if !rows[4][drafts[1][0]] {
			t.Fatal("row 1 position 0 masks its own draft")
		}
		// The bands differ where the schemas do: only the permissive object may
		// close immediately after the opener.
		closeID := draftPieceIDs(t, "}")[0]
		if rows[1][closeID] {
			t.Fatal("strict schema allows closing an empty object")
		}
		if !rows[5][closeID] {
			t.Fatal("permissive schema masks closing an empty object")
		}
		for id, ok := range rows[6] {
			if !ok {
				t.Fatalf("padding position masks token %d", id)
			}
		}

		// Each row's committed run reconciles against its own matcher.
		acceptRun(t, e, g0, append(drafts[0][:2:2], draftPieceIDs(t, "answer")[0]))
		acceptRun(t, e, g1, []int32{drafts[1][0], closeID})
		for _, id := range draftPieceIDs(t, "{", `"`, "answer") {
			if err := shadow0.m.Accept(id); err != nil {
				t.Fatalf("shadow accept %d: %v", id, err)
			}
		}
		for _, id := range []int32{draftPieceIDs(t, "{")[0], closeID} {
			if err := shadow1.m.Accept(id); err != nil {
				t.Fatalf("shadow accept %d: %v", id, err)
			}
		}
		requireSameMask(t, g0, shadow0, draftTestVocab)
		requireSameMask(t, g1, shadow1, draftTestVocab)
	})
}
