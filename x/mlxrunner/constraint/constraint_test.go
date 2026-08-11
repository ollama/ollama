package constraint_test

import (
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/ollama/ollama/x/mlxrunner/constraint"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

const (
	testEOS       int32 = 32                // Stop token in the second packed-mask word.
	testVocabSize       = 40                // Spans two packed-mask words, with the second only partly used.
	testPaddedID  int32 = testVocabSize - 1 // Empty padding at the vocabulary boundary must stay disallowed.
)

func testVocabulary() []string {
	pieces := []string{
		"{", "}", `"`, ":", ",", "a", "n", "s", "w", "e", "r", "o", "k", " ",
		"1", "2", "[", "]", "true", "false", "ok", "answer", "\x00", "\xff",
	}
	for len(pieces) < testVocabSize {
		pieces = append(pieces, "")
	}
	pieces[testEOS] = "<eos>"
	return pieces
}

func testConstraintModel(t testing.TB) *constraint.Model {
	t.Helper()
	path, err := mlx.LoadedLibraryPath()
	if err != nil {
		t.Skipf("native MLX payload is not built: %v", err)
	}
	if err := constraint.Load(filepath.Dir(path)); err != nil {
		t.Skipf("native constraint payload is not built: %v", err)
	}
	if constraint.LoadedLibraryPath() == "" {
		t.Fatal("constraint library loaded without recording its path")
	}

	pieces := testVocabulary()
	model, err := constraint.NewModel(pieces, testVocabSize, []int32{testEOS})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(model.Close)
	return model
}

func FuzzJSONSchemaMatcher(f *testing.F) {
	model := testConstraintModel(f)
	for _, seed := range []struct {
		schema string
		tokens []byte
	}{
		{schema: `{}`, tokens: []byte{0, 1, byte(testEOS)}},                    // Accept {}, then EOS.
		{schema: `{"type":"string"}`, tokens: []byte{2, 20, 2, byte(testEOS)}}, // Accept "ok", then EOS.
		{schema: `{"$defs":{"node":{"type":"array","items":{"$ref":"#/$defs/node"}}},"$ref":"#/$defs/node"}`},
		{schema: `{"enum":["a","b","c"]}`},
		{schema: `{"type":`},
		{schema: string([]byte{'{', '"', 'x', '"', ':', '"', 0xff, '"', '}'})},
	} {
		f.Add(seed.schema, seed.tokens)
	}

	f.Fuzz(func(t *testing.T, schema string, tokens []byte) {
		if len(schema) > 1<<20 || len(tokens) > 256 {
			return
		}
		matcher, err := model.Compile(constraint.JSONSchema, schema)
		if err != nil {
			return
		}
		defer matcher.Close()

		for _, token := range tokens {
			id := int32(token) % testVocabSize
			if _, _, err := matcher.Fill(); err != nil {
				return
			}
			if err := matcher.Accept(id); err != nil {
				return
			}
		}
	})
}

func allowed(mask []int32, id int32) bool {
	return uint32(mask[id/32])&(uint32(1)<<uint(id%32)) != 0
}

func acceptPieces(t *testing.T, matcher *constraint.Matcher, pieces ...string) {
	t.Helper()
	vocab := testVocabulary()
	for _, piece := range pieces {
		id := int32(slices.Index(vocab, piece))
		if id < 0 {
			t.Fatalf("test vocabulary does not contain %q", piece)
		}
		mask, _, err := matcher.Fill()
		if err != nil {
			t.Fatal(err)
		}
		if !allowed(mask, id) {
			t.Fatalf("token %d is not allowed by mask %032b", id, uint32(mask[id/32]))
		}
		if err := matcher.Accept(id); err != nil {
			t.Fatal(err)
		}
	}
}

func TestJSONSchemaMatcher(t *testing.T) {
	model := testConstraintModel(t)
	schema := `{"type":"object","properties":{"answer":{"type":"string","enum":["ok"]}},"required":["answer"],"additionalProperties":false}`
	matcher, err := model.Compile(constraint.JSONSchema, schema)
	if err != nil {
		t.Fatal(err)
	}
	defer matcher.Close()
	if got := matcher.VocabSize(); got != testVocabSize {
		t.Fatalf("matcher vocabulary size = %d, want %d", got, testVocabSize)
	}

	mask, _, err := matcher.Fill()
	if err != nil {
		t.Fatal(err)
	}
	if allowed(mask, testEOS) {
		t.Fatal("EOS is allowed before the schema is complete")
	}
	if allowed(mask, testPaddedID) {
		t.Fatal("a padded vocabulary ID is allowed")
	}

	acceptPieces(t, matcher, "{", `"`, "answer", `"`, ":", `"`, "ok", `"`, "}")
	mask, _, err = matcher.Fill()
	if err != nil {
		t.Fatal(err)
	}
	if !allowed(mask, testEOS) {
		t.Fatal("EOS is not allowed after the schema is complete")
	}
	acceptPieces(t, matcher, "<eos>")
}

func TestBuiltinJSON(t *testing.T) {
	model := testConstraintModel(t)
	matcher, err := model.Compile(constraint.JSON, "")
	if err != nil {
		t.Fatal(err)
	}
	defer matcher.Close()
	acceptPieces(t, matcher, "{", "}", "<eos>")
}

func TestInvalidConstraintReturnsError(t *testing.T) {
	model := testConstraintModel(t)
	for _, tt := range []struct {
		name   string
		source string
	}{
		{name: "empty", source: ""},
		{name: "malformed", source: `{"type":`},
		{name: "too large", source: strings.Repeat(" ", (1<<20)+1)},
	} {
		t.Run(tt.name, func(t *testing.T) {
			matcher, err := model.Compile(constraint.JSONSchema, tt.source)
			if matcher != nil {
				matcher.Close()
			}
			if err == nil {
				t.Fatal("Compile unexpectedly succeeded")
			}
			if !strings.Contains(err.Error(), "compile constraint") {
				t.Fatalf("Compile error = %v", err)
			}
		})
	}
}

func TestModelValidationAndClosedState(t *testing.T) {
	loaded := testConstraintModel(t)
	for _, tt := range []struct {
		name      string
		pieces    []string
		vocabSize int
		stops     []int32
	}{
		{name: "zero vocabulary", vocabSize: 0, stops: []int32{0}},
		{name: "too many pieces", pieces: []string{"a", "b"}, vocabSize: 1, stops: []int32{0}},
		{name: "vocabulary exceeds native limit", pieces: []string{"a"}, vocabSize: (1 << 20) + 1, stops: []int32{0}},
		{name: "no stop tokens", pieces: []string{"a"}, vocabSize: 1},
		{name: "too many stop tokens", pieces: []string{"a"}, vocabSize: 1, stops: []int32{0, 0}},
		{name: "stop outside vocabulary", pieces: []string{"a"}, vocabSize: 1, stops: []int32{1}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			model, err := constraint.NewModel(tt.pieces, tt.vocabSize, tt.stops)
			if model != nil {
				model.Close()
			}
			if err == nil {
				t.Fatal("NewModel unexpectedly succeeded")
			}
		})
	}

	matcher, err := loaded.Compile(constraint.JSON, "")
	if err != nil {
		t.Fatal(err)
	}
	matcher.Close()
	matcher.Close()
	if _, _, err := matcher.Fill(); err == nil {
		t.Fatal("Fill on closed matcher unexpectedly succeeded")
	}
	if err := matcher.Accept(0); err == nil {
		t.Fatal("Accept on closed matcher unexpectedly succeeded")
	}
	loaded.Close()
	loaded.Close()
	if matcher, err := loaded.Compile(constraint.JSON, ""); err == nil || matcher != nil {
		t.Fatalf("Compile on closed model = %v, %v; want error", matcher, err)
	}
}
