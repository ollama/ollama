package xgrammar_test

import (
	"errors"
	"io/fs"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/xgrammar"
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

func testLibraryDir(t testing.TB) string {
	t.Helper()
	path, err := mlx.LoadedLibraryPath()
	if err != nil {
		t.Skipf("native MLX payload is not built: %v", err)
	}
	return filepath.Dir(path)
}

func testGrammarCompiler(t testing.TB) *xgrammar.Compiler {
	t.Helper()
	compiler, err := xgrammar.New(testLibraryDir(t), testVocabulary(), testVocabSize, []int32{testEOS}, 8, 128<<20)
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			t.Skipf("native xgrammar payload is not built: %v", err)
		}
		t.Fatal(err)
	}
	t.Cleanup(compiler.Close)
	return compiler
}

func FuzzJSONSchemaMatcher(f *testing.F) {
	compiler := testGrammarCompiler(f)
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
		matcher, err := compiler.Compile(xgrammar.JSONSchema, schema)
		if err != nil {
			return
		}
		defer matcher.Close()

		row := make([]int32, (testVocabSize+31)/32)
		for _, token := range tokens {
			id := int32(token) % testVocabSize
			if _, err := matcher.Fill(row); err != nil {
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

// fillMask fills a fresh mask row and reports whether it constrains.
func fillMask(t *testing.T, matcher *xgrammar.Matcher) ([]int32, bool) {
	t.Helper()
	row := make([]int32, (testVocabSize+31)/32)
	constrained, err := matcher.Fill(row)
	if err != nil {
		t.Fatal(err)
	}
	return row, constrained
}

func acceptPieces(t *testing.T, matcher *xgrammar.Matcher, pieces ...string) {
	t.Helper()
	vocab := testVocabulary()
	for _, piece := range pieces {
		id := int32(slices.Index(vocab, piece))
		if id < 0 {
			t.Fatalf("test vocabulary does not contain %q", piece)
		}
		mask, constrained := fillMask(t, matcher)
		if constrained && !allowed(mask, id) {
			t.Fatalf("token %d is not allowed by mask %032b", id, uint32(mask[id/32]))
		}
		if err := matcher.Accept(id); err != nil {
			t.Fatal(err)
		}
	}
}

func TestJSONSchemaMatcher(t *testing.T) {
	compiler := testGrammarCompiler(t)
	schema := `{"type":"object","properties":{"answer":{"type":"string","enum":["ok"]}},"required":["answer"],"additionalProperties":false}`
	matcher, err := compiler.Compile(xgrammar.JSONSchema, schema)
	if err != nil {
		t.Fatal(err)
	}
	defer matcher.Close()

	mask, constrained := fillMask(t, matcher)
	if !constrained {
		t.Fatal("the schema start does not constrain sampling")
	}
	if allowed(mask, testEOS) {
		t.Fatal("EOS is allowed before the schema is complete")
	}
	if allowed(mask, testPaddedID) {
		t.Fatal("a padded vocabulary ID is allowed")
	}

	acceptPieces(t, matcher, "{", `"`, "answer", `"`, ":", `"`, "ok", `"`, "}")
	mask, constrained = fillMask(t, matcher)
	if !constrained || !allowed(mask, testEOS) {
		t.Fatal("EOS is not allowed after the schema is complete")
	}
	acceptPieces(t, matcher, "<eos>")
}

// A terminated matcher no longer constrains: Fill reports no constraint, so
// a decoder can sample past the grammar's end.
func TestFillAfterTermination(t *testing.T) {
	compiler := testGrammarCompiler(t)
	matcher, err := compiler.Compile(xgrammar.JSONSchema, "{}")
	if err != nil {
		t.Fatal(err)
	}
	defer matcher.Close()
	acceptPieces(t, matcher, "{", "}", "<eos>")

	if _, constrained := fillMask(t, matcher); constrained {
		t.Fatal("terminated matcher still constrains sampling")
	}
}

func TestInvalidGrammarReturnsError(t *testing.T) {
	compiler := testGrammarCompiler(t)
	for _, tt := range []struct {
		name   string
		source string
	}{
		{name: "empty", source: ""},
		{name: "malformed", source: `{"type":`},
	} {
		t.Run(tt.name, func(t *testing.T) {
			matcher, err := compiler.Compile(xgrammar.JSONSchema, tt.source)
			if matcher != nil {
				matcher.Close()
			}
			if err == nil {
				t.Fatal("Compile unexpectedly succeeded")
			}
			if !strings.Contains(err.Error(), "compile grammar") {
				t.Fatalf("Compile error = %v", err)
			}
		})
	}
}

func TestCompilerValidationAndClosedState(t *testing.T) {
	loaded := testGrammarCompiler(t)
	for _, tt := range []struct {
		name      string
		pieces    []string
		vocabSize int
		stops     []int32
	}{
		{name: "zero vocabulary", vocabSize: 0, stops: []int32{0}},
		{name: "too many pieces", pieces: []string{"a", "b"}, vocabSize: 1, stops: []int32{0}},
		{name: "no stop tokens", pieces: []string{"a"}, vocabSize: 1},
		{name: "stop outside vocabulary", pieces: []string{"a"}, vocabSize: 1, stops: []int32{1}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			compiler, err := xgrammar.New(testLibraryDir(t), tt.pieces, tt.vocabSize, tt.stops, 8, 128<<20)
			if compiler != nil {
				compiler.Close()
			}
			if err == nil {
				t.Fatal("New unexpectedly succeeded")
			}
		})
	}

	matcher, err := loaded.Compile(xgrammar.JSONSchema, "{}")
	if err != nil {
		t.Fatal(err)
	}
	matcher.Close()
	matcher.Close()
	if _, err := matcher.Fill(make([]int32, (testVocabSize+31)/32)); err == nil {
		t.Fatal("Fill on closed matcher unexpectedly succeeded")
	}
	if err := matcher.Accept(0); err == nil {
		t.Fatal("Accept on closed matcher unexpectedly succeeded")
	}
	loaded.Close()
	loaded.Close()
	if matcher, err := loaded.Compile(xgrammar.JSONSchema, "{}"); err == nil || matcher != nil {
		t.Fatalf("Compile on closed compiler = %v, %v; want error", matcher, err)
	}
}
