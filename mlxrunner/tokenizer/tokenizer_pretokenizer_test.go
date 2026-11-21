package tokenizer

import (
	"encoding/json"
	"slices"
	"strings"
	"testing"
)

const rightAlignedDigitsPretokenizer = `{
  "type": "Sequence",
  "pretokenizers": [
    {"type":"Split","pattern":{"Regex":"\\d{1,3}(?=(?:\\d{3})*\\b)"},"behavior":"Isolated","invert":false},
    {"type":"Split","pattern":{"Regex":"[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"},"behavior":"Isolated","invert":false},
    {"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":false}
  ]
}`

const mergedNewlinesPretokenizer = `{
  "type": "Sequence",
  "pretokenizers": [
    {"type":"Split","pattern":{"Regex":"(?:\\r?\\n)+(?!\\r?\\n)"},"behavior":"MergedWithNext","invert":false},
    {"type":"Split","pattern":{"Regex":"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"},"behavior":"Isolated","invert":false},
    {"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":false}
  ]
}`

// Split expectations were checked with Hugging Face tokenizers 0.22.2.
func TestPretokenizerSplits(t *testing.T) {
	for _, tc := range []struct {
		name, pretokenizer, text string
		want                     []string
	}{
		{"whitespace suffix", `{"type":"Split","pattern":{"Regex":"\\s+(?!\\S)|\\s+"},"behavior":"Isolated"}`, "  x", []string{" ", " ", "x"}},
		{"alternative after literal backslash", `{"type":"Split","pattern":{"Regex":"a\\\\|\\s+(?!\\S)|\\s+"},"behavior":"Isolated"}`, `a\  x`, []string{`a\`, " ", " ", "x"}},
		{"punctuation optional space", `{"type":"Split","pattern":{"Regex":" ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s+(?!\\S)|\\s+"}}`, "    }\n", []string{"   ", " }\n"}},
		{"punctuation without optional space", `{"type":"Split","pattern":{"Regex":"[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s+(?!\\S)|\\s+"}}`, "    }\n", []string{"   ", " ", "}\n"}},
		{"delimiter Isolated", `{"type":"Split","pattern":{"String":" "},"behavior":"Isolated","invert":false}`, "  a  b  ", []string{" ", " ", "a", " ", " ", "b", " ", " "}},
		{"delimiter Removed", `{"type":"Split","pattern":{"String":" "},"behavior":"Removed","invert":false}`, "  a  b  ", []string{"a", "b"}},
		{"delimiter MergedWithPrevious", `{"type":"Split","pattern":{"String":" "},"behavior":"MergedWithPrevious","invert":false}`, "  a  b  ", []string{" ", " ", "a ", " ", "b ", " "}},
		{"delimiter MergedWithNext", `{"type":"Split","pattern":{"String":" "},"behavior":"MergedWithNext","invert":false}`, "  a  b  ", []string{" ", " a", " ", " b", " ", " "}},
		{"delimiter Contiguous", `{"type":"Split","pattern":{"String":" "},"behavior":"Contiguous","invert":false}`, "  a  b  ", []string{"  ", "a", "  ", "b", "  "}},
		{"inverted contiguous", `{"type":"Split","pattern":{"String":" "},"behavior":"Contiguous","invert":true}`, "  a  b  ", []string{"  ", "a", "  ", "b", "  "}},
		{"inverted words", `{"type":"Split","pattern":{"Regex":"\\w+"},"behavior":"Removed","invert":true}`, "abc ! def", []string{"abc", "def"}},
		{"Unicode \\s+", `{"type":"Split","pattern":{"Regex":"\\s+"},"behavior":"Isolated","invert":false}`, "a\u0085\u0085١٢३́ b", []string{"a", "\u0085\u0085", "١٢३́", " ", "b"}},
		{"Unicode [^\\s]+", `{"type":"Split","pattern":{"Regex":"[^\\s]+"},"behavior":"Isolated","invert":false}`, "a\u0085\u0085١٢३́ b", []string{"a", "\u0085\u0085", "١٢३́", " ", "b"}},
		{"Unicode \\S+", `{"type":"Split","pattern":{"Regex":"\\S+"},"behavior":"Isolated","invert":false}`, "a\u0085\u0085١٢३́ b", []string{"a", "\u0085\u0085", "١٢३́", " ", "b"}},
		{"Unicode \\d+", `{"type":"Split","pattern":{"Regex":"\\d+"},"behavior":"Isolated","invert":false}`, "a\u0085\u0085١٢३́ b", []string{"a\u0085\u0085", "١٢३", "́ b"}},
		{"Unicode \\w+", `{"type":"Split","pattern":{"Regex":"\\w+"},"behavior":"Isolated","invert":false}`, "a\u0085\u0085١٢३́ b", []string{"a", "\u0085\u0085", "١٢३́", " ", "b"}},
		{"right aligned digits", rightAlignedDigitsPretokenizer, "1234567", []string{"1", "234", "567"}},
		{"digit stage whitespace", rightAlignedDigitsPretokenizer, "  1234", []string{"  ", "1", "234"}},
		{"digit word suffix", rightAlignedDigitsPretokenizer, "1234a", []string{"123", "4", "a"}},
		{"digits combining mark", rightAlignedDigitsPretokenizer, "1234́", []string{"123", "4", "́"}},
		{"Unicode digits", rightAlignedDigitsPretokenizer, "١٢٣٤", []string{"١", "٢٣٤"}},
		{"digit joiner", rightAlignedDigitsPretokenizer, "1234\u200d", []string{"1", "234", "\u200d"}},
		{"newline stage 'a\\n\\nb'", mergedNewlinesPretokenizer, "a\n\nb", []string{"a", "\n\n", "b"}},
		{"newline stage 'a\\r\\n\\r\\n'", mergedNewlinesPretokenizer, "a\r\n\r\n", []string{"a", "\r\n\r\n"}},
		{"newline stage 'a\\n\\n 1'", mergedNewlinesPretokenizer, "a\n\n 1", []string{"a", "\n\n", " ", "1"}},
		{"ByteLevel regex False prefix False", `{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false}`, "hello world", []string{"hello world"}},
		{"ByteLevel regex True prefix False", `{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":true}`, "hello world", []string{"hello", " world"}},
		{"ByteLevel regex False prefix True", `{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":false,"use_regex":false}`, "hello world", []string{" hello world"}},
		{"ByteLevel regex True prefix True", `{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":false,"use_regex":true}`, "hello world", []string{" hello", " world"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			data := []byte(`{"model":{"type":"BPE","vocab":{},"merges":[]},"pre_tokenizer":` + tc.pretokenizer + `}`)
			tok, err := LoadFromBytes(data)
			if err != nil {
				t.Fatal(err)
			}
			var got []string
			tok.forEachPartChunk(tc.text, func(c encodeChunk) { got = append(got, c.text) })
			if !slices.Equal(got, tc.want) {
				t.Fatalf("chunks for %q = %q, want %q", tc.text, got, tc.want)
			}
		})
	}
}

func TestUnsupportedPretokenizerStage(t *testing.T) {
	for _, pattern := range []string{`x(?=y)`, `(?i:ss)`, `(?mi:ss)`, `(?m:.)`, `[[:space:]]+`, `[a-z&&[^aeiou]]+`, `[a&&b]+`, `^x`, `\b`, `a\|\s+(?!\S)|\s+`, `a\\\|\s+(?!\S)|\s+`} {
		t.Run(pattern, func(t *testing.T) {
			quoted, err := json.Marshal(pattern)
			if err != nil {
				t.Fatal(err)
			}
			data := `{"model":{"type":"BPE","vocab":{},"merges":[]},"pre_tokenizer":{"type":"Sequence","pretokenizers":[{"type":"Split","pattern":{"Regex":` + string(quoted) + `},"behavior":"Isolated","invert":false},{"type":"Split","pattern":{"Regex":".+"},"behavior":"Isolated","invert":false}]}}`
			if _, err := LoadFromBytes([]byte(data)); err == nil {
				t.Fatal("unsupported stage must not be skipped in favor of the later Split")
			}
		})
	}
}

func FuzzPretokenizer(f *testing.F) {
	for _, pretokenizer := range []string{
		`{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":true}`,
		`{"type":"Sequence","pretokenizers":[
  {"type":"Split","pattern":{"Regex":"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"},"behavior":"Isolated","invert":false},
  {"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":false}
]}`,
		`{"type":"Sequence","pretokenizers":[
  {"type":"Split","pattern":{"Regex":"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"},"behavior":"Isolated","invert":false},
  {"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false}
]}`,
		`{"type":"Sequence","pretokenizers":[
  {"type":"Split","pattern":{"Regex":"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"},"behavior":"Isolated","invert":false},
  {"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":false}
]}`,
		`{"type":"Sequence","pretokenizers":[
  {"type":"Split","pattern":{"Regex":"[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"},"behavior":"Isolated","invert":false},
  {"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":false}
]}`,
		`{"type":"Sequence","pretokenizers":[
  {"type":"Split","pattern":{"Regex":"[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"},"behavior":"Isolated","invert":false},
  {"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":false}
]}`,
		rightAlignedDigitsPretokenizer,
		mergedNewlinesPretokenizer,
	} {
		for _, text := range []string{"", "  1234\n\nword", "हिन्दी  \u0085!"} {
			f.Add([]byte(pretokenizer), text)
		}
	}
	f.Fuzz(func(t *testing.T, data []byte, text string) {
		stages, err := loadPretokenizers(data)
		if err != nil {
			return
		}
		preservesText := true
		for _, stage := range stages {
			if stage.behavior == "Removed" || stage.prefixSpace {
				preservesText = false
			}
		}
		tok := Tokenizer{pretokenizer: stages}
		var joined strings.Builder
		tok.forEachPartChunk(text, func(c encodeChunk) { joined.WriteString(c.text) })
		if preservesText && joined.String() != text {
			t.Fatalf("split input %q became %q", text, joined.String())
		}
	})
}
