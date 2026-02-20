package tokenizer

import (
	"testing"

	"github.com/google/go-cmp/cmp"
)

func TestSplitSpecialTokens(t *testing.T) {
	cases := []struct {
		name     string
		input    string
		values   []string
		types    []int32
		expected []fragment
	}{
		{
			name:   "no special tokens in text",
			input:  "hello world",
			values: []string{"<special>"},
			types:  []int32{TOKEN_TYPE_CONTROL},
			expected: []fragment{
				{value: "hello world"},
			},
		},
		{
			name:   "single special token at start",
			input:  "<bos>hello",
			values: []string{"<bos>"},
			types:  []int32{TOKEN_TYPE_CONTROL},
			expected: []fragment{
				{value: "<bos>", ids: []int32{0}},
				{value: "hello"},
			},
		},
		{
			name:   "single special token at end",
			input:  "hello<eos>",
			values: []string{"<eos>"},
			types:  []int32{TOKEN_TYPE_CONTROL},
			expected: []fragment{
				{value: "hello"},
				{value: "<eos>", ids: []int32{0}},
			},
		},
		{
			name:   "single special token in middle",
			input:  "hello<sep>world",
			values: []string{"<sep>"},
			types:  []int32{TOKEN_TYPE_CONTROL},
			expected: []fragment{
				{value: "hello"},
				{value: "<sep>", ids: []int32{0}},
				{value: "world"},
			},
		},
		{
			name:   "multiple occurrences of same token",
			input:  "<s>hello<s>world<s>",
			values: []string{"<s>"},
			types:  []int32{TOKEN_TYPE_CONTROL},
			expected: []fragment{
				{value: "<s>", ids: []int32{0}},
				{value: "hello"},
				{value: "<s>", ids: []int32{0}},
				{value: "world"},
				{value: "<s>", ids: []int32{0}},
			},
		},
		{
			name:   "multiple different special tokens",
			input:  "<bos>hello<sep>world<eos>",
			values: []string{"<bos>", "<sep>", "<eos>"},
			types:  []int32{TOKEN_TYPE_CONTROL, TOKEN_TYPE_CONTROL, TOKEN_TYPE_CONTROL},
			expected: []fragment{
				{value: "<bos>", ids: []int32{0}},
				{value: "hello"},
				{value: "<sep>", ids: []int32{1}},
				{value: "world"},
				{value: "<eos>", ids: []int32{2}},
			},
		},
		{
			name:   "non-overlapping special tokens with shared prefix",
			input:  "x<end_of_turn>y",
			values: []string{"<end>", "<end_of_turn>"},
			types:  []int32{TOKEN_TYPE_CONTROL, TOKEN_TYPE_CONTROL},
			expected: []fragment{
				// "<end>" is not a substring of "<end_of_turn>" (the > doesn't match _)
				// so only "<end_of_turn>" matches
				{value: "x"},
				{value: "<end_of_turn>", ids: []int32{1}},
				{value: "y"},
			},
		},
		{
			name:   "overlapping special tokens: earlier vocab entry wins",
			input:  "x<end>y",
			values: []string{"<end>", "<end_of_turn>"},
			types:  []int32{TOKEN_TYPE_CONTROL, TOKEN_TYPE_CONTROL},
			expected: []fragment{
				{value: "x"},
				{value: "<end>", ids: []int32{0}},
				{value: "y"},
			},
		},
		{
			name:   "true substring special token: shorter processed first",
			input:  "xABCy",
			values: []string{"AB", "ABC"},
			types:  []int32{TOKEN_TYPE_CONTROL, TOKEN_TYPE_CONTROL},
			expected: []fragment{
				// "AB" (vocabIdx=0) is processed first, claims "AB" from "ABC"
				{value: "x"},
				{value: "AB", ids: []int32{0}},
				{value: "Cy"},
			},
		},
		{
			name:   "input is exactly a special token",
			input:  "<special>",
			values: []string{"<special>"},
			types:  []int32{TOKEN_TYPE_USER_DEFINED},
			expected: []fragment{
				{value: "<special>", ids: []int32{0}},
			},
		},
		{
			name:   "empty input",
			input:  "",
			values: []string{"<special>"},
			types:  []int32{TOKEN_TYPE_CONTROL},
			expected: []fragment{
				{value: ""},
			},
		},
		{
			name:     "empty vocabulary",
			input:    "hello world",
			values:   []string{},
			types:    []int32{},
			expected: []fragment{{value: "hello world"}},
		},
		{
			name:   "special tokens not in text are skipped",
			input:  "hello<a>world",
			values: []string{"<a>", "<b>", "<c>", "<d>"},
			types:  []int32{TOKEN_TYPE_CONTROL, TOKEN_TYPE_CONTROL, TOKEN_TYPE_CONTROL, TOKEN_TYPE_CONTROL},
			expected: []fragment{
				{value: "hello"},
				{value: "<a>", ids: []int32{0}},
				{value: "world"},
			},
		},
		{
			name:   "only CONTROL and USER_DEFINED are special",
			input:  "hello<a><b><c>world",
			values: []string{"<a>", "<b>", "<c>"},
			types:  []int32{TOKEN_TYPE_CONTROL, TOKEN_TYPE_NORMAL, TOKEN_TYPE_USER_DEFINED},
			expected: []fragment{
				{value: "hello"},
				{value: "<a>", ids: []int32{0}},
				// <b> is NORMAL, not special — left as text
				{value: "<b>"},
				{value: "<c>", ids: []int32{2}},
				{value: "world"},
			},
		},
	}

	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			vocab := &Vocabulary{
				Values: tt.values,
				Types:  tt.types,
			}
			got := splitSpecialTokens(tt.input, vocab)
			if diff := cmp.Diff(tt.expected, got, cmp.AllowUnexported(fragment{})); diff != "" {
				t.Errorf("mismatch (-want +got):\n%s", diff)
			}
		})
	}
}
