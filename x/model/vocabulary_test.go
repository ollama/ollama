package model

import (
	"testing"

	"github.com/google/go-cmp/cmp"
)

func TestSpecialVocabulary(t *testing.T) {
	vocab := &Vocabulary{
		Values: []string{"<|startoftext|>", "<|endoftext|>", "<|tool_call_start|>", "<|tool_call_end|>", "hi"},
		Types:  []int32{TOKEN_TYPE_CONTROL, TOKEN_TYPE_CONTROL, TOKEN_TYPE_USER_DEFINED, TOKEN_TYPE_USER_DEFINED, TOKEN_TYPE_NORMAL},
	}

	specialVocab := vocab.SpecialVocabulary()

	if len(specialVocab) != 4 {
		t.Errorf("expected 4 special tokens, got %d", len(specialVocab))
	}
}

func TestAddSpecialVocabulary(t *testing.T) {
	cases := []struct {
		name  string
		vocab *Vocabulary
		input []int32
		want  []int32
	}{
		{
			name: "add bos",
			vocab: &Vocabulary{
				BOS:    []int32{0},
				EOS:    []int32{1},
				AddBOS: true,
				AddEOS: false,
			},
			input: []int32{2, 3, 4},
			want:  []int32{0, 2, 3, 4},
		},
		{
			// TODO(mxyng): this is to match previous behaviour
			name: "add bos when already present",
			vocab: &Vocabulary{
				BOS:    []int32{0},
				EOS:    []int32{1},
				AddBOS: true,
				AddEOS: false,
			},
			input: []int32{0, 2, 3, 4},
			want:  []int32{0, 0, 2, 3, 4},
		},
		{
			name: "add eos",
			vocab: &Vocabulary{
				BOS:    []int32{0},
				EOS:    []int32{1},
				AddBOS: false,
				AddEOS: true,
			},
			input: []int32{2, 3, 4},
			want:  []int32{2, 3, 4, 1},
		},
		{
			// TODO(mxyng): this is to match previous behaviour
			name: "add eos when already present",
			vocab: &Vocabulary{
				BOS:    []int32{0},
				EOS:    []int32{1},
				AddBOS: false,
				AddEOS: true,
			},
			input: []int32{2, 3, 4, 1},
			want:  []int32{2, 3, 4, 1, 1},
		},
		{
			name: "add both",
			vocab: &Vocabulary{
				BOS:    []int32{0},
				EOS:    []int32{1},
				AddBOS: true,
				AddEOS: true,
			},
			input: []int32{2, 3, 4},
			want:  []int32{0, 2, 3, 4, 1},
		},
		{
			name: "add bos to empty inputs",
			vocab: &Vocabulary{
				BOS:    []int32{0},
				EOS:    []int32{1},
				AddBOS: true,
				AddEOS: false,
			},
			input: []int32{},
			want:  []int32{0},
		},
	}

	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.vocab.addSpecials(tt.input)
			if diff := cmp.Diff(tt.want, got); diff != "" {
				t.Errorf("no match (-want +got):\n%s", diff)
			}
		})
	}
}
