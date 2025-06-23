package model

import "testing"

func TestVocabulary_SpecialVocabulary(t *testing.T) {
	vocab := &Vocabulary{
		Values: []string{"<|startoftext|>", "<|endoftext|>", "<|tool_call_start|>", "<|tool_call_end|>", "hi"},
		Types:  []int32{TOKEN_TYPE_CONTROL, TOKEN_TYPE_CONTROL, TOKEN_TYPE_USER_DEFINED, TOKEN_TYPE_USER_DEFINED, TOKEN_TYPE_NORMAL},
	}

	specialVocab := vocab.SpecialVocabulary()

	if len(specialVocab) != 4 {
		t.Errorf("expected 4 special tokens, got %d", len(specialVocab))
	}
}
