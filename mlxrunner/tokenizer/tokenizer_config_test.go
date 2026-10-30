package tokenizer

import (
	"slices"
	"testing"
)

func TestSpecialTokenConfig(t *testing.T) {
	const data = `{
		"model":{"type":"BPE","vocab":{"a":0},"merges":[]},
		"added_tokens":[
			{"id":1,"content":"<generation>"},
			{"id":2,"content":"<model>"},
			{"id":3,"content":"<tokenizer>"},
			{"id":4,"content":"<map>"},
			{"id":5,"content":"<pad>"}
		]
	}`
	for _, tc := range []struct {
		name           string
		config         *TokenizerConfig
		bos, pad       int32
		eos            []int32
		addBOS, addEOS bool
	}{
		{name: "no companion files", bos: -1, pad: -1},
		{
			name: "generation wins including token zero",
			config: &TokenizerConfig{
				GenerationConfigJSON: []byte(`{"bos_token_id":0,"eos_token_id":[1,2]}`),
				ConfigJSON:           []byte(`{"bos_token_id":2,"eos_token_id":2}`),
				TokenizerConfigJSON:  []byte(`{"bos_token":"<tokenizer>","eos_token":"<tokenizer>","pad_token":"<pad>","add_bos_token":false,"add_eos_token":true}`),
				SpecialTokensMapJSON: []byte(`{"bos_token":"<map>","eos_token":"<map>","pad_token":"<map>"}`),
			},
			bos: 0, eos: []int32{1, 2}, pad: 5, addEOS: true,
		},
		{
			name: "model fills only missing IDs",
			config: &TokenizerConfig{
				GenerationConfigJSON: []byte(`{"eos_token_id":1}`),
				ConfigJSON:           []byte(`{"bos_token_id":2,"eos_token_id":2}`),
			},
			bos: 2, eos: []int32{1}, pad: -1,
		},
		{
			name: "tokenizer strings and objects",
			config: &TokenizerConfig{
				TokenizerConfigJSON: []byte(`{"bos_token":{"content":"<tokenizer>"},"eos_token":"<tokenizer>","pad_token":{"content":"<pad>"},"add_bos_token":true,"add_eos_token":true}`),
			},
			bos: 3, eos: []int32{3}, pad: 5, addBOS: true, addEOS: true,
		},
		{
			name: "special token map fallback",
			config: &TokenizerConfig{
				SpecialTokensMapJSON: []byte(`{"bos_token":"<map>","eos_token":{"content":"<map>"},"pad_token":"<pad>"}`),
			},
			bos: 4, eos: []int32{4}, pad: 5,
		},
		{
			name: "map fills only missing tokens",
			config: &TokenizerConfig{
				TokenizerConfigJSON:  []byte(`{"bos_token":"<tokenizer>"}`),
				SpecialTokensMapJSON: []byte(`{"bos_token":"<map>","eos_token":"<map>","pad_token":"<pad>"}`),
			},
			bos: 3, eos: []int32{4}, pad: 5,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			tok, err := LoadFromBytesWithConfig([]byte(data), tc.config)
			if err != nil {
				t.Fatal(err)
			}
			if tok.BOS() != tc.bos || tok.PAD() != tc.pad || !slices.Equal(tok.EOSTokens(), tc.eos) {
				t.Fatalf("BOS/PAD/EOS = %d/%d/%v, want %d/%d/%v", tok.BOS(), tok.PAD(), tok.EOSTokens(), tc.bos, tc.pad, tc.eos)
			}
			if tok.AddBOS() != tc.addBOS || tok.vocab.AddEOS != tc.addEOS {
				t.Errorf("AddBOS/AddEOS = %v/%v, want %v/%v", tok.AddBOS(), tok.vocab.AddEOS, tc.addBOS, tc.addEOS)
			}
			for _, id := range []int32{0, 1, 2, 3, 4, 5, 6} {
				want := slices.Contains(tc.eos, id)
				if got := tok.IsEOS(id); got != want {
					t.Errorf("IsEOS(%d) = %v, want %v", id, got, want)
				}
			}
			for _, input := range []struct {
				text string
				want []int32
			}{
				{"", nil},
				{"a", []int32{0}},
			} {
				for _, addBOS := range []bool{false, true} {
					want := input.want
					if addBOS && tc.bos >= 0 {
						want = append([]int32{tc.bos}, want...)
					}
					if got := tok.Encode(input.text, addBOS); !slices.Equal(got, want) {
						t.Errorf("Encode(%q, %v) = %v, want %v", input.text, addBOS, got, want)
					}
				}
			}
		})
	}
}
