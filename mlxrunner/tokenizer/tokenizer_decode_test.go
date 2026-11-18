package tokenizer

import "testing"

func TestDecodeAddedTokensByteLevel(t *testing.T) {
	tok, err := LoadFromBytes([]byte(`{
		"model":{"type":"BPE","vocab":{"a":0,"Ġ":1,"Ã":2,"©":3},"merges":[]},
		"pre_tokenizer":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":true},
		"decoder":{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":true,"use_regex":true},
		"added_tokens":[
			{"id":4,"content":"〈|EOS|〉","special":true},
			{"id":5,"content":"<｜end▁of▁sentence｜>","special":true},
			{"id":6,"content":"<|im_end|>","special":true},
			{"id":7,"content":"<think>","special":false}
		]
	}`))
	if err != nil {
		t.Fatal(err)
	}
	for _, tt := range []struct {
		ids  []int32
		want string
	}{
		{[]int32{0, 1, 0}, "a a"},
		{[]int32{2, 3}, "é"},
		{[]int32{4}, "〈|EOS|〉"},
		{[]int32{0, 5}, "a<｜end▁of▁sentence｜>"},
		{[]int32{6, 7, 0}, "<|im_end|><think>a"},
	} {
		if got := tok.Decode(tt.ids); got != tt.want {
			t.Errorf("Decode(%v) = %q, want %q", tt.ids, got, tt.want)
		}
	}
}
